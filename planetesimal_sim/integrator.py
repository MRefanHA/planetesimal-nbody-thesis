"""
Leapfrog (kick-drift-kick) integrator and run driver.

Step structure, with the collision search placed exactly where Wallace &
Quinn (2019) put it:

    1. half kick        v -> v + a dt/2
    2. collision search over [t, t+dt] using the post-kick velocities
    3. merge any detected collisions
    4. drift            x -> x + v dt
    5. recompute a      (tree + star + planet + drag)
    6. half kick        v -> v + a dt/2

Gas drag is velocity-dependent, so it is not symplectic. Step 6 evaluates the
gravitational part first and then applies the drag using the half-updated
velocity (one fixed-point iteration). Because the drag term is many orders of
magnitude smaller than the stellar term for the body sizes considered here,
this is more than adequate; the code prints the drag stopping time at startup
so the assumption is checkable.
"""

import json
import os
import time
from datetime import datetime

import numpy as np

from . import units as u
from . import forces
from .collisions import resolve_collisions
from .config import SimConfig
from .ic import build_system
from .runlog import RunLog, find_previous_run
from .orbits import state_to_elements


class Simulation:
    def __init__(self, cfg: SimConfig, verbose=True, runlog=None):
        self.cfg = cfg
        self.verbose = verbose
        self.runlog = runlog
        self.state = build_system(cfg)
        self.t = 0.0
        self.n_collisions = 0        # accreted impacts (mergers)
        self.n_rejected = 0          # detected but disruptive (v_rel > v_esc)
        self.collision_log = []
        self.series = []
        self.wall0 = time.perf_counter()
        # Every file this run produces goes in one timestamped folder, so a
        # result in Chapter IV can always be traced to the run that made it.
        # If no RunLog was supplied the folder is STILL stamped, so a second
        # run can never silently overwrite the first, and the missing
        # provenance is announced now rather than discovered months later.
        if runlog is not None:
            self.rundir = runlog.dir
        else:
            stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S")
            self.rundir = os.path.join(cfg.outdir, f"{stamp}_{cfg.name}")
            if verbose:
                print("!! WARNING: Simulation built without a RunLog. "
                      "No run_info.txt,\n"
                      "   console.log, environment.txt or parameters.json "
                      "will be written\n"
                      "   for this run, and --resume will not find it.\n"
                      f"   Output folder: {self.rundir}")
        os.makedirs(self.rundir, exist_ok=True)
        os.makedirs(os.path.join(self.rundir, "snapshots"), exist_ok=True)
        if runlog is not None:
            runlog.save_initial_conditions(self.state)

    # ------------------------------------------------------------------
    def _accel(self):
        return forces.gravitational_accel(self.state, self.cfg)

    def _drag(self, vel):
        return forces.drag_accel(self.state, vel, self.cfg, self.t)

    # ------------------------------------------------------------------
    def step(self, dt):
        cfg, st = self.cfg, self.state

        a_p, a_star, a_planet = self._accel()
        a_p = a_p + self._drag(st["vel"])

        # 1. half kick
        st["vel"] = st["vel"] + 0.5 * dt * a_p
        st["star_vel"] = st["star_vel"] + 0.5 * dt * a_star
        if a_planet is not None:
            st["planet_vel"] = st["planet_vel"] + 0.5 * dt * a_planet

        # 2-3. collisions predicted over the coming drift, then merged
        nmerge, nreject, recs = resolve_collisions(
            st, cfg, dt, st["star_pos"], st["star_vel"])
        self.n_collisions += nmerge
        self.n_rejected += nreject
        if recs:
            for r in recs:
                r["t"] = self.t
            self.collision_log.extend(recs)

        # 4. drift
        st["pos"] = st["pos"] + dt * st["vel"]
        st["star_pos"] = st["star_pos"] + dt * st["star_vel"]
        if st["planet_pos"] is not None:
            st["planet_pos"] = st["planet_pos"] + dt * st["planet_vel"]

        # 5-6. recompute and half kick (drag uses the half-updated velocity)
        a_p, a_star, a_planet = self._accel()
        v_half = st["vel"] + 0.5 * dt * a_p
        st["vel"] = v_half + 0.5 * dt * self._drag(v_half)
        st["star_vel"] = st["star_vel"] + 0.5 * dt * a_star
        if a_planet is not None:
            st["planet_vel"] = st["planet_vel"] + 0.5 * dt * a_planet

        self.t += dt

    # ------------------------------------------------------------------
    def diagnostics(self):
        cfg, st = self.cfg, self.state
        pos_h = st["pos"] - st["star_pos"][None, :]
        vel_h = st["vel"] - st["star_vel"][None, :]
        mass = st["mass"]
        a, e, inc = state_to_elements(pos_h, vel_h, u.G * (cfg.m_star + mass))

        bound = (a > 0) & (a < 50.0)
        ab, eb, ib = a[bound], e[bound], inc[bound]
        wb = mass[bound]

        def rms(x, w):
            return float(np.sqrt(np.average(x ** 2, weights=w))) if len(x) else np.nan

        imax = int(np.argmax(mass))
        v_kep = np.sqrt(u.G * cfg.m_star)
        e_rms = rms(eb, np.ones_like(eb))
        i_rms = rms(ib, np.ones_like(ib))

        return dict(
            t=float(self.t),
            n_bodies=int(len(mass)),
            n_collisions=int(self.n_collisions),
            n_rejected=int(self.n_rejected),
            disruptive_frac=float(
                self.n_rejected / max(1, self.n_rejected + self.n_collisions)),
            m_max=float(mass[imax]),
            m_max_mearth=float(u.msun_to_mearth(mass[imax])),
            m_mean_mearth=float(u.msun_to_mearth(mass.mean())),
            a_of_largest=float(a[imax]),
            e_rms=e_rms,
            i_rms=i_rms,
            e_rms_massweighted=rms(eb, wb),
            a_mean=float(np.mean(ab)) if len(ab) else np.nan,
            v_disp_ms=float(u.auyr_to_ms(np.sqrt(e_rms ** 2 + i_rms ** 2) * v_kep)),
            m_bound_mearth=float(u.msun_to_mearth(mass[bound].sum())),
            n_unbound=int((~bound).sum()),
            wall_s=float(time.perf_counter() - self.wall0),
        )

    # ------------------------------------------------------------------
    # Checkpointing. Needed for any environment where the process can be
    # killed and restarted -- Google Colab session limits, a laptop that
    # sleeps, a shared machine. A checkpoint holds everything required to
    # continue the run bit-for-bit: the full particle state, the clock, the
    # collision tally, and the diagnostics accumulated so far.
    # ------------------------------------------------------------------
    def checkpoint_path(self):
        return os.path.join(self.rundir, "checkpoint.npz")

    def save_checkpoint(self):
        st = self.state
        np.savez(
            self.checkpoint_path(),
            t=self.t, n_collisions=self.n_collisions,
            n_rejected=self.n_rejected,
            pos=st["pos"], vel=st["vel"], mass=st["mass"],
            radius=st["radius"],
            star_pos=st["star_pos"], star_vel=st["star_vel"],
            planet_pos=(st["planet_pos"] if st["planet_pos"] is not None
                        else np.zeros(3)),
            planet_vel=(st["planet_vel"] if st["planet_vel"] is not None
                        else np.zeros(3)),
            has_planet=(st["planet_pos"] is not None),
            series=json.dumps(self.series),
            collision_log=json.dumps(self.collision_log))

    def load_checkpoint(self):
        """Restore from checkpoint. Returns True if one was found."""
        p = self.checkpoint_path()
        if not os.path.exists(p):
            return False
        z = np.load(p, allow_pickle=False)
        st = self.state
        self.t = float(z["t"])
        self.n_collisions = int(z["n_collisions"])
        self.n_rejected = int(z["n_rejected"]) if "n_rejected" in z else 0
        st["pos"] = np.ascontiguousarray(z["pos"])
        st["vel"] = np.ascontiguousarray(z["vel"])
        st["mass"] = np.ascontiguousarray(z["mass"])
        st["radius"] = np.ascontiguousarray(z["radius"])
        st["star_pos"] = z["star_pos"]
        st["star_vel"] = z["star_vel"]
        if bool(z["has_planet"]):
            st["planet_pos"] = z["planet_pos"]
            st["planet_vel"] = z["planet_vel"]
        self.series = json.loads(str(z["series"]))
        self.collision_log = json.loads(str(z["collision_log"]))
        return True

    def dump_snapshot(self, tag=None):
        cfg, st = self.cfg, self.state
        tag = tag or f"t{self.t:09.1f}"
        path = os.path.join(self.rundir, "snapshots", f"snap_{tag}.npz")
        np.savez_compressed(
            path, t=self.t, pos=st["pos"], vel=st["vel"],
            mass=st["mass"], radius=st["radius"],
            star_pos=st["star_pos"], star_vel=st["star_vel"],
            planet_pos=(st["planet_pos"] if st["planet_pos"] is not None
                        else np.zeros(3)),
            planet_on=cfg.planet_on)
        return path

    # ------------------------------------------------------------------
    def run(self, resume=False, checkpoint_every=250.0):
        cfg = self.cfg
        dt = cfg.dt

        resumed = self.load_checkpoint() if resume else False
        if resumed and self.verbose:
            print(f"resumed {cfg.name} from checkpoint at t = {self.t:.1f} yr "
                  f"({len(self.state['mass'])} bodies, "
                  f"{self.n_collisions} collisions so far)")
        if self.t >= cfg.t_end:
            if self.verbose:
                print(f"{cfg.name} already complete at t = {self.t:.1f} yr")
            return self.series

        next_series = self.t + (0.0 if not resumed else cfg.snapshot_every)
        next_dump = self.t + (0.0 if not resumed else cfg.dump_every)
        next_ckpt = self.t + checkpoint_every

        if self.verbose and not resumed:
            print(cfg.summary())
            ts = forces.stopping_time(cfg)
            print(f"merge criterion             : {cfg.merge_criterion}"
                  + (f" (v_rel < {cfg.merge_velocity_factor:g} v_esc)"
                     if cfg.merge_criterion == "velocity" else
                     " (every impact accretes)"))
            print(f"drag stopping time t_s      : {ts:.3e} yr "
                  f"({'NEGLIGIBLE' if ts > 5 * cfg.t_end else 'relevant'} "
                  f"vs t_end = {cfg.t_end:.1e} yr)")
            print("-" * 66)

        while self.t < cfg.t_end:
            if self.t >= next_series:
                d = self.diagnostics()
                self.series.append(d)
                if self.verbose:
                    print(f"t={d['t']:9.1f} yr  N={d['n_bodies']:5d}  "
                          f"acc={d['n_collisions']:5d}  "
                          f"disr={d['n_rejected']:5d}  "
                          f"Mmax={d['m_max_mearth']:.4e} Me  "
                          f"e_rms={d['e_rms']:.3e}  "
                          f"i_rms={d['i_rms']:.3e}  "
                          f"[{d['wall_s']:.0f}s]")
                next_series += cfg.snapshot_every
            if self.t >= next_dump:
                self.dump_snapshot()
                next_dump += cfg.dump_every
            if self.t >= next_ckpt:
                self.save_checkpoint()
                next_ckpt += checkpoint_every

            self.step(min(dt, cfg.t_end - self.t))

            if len(self.state["mass"]) < 2:
                break

        self.series.append(self.diagnostics())
        self.dump_snapshot("final")
        self.save_checkpoint()
        self.save()
        return self.series

    def save(self):
        cfg = self.cfg
        d = self.rundir
        with open(os.path.join(d, "config.json"), "w") as fh:
            json.dump(cfg.to_dict(), fh, indent=2)
        with open(os.path.join(d, "series.json"), "w") as fh:
            json.dump(self.series, fh, indent=2)
        with open(os.path.join(d, "collisions.json"), "w") as fh:
            json.dump(self.collision_log, fh)
