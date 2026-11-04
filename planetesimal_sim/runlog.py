"""
Per-run logging (thesis Section 3.5, stage 7).

Every production run writes a self-contained, timestamped folder so that any
result in Chapter IV can be traced back to the exact configuration, machine
and command that produced it. Nothing in a run folder depends on anything
outside it.

Folder name
-----------
    YYYY-MM-DD_HHMMSS_<scenario>

for example  2026-09-12_143022_S1_gravity_only

Folder contents
---------------
    run_info.txt            human-readable summary: start, end, duration,
                            machine, every parameter, and the final state
    parameters.json         the full configuration, machine-readable
    environment.txt         Python and package versions, CPU, OS
    command.txt            the exact command line that was run
    console.log             everything printed during the run
    initial_conditions.npz  the seeded swarm as generated, before any step
    series.json             diagnostic time series
    collisions.json         every impact, accreted or disruptive
    checkpoint.npz          latest checkpoint, for --resume
    snapshots/              periodic full particle dumps

Why this matters for the thesis
-------------------------------
A number in Chapter IV is only defensible if you can say which run produced
it. With this scheme the answer is always a folder name, and that folder
contains the parameters, the machine, the wall-clock time and the console
output. If an examiner asks "what timestep produced Figure 4.3?", the answer
is in run_info.txt of the folder named in the caption.
"""

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timedelta

import numpy as np


def _human(seconds):
    """Seconds as a readable duration, e.g. '8 h 34 min 12 s'."""
    td = timedelta(seconds=int(seconds))
    h, rem = divmod(int(td.total_seconds()), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} h {m} min {s} s"
    if m:
        return f"{m} min {s} s"
    return f"{s} s"


class Tee:
    """Duplicate stdout to a file, so console.log matches what you saw."""

    def __init__(self, path, stream):
        self.file = open(path, "a", encoding="utf-8", buffering=1)
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


def _environment():
    """Everything about the machine that could affect a result."""
    info = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count_logical": os.cpu_count(),
    }
    for pkg in ("numpy", "scipy", "matplotlib", "pytreegrav", "numba"):
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[pkg] = "not installed"
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).decode().strip()
    except Exception:
        info["git_commit"] = "not a git repository"
    return info


def find_previous_run(outdir, scenario):
    """Most recent run folder for this scenario that holds a checkpoint.

    Used by --resume so that continuing a run appends to the folder it
    started in, rather than scattering one simulation across several.
    """
    if not os.path.isdir(outdir):
        return None
    cands = []
    for name in os.listdir(outdir):
        p = os.path.join(outdir, name)
        if (os.path.isdir(p) and name.endswith("_" + scenario)
                and os.path.exists(os.path.join(p, "checkpoint.npz"))):
            cands.append(name)
    if not cands:
        return None
    return os.path.join(outdir, sorted(cands)[-1])


class RunLog:
    """Creates and owns one timestamped run folder."""

    def __init__(self, cfg, argv=None, resume_dir=None, verbose=True):
        self.cfg = cfg
        self.t_wall_start = time.perf_counter()
        self.t_clock_start = datetime.now().astimezone()
        self.resumed = resume_dir is not None
        self.events = []

        if resume_dir:
            self.dir = resume_dir
        else:
            stamp = self.t_clock_start.strftime("%Y-%m-%d_%H%M%S")
            self.dir = os.path.join(cfg.outdir, f"{stamp}_{cfg.name}")
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(os.path.join(self.dir, "snapshots"), exist_ok=True)

        self._tee = Tee(os.path.join(self.dir, "console.log"), sys.stdout)
        sys.stdout = self._tee

        self._write_static(argv)
        if verbose:
            tag = "RESUMED" if self.resumed else "NEW RUN"
            print(f"[{tag}]  log folder: {self.dir}")
        self.log_event("resumed" if self.resumed else "started")

    # ------------------------------------------------------------------
    def path(self, *parts):
        return os.path.join(self.dir, *parts)

    def log_event(self, text):
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        self.events.append(f"{stamp}   {text}")

    # ------------------------------------------------------------------
    def _write_static(self, argv):
        with open(self.path("command.txt"), "a", encoding="utf-8") as fh:
            fh.write(datetime.now().astimezone().isoformat(timespec="seconds")
                     + "\n")
            fh.write("cwd: " + os.getcwd() + "\n")
            fh.write(" ".join(argv or sys.argv) + "\n\n")

        env = _environment()
        with open(self.path("environment.txt"), "w", encoding="utf-8") as fh:
            fh.write("MACHINE AND SOFTWARE ENVIRONMENT\n")
            fh.write("=" * 58 + "\n")
            for k, v in env.items():
                fh.write(f"{k:22s} : {v}\n")

        with open(self.path("parameters.json"), "w", encoding="utf-8") as fh:
            json.dump({"config": self.cfg.to_dict(), "environment": env},
                      fh, indent=2)

    def save_initial_conditions(self, state):
        """The seeded swarm exactly as generated, before any integration."""
        np.savez_compressed(
            self.path("initial_conditions.npz"),
            pos=state["pos"], vel=state["vel"], mass=state["mass"],
            radius=state["radius"], a0=state["a0"], e0=state["e0"],
            i0=state["i0"],
            planet_pos=(state["planet_pos"] if state["planet_pos"] is not None
                        else np.zeros(3)),
            planet_on=self.cfg.planet_on)

    # ------------------------------------------------------------------
    def finalize(self, series, n_collisions=0, n_rejected=0, status="completed"):
        wall = time.perf_counter() - self.t_wall_start
        end = datetime.now().astimezone()
        self.log_event(f"{status} after {_human(wall)}")
        last = series[-1] if series else {}

        cfg = self.cfg
        steps = int(last.get("t", 0.0) / cfg.dt) if cfg.dt else 0
        ms = (wall / steps * 1e3) if steps else float("nan")

        L = []
        L.append("SIMULATION RUN RECORD")
        L.append("=" * 62)
        L.append("")
        L.append("[ WHEN ]")
        L.append(f"  started          : {self.t_clock_start.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        L.append(f"  ended            : {end.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        L.append(f"  wall-clock time  : {_human(wall)}   ({wall:.1f} s)")
        L.append(f"  status           : {status}")
        L.append(f"  resumed run      : {'yes' if self.resumed else 'no'}")
        L.append(f"  run folder       : {os.path.abspath(self.dir)}")
        L.append("")
        L.append("[ SPEED ]")
        L.append(f"  integration steps: {steps:,}")
        L.append(f"  per step         : {ms:.2f} ms")
        L.append(f"  simulated / real : {last.get('t', 0.0):.1f} yr in {_human(wall)}")
        L.append("")
        L.append("[ INPUT: configuration ]")
        for line in cfg.summary().split("\n"):
            L.append("  " + line)
        L.append("")
        L.append("[ INPUT: derived quantities ]")
        d = cfg.to_dict()
        for k in ("m_total_solid_mearth", "m_particle_msun", "r_particle_km",
                  "h_reduced", "e_rms", "i_rms", "dt_yr", "n_steps"):
            if k in d:
                L.append(f"  {k:24s} : {d[k]}")
        L.append("")
        L.append("[ OUTPUT: final state ]")
        for k, label in (("t", "simulated time (yr)"),
                         ("n_bodies", "bodies remaining"),
                         ("n_collisions", "impacts accreted"),
                         ("n_rejected", "impacts rejected (disruptive)"),
                         ("disruptive_frac", "disruptive fraction"),
                         ("m_max_mearth", "largest mass (M_earth)"),
                         ("e_rms", "e_rms"), ("i_rms", "i_rms"),
                         ("v_disp_ms", "velocity dispersion (m/s)"),
                         ("n_unbound", "unbound bodies")):
            if k in last:
                v = last[k]
                L.append(f"  {label:32s} : "
                         + (f"{v:.6g}" if isinstance(v, float) else str(v)))
        L.append("")
        L.append("[ FILES IN THIS FOLDER ]")
        for f in sorted(os.listdir(self.dir)):
            fp = self.path(f)
            if os.path.isdir(fp):
                n = len(os.listdir(fp))
                L.append(f"  {f + '/':26s} {n} file(s)")
            else:
                L.append(f"  {f:26s} {os.path.getsize(fp) / 1024:8.1f} KB")
        L.append("")
        L.append("[ EVENT LOG ]")
        L.extend("  " + e for e in self.events)
        L.append("")
        L.append("=" * 62)
        L.append("Cite this run in the thesis as: " + os.path.basename(self.dir))

        with open(self.path("run_info.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(L) + "\n")

        self.close()
        return self.dir

    def close(self):
        if getattr(self, "_tee", None) is not None:
            sys.stdout = self._tee.stream
            self._tee.close()
            self._tee = None
