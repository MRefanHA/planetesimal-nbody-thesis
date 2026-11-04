"""
Reproducible benchmarks for Chapter III.

Every number quoted in the thesis methodology chapter should come out of this
script, on the machine that actually ran the production simulations. Run:

    python benchmarks.py            # all benchmarks
    python benchmarks.py --quick    # skip the slow convergence test

Output is plain text, designed to be pasted into a log appendix, plus a
machine-readable benchmarks.json.

The benchmarks are:
  B1  Mass budget check against the published PENTACLE "model R" value
  B2  Barnes-Hut tree vs direct summation: wall-clock scaling with N
  B3  Barnes-Hut accuracy vs opening angle theta  (thesis Eq. 2.4)
  B4  Timestep convergence of the velocity dispersion
  B5  Gas drag stopping time vs body size and orbital radius
"""

import argparse
import json
import platform
import sys
import time
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import replace                                  # noqa: E402
from pytreegrav import Accel                                     # noqa: E402

from planetesimal_sim import units as u                          # noqa: E402
from planetesimal_sim.config import SimConfig                    # noqa: E402
from planetesimal_sim.ic import make_planetesimals               # noqa: E402
from planetesimal_sim.forces import (gravitational_accel,        # noqa: E402
                                     stopping_time)
from planetesimal_sim.integrator import Simulation               # noqa: E402

RESULTS = {}


def header(txt):
    print()
    print("=" * 74)
    print(txt)
    print("=" * 74, flush=True)


# ---------------------------------------------------------------------------
def b0_environment():
    header("B0  Environment")
    import scipy
    import pytreegrav
    env = dict(
        python=sys.version.split()[0],
        platform=platform.platform(),
        processor=platform.processor() or "unknown",
        cpu_count=os.cpu_count(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        pytreegrav=getattr(pytreegrav, "__version__", "unknown"),
    )
    for k, v in env.items():
        print(f"  {k:12s} : {v}")
    RESULTS["environment"] = env


# ---------------------------------------------------------------------------
def b1_mass_budget():
    """Independent check of Eqs. (3.1)-(3.3) against a published value.

    Iwasawa et al. (2017) PENTACLE, Table 1: their "model R" is an annulus
    0.95-1.05 AU with eta_ice = 1 in Sigma = 10 eta_ice (a/AU)^(-3/2) g/cm^2,
    and they quote a total solid mass of 0.236 M_earth. Our unit conversion
    and Eq. (3.2) must reproduce that number independently.
    """
    header("B1  Mass budget: Eqs. (3.1)-(3.3) vs published PENTACLE model R")
    cfg = SimConfig(a_in=0.95, a_out=1.05, sigma1_solid_cgs=10.0,
                    sigma_slope=-1.5)
    ours = u.msun_to_mearth(cfg.m_total_solid)
    published = 0.236
    print(f"  annulus                       : {cfg.a_in} - {cfg.a_out} AU")
    print(f"  Sigma_1                       : {cfg.sigma1_solid_cgs} g/cm^2"
          f"  ( = {cfg.sigma1_solid:.6e} Msun/AU^2 )")
    print(f"  M_tot from Eq. (3.2)          : {ours:.6f} M_earth")
    print(f"  Iwasawa et al. 2017, Table 1  : {published:.3f} M_earth")
    print(f"  relative difference           : "
          f"{abs(ours - published) / published * 100:.2f} %")

    print()
    print("  Consequence of Eqs. (3.2)-(3.3): N and R_p are NOT independent.")
    print(f"  {'N':>7}  {'m_p (Msun)':>12}  {'m_p (M_earth)':>14}  "
          f"{'R_p (km)':>9}  {'h = r_H/a':>10}  {'e_rms = 4h':>11}")
    rows = []
    for n in (500, 1000, 2000, 5000, 10000, 100000, 10**9):
        c = replace(cfg, n_particles=n)
        print(f"  {n:>7}  {c.m_particle:12.4e}  "
              f"{u.msun_to_mearth(c.m_particle):14.4e}  "
              f"{u.au_to_km(c.r_particle):9.1f}  {c.h_reduced:10.3e}  "
              f"{c.e_rms:11.3e}")
        rows.append(dict(N=n, m_p_msun=float(c.m_particle),
                         R_p_km=float(u.au_to_km(c.r_particle)),
                         h=float(c.h_reduced), e_rms=float(c.e_rms)))
    print()
    print("  Note the last row: reaching genuine 1 km planetesimals in this")
    print("  annulus requires N ~ 1e9. The simulated bodies are 100-600 km")
    print("  objects at the runaway/oligarchic boundary, not 1 km bodies.")

    RESULTS["b1_mass_budget"] = dict(
        m_tot_mearth=float(ours), published_mearth=published,
        rel_diff_percent=float(abs(ours - published) / published * 100),
        n_vs_radius=rows)


# ---------------------------------------------------------------------------
def _bench_accel(pos, mass, soft, method, theta, parallel, n_rep):
    Accel(pos, mass, softening=soft, G=u.G, theta=theta,
          method=method, parallel=parallel)      # warm up numba JIT
    t0 = time.perf_counter()
    for _ in range(n_rep):
        Accel(pos, mass, softening=soft, G=u.G, theta=theta,
              method=method, parallel=parallel)
    return (time.perf_counter() - t0) / n_rep * 1e3


def b2_tree_vs_direct():
    """Is the tree code actually faster at the N we intend to run?

    This is the empirical justification for Section 2.5 / 1.1. It must be
    re-run on the production machine, because the crossover N depends on
    core count and JIT behaviour.
    """
    header("B2  Barnes-Hut tree vs direct summation (mutual forces only)")
    print("  Swarm geometry from the real IC generator; theta = 0.5.")
    print(f"  {'N':>7}  {'tree (ms)':>10}  {'brute (ms)':>11}  {'speedup':>8}")
    rows = []
    for n in (500, 1000, 2000, 5000, 10000, 20000):
        cfg = SimConfig(n_particles=n, min_separation_hill=0.0)
        sw = make_planetesimals(cfg)
        pos = np.ascontiguousarray(sw["pos"])
        mass = np.ascontiguousarray(sw["mass"])
        soft = np.ascontiguousarray(
            cfg.softening_frac * cfg.f_radius * sw["radius"])
        n_rep = max(3, int(40000 / n))
        t_tree = _bench_accel(pos, mass, soft, "tree", 0.5,
                              cfg.parallel, n_rep)
        t_brute = _bench_accel(pos, mass, soft, "bruteforce", 0.5,
                               cfg.parallel, max(2, n_rep // 4))
        print(f"  {n:>7}  {t_tree:10.2f}  {t_brute:11.2f}  "
              f"{t_brute / t_tree:7.2f}x")
        rows.append(dict(N=n, tree_ms=t_tree, brute_ms=t_brute,
                         speedup=t_brute / t_tree))
    print()
    print("  If the speedup at your production N is near 1.0, the tree code")
    print("  cannot be justified on cost grounds at that N. Raise N until it")
    print("  can, or state the justification differently in Chapter III.")
    RESULTS["b2_tree_vs_direct"] = rows


# ---------------------------------------------------------------------------
def b3_theta_accuracy():
    """Accuracy of the opening-angle criterion, thesis Eq. (2.4).

    The error is measured on the MUTUAL planetesimal-planetesimal
    acceleration only. Measuring it against the total acceleration would be
    meaningless, because the stellar term is computed exactly by direct
    summation and dominates by ~6 orders of magnitude -- it would make any
    theta look perfect.
    """
    header("B3  Barnes-Hut accuracy vs opening angle theta (Eq. 2.4)")
    cfg = SimConfig(n_particles=2000)
    sw = make_planetesimals(cfg)
    st = dict(pos=sw["pos"], vel=sw["vel"], mass=sw["mass"],
              radius=sw["radius"], star_pos=np.zeros(3),
              star_vel=np.zeros(3), planet_pos=None, planet_vel=None)

    star_only = gravitational_accel(
        dict(st, mass=np.zeros_like(st["mass"])),
        replace(cfg, tree_method="bruteforce"))[0]
    ref = gravitational_accel(
        st, replace(cfg, tree_method="bruteforce"))[0] - star_only
    refmag = np.linalg.norm(ref, axis=1)

    print(f"  N = {cfg.n_particles}, error relative to exact direct summation")
    print(f"  {'theta':>6}  {'median err':>12}  {'90th pct':>11}  "
          f"{'99th pct':>11}  {'max err':>11}  {'wall (ms)':>10}")
    rows = []
    for th in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0):
        c = replace(cfg, theta=th)
        gravitational_accel(st, c)
        t0 = time.perf_counter()
        for _ in range(10):
            ap = gravitational_accel(st, c)[0] - star_only
        wall = (time.perf_counter() - t0) / 10 * 1e3
        err = np.linalg.norm(ap - ref, axis=1) / refmag
        print(f"  {th:6.1f}  {np.median(err):12.3e}  "
              f"{np.percentile(err, 90):11.3e}  "
              f"{np.percentile(err, 99):11.3e}  {err.max():11.3e}  "
              f"{wall:10.1f}")
        rows.append(dict(theta=th, median=float(np.median(err)),
                         p90=float(np.percentile(err, 90)),
                         p99=float(np.percentile(err, 99)),
                         max=float(err.max()), wall_ms=wall))
    RESULTS["b3_theta_accuracy"] = rows


# ---------------------------------------------------------------------------
def b4_timestep_convergence(n_particles=600, t_end=60.0,
                            spo_list=(32, 64, 128, 256, 512)):
    """Timestep convergence of the velocity dispersion.

    A fixed-step leapfrog cannot resolve a close encounter whose crossing
    time is shorter than dt. When that happens the swarm is heated
    NUMERICALLY: e_rms grows for reasons that have nothing to do with viscous
    stirring. The test is simply to halve dt until e_rms(t_end) stops moving.

    Identical seed and identical initial conditions in every row, so the only
    difference between rows is dt.
    """
    header("B4  Timestep convergence of the velocity dispersion")
    cfg0 = SimConfig(n_particles=n_particles)
    r_h = cfg0.h_reduced * 1.0
    v_ran = np.sqrt(cfg0.e_rms ** 2 + cfg0.i_rms ** 2) * np.sqrt(u.G)
    t_enc = r_h / v_ran
    print(f"  N = {n_particles}, t_end = {t_end:g} yr, f = {cfg0.f_radius:g}, "
          f"seed = {cfg0.seed}")
    print(f"  mutual Hill radius r_H          : {r_h:.3e} AU")
    print(f"  random velocity  v_ran          : {u.auyr_to_ms(v_ran):.1f} m/s")
    print(f"  Hill-radius crossing time t_enc : {t_enc:.4f} yr")
    print("  --> dt must be well below t_enc for encounters to be resolved.")
    print()
    print(f"  {'spo':>5}  {'dt (yr)':>9}  {'dt/t_enc':>9}  {'N_end':>6}  "
          f"{'coll':>5}  {'e_rms(0)':>10}  {'e_rms(end)':>11}  "
          f"{'ratio':>6}  {'wall(s)':>8}")
    rows = []
    for spo in spo_list:
        cfg = replace(cfg0, name=f"conv_spo{spo}", t_end=t_end,
                      steps_per_inner_orbit=spo, snapshot_every=1e9,
                      dump_every=1e9, outdir="/tmp/psim_bench")
        sim = Simulation(cfg, verbose=False)
        e0 = sim.diagnostics()["e_rms"]
        sim.run()
        d = sim.series[-1]
        print(f"  {spo:>5}  {cfg.dt:9.5f}  {cfg.dt / t_enc:9.3f}  "
              f"{d['n_bodies']:6d}  {d['n_collisions']:5d}  {e0:10.4e}  "
              f"{d['e_rms']:11.4e}  {d['e_rms'] / e0:6.2f}  "
              f"{d['wall_s']:8.1f}", flush=True)
        rows.append(dict(spo=spo, dt=float(cfg.dt),
                         dt_over_tenc=float(cfg.dt / t_enc),
                         n_end=d["n_bodies"], collisions=d["n_collisions"],
                         e_rms_0=float(e0), e_rms_end=float(d["e_rms"]),
                         ratio=float(d["e_rms"] / e0),
                         wall_s=float(d["wall_s"])))
    print()
    print("  Converged rows are those where 'ratio' stops changing as dt")
    print("  halves. Rows above that are numerically heated, not physical.")
    RESULTS["b4_timestep_convergence"] = rows


# ---------------------------------------------------------------------------
def b5_drag_timescales():
    """Is gas drag even relevant at the body size we are forced to simulate?

    Answer this BEFORE running Scenario 2 for a week. If the stopping time
    greatly exceeds t_end, Scenario 2 is guaranteed to be indistinguishable
    from Scenario 1, and that is a result to be reported rather than a bug.
    """
    header("B5  Gas drag stopping time vs body size and orbital radius")
    cfg = SimConfig(gas_on=True)
    print(f"  rho_gas(1 AU) = {cfg.rho_gas1_cgs:.1e} g/cm^3, "
          f"C_D = {cfg.c_drag:g}, rho_p = {cfg.rho_p_cgs:g} g/cm^3")
    print(f"  eta = {cfg.eta1:g} (a/AU)^(1/2)")
    print()
    print(f"  {'a (AU)':>7}  {'R (km)':>8}  {'m (M_earth)':>12}  "
          f"{'t_stop (yr)':>12}  {'t_stop / t_end':>14}")
    rows = []
    for a_ref in (0.4, 1.0, 5.0):
        c = replace(cfg, a_in=a_ref - 0.05, a_out=a_ref + 0.05)
        for R_km in (1.0, 10.0, 100.0, 438.0):
            R = R_km * u.KM_TO_AU
            m = u.mass_from_radius(R, c.rho_p)
            v_kep = np.sqrt(u.G * c.m_star / a_ref)
            v_rel = np.sqrt(c.e_rms ** 2 + c.i_rms ** 2 +
                            (2 * c.eta1 * np.sqrt(a_ref)) ** 2) * v_kep
            rho = c.rho_gas1 * a_ref ** c.rho_gas_slope
            ts = m / (0.5 * c.c_drag * np.pi * R ** 2 * rho * v_rel)
            print(f"  {a_ref:7.1f}  {R_km:8.1f}  "
                  f"{u.msun_to_mearth(m):12.3e}  {ts:12.3e}  "
                  f"{ts / 2.0e4:14.1f}")
            rows.append(dict(a_AU=a_ref, R_km=R_km,
                             m_mearth=float(u.msun_to_mearth(m)),
                             t_stop_yr=float(ts)))
        print()
    print("  t_stop / t_end >> 1 means drag cannot act within the")
    print("  integration. At 438 km and 1 AU it is ~13, so Scenario 2 with")
    print("  nominal gas is a NULL result by construction. Use gas_boost, or")
    print("  move the annulus inward, or report the null with this table.")
    RESULTS["b5_drag_timescales"] = rows


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the slow timestep convergence benchmark")
    ap.add_argument("--out", default="benchmarks.json")
    args = ap.parse_args()

    t0 = time.perf_counter()
    b0_environment()
    b1_mass_budget()
    b2_tree_vs_direct()
    b3_theta_accuracy()
    if not args.quick:
        b4_timestep_convergence()
    b5_drag_timescales()

    RESULTS["total_wall_s"] = time.perf_counter() - t0
    with open(args.out, "w") as fh:
        json.dump(RESULTS, fh, indent=2)
    header(f"Done in {RESULTS['total_wall_s']:.1f} s. "
           f"Machine-readable results written to {args.out}")


if __name__ == "__main__":
    main()
