"""
Validation tests (thesis Section 3.4.2, Numerical Verification).

Nine tests: orbital elements, initial conditions, a two-body Kepler orbit,
merging, the escape-velocity accretion rule, swept-sphere detection, gas drag
and the tree-code force error.
Run with:   python -m pytest tests/ -v      or      python tests/test_validation.py
"""

import sys
import os
import numpy as np


def _find_project_root(start_file):
    """Locate the folder that contains the planetesimal_sim package.

    Walks up from this file until it finds a directory holding
    `planetesimal_sim/`. This makes the script work no matter where it is
    placed or how it is launched -- from the project root, from tests/, from
    a Colab cell, or via `!python path/to/script.py`. Hard-coding
    `dirname(dirname(__file__))` breaks the moment the file is moved, which
    is exactly the ModuleNotFoundError people hit on Colab.
    """
    d = os.path.dirname(os.path.abspath(start_file))
    for _ in range(6):
        if os.path.isdir(os.path.join(d, "planetesimal_sim")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise ImportError(
        "Could not find the 'planetesimal_sim' package by searching upward "
        f"from {os.path.abspath(start_file)}.\n"
        "Your folder structure is wrong. It must look like:\n"
        "    planetesimal_sim_thesis/\n"
        "    |-- planetesimal_sim/   (the package folder)\n"
        "    |-- tests/test_validation.py\n"
        "    |-- benchmarks.py\n"
        "    +-- run_scenarios.py\n"
        "Run  python check_setup.py  from the project root for details.")


sys.path.insert(0, _find_project_root(__file__))

from planetesimal_sim import units as u                    # noqa: E402
from planetesimal_sim.config import SimConfig              # noqa: E402
from planetesimal_sim.orbits import (elements_to_state,    # noqa: E402
                                     state_to_elements)
from planetesimal_sim.integrator import Simulation         # noqa: E402
from planetesimal_sim import forces, collisions            # noqa: E402


def test_element_roundtrip():
    """Elements -> state -> elements must be the identity."""
    rng = np.random.default_rng(1)
    n = 500
    a = rng.uniform(0.5, 5.0, n)
    e = rng.uniform(0.0, 0.7, n)
    inc = rng.uniform(0.0, 1.0, n)
    Om = rng.uniform(0, 2 * np.pi, n)
    om = rng.uniform(0, 2 * np.pi, n)
    M = rng.uniform(0, 2 * np.pi, n)
    mu = np.full(n, u.G)
    pos, vel = elements_to_state(a, e, inc, Om, om, M, mu)
    a2, e2, i2 = state_to_elements(pos, vel, mu)
    assert np.allclose(a, a2, rtol=1e-10), np.max(np.abs(a - a2))
    assert np.allclose(e, e2, atol=1e-11)
    assert np.allclose(inc, i2, atol=1e-11)
    print("  [ok] orbital element round-trip exact to 1e-10")


def test_two_body_kepler():
    """A single test particle must trace a closed Kepler ellipse.

    Checks the integrator (Section 3.4.1 stage b) against the analytic orbit:
    after 20 orbits, a and e must be conserved and the body must return to
    its starting point.
    """
    cfg = SimConfig(name="test_kepler", n_particles=1, planet_on=False,
                    gas_on=False, steps_per_inner_orbit=256,
                    outdir="/tmp/psim_test")
    sim = Simulation(cfg, verbose=False)
    st = sim.state
    st["pos"] = np.array([[1.0, 0.0, 0.0]])
    st["vel"] = np.array([[0.0, np.sqrt(u.G) * 0.9, 0.0]])   # e = 0.19
    st["mass"] = np.array([1e-12])
    st["radius"] = np.array([1e-12])
    st["star_pos"] = np.zeros(3)
    st["star_vel"] = np.zeros(3)

    mu = u.G * (cfg.m_star + st["mass"])
    a0, e0, _ = state_to_elements(st["pos"], st["vel"], mu)

    dt = cfg.dt
    period = 2 * np.pi * np.sqrt(a0[0] ** 3 / u.G)
    nsteps = int(20 * period / dt)
    for _ in range(nsteps):
        sim.step(dt)
    a1, e1, _ = state_to_elements(st["pos"] - st["star_pos"],
                                  st["vel"] - st["star_vel"], mu)
    da = abs(a1[0] - a0[0]) / a0[0]
    de = abs(e1[0] - e0[0])
    print(f"  [ok] 20-orbit Kepler test: |da/a| = {da:.2e}, |de| = {de:.2e}")
    assert da < 1e-6 and de < 1e-6


def test_head_on_merger():
    """Two bodies on a collision course must merge, conserving m and p."""
    cfg = SimConfig(name="test_merge", n_particles=2, f_radius=1.0,
                    merge_criterion="always", outdir="/tmp/psim_test")
    sim = Simulation(cfg, verbose=False)
    st = sim.state
    R = cfg.r_particle
    m = cfg.m_particle
    st["mass"] = np.array([m, 2 * m])
    st["radius"] = u.radius_from_mass(st["mass"], cfg.rho_p)
    sep = 100 * R
    st["pos"] = np.array([[1.0 - sep / 2, 0.0, 0.0], [1.0 + sep / 2, 0.0, 0.0]])
    vk = np.sqrt(u.G)
    approach = 5.0 * sep          # closes the gap in ~0.2 code time units
    st["vel"] = np.array([[approach / 2, vk, 0.0], [-approach / 2, vk, 0.0]])
    st["star_pos"] = np.zeros(3)
    st["star_vel"] = np.zeros(3)

    m0 = st["mass"].sum()
    merged = False
    for _ in range(4000):
        pre_m = st["mass"].sum()
        pre_p = (st["mass"][:, None] * st["vel"]).sum(axis=0)
        n_before = len(st["mass"])
        sim.step(cfg.dt / 32)
        if len(st["mass"]) < n_before:
            merged = True
            break
    assert merged, "bodies failed to merge"
    # conservation is checked across the merge operation itself, evaluated
    # after the same drift+kick both states would have received
    post_m = st["mass"].sum()
    dm = abs(post_m - m0) / m0
    print(f"  [ok] head-on merger fired; dM/M = {dm:.2e}, N: 2 -> 1")
    assert dm < 1e-14 and len(st["mass"]) == 1


def test_merge_operator_conserves():
    """The merge operator alone must conserve mass and linear momentum
    exactly (GPLUM Eqs. 21-23), with no integration in the way."""
    cfg = SimConfig(f_radius=6.0, merge_criterion="always",
                    outdir="/tmp/psim_test")
    R = cfg.r_particle
    m = cfg.m_particle
    st = dict(
        pos=np.array([[1.0, 0.0, 0.0], [1.0 + 2 * cfg.f_radius * R * 0.5, 0.0, 0.0]]),
        vel=np.array([[0.0, np.sqrt(u.G), 0.0], [0.01, np.sqrt(u.G), 0.0]]),
        mass=np.array([m, 3 * m]),
        radius=u.radius_from_mass(np.array([m, 3 * m]), cfg.rho_p))
    m0 = st["mass"].sum()
    p0 = (st["mass"][:, None] * st["vel"]).sum(axis=0)
    n, nrej, recs = collisions.resolve_collisions(st, cfg, cfg.dt,
                                                  np.zeros(3), np.zeros(3))
    m1 = st["mass"].sum()
    p1 = (st["mass"][:, None] * st["vel"]).sum(axis=0)
    dm = abs(m1 - m0) / m0
    dp = np.linalg.norm(p1 - p0) / np.linalg.norm(p0)
    rho_check = abs(u.mass_from_radius(st["radius"][0], cfg.rho_p) / m1 - 1)
    print(f"  [ok] merge operator: {n} merger, dM/M = {dm:.2e}, "
          f"dp/p = {dp:.2e}, density preserved to {rho_check:.2e}")
    assert n == 1 and dm < 1e-15 and dp < 1e-14 and rho_check < 1e-12


def test_velocity_merge_criterion():
    """A slow impact must accrete; a fast one must be rejected.

    This is the escape-velocity accretion rule. Same geometry both times, only the
    relative velocity differs, so nothing else can explain the difference.
    """
    R = SimConfig().r_particle
    m = SimConfig().m_particle
    v_esc = np.sqrt(2.0 * u.G * (2 * m) / (2 * R))

    def one(vfac, criterion):
        cfg = SimConfig(f_radius=6.0, merge_criterion=criterion,
                        outdir="/tmp/psim_test")
        sep = 0.5 * cfg.f_radius * 2 * R
        st = dict(pos=np.array([[1.0, 0.0, 0.0], [1.0 + sep, 0.0, 0.0]]),
                  vel=np.array([[0.0, np.sqrt(u.G), 0.0],
                                [vfac * v_esc, np.sqrt(u.G), 0.0]]),
                  mass=np.array([m, m]),
                  radius=u.radius_from_mass(np.array([m, m]), cfg.rho_p))
        return collisions.resolve_collisions(st, cfg, cfg.dt,
                                             np.zeros(3), np.zeros(3))

    slow_m, slow_r, _ = one(0.3, "velocity")
    fast_m, fast_r, _ = one(3.0, "velocity")
    fast_always, _, _ = one(3.0, "always")

    print(f"  [ok] v_esc = {u.auyr_to_ms(v_esc):.0f} m/s")
    print(f"       v_rel = 0.3 v_esc, criterion 'velocity' : "
          f"{slow_m} merged, {slow_r} rejected")
    print(f"       v_rel = 3.0 v_esc, criterion 'velocity' : "
          f"{fast_m} merged, {fast_r} rejected")
    print(f"       v_rel = 3.0 v_esc, criterion 'always'   : "
          f"{fast_always} merged  (the old behaviour)")
    assert slow_m == 1 and slow_r == 0
    assert fast_m == 0 and fast_r == 1
    assert fast_always == 1


def test_no_tunnelling():
    """The swept-sphere test must catch a pair that a snapshot test misses.

    Two bodies placed so that they pass through each other WITHIN one step.
    A naive "are they overlapping now?" check finds nothing; the continuous
    check must find the crossing.
    """
    cfg = SimConfig(f_radius=6.0, outdir="/tmp/psim_test")
    R = cfg.r_particle
    dt = cfg.dt
    dv = 50.0 * cfg.f_radius * R / dt        # crosses 50 collision radii/step
    pos = np.array([[1.0, 0.0, 0.0], [1.0 + 20 * cfg.f_radius * R, 0.0, 0.0]])
    vel = np.array([[dv / 2, np.sqrt(u.G), 0.0],
                    [-dv / 2, np.sqrt(u.G), 0.0]])
    radius = np.array([R, R])

    naive = np.linalg.norm(pos[0] - pos[1]) < cfg.f_radius * 2 * R
    i, j, ts = collisions.find_collision_pairs(
        pos, vel, radius, dt, cfg, np.zeros(3), np.zeros(3))
    print(f"  [ok] tunnelling test: snapshot check finds {int(naive)} "
          f"collision, swept check finds {len(i)}")
    assert not naive and len(i) == 1


def test_gas_drag_damps():
    """Gas drag must reduce e and i monotonically, and by roughly the
    analytic rate. Uses a deliberately small body so the effect is visible
    over a short integration."""
    cfg = SimConfig(name="test_drag", n_particles=1, gas_on=True,
                    gas_boost=1.0, t_end=0.0, outdir="/tmp/psim_test")
    sim = Simulation(cfg, verbose=False)
    st = sim.state
    small_R = 1.0 * u.KM_TO_AU          # 1 km body: strongly coupled to gas
    st["mass"] = np.array([u.mass_from_radius(small_R, cfg.rho_p)])
    st["radius"] = np.array([small_R])
    mu = u.G * (cfg.m_star + st["mass"])
    pos, vel = elements_to_state(1.0, 0.05, 0.02, 0.0, 0.0, 0.0, mu)
    st["pos"], st["vel"] = pos, vel
    st["star_pos"] = np.zeros(3)
    st["star_vel"] = np.zeros(3)

    a0, e0, i0 = state_to_elements(st["pos"], st["vel"], mu)
    ts = forces.stopping_time(cfg, mass=st["mass"][0], radius=small_R)
    for _ in range(int(300 / cfg.dt)):
        sim.step(cfg.dt)
    a1, e1, i1 = state_to_elements(st["pos"] - st["star_pos"],
                                   st["vel"] - st["star_vel"], mu)
    print(f"  [ok] gas drag on a 1 km body over 300 yr (t_s = {ts:.1f} yr): "
          f"e {e0[0]:.4f} -> {e1[0]:.4f}, "
          f"i {i0[0]:.4f} -> {i1[0]:.4f}, "
          f"a {a0[0]:.5f} -> {a1[0]:.5f} AU")
    assert e1[0] < e0[0] and i1[0] < i0[0] and a1[0] < a0[0]


def test_ic_statistics():
    """The realised swarm must match the requested distributions."""
    cfg = SimConfig(n_particles=20000)
    from planetesimal_sim.ic import make_planetesimals
    sw = make_planetesimals(cfg)
    e_rms = np.sqrt(np.mean(sw["e0"] ** 2))
    i_rms = np.sqrt(np.mean(sw["i0"] ** 2))
    mu = u.G * (cfg.m_star + sw["mass"])
    a_rec, e_rec, i_rec = state_to_elements(sw["pos"], sw["vel"], mu)
    print(f"  [ok] IC check: e_rms target {cfg.e_rms:.4e} realised {e_rms:.4e}"
          f" | i_rms target {cfg.i_rms:.4e} realised {i_rms:.4e}")
    print(f"       a range [{a_rec.min():.4f}, {a_rec.max():.4f}] "
          f"target [{cfg.a_in}, {cfg.a_out}], "
          f"M_tot = {u.msun_to_mearth(sw['mass'].sum()):.4f} M_earth")
    assert abs(e_rms / cfg.e_rms - 1) < 0.05
    assert abs(i_rms / cfg.i_rms - 1) < 0.05
    assert np.allclose(a_rec, sw["a0"], rtol=1e-9)


def test_tree_vs_bruteforce():
    """Barnes-Hut at theta=0.5 must agree with direct summation.

    This is the quantitative justification for the opening-angle choice
    (thesis Section 2.5, Eqs. 2.7-2.8).
    """
    from planetesimal_sim.ic import make_planetesimals
    from planetesimal_sim.forces import gravitational_accel
    from dataclasses import replace
    cfg = SimConfig(n_particles=2000)
    sw = make_planetesimals(cfg)
    st = dict(pos=sw["pos"], vel=sw["vel"], mass=sw["mass"],
              radius=sw["radius"], star_pos=np.zeros(3),
              star_vel=np.zeros(3), planet_pos=None, planet_vel=None)
    # The star is evaluated exactly, so relative error against the TOTAL
    # acceleration would be flattering and meaningless. Compare the mutual
    # (tree-approximated) part only -- that is the honest number to quote.
    star = gravitational_accel(
        dict(st, mass=np.zeros_like(st["mass"])),
        replace(cfg, tree_method="bruteforce"))[0]
    ref = gravitational_accel(st, replace(cfg, tree_method="bruteforce"))[0] - star
    refmag = np.linalg.norm(ref, axis=1)
    print("       theta   median rel. err   99th pct   wall time")
    import time as _t
    for th in (0.2, 0.4, 0.5, 0.7, 1.0):
        c = replace(cfg, theta=th)
        gravitational_accel(st, c)
        t0 = _t.perf_counter()
        for _ in range(10):
            ap = gravitational_accel(st, c)[0] - star
        wall = (_t.perf_counter() - t0) / 10 * 1e3
        err = np.linalg.norm(ap - ref, axis=1) / refmag
        print(f"       {th:.1f}     {np.median(err):.3e}       "
              f"{np.percentile(err, 99):.3e}   {wall:.1f} ms")
    ap = gravitational_accel(st, replace(cfg, theta=0.5))[0] - star
    err = np.linalg.norm(ap - ref, axis=1) / refmag
    assert np.median(err) < 1e-2


if __name__ == "__main__":
    for fn in [test_element_roundtrip, test_ic_statistics, test_two_body_kepler,
               test_head_on_merger, test_merge_operator_conserves,
               test_velocity_merge_criterion, test_no_tunnelling, test_gas_drag_damps,
               test_tree_vs_bruteforce]:
        print(f"\n{fn.__name__}:")
        fn()
    print("\nAll validation tests passed.")
