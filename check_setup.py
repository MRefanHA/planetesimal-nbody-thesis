#!/usr/bin/env python3
"""
Setup checker. Run this FIRST, before anything else:

    python check_setup.py

It verifies, in order:
  1. Python version
  2. That every file is where it should be
  3. That the required packages are installed
  4. That the package imports correctly
  5. That a 2-second mini-simulation actually runs

Every failure prints the exact command or action that fixes it. Nothing here
takes more than a few seconds, and it will save you from debugging a
2,700,000-step production run that was never going to work.

This file must stay in the PROJECT ROOT, next to run_scenarios.py.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

GREEN = "\033[92m" if sys.stdout.isatty() else ""
RED = "\033[91m" if sys.stdout.isatty() else ""
YELLOW = "\033[93m" if sys.stdout.isatty() else ""
RESET = "\033[0m" if sys.stdout.isatty() else ""

failures = []
warnings = []


def ok(msg):
    print(f"  {GREEN}[ OK ]{RESET} {msg}")


def fail(msg, fix):
    print(f"  {RED}[FAIL]{RESET} {msg}")
    print(f"         fix: {fix}")
    failures.append(msg)


def warn(msg, note):
    print(f"  {YELLOW}[warn]{RESET} {msg}")
    print(f"         {note}")
    warnings.append(msg)


def section(title):
    print()
    print(title)
    print("-" * len(title))


# ---------------------------------------------------------------------------
section("1. Python version")

v = sys.version_info
if v < (3, 9):
    fail(f"Python {v.major}.{v.minor} is too old",
         "install Python 3.9 or newer from python.org")
elif v >= (3, 14):
    warn(f"Python {v.major}.{v.minor} is very new",
         "numba may not support it yet; 3.10-3.12 are the safest choices")
else:
    ok(f"Python {v.major}.{v.minor}.{v.micro}")
print(f"         interpreter: {sys.executable}")

# ---------------------------------------------------------------------------
section("2. Folder structure")

REQUIRED = [
    "run_scenarios.py",
    "benchmarks.py",
    "requirements.txt",
    "planetesimal_sim/__init__.py",
    "planetesimal_sim/units.py",
    "planetesimal_sim/config.py",
    "planetesimal_sim/orbits.py",
    "planetesimal_sim/ic.py",
    "planetesimal_sim/forces.py",
    "planetesimal_sim/collisions.py",
    "planetesimal_sim/integrator.py",
    "planetesimal_sim/analysis.py",
    "tests/test_validation.py",
]
OPTIONAL = [
    ".vscode/settings.json",
    ".vscode/launch.json",
    "README.md",
    "planetesimal_sim_colab.ipynb",
]

missing = [p for p in REQUIRED if not os.path.exists(os.path.join(ROOT, p))]
if missing:
    for p in missing:
        fail(f"missing: {p}",
             f"download it and place it at {os.path.join(ROOT, p)}")
    print()
    print("  The folder must look EXACTLY like this:")
    print("""
    planetesimal_sim_thesis/
    |-- .vscode/
    |   |-- settings.json
    |   +-- launch.json
    |-- planetesimal_sim/
    |   |-- __init__.py        <- this one is easy to forget
    |   |-- units.py
    |   |-- config.py
    |   |-- orbits.py
    |   |-- ic.py
    |   |-- forces.py
    |   |-- collisions.py
    |   |-- integrator.py
    |   +-- analysis.py
    |-- tests/
    |   +-- test_validation.py <- must be INSIDE tests/, not in the root
    |-- benchmarks.py
    |-- check_setup.py
    |-- run_scenarios.py
    +-- requirements.txt
    """)
else:
    ok(f"all {len(REQUIRED)} required files present")

for p in OPTIONAL:
    if not os.path.exists(os.path.join(ROOT, p)):
        warn(f"optional file missing: {p}",
             "not fatal, but VS Code settings / Colab notebook won't be there")

stray = os.path.join(ROOT, "test_validation.py")
if os.path.exists(stray):
    fail("test_validation.py is in the project root",
         "move it into the tests/ folder. It computes its import path by "
         "going up two directories, so it only works from tests/")

# ---------------------------------------------------------------------------
section("3. Required packages")

NEEDED = {
    "numpy": "1.24",
    "scipy": "1.9",
    "matplotlib": "3.6",
    "pytreegrav": "1.0",
}
for pkg, minver in NEEDED.items():
    try:
        mod = __import__(pkg)
        got = getattr(mod, "__version__", "unknown")
        ok(f"{pkg} {got}")
    except ImportError:
        fail(f"{pkg} is not installed",
             f'run:  "{sys.executable}" -m pip install -r requirements.txt')

try:
    import numba
    ok(f"numba {numba.__version__} (pulled in by pytreegrav)")
except ImportError:
    pass

# ---------------------------------------------------------------------------
section("4. Package imports")

sys.path.insert(0, ROOT)
try:
    from planetesimal_sim import SimConfig, Simulation, units      # noqa
    ok("from planetesimal_sim import SimConfig, Simulation")
except Exception as exc:
    fail(f"cannot import the package: {type(exc).__name__}: {exc}",
         "check that planetesimal_sim/__init__.py exists and that you are "
         "running this from the project root")

# ---------------------------------------------------------------------------
section("5. Physics smoke test")

if not failures:
    try:
        from dataclasses import replace
        import tempfile
        import time as _t

        cfg = SimConfig(name="setup_check", n_particles=120, t_end=2.0,
                        steps_per_inner_orbit=64, snapshot_every=1e9,
                        dump_every=1e9,
                        outdir=os.path.join(tempfile.gettempdir(), "psim_chk"))

        print(f"         M_tot in annulus : "
              f"{units.msun_to_mearth(cfg.m_total_solid):.4f} M_earth "
              f"(should be ~0.2355)")
        print(f"         particle radius  : "
              f"{units.au_to_km(cfg.r_particle):.0f} km")

        t0 = _t.perf_counter()
        sim = Simulation(cfg, verbose=False)
        d0 = sim.diagnostics()
        sim.run()
        d1 = sim.series[-1]
        wall = _t.perf_counter() - t0
        steps = int(cfg.t_end / cfg.dt)
        ok(f"integrated {steps} steps in {wall:.1f} s "
           f"(most of that is numba compiling, once)")
        ok(f"N: {d0['n_bodies']} -> {d1['n_bodies']}, "
           f"e_rms: {d0['e_rms']:.3e} -> {d1['e_rms']:.3e}")

        # Time steps AFTER compilation, otherwise the estimate is meaningless.
        t1 = _t.perf_counter()
        for _ in range(30):
            sim.step(cfg.dt)
        ms = (_t.perf_counter() - t1) / 30 * 1e3
        ok(f"warm speed: {ms:.2f} ms/step at N={cfg.n_particles}")

        mass_err = abs(sum(sim.state["mass"]) / (cfg.m_particle * 120) - 1)
        if mass_err < 1e-12:
            ok(f"mass conserved to {mass_err:.1e}")
        else:
            fail(f"mass NOT conserved ({mass_err:.1e})",
                 "something is badly wrong; re-download the package")

        # Cost is roughly linear-ish in N over this range (tree is N log N,
        # neighbour search is N). Scale from the measured warm speed.
        scale = (2000 / cfg.n_particles)
        ms2000 = ms * scale
        hours = 2764742 * ms2000 / 3.6e6
        print()
        print("         Very rough production estimate, N=2000, 128 spo,")
        print("         t_end=2e4 yr  (2,764,742 steps):")
        print(f"           ~{ms2000:.1f} ms/step  ->  ~{hours:.1f} h "
              f"per scenario, ~{4*hours:.0f} h for all four")
        if hours > 24:
            print(f"         {YELLOW}That is a lot.{RESET} Reduce --t-end, "
                  f"raise --f, or lower --n.")
        print("         Run benchmarks.py for the number you should actually")
        print("         quote in Chapter III.")
    except Exception as exc:
        import traceback
        fail(f"smoke test crashed: {type(exc).__name__}: {exc}",
             "paste the traceback below when asking for help")
        traceback.print_exc()
else:
    print("  skipped (fix the failures above first)")

# ---------------------------------------------------------------------------
print()
print("=" * 70)
if failures:
    print(f"{RED}{len(failures)} problem(s) found.{RESET} "
          f"Fix them in order, then run this again.")
    sys.exit(1)
else:
    print(f"{GREEN}Setup is good.{RESET}")
    if warnings:
        print(f"({len(warnings)} warning(s) above, none fatal.)")
    print()
    print("Next steps, in order:")
    print("  1.  python tests/test_validation.py     (~1 min)")
    print("  2.  python benchmarks.py --quick        (~40 s)")
    print("  3.  python benchmarks.py                (~6 min, full B4 table)")
    print("  4.  python run_scenarios.py --all --dry-run")
print("=" * 70)
