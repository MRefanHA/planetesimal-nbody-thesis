# planetesimal_sim

Tree-code N-body simulation of protoplanet formation from a swarm of 2,000
planetesimals near 1 AU, written for the undergraduate thesis

> **N-Body Simulation of Protoplanet Formation with Nebular Gas Drag and a Giant Planet**
> Muhammad Refan Haisar Abhiyasa, Institut Teknologi Sumatera (ITERA), 2026.

Four scenarios start from identical initial conditions and differ only in the
physics that is switched on:

| Scenario | Gas drag | Giant planet (1 M_J at 1.6 AU) |
|---|---|---|
| S1 `S1_gravity_only` | no | no |
| S2 `S2_gas` | yes | no |
| S3 `S3_planet` | no | yes |
| S4 `S4_gas_planet` | yes | yes |

Two sets of extra runs were made: perfect merging (all four scenarios) and
20 times the nominal gas density (S2 and S4).

## Method in one paragraph

Mutual gravity between planetesimals is computed with a Barnes–Hut tree
([pytreegrav](https://github.com/mikegrudic/pytreegrav), opening angle θ = 0.5);
the star and the giant planet are added by direct summation. Orbits are
advanced with a fixed-step leapfrog (kick–drift–kick, 128 steps per orbit at
the inner edge). Collisions are found with a swept-sphere test over each step,
with radii enlarged by f = 6. In the main runs two bodies merge only when their
impact speed is below their mutual escape speed; in the perfect-merging runs
every collision merges. Gas drag is quadratic in a minimum-mass solar nebula.
Chapter III of the thesis gives the full method; Table 3.1 lists every parameter.

## Install

Python 3.10–3.12 (numba, used by pytreegrav, may not support newer versions yet).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python check_setup.py              # checks files and packages, runs a 2 s simulation
python tests/test_validation.py    # 9 validation tests, about 1 minute
```

## Reproduce the thesis

```bash
# Chapter III: benchmarks (timestep convergence, tree vs direct, opening angle, drag)
python benchmarks.py --quick       # about 40 s
python benchmarks.py               # adds the timestep convergence test, about 6 min

# the four main scenarios; the dry run shows they differ only in gas and planet
python run_scenarios.py --all --dry-run
python run_scenarios.py --all                         # several hours per scenario

# extra runs
python run_scenarios.py --all --merge-criterion always            # perfect merging
python run_scenarios.py --scenario S2_gas --gas-boost 20          # 20x gas density
python run_scenarios.py --scenario S4_gas_planet --gas-boost 20

# Chapter IV figures and tables (give the run folders, see below)
python -m planetesimal_sim.analysis results                           # comparison table
python -m planetesimal_sim.analysis_stacked out --runs RUN_S1 RUN_S2 RUN_S3 RUN_S4
python -m planetesimal_sim.fig_ae_loglog RUN_S1 RUN_S2 RUN_S3 RUN_S4 --out fig_ae
python scripts/supervisor_checks.py RUN_S1 RUN_S2 RUN_S3 RUN_S4 --fig fig_initial

# Chapter II and III schematic figures
python make_thesis_figures.py --outdir figures
```

Each run writes a time-stamped folder `results/YYYY-MM-DD_HHMMSS_<scenario>/`
holding `config.json`, `series.json` (diagnostics every 100 yr),
`collisions.json` (every impact), `initial_conditions.npz`, `snapshots/`,
`checkpoint.npz` and a readable `run_info.txt`. Long runs can be stopped and
continued with `--resume`. `inspect_run.py RUN_FOLDER` looks for sudden jumps
in a finished run and lists the impacts around them.

## Repository layout

```
planetesimal_sim/          the simulation package
  units.py                 unit system (AU, solar mass, yr; G = 4 pi^2) and conversions
  config.py                every parameter (Table 3.1) and the four scenarios
  ic.py                    initial conditions
  orbits.py                orbital elements <-> positions and velocities
  forces.py                star, tree gravity, giant planet, gas drag
  collisions.py            swept-sphere collision detection and merging
  integrator.py            leapfrog driver, diagnostics, checkpoints
  runlog.py                time-stamped run folders and run_info.txt
  analysis.py              comparison table and first-look figures
  analysis_stacked.py      stacked time-series and impact-speed figures (Chapter IV)
  fig_ae_loglog.py         eccentricity vs semi-major axis figures (Chapter IV)
tests/test_validation.py   validation tests
scripts/supervisor_checks.py  extra diagnostics added during the thesis revision
run_scenarios.py           production driver
benchmarks.py              benchmarks B0-B5 (Chapter III)
inspect_run.py             finds jumps in a finished run
check_setup.py             environment check
make_thesis_figures.py     schematic figures of Chapters II-III
logs/                      benchmark and validation output from the thesis machine
```

## Simulation output

The run folders (about 20 MB for the four main runs, more for the extra runs)
are not stored in this repository. They are available from the author on
request.

## Licence

MIT, see `LICENSE`. If you use the code, please cite the thesis (see `CITATION.cff`).
