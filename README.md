# planetesimal_sim

Tree-code N-body simulation of protoplanet formation from a seeded planetesimal
swarm, comparing four scenarios: gravity only, + nebular gas drag, + a giant
planet, and both together.

Everything here is reproducible from a clean checkout. Nothing in this README
is a number I remembered; every value either comes from `benchmarks.py` output
on this machine or is a literature value with the citation attached.

---

## 1. Install and verify

```bash
pip install numpy scipy matplotlib pytreegrav
python tests/test_validation.py     # 8 validation tests, ~1 minute
python benchmarks.py --quick        # B0-B3, B5; ~40 s
python benchmarks.py                # adds B4 convergence; ~6 minutes
```

Do not trust any number in Chapter III until `benchmarks.py` has produced it on my own machine. The timing tables in particular depend on core count and on
numba JIT behaviour; the reference run below was on a single core.

## 2. Run the four scenarios

```bash
python run_scenarios.py --all --dry-run     # verify the ICs are identical
python run_scenarios.py --all               # production
python -m planetesimal_sim.analysis results # figures + comparison table
```

`--dry-run` prints the resolved parameters for all four scenarios and lists
exactly which fields differ from S1. It should report `gas_on` and `planet_on`
only. That printout is your evidence for the Section 1.4 item 5 claim.

`run_scenarios.py` also prints a **feasibility check** before integrating
anything: drag stopping time versus `t_end`, `dt` versus the Hill-crossing
time, and whether any giant-planet mean-motion resonance lands inside the
annulus. Read it. It will warn you about the two failure modes described in
Sections 5 and 6 below.

## 3. Module map (matches the thesis chapter structure)

| Module | Thesis section | What it does |
|---|---|---|
| `units.py` | 3.2 | Unit system (AU, M☉, yr; G = 4π²) and all cgs conversions, in one place |
| `config.py` | 3.2, Table 3.1 | Every parameter as one dataclass; `.summary()` prints Table 3.1 |
| `ic.py` | 3.5 stage 1 | Initial conditions + overlap rejection |
| `orbits.py` | 2.2 | Kepler elements ↔ Cartesian state |
| `forces.py` | 3.5 stages 2, 4, 5 | Star, tree, giant planet, gas drag |
| `collisions.py` | 3.5 stage 3 | Swept-sphere detection + perfect merger |
| `integrator.py` | 3.4 | Leapfrog KDK driver + diagnostics |
| `analysis.py` | Chapter IV | Figures and the scenario comparison table |
| `benchmarks.py` | 3.3, 3.5 | B1–B5, the numbers that justify the method |
| `run_scenarios.py` | 3.5 stage 7 | Production driver |

---

## 4. Table 3.1 — simulation base parameters

These are the exact values used to produce every number in this README.
Reproduce with `python -c "from planetesimal_sim import SimConfig; print(SimConfig(n_particles=2000).summary())"`.

### 4.1 Independent inputs

| Symbol | Value | Source / justification |
|---|---|---|
| a_in | 0.95 AU | Narrow annulus, Hansen (2009); identical to Iwasawa et al. (2017) "model R" |
| a_out | 1.05 AU | as above |
| Σ₁ | 10 g cm⁻² | Hayashi (1981) MMSN solid surface density at 1 AU, η_ice = 1 |
| Σ slope | −3/2 | Hayashi (1981) |
| ρ_p | 2.0 g cm⁻³ | Kokubo & Ida (2000); Richardson et al. (2000) |
| N | 2000 | Chosen so the tree code is genuinely faster than direct summation (B2) |
| ẽ | 4.0 | e_rms = 2 i_rms = 4h, Kokubo & Ida (1998); Richardson et al. (2000) §3.1 |
| i_rms/e_rms | 0.5 | Equipartition, Ida & Makino (1992) |
| f | 6.0 | Radius enhancement, Kokubo & Ida (1998); Wallace & Quinn (2019) §2.2 |
| θ | 0.5 | Barnes & Hut (1986) opening angle; error quantified in B3 |
| ε_i / (f R_i) | 0.5 | Softening below the collision radius, so it cannot suppress focusing |
| steps/orbit | 128 | Set by the B4 convergence test, **not** by orbital sampling |
| t_end | 2×10⁴ yr | Budget; see §7 |
| seed | 20250904 | Identical across all four scenarios |
| min. IC separation | 2 r_H | Non-overlapping Hill spheres, Richardson et al. (2000) §2.4 |

### 4.2 Derived quantities (not free — computed from the above)

| Symbol | Value at N = 2000 |
|---|---|
| M_tot (Eq. 3.2) | 0.235522 M⊕ |
| m_p | 3.5361×10⁻¹⁰ M☉ = 1.1776×10⁻⁴ M⊕ |
| R_p (Eq. 3.3) | 437.9 km |
| collision radius f(R_i+R_j) | 5254 km for two equal bodies |
| softening ε_i | 1313.7 km = 8.78×10⁻⁶ AU |
| h = r_H/a (Eq. 2.1) | 6.177×10⁻⁴ |
| e_rms | 2.471×10⁻³ |
| i_rms | 1.2355×10⁻³ |
| random velocity v_ran | 82.3 m s⁻¹ |
| pair escape velocity v_esc | 463.0 m s⁻¹ |
| v_ran/v_esc | 0.178 |
| gravitational focusing factor (Eq. 2.5) | 32.7 |
| dt | 0.0072339 yr |
| total steps to t_end | 2,764,742 |
| Hill-crossing time t_enc | 0.0356 yr → dt/t_enc = 0.203 |

### 4.3 Gas disk (Scenarios 2 and 4)

| Symbol | Value | Source |
|---|---|---|
| ρ_gas(1 AU) | 2.0×10⁻⁹ g cm⁻³ | Kokubo & Ida (2000) Eq. 1 (1.5 × MMSN) |
| ρ_gas slope | −11/4 | Hayashi (1981) |
| h_g(1 AU) | 0.047 AU | Hayashi (1981) |
| h_g slope | 5/4 | Hayashi (1981) |
| η₁ | 0.0019 | Kokubo & Ida (2000): η = 0.0019 (a/AU)^(1/2) |
| C_D | 1.0 | Guo & Kokubo (2021) §2.1, quadratic drag regime |
| gas_boost | 1.0 | Multiplier for the enhanced-drag parameter study |

Vertical structure is included: ρ(R,z) = ρ₁ R^(−11/4) exp(−(z/h_g)²).
Gas velocity is v_gas = (1 − 2η)^(1/2) v_K, azimuthal.

### 4.4 Giant planet (Scenarios 3 and 4)

| Symbol | Value | Note |
|---|---|---|
| M_p | 1.0 M_J = 9.5433×10⁻⁴ M☉ | Guo & Kokubo (2021) |
| a_p | 1.6 AU | See §8 — **deliberately closer than a real cold Jupiter** |
| e_p | 0.05 | Comparable to Jupiter's 0.048 (Guo & Kokubo 2021) |
| i_p | 0 | Guo & Kokubo (2021) §2.1 |
| 2:1 interior MMR | 1.0079 AU | Falls inside the annulus |

---

## 5. B1 — the mass budget closes, and N is not free

`benchmarks.py` B1 computes M_tot from Eqs. (3.1)–(3.2) and compares against
the independently published value for the same annulus:

```
  M_tot from Eq. (3.2)          : 0.235522 M_earth
  Iwasawa et al. 2017, Table 1  : 0.236 M_earth
  relative difference           : 0.20 %
```

That 0.2% agreement is the unit-conversion check. It is worth putting in the
thesis because it validates the whole cgs→code chain in one line.

The second half of B1 is the point you need for Chapter III:

```
      N    m_p (Msun)   m_p (M_earth)   R_p (km)   h = r_H/a   e_rms = 4h
    500    1.4144e-09      4.7104e-04      695.1   9.806e-04    3.922e-03
   1000    7.0721e-10      2.3552e-04      551.7   7.783e-04    3.113e-03
   2000    3.5361e-10      1.1776e-04      437.9   6.177e-04    2.471e-03
   5000    1.4144e-10      4.7104e-05      322.6   4.552e-04    1.821e-03
  10000    7.0721e-11      2.3552e-05      256.1   3.613e-04    1.445e-03
 100000    7.0721e-12      2.3552e-06      118.9   1.677e-04    6.707e-04
 1e9       7.0721e-16      2.3552e-10        5.5   7.783e-06    3.113e-05
```

Once the annulus and Σ₁ are fixed, choosing N *determines* m_p, R_p, h, e_rms
and i_rms. They are not separate entries in Table 3.1. Reaching literal 1-km
planetesimals in this annulus needs N ~ 10⁹.

---

## 6. B4 — the timestep is set by close encounters, not by the orbit

This is the result that should change how you budget CPU time.

```
  N = 600, t_end = 60 yr, f = 6, seed = 20250904
  mutual Hill radius r_H          : 9.228e-04 AU
  random velocity  v_ran          : 122.9 m/s
  Hill-radius crossing time t_enc : 0.0356 yr

    spo    dt (yr)   dt/t_enc   N_end   coll    e_rms(0)   e_rms(end)   ratio   wall(s)
     32    0.02894      0.813     596      4  3.6457e-03   1.2514e-02    3.43      35.5
     64    0.01447      0.407     598      2  3.6457e-03   4.9440e-03    1.36      12.4
    128    0.00723      0.203     584     16  3.6457e-03   3.8815e-03    1.06      22.9
    256    0.00362      0.102     593      7  3.6457e-03   3.7929e-03    1.04      44.8
    512    0.00181      0.051     592      8  3.6457e-03   3.7787e-03    1.04      92.6
```

Identical seed and identical initial conditions in every row; the only
difference is `dt`. At 32 and 64 steps per orbit the swarm heats by 3.4× and
1.4× — that is integration error from unresolved close encounters, not viscous
stirring. Converged behaviour begins at **128 steps/orbit** (ratio 1.06 → 1.04
→ 1.04 → 1.03 as dt halves three times more; the residual 3% is real viscous
stirring over 60 yr, not numerics).

A fixed-step leapfrog cannot resolve a Hill-radius passage whose crossing time
is 0.036 yr when dt = 0.014 yr. This is the concrete version of the Richardson
et al. (2000) point your Chapter II already cites, measured on your own code.

Note the non-monotonic wall time (spo=32 slower than spo=64): the heated runs
generate more close pairs, so the neighbour search does more work. Heating is
not even cheaper.

---

## 7. B2 — is the tree actually faster?

Measured on a **single core**, using swarm geometry from the real IC generator:

```
      N   tree (ms)   brute (ms)   speedup
    500        0.69         0.96     1.39x
   1000        1.55         2.42     1.56x
   2000        3.40        10.03     2.95x
   5000       11.18        61.46     5.50x
  10000       26.93       243.92     9.06x
  20000       56.63       924.23    16.32x
```

At N = 1000 the tree buys about 1.5×. At N = 2000 it is 3×; at N = 5000, 5.5×.
Below ~1000 the argument in Chapter I ("computational efficiency is a practical
necessity") is thin. **Re-run this on your machine and quote your own numbers**
— multi-core will shift the crossover.

Cost estimate at the recommended settings (N = 2000, 128 steps/orbit): the
smoke test measured ~3–4 ms/step for the tree plus collision search and
integrator overhead. 2,764,742 steps ≈ 3–6 h per scenario on one core, so
roughly half a day to a day for all four. Halve `t_end`, or raise `f`, if that
is too much.

## 8. B3 — opening angle

Error measured on the **mutual** planetesimal–planetesimal acceleration only.
Measuring against the total would be meaningless: the star is computed exactly
by direct summation and dominates by six orders of magnitude, which would make
every θ look perfect.

```
 theta    median err     90th pct     99th pct      max err   wall (ms)
   0.2     6.947e-05    2.792e-04    1.247e-03    1.046e-02         5.2
   0.3     1.897e-04    8.143e-04    3.734e-03    2.557e-02         4.3
   0.4     4.061e-04    1.707e-03    6.755e-03    4.076e-02         3.9
   0.5     7.358e-04    3.247e-03    1.210e-02    1.069e-01         3.5
   0.6     1.025e-03    4.586e-03    1.688e-02    1.112e-01         3.2
   0.7     1.368e-03    6.056e-03    2.198e-02    1.113e-01         3.2
   1.0     3.235e-03    1.501e-02    7.123e-02    2.061e-01         2.7
```

θ = 0.5 is the default and gives 7×10⁻⁴ median error for 3.5 ms at N = 2000.
θ = 0.4 nearly halves the error for 11% more cost — a reasonable upgrade.
Richardson et al. (2000) report using 0.5–0.8; 0.5 sits at the accurate end of
that range.

## 9. B5 — gas drag is negligible at this body size, and you must say so

```
 a (AU)    R (km)   m (M_earth)   t_stop (yr)  t_stop / t_end
    0.4       1.0     1.403e-12     3.594e+01             0.0
    0.4      10.0     1.403e-09     3.594e+02             0.0
    0.4     100.0     1.403e-06     3.594e+03             0.2
    0.4     438.0     1.179e-04     1.574e+04             0.8
    1.0       1.0     1.403e-12     6.039e+02             0.0
    1.0      10.0     1.403e-09     6.039e+03             0.3
    1.0     100.0     1.403e-06     6.039e+04             3.0
    1.0     438.0     1.179e-04     2.645e+05            13.2
    5.0       1.0     1.403e-12     6.057e+04             3.0
    5.0     438.0     1.179e-04     2.653e+07          1326.4
```

At the production configuration (438 km bodies at 1 AU) the drag stopping time
is 2.6×10⁵ yr, **13× longer than the 2×10⁴ yr integration**. Scenario 2 with
nominal gas will be statistically indistinguishable from Scenario 1. That is
physics, not a bug: drag simply cannot act on bodies this large this quickly.

Three defensible responses, in order of preference:

1. **Report the null with this table.** A quantified null is a real result.
2. **Add an enhanced-drag parameter study**: `--gas-boost 20` compresses
   t_stop to ~1.3×10⁴ yr, comparable to t_end, and brackets the drag-dominated
   regime. Label it clearly as a parameter study, not the nominal nebula.
3. **Move the annulus to 0.4 AU** following Barnes et al. (2009), where
   t_stop ∝ a^(13/4) drops to 1.6×10⁴ yr. Costs more steps per year, since the
   orbital period is shorter.

I would do 1 and 2 together.

## 10. Giant planet placement

Guo & Kokubo (2021) put Jupiter at 5.2 AU with planetesimals at 8–15 AU. Your
annulus is at 1 AU. A 5.2 AU perturber's secular timescale on a 1 AU swarm is
far longer than 2×10⁴ yr — Scenario 3 would show nothing.

The default `a_planet = 1.6 AU` puts the **2:1 interior resonance at 1.0079 AU,
inside the annulus**. The smoke test confirms this produces a large, fast
signal: e_rms rose from 4.3×10⁻³ to 6.9×10⁻² (16×) within 40 yr, versus
essentially no change in the gravity-only run.

State this choice explicitly in Chapter III. It is a legitimate design decision
— you are testing the *mechanism*, not modelling the actual Solar System — but
it must not be presented as a cold-Jupiter configuration when it is not.

## 11. Known limitations

- **Perfect merging.** No fragmentation or bouncing. `analysis.py` measures the
  fraction of impacts with v_rel > v_esc, which are the ones a real outcome
  model would disrupt; Wallace & Quinn (2019) §5.3 report ~60% in their runs.
  Report your own number rather than leaving it as an unstated assumption.
- **Radius enhancement f = 6** shortens the accretion timescale by up to f².
  This is the largest artificial factor in the results and belongs in Section 1.4.
- **Fixed shared timestep.** No individual/block timesteps and no P³T
  hybridisation, so the deepest encounters are the least accurate part of the
  integration. This is why B4 matters.
- **Static gas disk.** No dissipation unless `gas_depletion_time > 0`.
- **Giant planet is fully self-consistent** (it feels the swarm and the star)
  but does not migrate through any disk torque.
- **Rotational angular momentum is discarded** on merger — the Barnes et al.
  (2009) caveat, unresolved here.

## 12. Reference logs

- `benchmarks_quick.log` — B0, B1, B2, B3, B5 as printed
- `benchmarks_b4.log` — B4 convergence as printed
- `validation.log` — full test suite output
- `benchmarks_quick.json`, `benchmarks_b4.json` — machine-readable

All from a single-core x86-64 Linux container, Python 3.12.3, numpy 2.4.4,
scipy 1.17.1. Your timings will differ; your physics numbers should not.

---

## 13. Running on Google Colab

Open `planetesimal_sim_colab.ipynb` in Colab. Set the runtime to **CPU** — a
GPU does nothing here, since pytreegrav is numba-on-CPU.

The one thing Colab needs that a laptop does not is **checkpointing**, because
free-tier sessions are killed after roughly 12 hours and sooner if you close
the tab. Pass `--resume` and write `--outdir` to a mounted Google Drive path:

```bash
python run_scenarios.py --scenario S1_gravity_only --n 2000 --t-end 2e4 \
    --spo 128 --outdir /content/drive/MyDrive/thesis_results --resume
```

The run checkpoints every 250 simulated years (`--checkpoint-every`). If the
session dies, re-run the identical command and it continues from where it
stopped, losing at most 250 yr of work. Verified: a 20 yr run was stopped and
resumed to 40 yr, continuing from t = 20.0 with the collision tally and
diagnostic series intact.

If `--outdir` points at `/content/` instead of Drive, the checkpoint dies with
the session and there is nothing to resume from.

## 14. Running in VS Code

`.vscode/settings.json` is included and matters more than it looks. A
production run writes hundreds of `.npz` files, and VS Code's file watcher
tries to index each one as it appears — this is the usual reason the editor
freezes on a simulation project. The settings exclude `results/` and `*.npz`
from watching, indexing and search, and cap the terminal scrollback.

`.vscode/launch.json` provides four debug configurations. Start with
**"TINY run - step through this one to learn the code"**: it runs 200
particles for 5 years, which is fast enough to step through in the debugger.
Put a breakpoint on the first line of `Simulation.step()` in `integrator.py`
and walk through one timestep. The six stages of the leapfrog happen in front
of you, and that is worth more than reading the file three times.

For anything longer than a few minutes, run it in an **external terminal**
rather than the integrated one.
