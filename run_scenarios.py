#!/usr/bin/env python3
"""
Four-scenario production runner.

    python run_scenarios.py --all
    python run_scenarios.py --scenario S3_planet
    python run_scenarios.py --all --n 2000 --t-end 2e4 --spo 128
    python run_scenarios.py --all --dry-run          # print parameters only

The four scenarios differ ONLY in the flags `gas_on` and `planet_on`. Every
other parameter -- seed, initial population, collision model, tree settings,
timestep, integration time -- is identical by construction, which is the
methodological claim made in Section 1.4 item 5. `--dry-run` prints the
resolved parameters for all four so you can verify that claim before
committing CPU time.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import replace


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

from planetesimal_sim import units as u                 # noqa: E402
from planetesimal_sim.config import SimConfig, four_scenarios   # noqa: E402
from planetesimal_sim.forces import stopping_time       # noqa: E402
from planetesimal_sim.integrator import Simulation      # noqa: E402
from planetesimal_sim.runlog import RunLog, find_previous_run  # noqa: E402


def build_base(args):
    base = SimConfig(
        seed=args.seed,
        a_in=args.a_in,
        a_out=args.a_out,
        sigma1_solid_cgs=args.sigma1,
        n_particles=args.n,
        rho_p_cgs=args.rho_p,
        e_tilde=args.e_tilde,
        f_radius=args.f,
        theta=args.theta,
        steps_per_inner_orbit=args.spo,
        t_end=args.t_end,
        snapshot_every=args.snapshot_every,
        dump_every=args.dump_every,
        outdir=args.outdir,
        m_planet=args.m_planet * u.MJUP,
        a_planet=args.a_planet,
        e_planet=args.e_planet,
        gas_boost=args.gas_boost,
        c_drag=args.c_drag,
        merge_criterion=args.merge_criterion,
        merge_velocity_factor=args.merge_velocity_factor,
        tree_method=args.force_method,
        parallel=not args.no_parallel,
    )
    return base


def feasibility_report(base: SimConfig):
    """Print the cost and physics sanity checks before anything is run."""
    print("=" * 74)
    print("FEASIBILITY CHECK  (read this before starting a multi-day run)")
    print("=" * 74)
    print(base.summary())
    ts = stopping_time(base)
    print(f"merge criterion             : {base.merge_criterion}"
          + (f" (accrete only if v_rel < "
             f"{base.merge_velocity_factor:g} v_esc)"
             if base.merge_criterion == "velocity"
             else " (PERFECT MERGING: every impact accretes)"))
    if base.merge_criterion == "always":
        print("  !! NOTE: with a giant planet present, most impacts exceed")
        print("     the escape velocity and would fragment in reality.")
        print("     Perfect merging will overstate growth in Scenarios 3")
        print("     and 4. Run --merge-criterion velocity for comparison.")
    print(f"drag stopping time t_s      : {ts:.3e} yr  "
          f"(t_s / t_end = {ts / base.t_end:.1f})")
    if ts > 5 * base.t_end:
        print("  !! WARNING: gas drag cannot act on this body size within")
        print("     t_end. Scenario 2 will be indistinguishable from")
        print("     Scenario 1. Either report that as a quantified null")
        print("     result, raise --gas-boost, or move the annulus inward.")
    import numpy as np
    v_ran = np.sqrt(base.e_rms ** 2 + base.i_rms ** 2) * np.sqrt(u.G)
    t_enc = base.h_reduced / v_ran
    print(f"Hill-crossing time t_enc    : {t_enc:.4f} yr   "
          f"(dt / t_enc = {base.dt / t_enc:.3f})")
    if base.dt > 0.15 * t_enc:
        print("  !! WARNING: dt is a large fraction of the encounter")
        print("     crossing time. Close encounters will be under-resolved")
        print("     and the swarm will be heated numerically. Raise --spo")
        print("     and confirm with benchmarks.py B4.")
    if base.planet_on or True:
        for p, q in ((2, 1), (3, 1), (5, 2), (3, 2)):
            a_res = base.a_planet * (q / p) ** (2.0 / 3.0)
            if base.a_in < a_res < base.a_out:
                print(f"giant-planet {p}:{q} resonance  : "
                      f"{a_res:.4f} AU  (inside the annulus)")
    print("=" * 74)
    print()


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--all", action="store_true", help="run all four scenarios")
    ap.add_argument("--scenario", default=None,
                    help="run one scenario by name")
    ap.add_argument("--dry-run", action="store_true",
                    help="print resolved parameters and exit")

    ap.add_argument("--seed", type=int, default=20250904)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--a-in", type=float, default=0.95)
    ap.add_argument("--a-out", type=float, default=1.05)
    ap.add_argument("--sigma1", type=float, default=10.0,
                    help="solid surface density at 1 AU, g/cm^2")
    ap.add_argument("--rho-p", type=float, default=2.0,
                    help="planetesimal bulk density, g/cm^3")
    ap.add_argument("--e-tilde", type=float, default=4.0,
                    help="e_rms in units of the reduced Hill radius")
    ap.add_argument("--f", type=float, default=6.0,
                    help="collision radius enhancement factor")
    ap.add_argument("--merge-criterion", default="velocity",
                    choices=["always", "velocity"],
                    help="'always' = perfect merging (every impact accretes); "
                         "'velocity' = accrete only when v_rel < v_esc")
    ap.add_argument("--merge-velocity-factor", type=float, default=1.0,
                    help="accrete when v_rel < factor * v_esc")
    ap.add_argument("--theta", type=float, default=0.5,
                    help="Barnes-Hut opening angle")
    ap.add_argument("--spo", type=int, default=128,
                    help="integration steps per inner orbital period")
    ap.add_argument("--t-end", type=float, default=2.0e4, help="yr")
    ap.add_argument("--snapshot-every", type=float, default=100.0, help="yr")
    ap.add_argument("--dump-every", type=float, default=2000.0, help="yr")

    ap.add_argument("--m-planet", type=float, default=1.0,
                    help="giant planet mass in Jupiter masses")
    ap.add_argument("--a-planet", type=float, default=1.6, help="AU")
    ap.add_argument("--e-planet", type=float, default=0.05)

    ap.add_argument("--gas-boost", type=float, default=1.0,
                    help="multiplies the nominal nebular gas density")
    ap.add_argument("--c-drag", type=float, default=1.0)

    ap.add_argument("--force-method", default="tree",
                    choices=["tree", "bruteforce"])
    ap.add_argument("--no-parallel", action="store_true")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--resume", action="store_true",
                    help="continue from checkpoint.npz if one exists "
                         "(use this on Google Colab, where sessions die)")
    ap.add_argument("--checkpoint-every", type=float, default=250.0,
                    help="yr between checkpoint writes")
    args = ap.parse_args()

    base = build_base(args)
    scenarios = four_scenarios(base)
    if args.scenario:
        scenarios = [s for s in scenarios if s.name == args.scenario]
        if not scenarios:
            raise SystemExit(
                f"unknown scenario {args.scenario!r}; choose from "
                + ", ".join(s.name for s in four_scenarios(base)))
    elif not args.all:
        raise SystemExit("pass --all or --scenario NAME (see --help)")

    feasibility_report(base)

    if args.dry_run:
        print("Resolved parameters for each scenario "
              "(differences from S1 marked '<-- differs'):")
        ref = scenarios[0].to_dict()
        for s in scenarios:
            d = s.to_dict()
            diffs = [k for k in d
                     if k not in ("name",) and d[k] != ref[k]]
            print(f"\n  {s.name}: "
                  + (", ".join(f"{k}={d[k]} <-- differs" for k in diffs)
                     if diffs else "identical to S1"))
        print("\n(dry run: nothing was integrated)")
        return

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "run_command.json"), "w") as fh:
        json.dump(dict(argv=sys.argv, args=vars(args)), fh, indent=2)

    summary = []
    for cfg in scenarios:
        print()
        print("#" * 74)
        print(f"# RUNNING {cfg.name}")
        print("#" * 74, flush=True)
        t0 = time.perf_counter()

        # --- provenance: one timestamped folder per run ------------------
        # RunLog owns the output directory. Simulation MUST be handed that
        # RunLog; if it is not, Simulation falls back to results/<scenario>,
        # which carries no timestamp and silently overwrites the previous
        # run of the same scenario. `runlog=log` below is not optional.
        prev = find_previous_run(args.outdir, cfg.name) if args.resume else None
        if args.resume and prev is None:
            print(f"--resume: no previous {cfg.name} run with a checkpoint "
                  f"found in {args.outdir!r}; starting a fresh run.")
        log = RunLog(cfg, argv=sys.argv, resume_dir=prev, verbose=True)

        sim = None              # bound before the try so the handlers below
        status = "completed"    # can never raise NameError and hide the
        try:                    # exception that actually killed the run
            sim = Simulation(cfg, verbose=True, runlog=log)
            sim.run(resume=args.resume,
                    checkpoint_every=args.checkpoint_every)
        except KeyboardInterrupt:
            status = "interrupted by user (Ctrl+C); resume with --resume"
            log.log_event(status)
            if sim is not None:
                sim.save_checkpoint()
                sim.save()
            print(f"\nInterrupted. Progress saved in {log.dir}")
            log.finalize(getattr(sim, "series", []), status=status)
            raise
        except Exception as exc:
            status = f"FAILED: {type(exc).__name__}: {exc}"
            log.log_event(status)
            log.finalize(getattr(sim, "series", []), status=status)
            raise
        wall = time.perf_counter() - t0
        last = sim.series[-1]
        rundir = log.finalize(sim.series, status=status)
        summary.append(dict(last, scenario=cfg.name, wall_total_s=wall,
                            run_folder=os.path.basename(rundir)))
        print(f"\n{cfg.name} finished in {wall / 3600:.2f} h: "
              f"N {cfg.n_particles} -> {last['n_bodies']}, "
              f"{last['n_collisions']} accreted, {last['n_rejected']} "
              f"disruptive, M_max = {last['m_max_mearth']:.4e} M_earth")
        print(f"log folder: {rundir}")

    with open(os.path.join(args.outdir, "run_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nAll requested scenarios complete. Now run:")
    print(f"    python -m planetesimal_sim.analysis {args.outdir}")


if __name__ == "__main__":
    main()
