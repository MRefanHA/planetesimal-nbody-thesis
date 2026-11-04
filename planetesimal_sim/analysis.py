"""
Analysis and plotting (thesis Chapter IV).

Reads the JSON time series and .npz snapshots written by
`planetesimal_sim.integrator.Simulation` and produces the comparison figures
the results chapter needs. Nothing here re-runs physics, so it is safe to
iterate on the figures without touching the simulations.

Usage:
    python -m planetesimal_sim.analysis results
    python -m planetesimal_sim.analysis results --scenarios S1_gravity_only S3_planet
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

from . import units as u                 # noqa: E402
from .orbits import state_to_elements    # noqa: E402

SCENARIO_LABELS = {
    "S1_gravity_only": "S1: gravity only",
    "S2_gas": "S2: + nebular gas",
    "S3_planet": "S3: + giant planet",
    "S4_gas_planet": "S4: + gas and giant planet",
}
SCENARIO_COLOURS = {
    "S1_gravity_only": "#000000",
    "S2_gas": "#1f77b4",
    "S3_planet": "#d62728",
    "S4_gas_planet": "#9467bd",
}
SCENARIO_STYLES = {
    "S1_gravity_only": "-",
    "S2_gas": "--",
    "S3_planet": "-.",
    "S4_gas_planet": ":",
}


# ---------------------------------------------------------------------------
def _scenario_from_folder(n):
    """'2026-09-12_031207_S1_gravity_only' -> 'S1_gravity_only'."""
    return n.split("_", 2)[2] if n[:2].isdigit() and n.count("_") >= 2 else n


def load_run(outdir, name):
    """Load one scenario's config, time series and collision log."""
    d = os.path.join(outdir, name)
    with open(os.path.join(d, "series.json")) as fh:
        series = json.load(fh)
    with open(os.path.join(d, "config.json")) as fh:
        cfg = json.load(fh)
    colls = []
    cp = os.path.join(d, "collisions.json")
    if os.path.exists(cp):
        with open(cp) as fh:
            colls = json.load(fh)
    # `name` must be the SCENARIO ("S1_gravity_only"), not the timestamped
    # folder name. Every colour, line style, legend label and the Mmax/S1
    # column of the comparison table is keyed on it. config.json is
    # authoritative; the folder name is only a fallback.
    scen = cfg.get("name") or _scenario_from_folder(name)
    return dict(name=scen, folder=name, cfg=cfg, series=series,
                collisions=colls, dir=d)


def _is_timestamped(n):
    """True for folders written by the current RunLog scheme."""
    return len(n) > 18 and n[:4].isdigit() and n[4] == "-"


def discover_runs(outdir):
    """Map scenario -> [candidate folders], best candidate last.

    A folder counts as a finished run if it holds series.json. The scenario
    is taken from that folder's own config.json, NOT from its name, so a
    folder you renamed by hand (S1_gravity_only_OLD_20kyr) is still
    recognised as the scenario it actually is and does not masquerade as a
    fifth scenario. Ranking puts timestamped folders above untimestamped
    ones and the latest timestamp last, so the newest complete run wins.
    """
    found = {}
    if not os.path.isdir(outdir):
        return found
    for n in sorted(os.listdir(outdir)):
        d = os.path.join(outdir, n)
        if not os.path.isfile(os.path.join(d, "series.json")):
            continue
        scen = None
        cfgp = os.path.join(d, "config.json")
        if os.path.isfile(cfgp):
            try:
                with open(cfgp) as fh:
                    scen = json.load(fh).get("name")
            except (OSError, ValueError):
                scen = None
        found.setdefault(scen or _scenario_from_folder(n), []).append(n)
    for folders in found.values():
        folders.sort(key=lambda n: (_is_timestamped(n), n))
    return found


def resolve_run(outdir, token, found=None):
    """Accept either an exact folder name or a scenario name."""
    if os.path.isfile(os.path.join(outdir, token, "series.json")):
        return token
    found = discover_runs(outdir) if found is None else found
    if token in found:
        return found[token][-1]
    raise SystemExit(
        f"{token!r} is neither a run folder nor a scenario in {outdir!r}.\n"
        "Available:\n"
        + "\n".join(f"  {k:<18} {', '.join(v)}" for k, v in found.items())
        + "\nPass an exact folder name, or a scenario name to get its "
          "most recent run.")


def load_all(outdir, names=None):
    """Find completed runs under outdir.

    Run folders are named YYYY-MM-DD_HHMMSS_<scenario>. When several runs of
    the same scenario exist, the most recent one is used, so re-running a
    scenario supersedes the earlier attempt without deleting its record.
    """
    found = discover_runs(outdir)
    if names:
        return [load_run(outdir, resolve_run(outdir, n, found)) for n in names]
    order = ["S1_gravity_only", "S2_gas", "S3_planet", "S4_gas_planet"]
    keys = [k for k in order if k in found] + \
           [k for k in sorted(found) if k not in order]
    return [load_run(outdir, found[k][-1]) for k in keys]


def arr(run, key):
    return np.array([s[key] for s in run["series"]], dtype=float)


def style(name):
    return dict(color=SCENARIO_COLOURS.get(name, None),
                ls=SCENARIO_STYLES.get(name, "-"),
                label=SCENARIO_LABELS.get(name, name), lw=1.6)


# ---------------------------------------------------------------------------
def fig_growth(runs, path):
    """Number of bodies, largest mass, cumulative collisions vs time.

    This is the figure that directly answers Research Questions 1-4: if the
    four curves separate, the added physics changed the accretion rate.
    """
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0))
    for r in runs:
        t = arr(r, "t")
        ax[0].plot(t, arr(r, "n_bodies"), **style(r["name"]))
        ax[1].plot(t, arr(r, "m_max_mearth"), **style(r["name"]))
        ax[2].plot(t, arr(r, "n_collisions"), **style(r["name"]))
    ax[0].set_ylabel("number of bodies $N$")
    ax[1].set_ylabel(r"largest mass $M_{\max}$  [$M_\oplus$]")
    ax[1].set_yscale("log")
    ax[2].set_ylabel("cumulative collisions")
    for a in ax:
        a.set_xlabel("time [yr]")
        a.grid(alpha=0.25)
    ax[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_velocity(runs, path):
    """RMS eccentricity/inclination and velocity dispersion vs time.

    Answers the "stirring versus damping" half of the research questions:
    a giant planet should raise these curves, gas drag should lower them.
    The horizontal line marks the seeded initial value common to all runs.
    """
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0))
    for r in runs:
        t = arr(r, "t")
        ax[0].plot(t, arr(r, "e_rms"), **style(r["name"]))
        ax[1].plot(t, arr(r, "i_rms"), **style(r["name"]))
        ax[2].plot(t, arr(r, "v_disp_ms"), **style(r["name"]))
    e0 = runs[0]["cfg"]["e_rms"]
    i0 = runs[0]["cfg"]["i_rms"]
    ax[0].axhline(e0, color="grey", lw=0.8, ls=":")
    ax[1].axhline(i0, color="grey", lw=0.8, ls=":")
    ax[0].set_ylabel(r"$\langle e^2\rangle^{1/2}$")
    ax[1].set_ylabel(r"$\langle i^2\rangle^{1/2}$")
    ax[2].set_ylabel(r"velocity dispersion [m s$^{-1}$]")
    for a in ax:
        a.set_xlabel("time [yr]")
        a.set_yscale("log")
        a.grid(alpha=0.25)
    ax[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_ae_snapshots(runs, path, snapshot="final"):
    """a-e scatter, one panel per scenario, marker size ~ body radius.

    This is the plot that makes a giant planet's mean-motion resonances
    visible as vertical spikes in eccentricity, in the style of Guo & Kokubo
    (2021) Fig. 2 and Grimm & Stadel (2014) Fig. 6.
    """
    fig, axes = plt.subplots(1, len(runs), figsize=(4.0 * len(runs), 3.8),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for a, r in zip(axes, runs):
        f = os.path.join(r["dir"], "snapshots", f"snap_{snapshot}.npz")
        if not os.path.exists(f):
            cand = sorted(glob.glob(os.path.join(r["dir"], "snapshots",
                                                "snap_*.npz")))
            if not cand:
                continue
            f = cand[-1]
        z = np.load(f)
        pos = z["pos"] - z["star_pos"]
        vel = z["vel"] - z["star_vel"]
        mass = z["mass"]
        mu = u.G * (1.0 + mass)
        sa, se, si = state_to_elements(pos, vel, mu)
        ok = (sa > 0) & (sa < 5)
        s = 4.0 * (mass[ok] / mass.min()) ** (1.0 / 3.0)
        a.scatter(sa[ok], se[ok], s=s, alpha=0.55,
                  color=SCENARIO_COLOURS.get(r["name"], "k"),
                  edgecolors="none")
        if bool(z["planet_on"]):
            ap = r["cfg"]["a_planet"]
            for p, q in ((2, 1), (3, 1), (5, 2), (3, 2)):
                a_res = ap * (q / p) ** (2.0 / 3.0)
                if r["cfg"]["a_in"] * 0.8 < a_res < r["cfg"]["a_out"] * 1.2:
                    a.axvline(a_res, color="grey", lw=0.7, ls="--")
                    a.text(a_res, 0.9, f"{p}:{q}", fontsize=7,
                           rotation=90, va="top", ha="right",
                           transform=a.get_xaxis_transform())
        a.set_title(SCENARIO_LABELS.get(r["name"], r["name"]), fontsize=9)
        a.set_xlabel("semi-major axis [AU]")
        a.grid(alpha=0.25)
    axes[0].set_ylabel("eccentricity")
    axes[0].set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_mass_function(runs, path, snapshot="final"):
    """Cumulative mass distribution N(>m) at the end of each run."""
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for r in runs:
        f = os.path.join(r["dir"], "snapshots", f"snap_{snapshot}.npz")
        if not os.path.exists(f):
            continue
        m = np.sort(np.load(f)["mass"])[::-1]
        ax.step(u.msun_to_mearth(m), np.arange(1, len(m) + 1),
                where="post", **style(r["name"]))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"mass $m$  [$M_\oplus$]")
    ax.set_ylabel(r"$N(>m)$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_collision_velocities(runs, path):
    """Distribution of v_rel / v_esc at impact.

    Impacts with v_rel > v_esc would erode or fragment the bodies in reality
    (thesis Section 1.4, item 2). With merge_criterion="velocity" they are
    rejected; with "always" (perfect merging) they are merged anyway, so this
    figure shows how many impacts that assumption affects.
    """
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    txt = []
    for r in runs:
        c = r["collisions"]
        if not c:
            continue
        ratio = np.array([x["v_rel"] / x["v_esc"] for x in c])
        acc = np.array([x.get("accreted", True) for x in c])
        bins = np.logspace(-2, 1.2, 45)
        ax.hist(ratio, bins=bins, histtype="step", density=True,
                color=SCENARIO_COLOURS.get(r["name"], "k"),
                ls=SCENARIO_STYLES.get(r["name"], "-"),
                label=SCENARIO_LABELS.get(r["name"], r["name"]))
        txt.append(f"{r['name']}: {100 * np.mean(ratio > 1):.1f}% of "
                   f"{len(ratio)} detected impacts had v_rel > v_esc; "
                   f"{int(acc.sum())} accreted, {int((~acc).sum())} rejected")
    ax.axvline(1.0, color="k", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlabel(r"$v_{\rm rel}/v_{\rm esc}$ at impact")
    ax.set_ylabel("probability density")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    for line in txt:
        print("   ", line)
    return path


# ---------------------------------------------------------------------------
def comparison_table(runs):
    """Scenario comparison table for Chapter IV, as plain text + dict."""
    rows = []
    base = None
    for r in runs:
        s = r["series"][-1]
        s0 = r["series"][0]
        m_final = s["m_max_mearth"]
        # time to reach 10x the initial particle mass, the simplest
        # scenario-independent definition of a "protoplanet formation time"
        target = 10.0 * s0["m_max_mearth"]
        t10 = np.nan
        for x in r["series"]:
            if x["m_max_mearth"] >= target:
                t10 = x["t"]
                break
        row = dict(scenario=r["name"], N_end=s["n_bodies"],
                   collisions=s["n_collisions"],
                   rejected=s.get("n_rejected", 0),
                   disruptive_frac=s.get("disruptive_frac", 0.0),
                   m_max_mearth=m_final,
                   t_10x_yr=t10, e_rms_end=s["e_rms"],
                   i_rms_end=s["i_rms"], v_disp_ms=s["v_disp_ms"],
                   n_unbound=s["n_unbound"],
                   m_bound_mearth=s["m_bound_mearth"],
                   wall_s=s["wall_s"])
        if r["name"] == "S1_gravity_only":
            base = row
        rows.append(row)

    if base is not None:
        for row in rows:
            row["m_max_rel_S1"] = row["m_max_mearth"] / base["m_max_mearth"]
            row["coll_rel_S1"] = (row["collisions"] / base["collisions"]
                                  if base["collisions"] else np.nan)

    hdr = (f"{'scenario':<20} {'N_end':>6} {'acc':>5} {'disr':>5} "
           f"{'disr%':>6} {'Mmax(Me)':>11} "
           f"{'Mmax/S1':>8} {'t(10x)/yr':>10} {'e_rms':>10} "
           f"{'vdisp m/s':>10}")
    lines = [hdr, "-" * len(hdr)]
    for row in rows:
        lines.append(
            f"{row['scenario']:<20} {row['N_end']:>6} {row['collisions']:>5} "
            f"{row['rejected']:>5} {100*row['disruptive_frac']:>5.1f}% "
            f"{row['m_max_mearth']:>11.4e} "
            f"{row.get('m_max_rel_S1', float('nan')):>8.3f} "
            f"{row['t_10x_yr']:>10.1f} {row['e_rms_end']:>10.3e} "
            f"{row['v_disp_ms']:>10.1f}")
    return "\n".join(lines), rows


# ---------------------------------------------------------------------------
# Keys that MUST be identical across the four scenarios. This is the
# methodological claim of Section 1.4 item 5: the scenarios differ only in
# `gas_on` and `planet_on`. The claim is checked here and the verdict is
# written into every analysis folder, so it is evidence rather than an
# assertion.
SHARED_KEYS = ("seed", "n_particles", "a_in", "a_out", "sigma1_solid_cgs",
               "rho_p_cgs", "e_tilde", "i_over_e", "f_radius",
               "merge_criterion", "merge_velocity_factor", "theta",
               "softening_frac", "tree_method", "steps_per_inner_orbit",
               "t_end", "dt_yr", "m_particle_msun", "r_particle_km")


def consistency_check(runs):
    """Report any shared parameter that is not identical across the runs."""
    lines, ok = [], True
    for k in SHARED_KEYS:
        vals = {r["name"]: r["cfg"].get(k) for r in runs if k in r["cfg"]}
        if not vals:
            continue
        uniq = set(map(repr, vals.values()))
        if len(uniq) == 1:
            lines.append(f"  {k:24s} : {next(iter(vals.values()))}")
        else:
            ok = False
            lines.append(f"  {k:24s} : !! DIFFERS -> "
                         + ", ".join(f"{n}={v}" for n, v in vals.items()))
    verdict = ("All shared parameters identical across the runs above."
               if ok else
               "!! WARNING: the runs above are NOT a controlled comparison.")
    return lines, verdict, ok


def write_analysis_info(figdir, runs, made, table, argv):
    """Record which runs produced this figure set, and on what."""
    import matplotlib
    now = datetime.now().astimezone()
    L = ["FIGURE SET RECORD",
         "=" * 62, "",
         "[ WHEN ]",
         f"  generated        : {now.strftime('%Y-%m-%d %H:%M:%S %Z')}",
         f"  command          : {' '.join(argv)}",
         f"  output folder    : {os.path.abspath(figdir)}",
         f"  python / numpy   : {sys.version.split()[0]} / {np.__version__}",
         f"  matplotlib       : {matplotlib.__version__}",
         "",
         "[ SOURCE RUNS ]",
         "  Every curve in every figure here came from one of these folders.",
         ""]
    for r in runs:
        last = r["series"][-1] if r["series"] else {}
        L.append(f"  {r['name']}")
        L.append(f"      folder       : {r['folder']}")
        L.append(f"      simulated to : {last.get('t', 0.0):.1f} yr")
        L.append(f"      N            : {r['cfg'].get('n_particles')} -> "
                 f"{last.get('n_bodies', '?')}")
        L.append(f"      M_max        : {last.get('m_max_mearth', float('nan')):.4e} M_earth")
        L.append("")
    L.append("[ CONTROLLED-COMPARISON CHECK ]")
    ck, verdict, _ = consistency_check(runs)
    L.extend(ck)
    L.append("")
    L.append("  " + verdict)
    L.append("")
    L.append("[ FIGURES ]")
    L.extend("  " + os.path.basename(m) for m in made)
    L.append("")
    L.append("[ COMPARISON TABLE ]")
    L.extend("  " + ln for ln in table.split("\n"))
    L.append("")
    L.append("=" * 62)
    L.append("Cite this figure set in the thesis as: "
             + os.path.basename(os.path.abspath(figdir)))
    with open(os.path.join(figdir, "analysis_info.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return verdict


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", nargs="?", default="results")
    ap.add_argument("--runs", "--scenarios", nargs="*", default=None,
                    dest="runs",
                    help="analyse only these runs. Each item may be an exact "
                         "folder name (2026-09-15_060739_S1_gravity_only) or "
                         "a scenario name (S1_gravity_only), which selects "
                         "that scenario's most recent run")
    ap.add_argument("--list", action="store_true",
                    help="show every run found under outdir and which folder "
                         "would be used for each scenario, then exit")
    ap.add_argument("--figdir", default=None)
    ap.add_argument("--tag", default=None,
                    help="short label appended to the analysis folder name, "
                         "e.g. --tag draft2")
    args = ap.parse_args()

    if args.list:
        found = discover_runs(args.outdir)
        if not found:
            raise SystemExit(f"no finished runs found in {args.outdir!r}")
        print(f"runs found in {args.outdir!r}:\n")
        for scen in sorted(found):
            for n in found[scen]:
                mark = "  <- used" if n == found[scen][-1] else ""
                print(f"  {scen:<18} {n}{mark}")
        print("\nPass --runs FOLDER to analyse one of them on its own.")
        return

    runs = load_all(args.outdir, args.runs)
    if not runs:
        raise SystemExit(f"no completed runs found in {args.outdir!r}")

    # One timestamped folder per analysis. The figures are a function of a
    # SET of runs, not of any single run, so they get their own folder that
    # names the runs that produced them. Re-running the analysis therefore
    # never overwrites an earlier figure set, and any figure in Chapter IV
    # can be traced to the exact simulations behind it.
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S")
    figdir = args.figdir or os.path.join(
        args.outdir,
        f"analysis_{stamp}" + (f"_{args.tag}" if args.tag else ""))
    os.makedirs(figdir, exist_ok=True)

    made = [
        fig_growth(runs, os.path.join(figdir, "fig_growth.png")),
        fig_velocity(runs, os.path.join(figdir, "fig_velocity.png")),
        fig_ae_snapshots(runs, os.path.join(figdir, "fig_ae_final.png")),
        fig_mass_function(runs, os.path.join(figdir, "fig_massfunc.png")),
        fig_collision_velocities(
            runs, os.path.join(figdir, "fig_impact_velocities.png")),
    ]

    table, rows = comparison_table(runs)
    print()
    print(table)
    with open(os.path.join(figdir, "comparison_table.txt"), "w") as fh:
        fh.write(table + "\n")
    with open(os.path.join(figdir, "comparison_table.json"), "w") as fh:
        json.dump(rows, fh, indent=2)

    verdict = write_analysis_info(figdir, runs, made, table, sys.argv)

    # Back-reference, so that from any run folder you can see which figure
    # sets used it. Provenance has to work in both directions.
    for r in runs:
        try:
            with open(os.path.join(r["dir"], "analyses.txt"), "a",
                      encoding="utf-8") as fh:
                fh.write(f"{stamp}  ->  {os.path.abspath(figdir)}\n")
        except OSError:
            pass

    print()
    for m in made:
        print("wrote", m)
    print("wrote", os.path.join(figdir, "comparison_table.txt"))
    print("wrote", os.path.join(figdir, "analysis_info.txt"))
    print()
    print(verdict)
    print(f"figure set: {figdir}")


if __name__ == "__main__":
    main()
