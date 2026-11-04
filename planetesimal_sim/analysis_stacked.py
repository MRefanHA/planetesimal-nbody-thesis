"""
Stacked comparison figures for thesis Chapter IV.

Same data as analysis.py, different layout: every scenario gets its own row
(S1 at the top, S4 at the bottom), all rows share ONE x axis and, column by
column, ONE y axis, so a reader can compare scenarios by looking straight
down a column. Time is plotted on a logarithmic axis so the early, fast
evolution (the first few thousand years) is not crushed against the left
edge.

Nothing here re-runs physics. It only reads series.json, collisions.json,
config.json and snapshots/ from finished run folders.

Usage (from C:\\thesis\\planetesimal_sim_thesis):

    # the four newest runs in results/
    python -m planetesimal_sim.analysis_stacked results

    # or name the exact run folders, in any location, in any order
    python -m planetesimal_sim.analysis_stacked --runs \\
        results/2026-09-15_124025_S1_gravity_only \\
        results/2026-10-25_111402_S2_gas \\
        results/2026-09-17_062814_S3_planet \\
        results/2026-10-27_210615_S4_gas_planet  --tag gasboost

Options:
    --no-ref       do not draw S1 as a faint grey reference in rows S2-S4
    --tmin 100     left edge of the log time axis in years (default: first
                   output time after t = 0)
    --ae-xscale    'linear' (default) or 'log' for the a-e figure
    --ae-xlim      e.g. --ae-xlim 0.5 1.3

Every figure is written as PNG (300 dpi) and PDF (vector, for LaTeX/Word).
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

from . import units as u                        # noqa: E402
from .orbits import state_to_elements           # noqa: E402

ORDER = ["S1_gravity_only", "S2_gas", "S3_planet", "S4_gas_planet"]
LABELS = {
    "S1_gravity_only": "S1: gravity only",
    "S2_gas": "S2: + nebular gas",
    "S3_planet": "S3: + giant planet",
    "S4_gas_planet": "S4: + gas & giant planet",
}
COLOURS = {
    "S1_gravity_only": "#000000",
    "S2_gas": "#1f77b4",
    "S3_planet": "#d62728",
    "S4_gas_planet": "#9467bd",
}
REF_COLOUR = "#b0b0b0"

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "savefig.dpi": 300,
})


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_run(d):
    with open(os.path.join(d, "series.json")) as fh:
        series = json.load(fh)
    with open(os.path.join(d, "config.json")) as fh:
        cfg = json.load(fh)
    colls = []
    cp = os.path.join(d, "collisions.json")
    if os.path.exists(cp):
        with open(cp) as fh:
            colls = json.load(fh)
    name = cfg.get("name") or os.path.basename(os.path.normpath(d))
    return dict(name=name, folder=os.path.basename(os.path.normpath(d)),
                dir=d, cfg=cfg, series=series, collisions=colls)


def find_newest(outdir):
    """Newest finished run folder for each scenario under outdir."""
    best = {}
    for n in sorted(os.listdir(outdir)):
        d = os.path.join(outdir, n)
        if not os.path.isfile(os.path.join(d, "series.json")):
            continue
        try:
            with open(os.path.join(d, "config.json")) as fh:
                scen = json.load(fh).get("name")
        except (OSError, ValueError):
            continue
        best[scen] = d                          # sorted -> newest wins
    return [best[k] for k in ORDER if k in best]


def sort_runs(runs):
    return sorted(runs, key=lambda r: ORDER.index(r["name"])
                  if r["name"] in ORDER else 99)


def arr(run, key):
    return np.array([s[key] for s in run["series"]], dtype=float)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _row_label(ax, name):
    ax.annotate(LABELS.get(name, name), xy=(1.03, 0.5),
                xycoords="axes fraction", rotation=270, va="center",
                ha="left", fontsize=8.5, color=COLOURS.get(name, "k"),
                fontweight="bold")


def _log_time_axis(ax, tmin, tmax):
    ax.set_xscale("log")
    ax.set_xlim(tmin, tmax)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _save(fig, path_noext):
    out = []
    for ext in ("png", "pdf"):
        p = f"{path_noext}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        out.append(p)
    plt.close(fig)
    return out


def _stacked_timeseries(runs, columns, path_noext, tmin=None, ref=True, base=None):
    """Generic rows = scenarios, columns = quantities, log time axis.

    columns: list of dicts with keys key, title, yscale, (optional) init
    where init is a callable(run) -> horizontal reference value.
    """
    nrow, ncol = len(runs), len(columns)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.55 * ncol + 0.6,
                                                  1.75 * nrow + 0.5),
                             sharex=True, sharey="col", squeeze=False)

    tmax = max(arr(r, "t")[-1] for r in runs)
    if tmin is None:
        t1 = min(arr(r, "t")[arr(r, "t") > 0][0] for r in runs)
        tmin = 10.0 ** np.floor(np.log10(t1) + 1e-6)   # snap to a decade

    if base is None:
        base = next((r for r in runs if r["name"] == "S1_gravity_only"), None)

    for i, r in enumerate(runs):
        t = arr(r, "t")
        keep = t > 0                               # log axis: drop t = 0
        for j, col in enumerate(columns):
            ax = axes[i, j]
            if ref and base is not None and r is not base:
                tb = arr(base, "t")
                kb = tb > 0
                ax.plot(tb[kb], arr(base, col["key"])[kb], color=REF_COLOUR,
                        lw=1.1, ls="--", label="S1 (reference)", zorder=1)
            ax.plot(t[keep], arr(r, col["key"])[keep],
                    color=COLOURS.get(r["name"], "k"), lw=1.5, zorder=3,
                    label=LABELS.get(r["name"], r["name"]))
            if col.get("init"):
                ax.axhline(col["init"](r), color="grey", lw=0.7, ls=":",
                           zorder=0)
            ax.set_yscale(col.get("yscale", "linear"))
            _log_time_axis(ax, tmin, tmax)
            ax.grid(alpha=0.25, which="major")
            if i == 0:
                ax.set_title(col["title"])
            if i == nrow - 1:
                ax.set_xlabel("time [yr]")
        _row_label(axes[i, -1], r["name"])

    fig.tight_layout(h_pad=0.4, w_pad=0.8)
    from matplotlib.lines import Line2D
    handles = []
    if ref and base is not None and nrow > 1:
        handles.append(Line2D([], [], color=REF_COLOUR, ls="--", lw=1.1,
                              label="S1 gravity only (reference)"))
    if any(c.get("init") for c in columns):
        handles.append(Line2D([], [], color="grey", ls=":", lw=0.7,
                              label="initial value"))
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   frameon=False, bbox_to_anchor=(0.5, 1.0))
    return _save(fig, path_noext)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_growth_stacked(runs, path_noext, tmin=None, ref=True, base=None):
    cols = [
        dict(key="n_bodies", title="number of bodies $N$"),
        dict(key="m_max_mearth", yscale="log",
             title=r"largest mass $M_{\max}$ [$M_\oplus$]"),
        dict(key="n_collisions", title="cumulative mergers"),
    ]
    return _stacked_timeseries(runs, cols, path_noext, tmin, ref, base)


def fig_velocity_stacked(runs, path_noext, tmin=None, ref=True, base=None):
    cols = [
        dict(key="e_rms", yscale="log",
             title=r"$\langle e^2\rangle^{1/2}$ (dimensionless)",
             init=lambda r: r["cfg"]["e_rms"]),
        dict(key="i_rms", yscale="log",
             title=r"$\langle i^2\rangle^{1/2}$ [rad]",
             init=lambda r: r["cfg"]["i_rms"]),
        dict(key="v_disp_ms", yscale="log",
             title=r"velocity dispersion [m s$^{-1}$]"),
    ]
    return _stacked_timeseries(runs, cols, path_noext, tmin, ref)


def _final_elements(r):
    snapdir = os.path.join(r["dir"], "snapshots")
    f = os.path.join(snapdir, "snap_final.npz")
    if not os.path.exists(f):
        cand = sorted(x for x in os.listdir(snapdir) if x.endswith(".npz"))
        if not cand:
            return None
        f = os.path.join(snapdir, cand[-1])
    z = np.load(f)
    pos = z["pos"] - z["star_pos"]
    vel = z["vel"] - z["star_vel"]
    mass = z["mass"]
    a, e, _ = state_to_elements(pos, vel, u.G * (1.0 + mass))
    return dict(a=a, e=e, mass=mass, t=float(z["t"]),
                planet_on=bool(z["planet_on"]))


def fig_ae_stacked(runs, path_noext, xlim=(0.5, 1.3), xscale="linear"):
    """Final a-e plane, one row per scenario, shared a and e axes.

    Bodies outside the plotted a-range (scattered or ejected) cannot share
    the axis without crushing the annulus, so they are COUNTED in each panel
    instead of silently dropped.
    """
    nrow = len(runs)
    fig, axes = plt.subplots(nrow, 1, figsize=(6.2, 1.75 * nrow + 0.5),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[:, 0]
    mmin = min(r["cfg"]["m_particle_msun"] for r in runs)
    for ax, r in zip(axes, runs):
        el = _final_elements(r)
        if el is None:
            continue
        a, e, m = el["a"], el["e"], el["mass"]
        bound = (a > 0) & (e < 1)
        inside = bound & (a >= xlim[0]) & (a <= xlim[1])
        s = 3.0 * (m[inside] / mmin) ** (1.0 / 3.0)
        ax.scatter(a[inside], e[inside], s=s, alpha=0.6, edgecolors="none",
                   color=COLOURS.get(r["name"], "k"))
        n_out = int((bound & ~inside).sum())
        n_unb = int((~bound).sum())
        note = f"N = {len(m)}"
        if n_out:
            note += f"\n{n_out} bound outside {xlim[0]}–{xlim[1]} AU"
        if n_unb:
            note += f"\n{n_unb} unbound (ejected)"
        ax.text(0.015, 0.06, note, transform=ax.transAxes, fontsize=7,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                          alpha=0.8))
        ax.axvspan(r["cfg"]["a_in"], r["cfg"]["a_out"], color="grey",
                   alpha=0.10, lw=0, zorder=0)
        if el["planet_on"]:
            ap = r["cfg"]["a_planet"]
            for p, q in ((3, 1), (5, 2), (2, 1), (5, 3), (3, 2)):
                a_res = ap * (q / p) ** (2.0 / 3.0)
                if xlim[0] < a_res < xlim[1]:
                    ax.axvline(a_res, color="grey", lw=0.7, ls="--")
                    ax.text(a_res, 0.97, f" {p}:{q}", fontsize=7,
                            va="top", ha="left",
                            transform=ax.get_xaxis_transform())
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_ylabel("eccentricity $e$")
        _row_label(ax, r["name"])
    axes[0].set_xscale(xscale)
    axes[0].set_xlim(*xlim)
    axes[0].set_ylim(1e-4, 1.0)
    axes[-1].set_xlabel("semi-major axis $a$ [AU]")
    fig.tight_layout(h_pad=0.4)
    return _save(fig, path_noext)


def fig_impact_stacked(runs, path_noext):
    """v_rel/v_esc at every detected impact, one row per scenario.

    Filled = impacts that merged, outline = all detected impacts. In
    perfect-merging runs the two coincide.
    """
    nrow = len(runs)
    fig, axes = plt.subplots(nrow, 1, figsize=(6.2, 1.6 * nrow + 0.5),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[:, 0]
    bins = np.logspace(-1, 2, 49)   # 0.1-100: no impact falls outside
    for ax, r in zip(axes, runs):
        c = r["collisions"]
        if c:
            ratio = np.array([x["v_rel"] / x["v_esc"] for x in c])
            acc = np.array([x.get("accreted", True) for x in c], dtype=bool)
            w = np.full(len(ratio), 1.0 / len(ratio))   # fraction per bin
            ax.hist(ratio[acc], bins=bins, weights=w[acc],
                    color=COLOURS.get(r["name"], "k"), alpha=0.35,
                    label="merged")
            ax.hist(ratio, bins=bins, weights=w, histtype="step",
                    color=COLOURS.get(r["name"], "k"), lw=1.2,
                    label="all impacts")
            ax.text(0.985, 0.92,
                    f"{len(ratio)} impacts\n"
                    f"{100*np.mean(ratio > 1):.1f}% with $v_{{rel}}>v_{{esc}}$\n"
                    f"{int(acc.sum())} merged, {int((~acc).sum())} rejected",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7)
        ax.axvline(1.0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.set_ylabel("fraction")
        _row_label(ax, r["name"])
    axes[0].legend(loc="upper left", frameon=False)
    axes[-1].set_xlabel(r"$v_{\rm rel}/v_{\rm esc}$ at impact")
    fig.tight_layout(h_pad=0.4)
    return _save(fig, path_noext)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
SHARED_KEYS = ("seed", "n_particles", "a_in", "a_out", "sigma1_solid_cgs",
               "rho_p_cgs", "e_tilde", "i_over_e", "f_radius",
               "merge_criterion", "merge_velocity_factor", "theta",
               "softening_frac", "tree_method", "steps_per_inner_orbit",
               "t_end", "dt_yr", "m_particle_msun", "r_particle_km")
SCENARIO_KEYS = ("gas_on", "gas_boost", "planet_on", "m_planet", "a_planet",
                 "e_planet")


def write_info(figdir, runs, made, argv):
    L = ["STACKED FIGURE SET RECORD", "=" * 62, "",
         f"generated : {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}",
         f"command   : {' '.join(argv)}", "", "[ SOURCE RUNS ]"]
    for r in runs:
        s = r["series"][-1]
        L.append(f"  {r['name']:<16} {r['folder']}")
        L.append(f"      t_end {s['t']:.0f} yr | N {r['cfg']['n_particles']}"
                 f" -> {s['n_bodies']} | M_max {s['m_max_mearth']:.4e} M_E")
        L.append("      " + ", ".join(f"{k}={r['cfg'].get(k)}"
                                      for k in SCENARIO_KEYS))
    L += ["", "[ CONTROLLED-COMPARISON CHECK ]"]
    ok = True
    for k in SHARED_KEYS:
        vals = {r["name"]: r["cfg"].get(k) for r in runs}
        if len(set(map(repr, vals.values()))) == 1:
            L.append(f"  {k:24s}: {next(iter(vals.values()))}")
        else:
            ok = False
            L.append(f"  {k:24s}: !! DIFFERS " + ", ".join(
                f"{n}={v}" for n, v in vals.items()))
    L.append("")
    L.append("  All shared parameters identical." if ok else
             "  !! WARNING: NOT a controlled comparison.")
    L += ["", "[ FIGURES ]"] + ["  " + os.path.basename(m) for m in made]
    with open(os.path.join(figdir, "stacked_info.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return ok


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("outdir", nargs="?", default="results")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="explicit run folders (paths)")
    ap.add_argument("--figdir", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--ref-run", default=None,
                    help="run folder drawn only as the grey S1 reference, "
                         "without a row of its own (e.g. when plotting S2 and S4 only)")
    ap.add_argument("--tmin", type=float, default=None)
    ap.add_argument("--ae-xscale", default="linear",
                    choices=["linear", "log"])
    ap.add_argument("--ae-xlim", nargs=2, type=float, default=(0.5, 1.3))
    args = ap.parse_args(argv)

    dirs = args.runs if args.runs else find_newest(args.outdir)
    if not dirs:
        raise SystemExit("no finished runs found")
    runs = sort_runs([load_run(d) for d in dirs])

    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S")
    figdir = args.figdir or os.path.join(
        args.outdir, f"stacked_{stamp}" + (f"_{args.tag}" if args.tag else ""))
    os.makedirs(figdir, exist_ok=True)

    ref = not args.no_ref
    base = load_run(args.ref_run) if args.ref_run else None
    made = []
    made += fig_growth_stacked(runs, os.path.join(figdir, "fig_growth_stacked"),
                               args.tmin, ref, base)
    made += fig_velocity_stacked(runs, os.path.join(figdir,
                                                    "fig_velocity_stacked"),
                                 args.tmin, ref, base)
    made += fig_ae_stacked(runs, os.path.join(figdir, "fig_ae_stacked"),
                           tuple(args.ae_xlim), args.ae_xscale)
    made += fig_impact_stacked(runs, os.path.join(figdir,
                                                  "fig_impact_stacked"))
    ok = write_info(figdir, runs, made, sys.argv if argv is None else argv)
    for m in made:
        print("wrote", m)
    print("controlled comparison:", "OK" if ok else "!! parameters differ")
    print("figure set:", figdir)


if __name__ == "__main__":
    main()
