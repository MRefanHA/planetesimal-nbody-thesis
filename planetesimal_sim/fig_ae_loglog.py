"""
Final e vs a, four scenarios stacked top to bottom on shared log-log axes.

Usage (from C:\\thesis\\planetesimal_sim_thesis):

    python -m planetesimal_sim.fig_ae_loglog \\
        results/2026-09-15_124025_S1_gravity_only \\
        results/2026-09-16_064320_S2_gas \\
        results/2026-09-17_062814_S3_planet \\
        results/2026-09-19_061113_S4_gas_planet \\
        --out fig_ae_loglog

Give one to four run folders (any of S1-S4, in the order the panels should
appear). Writes <out>.png (300 dpi) and <out>.pdf (vector).

Markers: area grows with mass (m/m_min)^(2/3); each marker has a solid
outline and a transparent fill, and bodies are drawn from the lightest to the
heaviest, so a massive body is never hidden under the small ones.
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.colors import to_rgba          # noqa: E402
from matplotlib.ticker import ScalarFormatter, NullFormatter, FixedLocator  # noqa: E402

from . import units as u                              # noqa: E402
from .orbits import state_to_elements                 # noqa: E402

LABELS = ["S1: gravity only", "S2: + nebular gas",
          "S3: + giant planet", "S4: + gas and giant planet"]
COLOURS = ["0.25", "tab:blue", "tab:red", "tab:purple"]
XTICKS = [0.6, 0.7, 0.8, 1.0, 1.2, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
RESONANCES = ((3, 1), (5, 2), (2, 1), (3, 2))
A_MAX = 5.0


def final_elements(run_dir):
    f = os.path.join(run_dir, "snapshots", "snap_final.npz")
    z = np.load(f)
    pos = z["pos"] - z["star_pos"]
    vel = z["vel"] - z["star_vel"]
    m = z["mass"]
    a, e, _ = state_to_elements(pos, vel, u.G * (1.0 + m))
    return a, e, m, bool(z["planet_on"])


def scenario_index(run_dir):
    try:
        with open(os.path.join(run_dir, "config.json")) as fh:
            name = json.load(fh).get("name", "")
    except (OSError, ValueError):
        name = os.path.basename(run_dir.rstrip("/"))
    for k in range(4):
        if f"S{k + 1}_" in name or name.startswith(f"S{k + 1}"):
            return k
    raise SystemExit(f"cannot tell the scenario of {run_dir}")


def planet_a(run_dir, default=1.6):
    try:
        with open(os.path.join(run_dir, "config.json")) as fh:
            return float(json.load(fh).get("a_planet", default))
    except (OSError, ValueError):
        return default


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run folders, one panel each (S1-S4)")
    ap.add_argument("--fill-alpha", type=float, default=0.25)
    ap.add_argument("--out", default="fig_ae_loglog")
    ap.add_argument("--xmax", type=float, default=4.8,
                    help="right end of the a axis in AU (default 4.8)")
    args = ap.parse_args(argv)

    n = len(args.runs)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8.5, 2.45 * n + 0.7),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[:, 0]
    top = 1.0 - 0.85 / (2.45 * n + 0.7)
    fig.subplots_adjust(hspace=0.08, top=top)

    for ax, run in zip(axes, args.runs):
        k = scenario_index(run)
        a, e, m, planet_on = final_elements(run)
        ok = (a > 0) & (a < A_MAX) & (e < 1.0)
        order = np.argsort(m[ok])                 # heaviest drawn last, on top
        aa, ee, mm = a[ok][order], e[ok][order], m[ok][order]
        size = 6.0 * (mm / m.min()) ** (2.0 / 3.0)
        big = mm >= 10.0 * m.min() * (1 - 1e-9)       # protoplanet seeds
        dark = tuple(0.55 * c for c in to_rgba(COLOURS[k])[:3]) + (1.0,)
        ax.scatter(aa[~big], ee[~big], s=size[~big],
                   facecolors=[to_rgba(COLOURS[k], args.fill_alpha)],
                   edgecolors=[dark], linewidths=0.35, zorder=3)
        ax.scatter(aa[big], ee[big], s=size[big],
                   facecolors=[to_rgba(COLOURS[k], 0.55)],
                   edgecolors=[dark], linewidths=1.2, zorder=4)

        ax.axvspan(0.95, 1.05, color="0.85", alpha=0.6, lw=0, zorder=0)

        ap_ = planet_a(run)
        for p, q in RESONANCES:
            a_res = ap_ * (q / p) ** (2.0 / 3.0)
            if planet_on:
                ax.axvline(a_res, color="0.4", ls="--", lw=1.0, zorder=1)
                ax.text(a_res, 0.93, f"{p}:{q}", rotation=90,
                        transform=ax.get_xaxis_transform(),
                        ha="right", va="top", fontsize=8)
            else:
                ax.axvline(a_res, color="0.6", ls=":", lw=0.8, zorder=1)
        if planet_on:
            ax.axvline(ap_, color="darkorange", lw=1.4, zorder=2)
            ax.text(ap_ * 1.012, 0.04, f"Giant Planet ({ap_:g} AU)",
                    rotation=90, color="darkorange", fontsize=9,
                    transform=ax.get_xaxis_transform(),
                    ha="left", va="bottom")

        a_cut = min(A_MAX, args.xmax)
        n_out = int((~(ok & (a <= a_cut))).sum())
        if n_out:
            ax.text(0.985, 0.05 if args.xmax > 3 else 0.68, f"{n_out} at a ≥ {a_cut:g} AU / unbound →",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8.5, color=COLOURS[k], zorder=5,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))

        ax.text(0.985, 0.90, LABELS[k], transform=ax.transAxes,
                ha="right", va="top", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8"))
        ax.set_ylabel("eccentricity $e$ (dimensionless)")
        ax.grid(alpha=0.25, which="major")

    m0 = final_elements(args.runs[0])[2].min()
    for mk, lab in ((1, "1 $m_0$"), (10, "10 $m_0$"), (100, "100 $m_0$")):
        axes[0].scatter([], [], s=6.0 * mk ** (2.0 / 3.0), facecolors="none",
                        edgecolors="0.3", linewidths=0.8, label=lab)
    axes[0].scatter([], [], s=6.0 * 10 ** (2.0 / 3.0), facecolors="0.6",
                    edgecolors="0.2", linewidths=1.2, label=r"$\geq 10\,m_0$ (thick outline)")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.5, top + 0.005),
               title="marker size = mass", fontsize=9, title_fontsize=9,
               frameon=False, ncol=4, columnspacing=2.0)

    ax0 = axes[0]
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlim(0.58, args.xmax)
    ax0.set_ylim(1e-4, 1.8)
    ax0.xaxis.set_major_locator(FixedLocator([x for x in XTICKS if x <= args.xmax]))
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    ax0.xaxis.set_major_formatter(fmt)
    ax0.xaxis.set_minor_formatter(NullFormatter())
    axes[-1].set_xlabel("semi-major axis [AU]")

    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", f"{args.out}.png", f"{args.out}.pdf")


if __name__ == "__main__":
    main()
