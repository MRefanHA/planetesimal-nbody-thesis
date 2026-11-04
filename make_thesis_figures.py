#!/usr/bin/env python3
"""
Generate the schematic figures required by Chapters II and III.

    python make_thesis_figures.py [--outdir figures]

Produces each figure as both PNG (for Word) and PDF (vector, for LaTeX later).
Greyscale-safe: nothing depends on colour to be readable when printed.

Figures produced
----------------
  Figure 2.1  Barnes-Hut opening-angle criterion          (Section 2.5, Eq. 2.4)
  Figure 3.1  Simulated system geometry                   (Section 3.8)
  Figure 3.2  Research workflow flowchart                 (Section 3.9)
  Figure 3.3  Within-timestep collision detection         (Section 3.5 stage 3)
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import (Rectangle, Circle,   # noqa: E402
                                FancyArrowPatch, Wedge, FancyBboxPatch)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.8,
})

GREY = "0.55"
LIGHT = "0.88"


def save(fig, outdir, name):
    for ext in ("png", "pdf"):
        p = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png and {name}.pdf")


# ---------------------------------------------------------------------------
def figure_2_1(outdir):
    """Barnes-Hut opening angle: theta = s/d, thesis Eq. (2.4)."""
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    rng = np.random.default_rng(7)

    # a distant cluster of bodies, approximated as one cell
    cx, cy, s = 6.4, 1.7, 2.0
    cluster = rng.normal([cx, cy], 0.42, size=(14, 2))

    # quadtree-style subdivision inside the cell
    ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s, fill=False,
                           ec="k", lw=1.4))
    for k in (0, 1):
        ax.plot([cx - s / 2 + s / 2, cx - s / 2 + s / 2],
                [cy - s / 2, cy + s / 2], color=GREY, lw=0.6)
        ax.plot([cx - s / 2, cx + s / 2],
                [cy - s / 2 + s / 2, cy - s / 2 + s / 2],
                color=GREY, lw=0.6)
    for qx in (cx - s / 2, cx):
        for qy in (cy - s / 2, cy):
            ax.plot([qx + s / 4, qx + s / 4], [qy, qy + s / 2],
                    color=LIGHT, lw=0.4)
            ax.plot([qx, qx + s / 2], [qy + s / 4, qy + s / 4],
                    color=LIGHT, lw=0.4)

    ax.scatter(cluster[:, 0], cluster[:, 1], s=9, c="k", zorder=3)

    # centre of mass of the cell
    com = cluster.mean(axis=0)
    ax.scatter(*com, s=90, marker="x", c="k", lw=1.6, zorder=4)
    ax.text(com[0] + 0.12, com[1] + 0.55, "centre of mass\nof cell",
            fontsize=7.5, ha="left")

    # the body being acted on
    bx, by = 0.7, 1.0
    ax.scatter([bx], [by], s=42, c="k", zorder=4)
    ax.text(bx, by - 0.45, "body $i$", fontsize=8.5, ha="center")

    # distance d
    ax.annotate("", xy=(com[0], com[1]), xytext=(bx, by),
                arrowprops=dict(arrowstyle="<->", lw=1.0, color="k"))
    ax.text((bx + com[0]) / 2, (by + com[1]) / 2 + 0.22, "$d$",
            fontsize=11, ha="center")

    # cell size s
    ax.annotate("", xy=(cx - s / 2, cy - s / 2 - 0.30),
                xytext=(cx + s / 2, cy - s / 2 - 0.30),
                arrowprops=dict(arrowstyle="<->", lw=1.0, color="k"))
    ax.text(cx, cy - s / 2 - 0.62, "$s$", fontsize=11, ha="center")

    ax.text(3.5, 3.35,
            r"$\theta = \dfrac{s}{d}$", fontsize=14, ha="center")
    ax.text(3.5, 2.62,
            "if $\\theta <$ threshold: treat the whole cell as a single\n"
            "mass at its centre of mass.  Otherwise: open the cell\n"
            "and evaluate its children individually.",
            fontsize=7.5, ha="center", va="top")

    ax.set_xlim(-0.2, 8.2)
    ax.set_ylim(-0.35, 3.9)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, outdir, "figure_2_1_opening_angle")


# ---------------------------------------------------------------------------
def figure_3_1(outdir, a_in=0.95, a_out=1.05, a_p=1.6, e_p=0.05):
    """Geometry of the simulated system (Section 3.8)."""
    fig, ax = plt.subplots(figsize=(5.4, 5.0))

    # the annulus, drawn to scale
    ax.add_patch(Wedge((0, 0), a_out, 0, 360, width=a_out - a_in,
                       facecolor=LIGHT, edgecolor="k", lw=0.9, zorder=1))

    # planetesimals sampled from the real IC distribution
    rng = np.random.default_rng(3)
    n = 260
    q = 0.5
    a = (a_in ** q + rng.random(n) * (a_out ** q - a_in ** q)) ** (1 / q)
    th = rng.uniform(0, 2 * np.pi, n)
    ax.scatter(a * np.cos(th), a * np.sin(th), s=2.2, c="k", zorder=3)

    # central star
    ax.scatter([0], [0], s=190, marker="*", c="k", zorder=5)
    ax.text(0.0, -0.16, r"$M_\star$", fontsize=10, ha="center", va="top")

    # giant planet orbit
    nu = np.linspace(0, 2 * np.pi, 400)
    r = a_p * (1 - e_p ** 2) / (1 + e_p * np.cos(nu))
    ax.plot(r * np.cos(nu), r * np.sin(nu), "k--", lw=1.0, zorder=2)
    ax.scatter([a_p * (1 - e_p)], [0], s=90, c="k", zorder=5)
    ax.text(a_p * (1 - e_p), 0.13, "giant planet\n(Scenarios 3, 4)",
            fontsize=7.5, ha="center", va="bottom")

    # the 2:1 interior mean-motion resonance (falls inside the annulus)
    a_res = a_p * (1 / 2) ** (2 / 3)
    ax.plot(a_res * np.cos(nu), a_res * np.sin(nu), color="0.25",
            lw=1.0, ls=":", zorder=4)
    ax.annotate(f"2:1 MMR at {a_res:.3f} AU",
                xy=(-a_res * 0.707, a_res * 0.707),
                xytext=(-1.55, 1.30), fontsize=7.5, color="0.15",
                ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="0.35",
                                connectionstyle="arc3,rad=0.2"))

    # radial labels, each along its own direction so they cannot collide
    for val, lab, ang in ((a_in, r"$a_{\rm in}$", 118),
                          (a_out, r"$a_{\rm out}$", 62),
                          (a_p, r"$a_{\rm p}$", 250)):
        th_r = np.radians(ang)
        ex, ey = val * np.cos(th_r), val * np.sin(th_r)
        ax.annotate("", xy=(ex, ey), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", lw=0.9, color="0.25"))
        ax.text(ex * 1.10, ey * 1.10, lab, fontsize=11, color="k",
                ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=0.9, alpha=0.85))

    lim = a_p * 1.25
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ [AU]")
    ax.set_ylabel("$y$ [AU]")
    ax.grid(alpha=0.18, lw=0.5)
    ax.set_axisbelow(True)
    save(fig, outdir, "figure_3_1_system_geometry")


# ---------------------------------------------------------------------------
def figure_3_2(outdir):
    """Research workflow flowchart (Section 3.9)."""
    fig, ax = plt.subplots(figsize=(5.0, 8.4))

    steps = [
        ("Initial-condition generator\nEqs. (3.1)-(3.3), (2.2)", "box"),
        ("Validate:\ne, i, a distributions match targets", "check"),
        ("Gravity-only integrator (Scenario 1)\nleapfrog KDK + pytreegrav", "box"),
        ("Validate:\ntwo-body Kepler orbit", "check"),
        ("Collision + merger module\nswept-sphere detection, Eq. (3.4)", "box"),
        ("Validate:\nhead-on merger, mass and momentum", "check"),
        ("Gas-drag module (Scenario 2)\nEq. (3.5)", "box"),
        ("Giant-planet module (Scenario 3)", "box"),
        ("Combined run (Scenario 4)", "box"),
        ("Timestep convergence test\nselect $dt$", "check"),
        ("Production runs\nall four scenarios, identical settings", "box"),
        ("Comparative analysis\ntimescale, $M_{\\max}$, $e_{\\rm rms}$", "box"),
    ]

    y = len(steps)
    dy = 1.0
    for i, (label, kind) in enumerate(steps):
        yy = y - i * dy
        if kind == "check":
            box = FancyBboxPatch((0.14, yy - 0.30), 0.72, 0.56,
                                 boxstyle="round,pad=0.015",
                                 fc="white", ec="0.35", lw=0.9, ls="--")
        else:
            box = FancyBboxPatch((0.06, yy - 0.32), 0.88, 0.60,
                                 boxstyle="round,pad=0.015",
                                 fc=LIGHT, ec="k", lw=1.1)
        ax.add_patch(box)
        ax.text(0.5, yy - 0.02, label, ha="center", va="center",
                fontsize=7.8)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch(
                (0.5, yy - 0.33), (0.5, yy - dy + 0.30),
                arrowstyle="-|>", mutation_scale=11, lw=0.9, color="k"))

    ax.text(0.965, y - 1 * dy, "solid = implement\ndashed = validate",
            fontsize=6.8, ha="right", va="center", color="0.35")

    ax.set_xlim(0, 1)
    ax.set_ylim(y - len(steps) * dy - 0.1, y + 0.55)
    ax.axis("off")
    save(fig, outdir, "figure_3_2_workflow")


# ---------------------------------------------------------------------------
def figure_3_3(outdir):
    """Why a snapshot overlap test fails and the swept test does not."""
    fig, axes = plt.subplots(2, 1, figsize=(5.8, 5.6))

    r_coll = 0.34                       # f (R_i + R_j), drawn to scale
    p1a, p1b = np.array([0.5, 0.55]), np.array([3.5, 1.75])
    p2a, p2b = np.array([3.5, 0.55]), np.array([0.5, 1.75])

    for ax, mode in zip(axes, ("snapshot", "swept")):
        for pa, pb, lab in ((p1a, p1b, "i"), (p2a, p2b, "j")):
            ax.annotate("", xy=pb, xytext=pa,
                        arrowprops=dict(arrowstyle="-|>", lw=1.0,
                                        color="0.45"))
            ax.add_patch(Circle(pa, r_coll / 2, fc="white", ec="k", lw=1.2,
                                zorder=6))
            ax.add_patch(Circle(pb, r_coll / 2, fc=LIGHT, ec="k", lw=1.2,
                                ls=(0, (2, 1.6)), zorder=6))
            ax.text(pa[0], pa[1], f"${lab}$", fontsize=8, ha="center",
                    va="center", zorder=7)
            ax.text(pb[0], pb[1], f"${lab}$", fontsize=8, ha="center",
                    va="center", zorder=7)

        ax.text(p1a[0] - 0.28, p1a[1] - 0.26, "at $t$", fontsize=7)
        ax.text(p1b[0] - 0.10, p1b[1] + 0.24, "at $t+\\Delta t$", fontsize=7)
        ax.text(p2a[0] - 0.30, p2a[1] - 0.26, "at $t$", fontsize=7)
        ax.text(p2b[0] - 0.70, p2b[1] + 0.24, "at $t+\\Delta t$", fontsize=7)

        if mode == "snapshot":
            for pa, pb in ((p1a, p2a), (p1b, p2b)):
                ax.annotate("", xy=pb, xytext=pa,
                            arrowprops=dict(arrowstyle="<->", lw=0.9,
                                            color="0.25", ls=":"))
            ax.text(2.0, 0.30, "separation large at $t$", fontsize=7.5,
                    ha="center", color="0.2")
            ax.text(2.0, 1.99, "separation large at $t+\\Delta t$",
                    fontsize=7.5, ha="center", color="0.2")
            ax.set_title("(a)  Snapshot test: overlap checked only at the step "
                         "boundaries.  NOT DETECTED",
                         fontsize=8.2, loc="left")
        else:
            dr = p2a - p1a
            dv = (p2b - p2a) - (p1b - p1a)
            tstar = float(np.clip(-dr @ dv / (dv @ dv), 0, 1))
            c1 = p1a + (p1b - p1a) * tstar
            c2 = p2a + (p2b - p2a) * tstar
            ax.add_patch(Circle(c1, r_coll / 2, fc="0.6", ec="k", lw=1.3,
                                zorder=7))
            ax.add_patch(Circle(c2, r_coll / 2, fc="0.6", ec="k", lw=1.3,
                                zorder=7))
            ax.plot([c1[0], c2[0]], [c1[1], c2[1]], "k-", lw=1.3, zorder=6)
            ax.annotate("$d_{\\min} < f(R_i+R_j)$\n$\\Rightarrow$ COLLISION",
                        xy=(0.5 * (c1[0] + c2[0]), 0.5 * (c1[1] + c2[1])),
                        xytext=(3.15, 0.98), fontsize=8, ha="center",
                        va="center",
                        arrowprops=dict(arrowstyle="->", lw=0.8,
                                        color="0.3",
                                        connectionstyle="arc3,rad=-0.25"))
            ax.text(0.72, 1.13,
                    "$t^{\\ast}=\\mathrm{clip}\\!\\left(-\\dfrac{\\Delta r\\cdot"
                    "\\Delta v}{\\Delta v\\cdot\\Delta v},\\,0,\\,\\Delta t"
                    "\\right)$",
                    fontsize=8.5, ha="center", va="center",
                    bbox=dict(fc="white", ec="0.7", lw=0.6, pad=2.5))
            ax.set_title("(b)  Swept test: minimum separation evaluated along "
                         "the step.  DETECTED",
                         fontsize=8.2, loc="left")

        ax.set_xlim(-0.1, 4.5)
        ax.set_ylim(0.12, 2.22)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.tight_layout(h_pad=1.6)
    save(fig, outdir, "figure_3_3_collision_detection")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    print(f"writing figures to {args.outdir}/")
    figure_2_1(args.outdir)
    figure_3_1(args.outdir)
    figure_3_2(args.outdir)
    figure_3_3(args.outdir)
    print("done")


if __name__ == "__main__":
    main()
