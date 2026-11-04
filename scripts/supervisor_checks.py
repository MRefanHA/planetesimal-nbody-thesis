"""
supervisor_checks.py -- extra diagnostics asked for in the supervisor review
(Chapter IV revision, November 2026).

Every number quoted in the new parts of Chapter IV comes from this script,
run on the original run folders. Run it yourself to reproduce them:

    python supervisor_checks.py RUN_FOLDER [RUN_FOLDER ...] --fig fig_initial

A run folder is one that contains config.json, series.json, checkpoint.npz
and snapshots/. The script prints a report and, with --fig, draws the
initial-condition histograms (semi-major axis and random velocity at t = 0).

What it checks
  1. Are the starting positions and velocities identical in every run?
     (compared relative to the Sun, because runs with a planet are set up in
      the barycentric frame, which shifts the Sun slightly)
  2. Formation time for several thresholds: t5, t10, t20, t50, t100
     (time at which the largest body first reaches k times the initial mass,
      read from series.json at the 100 yr output interval)
  3. How many bodies have grown to 10 m0 or more (final state and snapshots)
  4. Bodies far from the ring: a >= 5 AU, a < 0 (unbound), a > 50 AU, and
     bound bodies whose orbit reaches beyond 5 AU (apocentre Q > 5 AU) but
     still passes through the ring (pericentre q < 1.05 AU). Also e_rms with
     and without the bodies at a >= 5 AU.
"""
import argparse, glob, json, os, re
import numpy as np

G = 4 * np.pi ** 2                      # AU^3 / (Msun yr^2)
AUYR_TO_MS = 1.495978707e11 / 3.15576e7
MSUN_TO_MEARTH = 332946.0487


def load_run(d):
    cfg = json.load(open(os.path.join(d, "config.json")))
    s = json.load(open(os.path.join(d, "series.json")))
    series = s if isinstance(s, list) else s["series"]
    snaps = glob.glob(os.path.join(d, "snapshots", "snap_t*.npz"))
    snaps = sorted(snaps, key=lambda f: float(re.search(r"snap_t([\d.]+)\.npz", f).group(1)))
    return dict(dir=d, cfg=cfg, series=series, snaps=snaps,
                final=np.load(os.path.join(d, "checkpoint.npz")))


def label(r):
    c = r["cfg"]
    rule = "perfect" if c["merge_criterion"] == "always" else "velocity"
    boost = "" if c.get("gas_boost", 1.0) == 1.0 or not c["gas_on"] else f", gas x{c['gas_boost']:g}"
    return f"{c['name'][:2]} ({rule}{boost})"


def elements(z):
    """heliocentric a, e, i (rad) of every body in a snapshot/checkpoint"""
    r = z["pos"] - z["star_pos"]
    v = z["vel"] - z["star_vel"]
    mu = G * (1.0 + z["mass"])
    rr = np.linalg.norm(r, axis=1)
    v2 = (v * v).sum(1)
    a = 1.0 / (2.0 / rr - v2 / mu)
    h = np.cross(r, v)
    hm = np.linalg.norm(h, axis=1)
    e = np.linalg.norm(np.cross(v, h) / mu[:, None] - r / rr[:, None], axis=1)
    inc = np.arccos(np.clip(h[:, 2] / hm, -1, 1))
    return a, e, inc, r, v


def v_random(r, v):
    """speed relative to the local circular Keplerian velocity (m/s)"""
    R = np.hypot(r[:, 0], r[:, 1])
    vc = np.sqrt(G / R)
    vcirc = np.stack([-vc * r[:, 1] / R, vc * r[:, 0] / R, np.zeros_like(R)], 1)
    return np.linalg.norm(v - vcirc, axis=1) * AUYR_TO_MS


def t_threshold(series, k):
    m0 = series[0]["m_max_mearth"]
    for x in series:
        if x["m_max_mearth"] >= k * m0 * (1 - 1e-9):
            return x["t"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--fig", default=None)
    args = ap.parse_args()
    runs = [load_run(d) for d in args.runs]

    print("=" * 78)
    print("1. STARTING POSITIONS AND VELOCITIES (relative to the Sun, t = 0)")
    print("=" * 78)
    ref = np.load(runs[0]["snaps"][0])
    _, _, _, r0, v0 = elements(ref)
    for r in runs:
        z = np.load(r["snaps"][0])
        _, _, _, rr, vv = elements(z)
        print(f"  {label(r):28s} t={float(z['t']):.0f}  N={len(rr)}  "
              f"max |dr| = {np.abs(rr - r0).max():.1e} AU   "
              f"max |dv| = {np.abs(vv - v0).max() * AUYR_TO_MS:.1e} m/s   "
              f"Sun offset = {np.linalg.norm(z['star_pos']):.5f} AU")

    print()
    print("=" * 78)
    print("2. FORMATION TIME FOR SEVERAL THRESHOLDS (yr; '-' = not reached)")
    print("=" * 78)
    ks = (2, 5, 10, 20, 50, 100)
    print("  " + f"{'run':28s}" + "".join(f"{'t' + str(k):>9s}" for k in ks))
    for r in runs:
        vals = [t_threshold(r["series"], k) for k in ks]
        print("  " + f"{label(r):28s}" + "".join(f"{(f'{v:.0f}' if v is not None else '-'):>9s}" for v in vals))

    print()
    print("=" * 78)
    print("3. NUMBER OF BODIES WITH m >= 10 m0")
    print("=" * 78)
    for r in runs:
        m0 = r["series"][0]["m_max_mearth"] / MSUN_TO_MEARTH
        mf = r["final"]["mass"]
        big = int((mf >= 10 * m0 * (1 - 1e-9)).sum())
        hist = []
        for f in r["snaps"]:
            z = np.load(f)
            hist.append((float(z["t"]), int((z["mass"] >= 10 * m0 * (1 - 1e-9)).sum())))
        print(f"  {label(r):28s} end: {big:3d}   (>= 5 m0: {int((mf >= 5*m0*(1-1e-9)).sum()):3d}, "
              f">= 50 m0: {int((mf >= 50*m0*(1-1e-9)).sum()):2d})")
        print("      by snapshot: " + ", ".join(f"{t:.0f} yr: {n}" for t, n in hist))

    print()
    print("=" * 78)
    print("4. BODIES FAR FROM THE RING, final state (t = 2e4 yr)")
    print("=" * 78)
    for r in runs:
        a, e, inc, rr, vv = elements(r["final"])
        unb = a < 0
        far = (a >= 5) & ~unb
        a50 = a > 50
        bound = (a > 0) & (a < 50)          # the definition used by the code
        q, Q = a * (1 - e), a * (1 + e)
        cross = bound & (Q > 5) & (q < 1.05)
        erms_all = np.sqrt(np.mean(e[bound] ** 2))
        inner = (a > 0) & (a < 5)
        erms_in = np.sqrt(np.mean(e[inner] ** 2))
        irms_all = np.sqrt(np.mean(inc[bound] ** 2))
        irms_in = np.sqrt(np.mean(inc[inner] ** 2))
        print(f"  {label(r):28s} N={len(a)}  a>=5 AU (bound): {far.sum():3d}  "
              f"unbound (a<0): {unb.sum():3d}  a>50 AU: {a50.sum():3d}  "
              f"[code's 'unbound' = a<0 or a>50: {(~bound).sum()}]")
        print(f"      bound orbits reaching beyond 5 AU yet crossing the ring (Q>5, q<1.05): {cross.sum()}")
        print(f"      e_rms all bound = {erms_all:.3f}, only a<5 AU = {erms_in:.3f};  "
              f"i_rms all bound = {irms_all:.3f} rad, only a<5 AU = {irms_in:.3f} rad")

    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        z = np.load(runs[0]["snaps"][0])
        a, e, inc, rr, vv = elements(z)
        vr = v_random(rr, vv)
        cfg = runs[0]["cfg"]
        m = z["mass"][0]; R = z["radius"][0]
        vesc = np.sqrt(2 * G * 2 * m / (2 * R)) * AUYR_TO_MS
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        bins = np.linspace(cfg["a_in"], cfg["a_out"], 21)
        ax[0].hist(a, bins=bins, color="0.55", edgecolor="k", lw=0.6)
        # expected count per bin for Sigma ~ a^-3/2  ->  dN/da ~ a^-1/2
        x = np.linspace(cfg["a_in"], cfg["a_out"], 200)
        norm = len(a) * (bins[1] - bins[0]) / np.trapezoid(x ** -0.5, x)
        ax[0].plot(x, norm * x ** -0.5, "r--", lw=1.4, label=r"expected for $\Sigma \propto a^{-3/2}$")
        ax[0].set_xlabel("semi-major axis $a$ [AU]")
        ax[0].set_ylabel("number of bodies")
        ax[0].set_title("(a) semi-major axis at $t = 0$", pad=14)
        ax[0].set_ylim(0, 1.3 * np.histogram(a, bins)[0].max())
        ax[0].legend(loc="upper left", fontsize=8.5, frameon=False)
        # (b) orbital speed around the Sun, with the speed of a circular orbit
        #     at each body's own semi-major axis for comparison
        vorb = np.linalg.norm(vv, axis=1) * AUYR_TO_MS / 1e3          # km/s
        vcirc_a = np.sqrt(G * (1.0 + z["mass"]) / a) * AUYR_TO_MS / 1e3
        vk_in = np.sqrt(G / cfg["a_out"]) * AUYR_TO_MS / 1e3
        vk_out = np.sqrt(G / cfg["a_in"]) * AUYR_TO_MS / 1e3
        vb = np.linspace(min(vorb.min(), vk_in) - 0.05, max(vorb.max(), vk_out) + 0.05, 31)
        ax[1].hist(vorb, bins=vb, color="0.55", edgecolor="k", lw=0.6, label="orbital speed")
        ax[1].hist(vcirc_a, bins=vb, histtype="step", color="tab:red", lw=1.5,
                   label="circular speed at the same $a$")
        for v_, lab in ((vk_in, "1.05 AU"), (vk_out, "0.95 AU")):
            ax[1].axvline(v_, color="tab:blue", ls="--", lw=1.0)
            ax[1].text(v_, 1.02, lab, color="tab:blue", fontsize=8, ha="center", va="bottom",
                       transform=ax[1].get_xaxis_transform())
        ax[1].set_xlabel("orbital speed around the Sun [km s$^{-1}$]")
        ax[1].set_ylabel("number of bodies")
        ax[1].set_title("(b) orbital speed at $t = 0$", pad=14)
        ax[1].set_ylim(0, 1.35 * np.histogram(vorb, vb)[0].max())
        ax[1].legend(loc="upper left", fontsize=8.5, frameon=False)
        dv = (vorb - vcirc_a) * 1e3
        rms = np.sqrt(np.mean(vr ** 2))
        for a_ in ax:
            a_.tick_params(direction="in", top=True, right=True)
        fig.tight_layout()
        fig.savefig(args.fig + ".png", dpi=300)
        fig.savefig(args.fig + ".pdf")
        print(f"\norbital speed: min {vorb.min():.3f}, max {vorb.max():.3f}, mean {vorb.mean():.3f} km/s; "
              f"circular speed range {vk_in:.3f}-{vk_out:.3f} km/s; "
              f"orbital minus circular speed: rms {np.sqrt(np.mean(dv**2)):.1f} m/s, max |dv| {np.abs(dv).max():.1f} m/s")
        print(f"figure: {args.fig}.png  (random velocity rms {rms:.1f} m/s, "
              f"median {np.median(vr):.1f}, max {vr.max():.1f}; pair v_esc {vesc:.1f} m/s;"
              f" {np.sum((a >= cfg['a_in']) & (a <= cfg['a_out']))} of {len(a)} bodies inside {cfg['a_in']}-{cfg['a_out']} AU)")


if __name__ == "__main__":
    main()
