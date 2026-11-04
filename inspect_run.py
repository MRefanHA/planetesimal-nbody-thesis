#!/usr/bin/env python3
"""
Explain a jump in a run's diagnostic time series.

    python inspect_run.py results\\2026-09-15_060739_S1_gravity_only
    python inspect_run.py <folder> --key i_rms --window 8

Reads series.json, collisions.json and config.json from one run folder and
answers three questions about the largest jumps in a diagnostic:

  1. Did the whole swarm heat up, or did a few bodies get flung?
     Compares e_rms (every bound body counts equally) against
     e_rms_massweighted (big bodies dominate). If the unweighted value jumps
     and the weighted one does not, a small number of LOW-MASS bodies went
     onto eccentric orbits. That is a scattering event, not swarm-wide
     stirring.

  2. Was a collision or an ejection involved?
     Lists every impact inside the window with its v_rel / v_esc ratio, and
     tracks n_unbound and the mass of the largest body.

  3. Was the timestep still resolving close encounters at that point?
     Estimates dt / t_enc from the swarm's own velocity dispersion at each
     output time. t_enc is the Hill-radius crossing time of an encounter.
     Once dt / t_enc approaches 1 the integrator cannot resolve encounters
     and will heat the swarm numerically. This value gets WORSE as the run
     proceeds, because stirring raises the velocity dispersion and growth
     raises the Hill radius.

Stdlib + numpy only. Does not touch the simulation.
"""

import argparse
import json
import math
import os

TWO_PI = 2.0 * math.pi          # sqrt(G) with G = 4 pi^2 in AU, Msun, yr


def load(folder):
    def maybe(name):
        p = os.path.join(folder, name)
        if not os.path.isfile(p):
            return None
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    series = maybe("series.json")
    if not series:
        raise SystemExit(f"no series.json in {folder!r}")
    return series, maybe("collisions.json") or [], maybe("config.json") or {}


def dt_over_tenc(d, cfg, m_mean0):
    """dt / (Hill-crossing time), estimated from this output row.

    t_enc = r_H / v_ran, with r_H scaled from its initial value by the
    cube root of the mean mass, since r_H goes as m^(1/3).
    """
    h0 = cfg.get("h_reduced")
    dt = cfg.get("dt_yr")
    if not h0 or not dt or not m_mean0:
        return float("nan")
    grow = (d.get("m_mean_mearth", m_mean0) / m_mean0) ** (1.0 / 3.0)
    v_ran = math.hypot(d.get("e_rms", 0.0), d.get("i_rms", 0.0)) * TWO_PI
    if v_ran <= 0:
        return float("nan")
    return dt / ((h0 * grow) / v_ran)


def jumps(series, key, n=5):
    out = []
    for k in range(1, len(series)):
        a, b = series[k - 1].get(key), series[k].get(key)
        if not a or not b or a <= 0:
            continue
        out.append((b / a - 1.0, k))
    out.sort(reverse=True)
    return out[:n]


def table(series, cfg, lo, hi, m_mean0):
    hdr = (f"{'t (yr)':>9} {'N':>5} {'unb':>4} {'acc':>5} {'disr':>5} "
           f"{'e_rms':>10} {'e_rms(m-w)':>11} {'i_rms':>10} "
           f"{'v (m/s)':>8} {'a_mean':>7} {'Mmax(Me)':>10} {'dt/t_enc':>9}")
    print(hdr)
    print("-" * len(hdr))
    for d in series[lo:hi]:
        print(f"{d['t']:>9.1f} {d['n_bodies']:>5d} {d.get('n_unbound', 0):>4d} "
              f"{d['n_collisions']:>5d} {d.get('n_rejected', 0):>5d} "
              f"{d['e_rms']:>10.4e} "
              f"{d.get('e_rms_massweighted', float('nan')):>11.4e} "
              f"{d['i_rms']:>10.4e} {d.get('v_disp_ms', 0):>8.1f} "
              f"{d.get('a_mean', float('nan')):>7.4f} "
              f"{d['m_max_mearth']:>10.4e} "
              f"{dt_over_tenc(d, cfg, m_mean0):>9.3f}")


def verdict(before, after, colls, cfg, m_mean0):
    print()
    print("VERDICT")
    print("-" * 62)

    def ratio(key):
        a, b = before.get(key), after.get(key)
        return (b / a) if (a and b and a > 0) else float("nan")

    r_u, r_w = ratio("e_rms"), ratio("e_rms_massweighted")
    print(f"  e_rms              x{r_u:.2f}   (every bound body counts equally)")
    print(f"  e_rms mass-weighted x{r_w:.2f}   (large bodies dominate)")
    if r_u > 1.15 and r_w < 1.05:
        print("  -> A few LOW-MASS bodies were thrown onto eccentric orbits.")
        print("     The bulk of the swarm was not heated. Look for a close")
        print("     encounter with the largest body, not a global change.")
    elif r_u > 1.15 and r_w > 1.15:
        print("  -> The whole swarm heated, including the massive bodies.")
        print("     That is either strong global stirring or numerical")
        print("     heating. Check the dt/t_enc column above.")
    else:
        print("  -> No large jump in this window.")

    du = after.get("n_unbound", 0) - before.get("n_unbound", 0)
    dn = after["n_bodies"] - before["n_bodies"]
    dm = after["m_max_mearth"] / before["m_max_mearth"]
    da = after.get("a_mean", 0) - before.get("a_mean", 0)
    print()
    print(f"  bodies lost         {-dn}")
    print(f"  newly unbound       {du}")
    print(f"  M_max change        x{dm:.4f}")
    print(f"  mean a change       {da:+.5f} AU")
    if du > 0:
        print("  -> Bodies were ejected. Whatever happened was violent.")
    if da > 0.01:
        print("  -> Mean semi-major axis moved. Something was scattered far out.")

    t0, t1 = before["t"], after["t"]
    win = [c for c in colls if t0 <= c.get("t", -1) <= t1]
    print()
    print(f"  impacts in {t0:.0f}-{t1:.0f} yr : {len(win)}")
    if win:
        print(f"    {'t (yr)':>9} {'v_rel (m/s)':>12} {'v_esc (m/s)':>12} "
              f"{'ratio':>7}  outcome")
        for c in sorted(win, key=lambda c: -c.get("v_rel", 0))[:10]:
            vr, ve = c.get("v_rel", 0), c.get("v_esc", 1)
            au_yr_to_ms = 1.495978707e11 / 3.15576e7
            print(f"    {c.get('t', 0):>9.1f} {vr * au_yr_to_ms:>12.1f} "
                  f"{ve * au_yr_to_ms:>12.1f} {vr / ve:>7.2f}  "
                  + ("accreted" if vr < ve else "DISRUPTIVE, rejected"))

    d_end = dt_over_tenc(after, cfg, m_mean0)
    print()
    print(f"  dt / t_enc here     {d_end:.3f}")
    if d_end > 0.5:
        print("  -> The timestep is comparable to the encounter crossing time.")
        print("     Close encounters are NOT resolved. A spurious kick from an")
        print("     under-resolved encounter is a real possibility here, and")
        print("     this must be stated in the thesis as a limitation.")
    elif d_end > 0.15:
        print("  -> Marginal. Encounters are under-resolved; benchmark B4")
        print("     showed convergence needs dt/t_enc well below this.")
    else:
        print("  -> Timestep still adequate. A physical event is more likely.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--key", default="e_rms",
                    help="diagnostic to look for jumps in")
    ap.add_argument("--window", type=int, default=6,
                    help="output rows to show either side of the jump")
    ap.add_argument("--top", type=int, default=5,
                    help="how many jumps to rank")
    args = ap.parse_args()

    series, colls, cfg = load(args.folder)
    m_mean0 = series[0].get("m_mean_mearth")

    print(f"run folder : {os.path.abspath(args.folder)}")
    print(f"scenario   : {cfg.get('name', '?')}")
    print(f"outputs    : {len(series)} rows, "
          f"t = {series[0]['t']:.1f} to {series[-1]['t']:.1f} yr")
    print(f"dt         : {cfg.get('dt_yr', float('nan')):.5f} yr")
    print()

    ranked = jumps(series, args.key, args.top)
    if not ranked:
        raise SystemExit(f"no usable values for {args.key!r}")
    print(f"LARGEST JUMPS IN {args.key}")
    print("-" * 62)
    for frac, k in ranked:
        print(f"  {series[k-1]['t']:>9.1f} -> {series[k]['t']:>9.1f} yr : "
              f"{frac * 100:+7.1f} %")

    _, k = ranked[0]
    lo = max(0, k - args.window)
    hi = min(len(series), k + args.window)
    print()
    print(f"AROUND THE LARGEST JUMP (t = {series[k]['t']:.1f} yr)")
    print("-" * 62)
    table(series, cfg, lo, hi, m_mean0)
    verdict(series[k - 1], series[k], colls, cfg, m_mean0)

    print()
    print("dt/t_enc OVER THE WHOLE RUN")
    print("-" * 62)
    step = max(1, len(series) // 12)
    for d in series[::step]:
        v = dt_over_tenc(d, cfg, m_mean0)
        bar = "#" * min(40, int(v * 20))
        print(f"  t={d['t']:>8.0f}  dt/t_enc={v:>6.3f}  {bar}")
    print()
    print("  Rule of thumb: below 0.15 encounters are resolved; near or")
    print("  above 1.0 they are not resolved at all.")


if __name__ == "__main__":
    main()
