"""
Force evaluation (thesis Section 3.5, stages 2, 4 and 5).

Design decision worth stating explicitly in the thesis
-----------------------------------------------------
The star and the giant planet are NOT put into the tree. They are evaluated
by exact direct summation, because:

  * the star carries ~1e9 times the mass of one planetesimal, so any cell that
    lumped it together with planetesimals would produce a large multipole
    error precisely where accuracy matters most;
  * both are single bodies, so the direct cost is O(N), not O(N^2).

The tree therefore handles ONLY the mutual planetesimal-planetesimal forces,
which is where the O(N^2) cost actually lives and where all bodies have
comparable mass. This is the standard "dominant central mass handled exactly"
arrangement and it is what makes a Barnes-Hut solver defensible for a
planetary problem at all.
"""

import numpy as np
from pytreegrav import Accel

from . import units as u


# ---------------------------------------------------------------------------
# point-mass acceleration from a single body
# ---------------------------------------------------------------------------
def _accel_from_point(pos, src_pos, src_mass, soft=0.0):
    d = src_pos[None, :] - pos
    r2 = np.einsum("ij,ij->i", d, d) + soft ** 2
    inv_r3 = r2 ** -1.5
    return (u.G * src_mass * inv_r3)[:, None] * d


# ---------------------------------------------------------------------------
# gas disk (Hayashi 1981 / Kokubo & Ida 2000 / Guo & Kokubo 2021)
# ---------------------------------------------------------------------------
def gas_density(pos_helio, cfg, t=0.0):
    """Local gas density (Msun/AU^3) at each planetesimal position.

        rho(R, z) = rho_1 (R/AU)^(-11/4) exp(-(z/h_g)^2)
        h_g(R)    = h_1 (R/AU)^(5/4)

    Optional exponential disk dispersal with e-folding time
    `cfg.gas_depletion_time` (0 = no dispersal).
    """
    R = np.hypot(pos_helio[:, 0], pos_helio[:, 1])
    R = np.maximum(R, 1e-6)
    z = pos_helio[:, 2]
    h_g = cfg.h_gas1 * R ** cfg.h_gas_slope
    rho = cfg.rho_gas1 * R ** cfg.rho_gas_slope * np.exp(-(z / h_g) ** 2)
    if cfg.gas_depletion_time > 0.0:
        rho = rho * np.exp(-t / cfg.gas_depletion_time)
    return rho


def gas_velocity(pos_helio, cfg):
    """Gas velocity field: circular, sub-Keplerian by the factor (1-2 eta)^(1/2).

    Kokubo & Ida (2000):  v_gas = (1 - 2 eta)^(1/2) v_K,
                          eta   = 0.0019 (a/AU)^(1/2).
    """
    x, y = pos_helio[:, 0], pos_helio[:, 1]
    R = np.maximum(np.hypot(x, y), 1e-6)
    v_kep = np.sqrt(u.G * cfg.m_star / R)
    eta = cfg.eta1 * np.sqrt(R)
    v_gas_mag = np.sqrt(np.maximum(1.0 - 2.0 * eta, 0.0)) * v_kep
    vg = np.zeros_like(pos_helio)
    vg[:, 0] = -v_gas_mag * y / R
    vg[:, 1] = v_gas_mag * x / R
    return vg


def gas_drag_accel(pos_helio, vel_helio, radius, mass, cfg, t=0.0):
    """Aerodynamic drag acceleration (thesis Eq. 3.5).

        a_drag = -(1/2) C_D (pi R^2 / m) rho_gas |u| u,   u = v - v_gas

    NOTE: `radius` here must be the TRUE physical radius, never the
    collision-inflated radius f*R. Inflating R for collision detection is a
    numerical device for the accretion rate; it must not leak into the drag,
    which depends on the real cross-section.
    """
    rho = gas_density(pos_helio, cfg, t)
    vg = gas_velocity(pos_helio, cfg)
    uu = vel_helio - vg
    umag = np.linalg.norm(uu, axis=1)
    coeff = 0.5 * cfg.c_drag * np.pi * radius ** 2 / mass * rho * umag
    return -coeff[:, None] * uu


def stopping_time(cfg, mass=None, radius=None, v_rel=None):
    """Analytic drag stopping time t_s = m / (0.5 C_D pi R^2 rho |u|), in yr.

    Handy as a sanity check BEFORE running: if t_s (and the eccentricity
    damping time ~ t_s) greatly exceeds t_end, Scenario 2 cannot show a
    measurable effect and you should say so up front rather than discover it
    after a week of CPU time.
    """
    m = cfg.m_particle if mass is None else mass
    R = cfg.r_particle if radius is None else radius
    rho = cfg.rho_gas1
    if v_rel is None:
        v_kep = np.sqrt(u.G * cfg.m_star)
        v_rel = np.sqrt(cfg.e_rms ** 2 + cfg.i_rms ** 2 +
                        (2 * cfg.eta1) ** 2) * v_kep
    return m / (0.5 * cfg.c_drag * np.pi * R ** 2 * rho * v_rel)


# ---------------------------------------------------------------------------
# full acceleration
# ---------------------------------------------------------------------------
def gravitational_accel(state, cfg):
    """Gravitational accelerations of (planetesimals, star, planet).

    Returns (a_p, a_star, a_planet). a_planet is None when planet_on is False.
    """
    pos, mass, radius = state["pos"], state["mass"], state["radius"]
    star_pos = state["star_pos"]
    n = len(mass)

    # --- star on planetesimals (exact) ---
    a_p = _accel_from_point(pos, star_pos, cfg.m_star)

    # --- planetesimals on star (exact, O(N)) ---
    d = pos - star_pos[None, :]
    r2 = np.einsum("ij,ij->i", d, d)
    inv_r3 = r2 ** -1.5
    a_star = u.G * ((mass * inv_r3)[:, None] * d).sum(axis=0)

    # --- mutual planetesimal gravity (Barnes-Hut tree) ---
    if n > 1:
        soft = np.ascontiguousarray(
            cfg.softening_frac * cfg.f_radius * radius)
        a_p += Accel(np.ascontiguousarray(pos),
                     np.ascontiguousarray(mass),
                     softening=soft, G=u.G, theta=cfg.theta,
                     method=cfg.tree_method, parallel=cfg.parallel)

    # --- giant planet (exact, both directions) ---
    a_planet = None
    if cfg.planet_on and state["planet_pos"] is not None:
        p_pos = state["planet_pos"]
        a_p += _accel_from_point(pos, p_pos, cfg.m_planet)

        dp = pos - p_pos[None, :]
        r2p = np.einsum("ij,ij->i", dp, dp)
        inv_r3p = r2p ** -1.5
        a_planet = u.G * ((mass * inv_r3p)[:, None] * dp).sum(axis=0)
        a_planet += _accel_from_point(p_pos[None, :], star_pos,
                                      cfg.m_star)[0]
        a_star += _accel_from_point(star_pos[None, :], p_pos,
                                    cfg.m_planet)[0]

    return a_p, a_star, a_planet


def drag_accel(state, vel, cfg, t):
    """Drag on the planetesimals given a candidate velocity array.

    Kept separate from gravity so the second half-kick of the leapfrog can
    re-evaluate the velocity-dependent term without rebuilding the tree.
    """
    if not cfg.gas_on:
        return 0.0
    pos_h = state["pos"] - state["star_pos"][None, :]
    vel_h = vel - state["star_vel"][None, :]
    return gas_drag_accel(pos_h, vel_h, state["radius"], state["mass"], cfg, t)
