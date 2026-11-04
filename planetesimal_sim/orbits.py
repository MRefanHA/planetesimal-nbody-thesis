"""
Conversions between Keplerian orbital elements and Cartesian state vectors.

All angles in radians. Positions/velocities are HELIOCENTRIC (relative to the
central star), which is what the orbital elements describe.
"""

import numpy as np
from .units import G


def solve_kepler(M, e, tol=1e-13, itmax=100):
    """Newton solve of M = E - e sin E, vectorised."""
    M = np.mod(np.asarray(M, dtype=float) + np.pi, 2 * np.pi) - np.pi
    e = np.asarray(e, dtype=float)
    E = M + np.sign(M) * 0.85 * e          # Danby's starter
    for _ in range(itmax):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        fpp = e * np.sin(E)
        # Halley step: faster and safe for e < 0.9
        dE = -f / (fp - 0.5 * f * fpp / fp)
        E = E + dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def elements_to_state(a, e, inc, Omega, omega, M, mu):
    """Orbital elements -> heliocentric (pos, vel).

    Parameters
    ----------
    a, e, inc, Omega, omega, M : array_like
        Semi-major axis (AU), eccentricity, inclination, longitude of
        ascending node, argument of pericentre, mean anomaly (rad).
    mu : array_like
        G * (M_star + m_body).

    Returns
    -------
    pos, vel : (N, 3) arrays in AU and AU/yr.
    """
    a, e, inc = np.atleast_1d(a), np.atleast_1d(e), np.atleast_1d(inc)
    Omega, omega, M = np.atleast_1d(Omega), np.atleast_1d(omega), np.atleast_1d(M)
    mu = np.atleast_1d(mu) * np.ones_like(a)

    E = solve_kepler(M, e)
    cosE, sinE = np.cos(E), np.sin(E)
    b_over_a = np.sqrt(1.0 - e ** 2)

    # position/velocity in the orbital plane (pericentre along +x_orb)
    x_o = a * (cosE - e)
    y_o = a * b_over_a * sinE
    n = np.sqrt(mu / a ** 3)
    fac = n * a / (1.0 - e * cosE)
    vx_o = -fac * sinE
    vy_o = fac * b_over_a * cosE

    cO, sO = np.cos(Omega), np.sin(Omega)
    cw, sw = np.cos(omega), np.sin(omega)
    ci, si = np.cos(inc), np.sin(inc)

    # rotation matrix R = Rz(Omega) Rx(inc) Rz(omega), columns P and Q
    Px = cO * cw - sO * sw * ci
    Py = sO * cw + cO * sw * ci
    Pz = sw * si
    Qx = -cO * sw - sO * cw * ci
    Qy = -sO * sw + cO * cw * ci
    Qz = cw * si

    pos = np.stack([Px * x_o + Qx * y_o,
                    Py * x_o + Qy * y_o,
                    Pz * x_o + Qz * y_o], axis=-1)
    vel = np.stack([Px * vx_o + Qx * vy_o,
                    Py * vx_o + Qy * vy_o,
                    Pz * vx_o + Qz * vy_o], axis=-1)
    return pos, vel


def state_to_elements(pos, vel, mu):
    """Heliocentric (pos, vel) -> (a, e, inc). Diagnostics only.

    Returns arrays of semi-major axis (AU), eccentricity, inclination (rad).
    Bodies on hyperbolic orbits get a < 0, which is how ejections are flagged.
    """
    pos = np.atleast_2d(pos)
    vel = np.atleast_2d(vel)
    mu = np.atleast_1d(mu) * np.ones(len(pos))

    r = np.linalg.norm(pos, axis=1)
    v2 = np.einsum("ij,ij->i", vel, vel)

    a = 1.0 / (2.0 / r - v2 / mu)

    h = np.cross(pos, vel)
    hmag = np.linalg.norm(h, axis=1)

    e_vec = np.cross(vel, h) / mu[:, None] - pos / r[:, None]
    ecc = np.linalg.norm(e_vec, axis=1)

    inc = np.arccos(np.clip(h[:, 2] / np.maximum(hmag, 1e-300), -1.0, 1.0))
    return a, ecc, inc


def mutual_hill_radius(m1, m2, a1, a2, m_star):
    """Thesis Eq. (2.1): r_H = ((m1+m2)/(3 M*))^(1/3) * (a1+a2)/2."""
    return ((m1 + m2) / (3.0 * m_star)) ** (1.0 / 3.0) * 0.5 * (a1 + a2)
