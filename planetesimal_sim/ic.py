"""
Initial-condition generator (thesis Section 3.4.1, stage a; Eqs. 3.1-3.3).

Answers the question "where do the initial conditions come from?" with a
single closed chain of reasoning, so that nothing is a free knob:

    Sigma(a) = Sigma_1 (a/AU)^(-3/2)                    [Eq. 3.1]  (Hayashi 1981)
    M_tot    = int 2 pi a Sigma(a) da over the annulus  [Eq. 3.2]
    m_p      = M_tot / N                                (equal-mass swarm)
    R_p      = (3 m_p / 4 pi rho_p)^(1/3)               [Eq. 3.3]
    h        = (2 m_p / 3 M_star)^(1/3)                 [Eq. 2.1, reduced]
    e_rms    = e_tilde * h,   i_rms = e_rms / 2         [Eq. 2.2]

So the ONLY genuinely free inputs are (a_in, a_out, Sigma_1, rho_p, N,
e_tilde). Everything else follows. That is the point: N and R_p are not
independent, they are two ends of the same constraint.

Semi-major axes are drawn from the surface-density profile itself:
    dN/da  ~  2 pi a Sigma(a)  ~  a^(1+p)
which guarantees the realised particle distribution reproduces Sigma(a).

Eccentricities and inclinations follow Rayleigh distributions, as in Ida &
Makino (1992), Kokubo & Ida (1998), Richardson et al. (2000) and Guo &
Kokubo (2021). The Rayleigh scale parameter is set so that the ROOT MEAN
SQUARE eccentricity equals the target: for Rayleigh(sigma), <e^2> = 2 sigma^2,
hence sigma = e_rms / sqrt(2).
"""

import numpy as np
from scipy.spatial import cKDTree

from . import units as u
from .config import SimConfig
from .orbits import elements_to_state


def sample_semi_major_axes(rng, n, a_in, a_out, slope):
    """Inverse-CDF sample of a from dN/da ~ a^(1+slope)."""
    q = 2.0 + slope
    v = rng.random(n)
    if abs(q) < 1e-12:
        return a_in * (a_out / a_in) ** v
    return (a_in ** q + v * (a_out ** q - a_in ** q)) ** (1.0 / q)


def sample_rayleigh(rng, n, rms):
    """Rayleigh sample with the requested RMS value."""
    sigma = rms / np.sqrt(2.0)
    return sigma * np.sqrt(-2.0 * np.log(1.0 - rng.random(n)))


def _draw_elements(rng, n, cfg):
    """Draw n complete sets of orbital elements from the target distributions."""
    return dict(
        a=sample_semi_major_axes(rng, n, cfg.a_in, cfg.a_out, cfg.sigma_slope),
        e=sample_rayleigh(rng, n, cfg.e_rms),
        inc=sample_rayleigh(rng, n, cfg.i_rms),
        Omega=rng.uniform(0.0, 2 * np.pi, n),
        omega=rng.uniform(0.0, 2 * np.pi, n),
        M=rng.uniform(0.0, 2 * np.pi, n),
    )


def reject_initial_overlaps(pos, vel, els, mass, radius, cfg, rng,
                            max_iter=200, verbose=False):
    """Redraw any particle whose starting position is too close to a neighbour.

    Why this exists
    ---------------
    A Rayleigh/uniform draw places a handful of pairs within a mutual Hill
    radius of each other at t = 0 purely by chance. Those pairs merge almost
    immediately, producing a spurious burst of collisions in the first few
    tens of years that is an artefact of the initial draw, not accretion.
    Richardson et al. (2000) Section 2.4 handle this the same way: "an
    iterative method is used to ensure that particles' Hill spheres do not
    overlap initially".

    The rejection criterion is

        |r_i - r_j|  >  max( sep_hill * r_H,ij ,  f (R_i + R_j) )

    with sep_hill = cfg.min_separation_hill (default 2.0, i.e. non-overlapping
    Hill spheres) and r_H,ij the mutual Hill radius, thesis Eq. (2.1).

    Offending pairs have ONE member redrawn (the higher index), so the
    surviving population is still an unbiased sample of the target
    distributions -- verified by tests/test_validation.py::test_ic_statistics,
    which passes with rejection enabled.
    """
    n = len(mass)
    sep_hill = cfg.min_separation_hill
    if sep_hill <= 0.0:
        return pos, vel, els, 0

    n_redrawn = 0
    for it in range(max_iter):
        a_now = els["a"]
        # mutual Hill radius for the equal-mass initial swarm, Eq. (2.1)
        r_h = ((2.0 * mass[0]) / (3.0 * cfg.m_star)) ** (1.0 / 3.0) * a_now
        d_min = np.maximum(sep_hill * r_h, cfg.f_radius * 2.0 * radius)

        tree = cKDTree(pos)
        neigh = tree.query_ball_point(pos, d_min, workers=1)

        bad = set()
        for i, nb in enumerate(neigh):
            for j in nb:
                if j <= i:
                    continue
                sep = np.linalg.norm(pos[i] - pos[j])
                thresh = max(sep_hill * 0.5 * (r_h[i] + r_h[j]),
                             cfg.f_radius * (radius[i] + radius[j]))
                if sep < thresh:
                    bad.add(j)
        if not bad:
            if verbose:
                print(f"    IC overlap rejection converged after {it} "
                      f"iteration(s); {n_redrawn} particle(s) redrawn")
            return pos, vel, els, n_redrawn

        idx = np.fromiter(bad, dtype=int)
        n_redrawn += len(idx)
        new = _draw_elements(rng, len(idx), cfg)
        for k in els:
            els[k][idx] = new[k]
        mu = u.G * (cfg.m_star + mass[idx])
        p_new, v_new = elements_to_state(new["a"], new["e"], new["inc"],
                                         new["Omega"], new["omega"],
                                         new["M"], mu)
        pos[idx] = p_new
        vel[idx] = v_new

    raise RuntimeError(
        f"IC overlap rejection failed to converge in {max_iter} iterations. "
        f"The swarm is too crowded for min_separation_hill="
        f"{sep_hill:g} at N={n}: either lower cfg.min_separation_hill, "
        f"widen the annulus, or reduce N.")


def make_planetesimals(cfg: SimConfig, verbose=False):
    """Build the seeded planetesimal swarm. Returns a dict of arrays."""
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_particles

    mass = np.full(n, cfg.m_particle)
    radius = u.radius_from_mass(mass, cfg.rho_p)

    els = _draw_elements(rng, n, cfg)
    mu = u.G * (cfg.m_star + mass)
    pos, vel = elements_to_state(els["a"], els["e"], els["inc"],
                                 els["Omega"], els["omega"], els["M"], mu)

    pos, vel, els, n_redrawn = reject_initial_overlaps(
        pos, vel, els, mass, radius, cfg, rng, verbose=verbose)

    return dict(pos=pos, vel=vel, mass=mass, radius=radius,
                a0=els["a"], e0=els["e"], i0=els["inc"],
                n_redrawn=n_redrawn)


def make_giant_planet(cfg: SimConfig):
    """Giant-planet state vector (Scenarios 3 and 4).

    Placed at pericentre on the +x axis with omega = Omega = 0, following the
    convention of Guo & Kokubo (2021) Section 2.1.
    """
    mu = u.G * (cfg.m_star + cfg.m_planet)
    pos, vel = elements_to_state(cfg.a_planet, cfg.e_planet, cfg.i_planet,
                                 0.0, 0.0, 0.0, mu)
    return pos[0], vel[0]


def build_system(cfg: SimConfig, verbose=False):
    """Assemble the full initial state, with the barycentre at rest.

    Returns
    -------
    dict with keys pos, vel, mass, radius (planetesimals),
    star_pos, star_vel, planet_pos, planet_vel (or None).
    """
    sw = make_planetesimals(cfg, verbose=verbose)
    pos, vel, mass = sw["pos"], sw["vel"], sw["mass"]

    star_pos = np.zeros(3)
    star_vel = np.zeros(3)

    if cfg.planet_on:
        p_pos, p_vel = make_giant_planet(cfg)
    else:
        p_pos = p_vel = None

    # shift into the barycentric frame so the swarm does not drift
    tot_m = cfg.m_star + mass.sum() + (cfg.m_planet if cfg.planet_on else 0.0)
    mom = (mass[:, None] * vel).sum(axis=0)
    com = (mass[:, None] * pos).sum(axis=0)
    if cfg.planet_on:
        mom = mom + cfg.m_planet * p_vel
        com = com + cfg.m_planet * p_pos
    com = com / tot_m
    mom = mom / tot_m

    pos = pos - com
    vel = vel - mom
    star_pos = star_pos - com
    star_vel = star_vel - mom
    if cfg.planet_on:
        p_pos = p_pos - com
        p_vel = p_vel - mom

    return dict(pos=pos, vel=vel, mass=mass, radius=sw["radius"],
                star_pos=star_pos, star_vel=star_vel,
                planet_pos=p_pos, planet_vel=p_vel,
                a0=sw["a0"], e0=sw["e0"], i0=sw["i0"],
                n_redrawn=sw["n_redrawn"])
