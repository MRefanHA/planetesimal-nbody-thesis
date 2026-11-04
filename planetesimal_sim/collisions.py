"""
Collision detection and perfectly inelastic merging (thesis Section 3.5, stage 3).

This is the module that answers your supervisor's "effective area around each
planetesimal" suggestion. Two things are going on, and they are worth keeping
separate in the write-up because they are separate ideas:

1. THE EFFECTIVE (INFLATED) RADIUS.
   A collision is registered when two bodies approach within

       d_min  <  f * (R_i + R_j)                       [GPLUM Eq. 21]

   where f is the radius enhancement factor. f = 1 is the literal physical
   overlap condition. f > 1 artificially enlarges the target, which shortens
   the accretion timescale by up to f^2 (Kokubo & Ida 1996; Wallace & Quinn
   2019 use f = 6, as do Richardson et al. 2000 reproducing Kokubo & Ida 1998).
   This is a standard and citable device, not a fudge, but f must be reported
   and its effect on the growth timescale acknowledged.

2. SWEPT-SPHERE (CONTINUOUS) DETECTION -- and this is the part that people
   usually get wrong and then wonder why nothing ever collides.
   Checking "do the spheres overlap right now?" once per timestep does NOT
   work here. With dt ~ 1/64 yr the relative displacement of two neighbouring
   planetesimals in one step is ~1e-4 AU, while their collision radius is
   ~1e-5 AU. Bodies simply tunnel straight through each other and the
   simulation reports almost zero collisions.

   Instead, treat each body as moving in a straight line over the step and
   find the MINIMUM separation along that segment. For a pair with
   dr = r_j - r_i and dv = v_j - v_i at the start of the step, the closest
   approach happens at

       t*  = clip( -(dr . dv) / (dv . dv),  0,  dt )

   and a collision occurs if |dr + dv t*| < f (R_i + R_j). This is exactly the
   scheme described by Wallace & Quinn (2019) Section 2.2 and Richardson
   et al. (2000): "collisions are predicted at the beginning of each drift
   step by extrapolating the positions forward using the velocities
   calculated during the first kick".

NEIGHBOUR SEARCH
----------------
A cKDTree with a PER-PARTICLE search radius

    s_i = SAFETY * |v_i - v_circ(r_i)| * dt + f R_i

is provably sufficient: for any pair, the start-of-step separation of a
colliding pair is at most |dv| dt + f(R_i+R_j) <= s_i + s_j (up to the
Keplerian shear across s_i, which is ~1e-4 of the peculiar velocity here and
is covered by SAFETY).

ACCRETION CRITERION
-------------------
Detecting an impact is not the same as accreting it. Two prescriptions are
available, selected by `cfg.merge_criterion`:

  "always"    Every detected impact merges. This is perfect inelastic merging,
              the standard first-order prescription used by Kokubo & Ida (2000)
              and Wallace & Quinn (2019). Barnes et al. (2009) note that it can
              overestimate growth, because a real impact above the mutual escape
              velocity would fragment or erode rather than accrete, and the
              merged body would in any case retain unphysical rotational angular
              momentum.

  "velocity"  Merge only when v_rel < v_esc, where

                  v_esc = sqrt( 2 G (m_1 + m_2) / (R_1 + R_2) )

              is evaluated with the TRUE physical radii, not the inflated
              collision radii. Impacts at or above v_esc are recorded as
              disruptive and the bodies pass through unchanged. This is a
              first-order accretion filter, not a fragmentation model: it
              refuses to accrete mass that should not accrete, but it does not
              produce the debris that a real disruption would create.

The distinction matters most when an external perturber is present. A giant
planet raises relative velocities well above v_esc, so under "always" it appears
to accelerate growth while in fact it is driving the swarm into the erosive
regime. Running both prescriptions isolates that artefact.

MERGING
-------
When a pair does accrete, the merger is perfectly inelastic, conserving mass
and linear momentum (GPLUM Eqs. 21-23):

    m_new = m_1 + m_2
    r_new = (m_1 r_1 + m_2 r_2) / m_new
    v_new = (m_1 v_1 + m_2 v_2) / m_new
    R_new = (3 m_new / 4 pi rho)^(1/3)

Rotational angular momentum is discarded, which is the limitation flagged by
Barnes et al. (2009) and recorded in Section 1.4 item 2 of the thesis.
"""

import numpy as np
from scipy.spatial import cKDTree

from . import units as u


def _circular_velocity(pos_helio, m_star):
    """Local circular Keplerian velocity vector at each position."""
    x, y = pos_helio[:, 0], pos_helio[:, 1]
    R = np.maximum(np.hypot(x, y), 1e-9)
    vc = np.sqrt(u.G * m_star / R)
    out = np.zeros_like(pos_helio)
    out[:, 0] = -vc * y / R
    out[:, 1] = vc * x / R
    return out


def find_collision_pairs(pos, vel, radius, dt, cfg, star_pos, star_vel,
                         safety=1.5):
    """Return (i, j, t_star) arrays for pairs that collide during [t, t+dt]."""
    n = len(radius)
    if n < 2:
        return np.empty(0, int), np.empty(0, int), np.empty(0)

    f = cfg.f_radius
    pos_h = pos - star_pos[None, :]
    vel_h = vel - star_vel[None, :]
    v_pec = np.linalg.norm(vel_h - _circular_velocity(pos_h, cfg.m_star),
                           axis=1)
    s = safety * v_pec * dt + f * radius

    tree = cKDTree(pos)
    # workers=1, deliberately. SciPy spawns a fresh thread pool on every
    # query_ball_point call when workers=-1. This function is called once per
    # timestep, so a production run makes ~2.8 million such calls, and on
    # Windows the process eventually runs out of thread handles:
    #     RuntimeError: can't start new thread
    # Threading buys nothing here anyway. The per-particle search radius is
    # ~3e-4 AU, so the query returns almost no pairs and the work per call is
    # far too small to amortise the cost of starting threads. Single-threaded
    # is both safe and faster.
    neigh = tree.query_ball_point(pos, s, workers=1)

    ii, jj = [], []
    for i, nb in enumerate(neigh):
        for j in nb:
            if j > i:
                ii.append(i)
                jj.append(j)
    if not ii:
        return np.empty(0, int), np.empty(0, int), np.empty(0)

    i = np.asarray(ii)
    j = np.asarray(jj)

    dr = pos[j] - pos[i]
    dv = vel[j] - vel[i]
    rsum = f * (radius[i] + radius[j])

    dvdv = np.einsum("ij,ij->i", dv, dv)
    drdv = np.einsum("ij,ij->i", dr, dv)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstar = np.where(dvdv > 0, -drdv / dvdv, 0.0)
    tstar = np.clip(tstar, 0.0, dt)

    sep = dr + dv * tstar[:, None]
    d2 = np.einsum("ij,ij->i", sep, sep)
    hit = d2 < rsum ** 2

    return i[hit], j[hit], tstar[hit]


def resolve_collisions(state, cfg, dt, star_pos, star_vel):
    """Detect impacts and accrete those that qualify.

    Returns (n_merged, n_rejected, records). Called with the post-first-kick
    velocities, BEFORE the drift, so the linear extrapolation matches the
    drift the integrator is about to do.

    Every detected impact appears in `records` with an `accreted` flag, so the
    disruptive fraction can be reported even when those impacts do not
    contribute to growth.
    """
    pos, vel = state["pos"], state["vel"]
    mass, radius = state["mass"], state["radius"]

    i, j, tstar = find_collision_pairs(pos, vel, radius, dt, cfg,
                                       star_pos, star_vel)
    if len(i) == 0:
        return 0, 0, []

    order = np.argsort(tstar)
    alive = np.ones(len(mass), bool)
    records = []
    n_merged = 0
    n_rejected = 0

    new_pos = pos.copy()
    new_vel = vel.copy()
    new_mass = mass.copy()

    for k in order:
        a, b = int(i[k]), int(j[k])
        if not (alive[a] and alive[b]):
            continue          # one partner already consumed this step
        ma, mb = new_mass[a], new_mass[b]
        mtot = ma + mb
        v_rel = float(np.linalg.norm(new_vel[b] - new_vel[a]))
        # escape velocity of the pair, from the TRUE physical radii
        v_esc = float(np.sqrt(2.0 * u.G * mtot /
                              (radius[a] + radius[b])))

        if cfg.merge_criterion == "always":
            accrete = True
        elif cfg.merge_criterion == "velocity":
            accrete = v_rel < cfg.merge_velocity_factor * v_esc
        else:
            raise ValueError(
                f"unknown merge_criterion {cfg.merge_criterion!r}; "
                "expected 'always' or 'velocity'")

        records.append(dict(m1=float(ma), m2=float(mb),
                            v_rel=v_rel, v_esc=v_esc,
                            t_star=float(tstar[k]),
                            accreted=bool(accrete)))

        if not accrete:
            # Disruptive impact. The bodies are left untouched: this is an
            # accretion filter, not a fragmentation model. Because
            # v_rel > v_esc, the pair separates by many collision radii within
            # a single step, so it is not re-detected on the next one.
            n_rejected += 1
            continue

        new_pos[a] = (ma * new_pos[a] + mb * new_pos[b]) / mtot
        new_vel[a] = (ma * new_vel[a] + mb * new_vel[b]) / mtot
        new_mass[a] = mtot
        alive[b] = False
        n_merged += 1

    if n_merged:
        keep = np.where(alive)[0]
        state["pos"] = np.ascontiguousarray(new_pos[keep])
        state["vel"] = np.ascontiguousarray(new_vel[keep])
        state["mass"] = np.ascontiguousarray(new_mass[keep])
        state["radius"] = np.ascontiguousarray(
            u.radius_from_mass(new_mass[keep], cfg.rho_p))

    return n_merged, n_rejected, records
