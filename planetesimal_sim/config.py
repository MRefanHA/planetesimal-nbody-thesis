"""
All simulation parameters live here, in one dataclass.

This is deliberate: Chapter III Table 3.1 of the thesis should be generated
directly from this object (see `SimConfig.as_table()`), so that the numbers
in the written thesis can never drift away from the numbers that were
actually run.
"""

from dataclasses import dataclass, asdict, field, replace
import numpy as np

from . import units as u


@dataclass
class SimConfig:
    # ---------------- identity -------------------------------------------
    name: str = "S1_gravity_only"
    seed: int = 20250904          # same seed in every scenario -> identical ICs

    # ---------------- star -----------------------------------------------
    m_star: float = 1.0           # Msun

    # ---------------- annulus (Hansen 2009; PENTACLE "model R") -----------
    a_in: float = 0.95            # AU
    a_out: float = 1.05           # AU
    sigma1_solid_cgs: float = 10.0   # g/cm^2 at 1 AU (Hayashi 1981 MMSN)
    sigma_slope: float = -1.5        # Sigma ~ a^(-3/2)

    # ---------------- planetesimals --------------------------------------
    n_particles: int = 2000
    rho_p_cgs: float = 2.0        # g/cm^3 bulk density (Kokubo & Ida 2000)
    e_tilde: float = 4.0          # e_rms = e_tilde * h  (Kokubo & Ida 1998: 4h)
    i_over_e: float = 0.5         # i_rms = 0.5 * e_rms  (equipartition)
    min_separation_hill: float = 2.0   # reject ICs closer than this * r_H
                                       # (0 disables; Richardson et al. 2000)

    # ---------------- collisions -----------------------------------------
    f_radius: float = 6.0         # radius enhancement factor (K&I 1998, f=6)
    # collision when |r_i - r_j|_min over the step  <  f*(R_i + R_j)
    merge_criterion: str = "velocity"
    # "always"   : every detected impact merges (perfect inelastic merging).
    #              This is the classic prescription and the one whose
    #              limitation Barnes et al. (2009) flag.
    # "velocity" : merge only when v_rel < v_esc of the pair. Impacts above
    #              the mutual escape velocity are not accreted; in reality
    #              they would fragment or erode, and merging them
    #              overestimates growth. Bodies pass through unchanged.
    merge_velocity_factor: float = 1.0    # merge if v_rel < factor * v_esc

    # ---------------- gravity solver -------------------------------------
    theta: float = 0.5            # Barnes-Hut opening angle, thesis Eqs. (2.7)-(2.8)
    softening_frac: float = 0.5   # eps_i = softening_frac * f_radius * R_i
    tree_method: str = "tree"     # "tree" | "bruteforce"
    parallel: bool = True

    # ---------------- integration ----------------------------------------
    steps_per_inner_orbit: int = 128
    t_end: float = 2.0e4          # yr
    snapshot_every: float = 100.0   # yr, for the diagnostic time series
    dump_every: float = 2000.0      # yr, for full particle snapshots

    # ---------------- gas disk (Scenarios 2 and 4) ------------------------
    gas_on: bool = False
    rho_gas1_cgs: float = 2.0e-9  # g/cm^3 midplane at 1 AU (Kokubo & Ida 2000)
    rho_gas_slope: float = -2.75  # rho ~ a^(-11/4)
    h_gas1: float = 0.047         # AU, gas scale height at 1 AU (Hayashi 1981)
    h_gas_slope: float = 1.25     # h_g ~ a^(5/4)
    eta1: float = 0.0019          # eta = eta1 * (a/AU)^(1/2) (Kokubo & Ida 2000)
    c_drag: float = 1.0           # C_D, order unity (Guo & Kokubo 2021)
    gas_boost: float = 1.0        # multiplies rho_gas; 1.0 = nominal MMSN x1.5
    gas_depletion_time: float = 0.0   # yr; 0 = no disk dispersal

    # ---------------- giant planet (Scenarios 3 and 4) --------------------
    planet_on: bool = False
    m_planet: float = 1.0 * u.MJUP     # Msun
    a_planet: float = 1.6              # AU  (2:1 interior MMR at 1.008 AU)
    e_planet: float = 0.05
    i_planet: float = 0.0              # rad
    planet_fixed_orbit: bool = False   # True = planet on a fixed Kepler orbit

    # ---------------- output ----------------------------------------------
    outdir: str = "results"

    # ================= derived quantities =================================
    @property
    def rho_p(self):
        """Planetesimal bulk density in Msun/AU^3."""
        return self.rho_p_cgs * u.RHO_CGS_TO_CODE

    @property
    def sigma1_solid(self):
        """Solid surface density at 1 AU in Msun/AU^2."""
        return self.sigma1_solid_cgs * u.SIGMA_CGS_TO_CODE

    @property
    def rho_gas1(self):
        """Midplane gas density at 1 AU in Msun/AU^3, including gas_boost."""
        return self.rho_gas1_cgs * u.RHO_CGS_TO_CODE * self.gas_boost

    @property
    def m_total_solid(self):
        """Total solid mass in the annulus, thesis Eq. (3.2).

        M = int_{a_in}^{a_out} 2 pi a Sigma(a) da  with Sigma ~ a^p
          = 2 pi Sigma_1 [a^(2+p)/(2+p)]
        """
        p = self.sigma_slope
        q = 2.0 + p
        s1 = self.sigma1_solid
        if abs(q) < 1e-12:
            return 2.0 * np.pi * s1 * np.log(self.a_out / self.a_in)
        return 2.0 * np.pi * s1 * (self.a_out ** q - self.a_in ** q) / q

    @property
    def m_particle(self):
        """Equal-mass planetesimal mass in Msun."""
        return self.m_total_solid / self.n_particles

    @property
    def r_particle(self):
        """Planetesimal physical radius in AU, thesis Eq. (3.3)."""
        return u.radius_from_mass(self.m_particle, self.rho_p)

    @property
    def h_reduced(self):
        """Reduced mutual Hill radius h = r_H/a for two equal-mass bodies.

        Thesis Eq. (2.1) with M1 = M2 = m_particle, evaluated at a = 1.
        """
        return (2.0 * self.m_particle / (3.0 * self.m_star)) ** (1.0 / 3.0)

    @property
    def e_rms(self):
        return self.e_tilde * self.h_reduced

    @property
    def i_rms(self):
        return self.i_over_e * self.e_rms

    @property
    def dt(self):
        """Timestep in yr, set by the innermost orbital period."""
        p_in = 2.0 * np.pi * np.sqrt(self.a_in ** 3 / (u.G * self.m_star))
        return p_in / self.steps_per_inner_orbit

    @property
    def n_steps(self):
        return int(np.ceil(self.t_end / self.dt))

    # ================= reporting ==========================================
    def summary(self):
        """Human-readable parameter dump (paste into Chapter III)."""
        mp = self.m_particle
        Rp = self.r_particle
        v_esc = np.sqrt(2.0 * u.G * (2 * mp) / (2 * Rp))
        v_ran = np.sqrt(self.e_rms ** 2 + self.i_rms ** 2) * \
            np.sqrt(u.G * self.m_star / 1.0)
        lines = [
            f"scenario                    : {self.name}",
            f"annulus                     : {self.a_in} - {self.a_out} AU",
            f"total solid mass            : {u.msun_to_mearth(self.m_total_solid):.4f} M_earth",
            f"N particles                 : {self.n_particles}",
            f"particle mass               : {mp:.4e} Msun "
            f"= {u.msun_to_mearth(mp):.4e} M_earth",
            f"particle radius             : {u.au_to_km(Rp):.1f} km",
            f"collision radius (f={self.f_radius:g})     : {u.au_to_km(self.f_radius*Rp):.1f} km",
            f"reduced Hill radius h       : {self.h_reduced:.4e}",
            f"e_rms (= {self.e_tilde:g} h)             : {self.e_rms:.4e}",
            f"i_rms                       : {self.i_rms:.4e}",
            f"random velocity  v_ran      : {u.auyr_to_ms(v_ran):.1f} m/s",
            f"pair escape vel  v_esc      : {u.auyr_to_ms(v_esc):.1f} m/s",
            f"v_ran / v_esc               : {v_ran/v_esc:.3f}",
            f"grav. focusing 1+(vesc/vran)^2: {1 + (v_esc/v_ran)**2:.1f}",
            f"timestep dt                 : {self.dt:.5f} yr "
            f"({self.steps_per_inner_orbit} steps / inner orbit)",
            f"t_end                       : {self.t_end:.3e} yr "
            f"({self.n_steps} steps)",
            f"gas drag                    : {self.gas_on}"
            + (f" (boost x{self.gas_boost:g})" if self.gas_on else ""),
            f"giant planet                : {self.planet_on}"
            + (f" ({self.m_planet/u.MJUP:.2f} M_J at {self.a_planet} AU, "
               f"e={self.e_planet})" if self.planet_on else ""),
        ]
        return "\n".join(lines)

    def to_dict(self):
        d = asdict(self)
        d.update(dict(
            m_total_solid_mearth=float(u.msun_to_mearth(self.m_total_solid)),
            m_particle_msun=float(self.m_particle),
            r_particle_km=float(u.au_to_km(self.r_particle)),
            h_reduced=float(self.h_reduced),
            e_rms=float(self.e_rms),
            i_rms=float(self.i_rms),
            dt_yr=float(self.dt),
            n_steps=int(self.n_steps),
        ))
        return d


# ---------------------------------------------------------------------------
# The four scenarios of the thesis. They differ ONLY in gas_on / planet_on,
# which is exactly the claim made in Section 1.4 item 5.
# ---------------------------------------------------------------------------
def four_scenarios(base: SimConfig = None):
    base = base or SimConfig()
    return [
        replace(base, name="S1_gravity_only", gas_on=False, planet_on=False),
        replace(base, name="S2_gas",          gas_on=True,  planet_on=False),
        replace(base, name="S3_planet",       gas_on=False, planet_on=True),
        replace(base, name="S4_gas_planet",   gas_on=True,  planet_on=True),
    ]
