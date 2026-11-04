"""
Unit system for the simulation.

Internal working units are:
    length : AU
    mass   : Msun
    time   : yr

In these units the gravitational constant is exactly G = 4*pi^2, because a
circular orbit at a = 1 AU around a 1 Msun star has period 1 yr.

Everything that comes out of the literature in cgs (g/cm^2, g/cm^3, km, ...)
is converted here, once, so that no conversion factors are ever hard-coded
anywhere else in the code.
"""

import numpy as np

# --- SI reference constants ------------------------------------------------
AU_M = 1.495978707e11        # m
MSUN_KG = 1.98892e30         # kg
YR_S = 3.15576e7             # s  (Julian year)
MEARTH_KG = 5.9722e24        # kg
MJUP_KG = 1.89813e27         # kg

# --- Working-unit constants ------------------------------------------------
G = 4.0 * np.pi ** 2         # AU^3 Msun^-1 yr^-2
MEARTH = MEARTH_KG / MSUN_KG   # 1 Earth mass in Msun  ~ 3.0028e-6
MJUP = MJUP_KG / MSUN_KG       # 1 Jupiter mass in Msun ~ 9.5433e-4

# --- Conversion factors ----------------------------------------------------
# surface density: 1 g/cm^2 -> Msun/AU^2
SIGMA_CGS_TO_CODE = (10.0 * AU_M ** 2) / MSUN_KG          # ~1.1252e-7
# volume density: 1 g/cm^3 -> Msun/AU^3
RHO_CGS_TO_CODE = (1000.0 * AU_M ** 3) / MSUN_KG          # ~1.6833e+6
# length: 1 km -> AU
KM_TO_AU = 1.0e3 / AU_M                                   # ~6.6846e-9
# speed: 1 AU/yr -> m/s
AUYR_TO_MS = AU_M / YR_S                                  # ~4740.6


def au_to_km(x):
    return x / KM_TO_AU


def auyr_to_ms(x):
    return x * AUYR_TO_MS


def msun_to_mearth(x):
    return x / MEARTH


def radius_from_mass(mass, rho_code):
    """Sphere radius (AU) from mass (Msun) and bulk density (Msun/AU^3).

    Thesis Eq. (3.3):  m = (4/3) pi R^3 rho
    """
    return (3.0 * np.asarray(mass) / (4.0 * np.pi * rho_code)) ** (1.0 / 3.0)


def mass_from_radius(radius, rho_code):
    """Inverse of radius_from_mass."""
    return (4.0 / 3.0) * np.pi * np.asarray(radius) ** 3 * rho_code
