# biology.py
# =====================================================================
# Rate functions for the TECP V10.3 audit.
# All rates in s^-1 unless otherwise stated.
# Each function returns either a FiPy expression or a numpy array,
# exactly as in V10.3.
# =====================================================================

import numpy as np


def compute_photosynthesis(P, CO2, Temp, theta, bio_active, M_photo, day):
    """
    Photosynthesis (volume fraction of O2 per second).
    bio_active = max(0, biomass) WITHOUT soil_mask (as in V10.3).
    """
    f_T = np.maximum(0, np.exp(-((Temp.value - P['T_opt']) / P['T_range'])**2))
    Pmax_eff = P['Pmax']    ###P['Pmax'] * (1 + 0.1 * day)
    return (
        Pmax_eff
        * (CO2 / (CO2 + P['k_CO2']))
        * f_T
        * theta
        * bio_active
        * M_photo
    )


def compute_respiration(P, Temp, bio_active, M_resp):
    """Plant respiration."""
    resp_factor = np.maximum(0.1, 1 + 0.05 * (Temp.value - P['T_opt']))
    return P['R_maint'] * bio_active * resp_factor * M_resp


def compute_thermal_sources(P, photosynthesis, respiration):
    """Metabolic and radiative heat sources."""
    Q_bio = P['eta_heat'] * respiration
    Q_rad = P['alpha_heat'] * photosynthesis
    return Q_bio, Q_rad


def compute_stress(P, photosynthesis):
    """
    Stress signal: max(0, P_stress_threshold - photosynthesis).

    P_stress_threshold is a PHENOMENOLOGICAL threshold on the photosynthesis
    rate (s^-1). When the photosynthetic rate drops below this value, a
    stress signal is emitted, upstream of the mycorrhizal signalling chain.

    WARNING: This is NOT the C3 CO2 compensation point (30-50 ppm). For that
    concept, a separate parameter (P_co2_compensation, in concentration units)
    would need to be introduced in compute_photosynthesis.
    """
    return np.maximum(0.0, P['P_stress_threshold'] - photosynthesis.value)


def compute_hill_term(P, S):
    """Hill term from S values."""
    S_safe = np.maximum(0, S.value)
    return (
        S_safe**P['n_hill']
        / (P['Km']**P['n_hill'] + S_safe**P['n_hill'] + 1e-8)
    )


def compute_cooperation(P, hill_term, biomass, soil_mask, tau_base):
    """Cooperation term (divided by TAU_BASE)."""
    return (P['gamma'] * hill_term * biomass.value * soil_mask) / tau_base


def compute_evap(P, theta, Temp, day):
    """Evaporation."""
    return P['evap_coeff'] * (1 + day) * theta * (Temp - 273.0)


def compute_uptake(P, theta, soil_mask):
    """Root uptake (computed in V10.3 but not used in the theta equation)."""
    return P['uptake_coeff'] * theta * soil_mask


def compute_transpiration(P, biomass, day, M_transp):
    """Transpiration."""
    return P['transpiration_coeff'] * biomass * (1 + day) * M_transp


def compute_condensation(P, Cv, mesh):
    """Condensation term (returns a CellVariable)."""
    from fipy import CellVariable
    RH = float(np.mean(Cv.value) / 0.03)
    cond_val = max(0, P['k_cond'] * (RH - 0.6))
    return CellVariable(mesh=mesh, value=cond_val)