# equations.py
# =====================================================================
# FiPy equation builders for the TECP V10.3 audit.
# Each function returns a fresh FiPy equation object, to be built
# at each time step (FiPy expressions depend on the current values).
# =====================================================================

from fipy import TransientTerm, DiffusionTerm, ConvectionTerm


def build_theta_eq(transpiration, evap, cond_term):
    """Theta equation (water balance with transpiration, not uptake)."""
    return (
        TransientTerm() ==
        DiffusionTerm(coeff=1e-6) - transpiration - evap + cond_term
    )


def build_vapor_eq(evap, transpiration, cond_term):
    """Vapor equation."""
    return (
        TransientTerm() ==
        DiffusionTerm(coeff=1e-5) + evap + transpiration - cond_term
    )


def build_temp_eq(Q_bio, Q_rad, cond_term, ceiling_mask, P):
    """Thermal equation with Q_bio and Q_rad (V10.3 correction)."""
    return (
        TransientTerm() ==
        DiffusionTerm(coeff=2e-5)
        + Q_bio + Q_rad
        - P['Lv'] * cond_term
        - P['P_elt'] * ceiling_mask
    )


def build_O2_eq(photosynthesis, respiration, O2_cons_eff):
    """O2 equation."""
    return (
        TransientTerm() ==
        DiffusionTerm(coeff=2e-5) + photosynthesis - respiration - O2_cons_eff
    )


def build_CO2_eq(photosynthesis, respiration, CO2_prod_eff):
    """CO2 equation."""
    return (
        TransientTerm() ==
        DiffusionTerm(coeff=2e-5) - photosynthesis + respiration + CO2_prod_eff
    )


def build_S_eq(S, M, source_S, lambda_S_eff):
    """Signal equation (S)."""
    convection_S = ConvectionTerm(coeff=-1e-3 * M.faceGrad, var=S)
    return (
        TransientTerm(var=S) ==
        DiffusionTerm(coeff=1e-4, var=S) + convection_S +
        source_S - lambda_S_eff * S
    )


def build_M_eq(M, S, soil_mask, P):
    """Mycelium equation (M)."""
    return (
        TransientTerm(var=M) ==
        DiffusionTerm(coeff=1e-6, var=M) +
        P['gamma_M'] * S * (1 - M / 1.0) * soil_mask -
        P['delta_M'] * M
    )


def build_bio_eq(biomass, photosynthesis, respiration, cooperation_term, P):
    """Biomass equation (V10.3: uses Y_B from params)."""
    return (
        TransientTerm(var=biomass) ==
        P['Y_B'] * photosynthesis - respiration + cooperation_term
    )