# =====================================================================
# COMPUTE_VOLUME_FRACTIONS — ANALYTICAL VOLUME FRACTION CALCULATOR
# 
# Description:
#   Computes the effective volume fractions (f_ceil, f_photo, f_resp,
#   f_transp, f_uptake) from the geometric parameters. Used for the
#   homogeneous reduction and for normalization in the 3D model.
#
# Version: 1.0
# Reference: Trouillefou, C.M. et al. (2026). Parametric resilience
#            in a closed ecological reactor: The Wood Wide Web as a
#            distributed PID controller for life support.
#            npj Microgravity (submitted).
#
# Key features:
#   1. Analytical integration over the two biomes.
#   2. Returns a dictionary of fractions.
#   3. Used in algo_2_3d_bio_V8.3.py for homogeneous analysis.
# =====================================================================

def compute_volume_fractions(params):
    """
    Calcule les fractions volumiques effectives pour chaque processus
    à partir des paramètres géométriques adimensionnés.

    Parameters
    ----------
    params : dict
        Clés attendues :
        - 'x_split' : position horizontale séparant continental (gauche) et marais (droite)
        - 'z_ceil'  : seuil vertical pour le module Peltier (z > z_ceil*z_max)
        - 'z_sol_continental' : surface du sol continental
        - 'z_plant_max_continental' : hauteur max du feuillage continental
        - 'z_sol_marais' : surface du sol marais
        - 'z_plant_max_marais' : hauteur max du feuillage marais

    Returns
    -------
    dict avec les clés 'f_ceil', 'f_photo', 'f_resp', 'f_transp', 'f_uptake'
    """
    x_split = params.get('x_split', 0.5)
    w_continental = x_split
    w_marais = 1.0 - x_split

    # Seuils normalisés pour le biome continental
    z_sol_c = params.get('z_sol_continental', 0.4)
    z_max_c = params.get('z_plant_max_continental', 0.8)
    z_resp_c = (z_sol_c + z_max_c) / 2.0

    # Seuils normalisés pour le biome marais
    z_sol_m = params.get('z_sol_marais', 0.2)
    z_max_m = params.get('z_plant_max_marais', 0.6)
    z_resp_m = (z_sol_m + z_max_m) / 2.0

    # Fractions pour chaque processus
    f_photo  = w_continental * (z_max_c - z_sol_c) + w_marais * (z_max_m - z_sol_m)
    f_resp   = w_continental * z_resp_c + w_marais * z_resp_m
    f_transp = w_continental * (z_resp_c - z_sol_c) + w_marais * (z_resp_m - z_sol_m)
    f_uptake = w_continental * z_sol_c + w_marais * z_sol_m
    f_ceil   = 1.0 - params.get('z_ceil', 0.9)

    return {
        'f_ceil':   f_ceil,
        'f_photo':  f_photo,
        'f_resp':   f_resp,
        'f_transp': f_transp,
        'f_uptake': f_uptake
    }