# build_masks.py — Reusable Boolean mask builder for the TECP
# (unchanged except for the optional extend_photo_to_plant_max flag)

import numpy as np
from fipy import CellVariable

def build_masks(mesh, params, extend_photo_to_plant_max=False):
    """
    Construct the five Boolean masks from geometric parameters.

    Parameters
    ----------
    mesh : fipy.meshes.grid3D.Grid3D
        The 3D mesh on which masks are defined.
    params : dict
        Dictionary containing geometric thresholds.
    extend_photo_to_plant_max : bool, optional
        If True, the photosynthesis mask (M_photo) is extended from
        z_sol to z_plant_max (full foliage height) instead of
        z_sol to z_resp (lower half). Default: False.

    Returns
    -------
    M_ceil, M_photo, M_resp, M_transp, M_uptake : tuple of CellVariable
    """
    x = mesh.cellCenters[0]
    z = mesh.cellCenters[2]
    z_max = z.max()
    x_mid = params['x_split'] * x.max()

    continental = (x < x_mid)
    marsh = (x >= x_mid)

    # Continental thresholds
    z_sol_c  = params['z_sol_continental'] * z_max
    z_max_c  = params['z_plant_max_continental'] * z_max
    z_resp_c = (z_sol_c + z_max_c) / 2.0

    # Marsh thresholds
    z_sol_m  = params['z_sol_marsh'] * z_max
    z_max_m  = params['z_plant_max_marsh'] * z_max
    z_resp_m = (z_sol_m + z_max_m) / 2.0

    # 1. Peltier ceiling mask (unchanged)
    M_ceil = CellVariable(mesh=mesh, value=0.0)
    M_ceil.setValue(1.0, where=(z > params['z_ceil'] * z_max))

    # 2. Photosynthesis mask (extendable)
    M_photo = CellVariable(mesh=mesh, value=0.0)
    if extend_photo_to_plant_max:
        z_top_c = z_max_c
        z_top_m = z_max_m
    else:
        z_top_c = z_resp_c
        z_top_m = z_resp_m
    M_photo.setValue(1.0, where=(continental & (z >= z_sol_c) & (z <= z_top_c)) |
                                (marsh       & (z >= z_sol_m) & (z <= z_top_m)))

    # 3. Respiration mask (extendable)
    if extend_photo_to_plant_max:
        z_top_c = z_max_c
        z_top_m = z_max_m
    else:
        z_top_c = z_resp_c
        z_top_m = z_resp_m
    M_resp = CellVariable(mesh=mesh, value=0.0)
    M_resp.setValue(1.0, where=(continental & (z <= z_top_c)) |
                                (marsh       & (z <= z_top_m)))

    # 4. Transpiration mask (unchanged: same vertical range as M_photo in V10.3)
    if extend_photo_to_plant_max:
        z_top_c = z_max_c
        z_top_m = z_max_m
    else:
        z_top_c = z_resp_c
        z_top_m = z_resp_m
    M_transp = CellVariable(mesh=mesh, value=0.0)
    M_transp.setValue(1.0, where=(continental & (z >= z_sol_c) & (z <= z_top_c)) |
                                  (marsh       & (z >= z_sol_m) & (z <= z_top_m)))

    # 5. Root uptake mask (unchanged)
    M_uptake = CellVariable(mesh=mesh, value=0.0)
    M_uptake.setValue(1.0, where=(continental & (z <= z_sol_c)) |
                                  (marsh       & (z <= z_sol_m)))

    return M_ceil, M_photo, M_resp, M_transp, M_uptake