#!/usr/bin/env python3
# =====================================================================
# run_simulation.py
# TECP V10.3 audit — modular entry point.
#
# Assembles config.py, biology.py, equations.py, diagnostics.py
# and reproduces the exact behavior of algo_3_3d_bio_V10.3_audit.py.
# =====================================================================

import json
import time
import os
import sys
import csv
import numpy as np
from fipy import CellVariable, Grid3D
from fipy.solvers import DefaultSolver
import matplotlib
matplotlib.use('Agg')

import config as cfg
from biology import (
    compute_photosynthesis, compute_respiration, compute_thermal_sources,
    compute_stress, compute_hill_term, compute_cooperation,
    compute_evap, compute_uptake, compute_transpiration,
    compute_condensation,
)
from equations import (
    build_theta_eq, build_vapor_eq, build_temp_eq,
    build_O2_eq, build_CO2_eq, build_S_eq, build_M_eq, build_bio_eq,
)
from diagnostics import (
    report_masks, print_diagnostic_summary,
    save_timeseries_csv, save_figure,
)
from build_masks import build_masks

from diagnostics import get_window_mask

# ---------------------------------------------------------------------
# SIMULATION FUNCTION
# ---------------------------------------------------------------------
def run_simulation(crew_mult, P_base, mesh, masks, tight_solver, timestamp):
    """Run 30 days of simulation for a given crew multiplier."""
    M_ceil, M_photo, M_resp, M_transp, M_uptake = masks
    soil_mask = M_uptake
    ceiling_mask = M_ceil

    # --- Local copy of parameters (fixes cumulative-mutation bug) ---
    P = P_base.copy()
    if not np.isclose(P['k_CO2'], cfg.K_CO2_EXPECTED):
        raise ValueError(
            f"Configuration {cfg.CONFIG_NAME}: "
            f"k_CO2 attendu = {cfg.K_CO2_EXPECTED}, "
            f"valeur reçue = {P['k_CO2']}"
        )
    P['Pmax'] = P_base['Pmax'] * cfg.PMAX_FACTOR
    P['R_maint'] = P_base['R_maint'] / cfg.RMAINT_DIV
    P['gamma'] = cfg.GAMMA
    P['transpiration_coeff'] = P_base['uptake_coeff'] * 0.9
    P['evap_coeff'] = P_base['evap_coeff'] / 10.0

    O2_cons_eff = P_base['O2_cons'] * crew_mult
    CO2_prod_eff = cfg.RQ * O2_cons_eff

    print(f"Configuration     = {cfg.CONFIG_NAME}")
    print(f"Pmax factor       = {cfg.PMAX_FACTOR}")
    print(f"Rmaint divisor    = {cfg.RMAINT_DIV}")
    print(f"k_CO2             = {P['k_CO2']:.6e}")
    print(f"RQ                = {cfg.RQ}")
    print(f"gamma             = {P['gamma']:.6e}")
    print(f"B_ROOT_FRACTION   = {cfg.B_ROOT_FRACTION}")
    print(f"S_INIT            = {cfg.S_INIT:.6e}")
    print(f"M_INIT            = {cfg.M_INIT:.6e}")
    print(f"Duration          = {cfg.DURATION_DAYS} days ({cfg.N_STEPS} steps)")
    print(f"\n--- crew_mult = {crew_mult} ---")
    print(f"O2_cons (test)  = {O2_cons_eff:.6e} s^-1")
    print(f"CO2_prod (test) = {CO2_prod_eff:.6e} s^-1")

    scale_ratio = cfg.BASELINE_SPEED / cfg.SPEED
    lambda_S_eff = P['lambda_S'] * scale_ratio
    k_prod_eff   = P['k_prod']   * scale_ratio
    k_basal_eff  = P['k_basal']  * scale_ratio

    # --- Variables (reinitialised per crew) ---
    theta   = CellVariable(mesh=mesh, value=0.3,    hasOld=True)
    Temp    = CellVariable(mesh=mesh, value=298.0,  hasOld=True)
    Cv      = CellVariable(mesh=mesh, value=0.02,   hasOld=True)
    O2      = CellVariable(mesh=mesh, value=0.21,   hasOld=True)
    CO2     = CellVariable(mesh=mesh, value=0.0004, hasOld=True)

    f_root = cfg.B_ROOT_FRACTION

    biomass_init = cfg.BIOMASS_INIT_SCALE * (
        (1.0 - f_root) * np.asarray(M_photo.value)
        + f_root * np.asarray(M_uptake.value)
    )
    biomass = CellVariable(mesh=mesh, value=biomass_init, hasOld=True)
    print(
        f"Biomass initialised (80% M_photo + 20% M_uptake). "
        f"Mean = {np.mean(biomass.value):.6f}, "
        f"Max = {np.max(biomass.value):.6f}"
    )

    S = CellVariable(mesh=mesh, value=cfg.S_INIT, hasOld=True)
    M = CellVariable(mesh=mesh, value=cfg.M_INIT, hasOld=True)

    # --- Storage ---
    hist = {k: [] for k in [
        'time_h', 'CO2', 'O2', 'Temp', 'theta', 'biomass', 'S', 'M',
        'P_photo', 'R_plant', 'R_crew', 'Q_bio', 'Q_rad', 'Cv', 'C_coop', 'net_gas', 'net_biomass'
    ]}

    print(f"🚀 Running crew ×{crew_mult}...")

    for step in range(cfg.N_STEPS):
        t = step * cfg.DT

        theta.updateOld(); Temp.updateOld(); Cv.updateOld()
        O2.updateOld(); CO2.updateOld(); biomass.updateOld()
        S.updateOld(); M.updateOld()

        day = 0.5 + 0.5 * np.sin(2 * np.pi * t / cfg.FORCING_PERIOD)

        # Condensation
        cond_term = compute_condensation(P, Cv, mesh)

        # Evapotranspiration and uptake
        evap = compute_evap(P, theta, Temp, day)
        uptake = compute_uptake(P, theta, soil_mask)  # computed, not used (as in V10.3)
        transpiration = compute_transpiration(P, biomass, day, M_transp)

        # Photosynthesis and respiration
        bio_active = np.maximum(0, biomass.value)   # no soil_mask
        photosynthesis = compute_photosynthesis(P, CO2, Temp, theta, bio_active, M_photo, day)
        respiration = compute_respiration(P, Temp, bio_active, M_resp)

        # Thermal sources
        Q_bio, Q_rad = compute_thermal_sources(P, photosynthesis, respiration)

        # WWW
        stress = compute_stress(P, photosynthesis)
        source_S = (k_prod_eff * stress + k_basal_eff) * soil_mask
        hill_term = compute_hill_term(P, S)
        cooperation_term = compute_cooperation(P, hill_term, biomass, soil_mask, cfg.TAU_BASE)
        YB = P['Y_B']
        net_gas = photosynthesis - respiration


        # Build equations
        eq_theta = build_theta_eq(transpiration, evap, cond_term)
        eq_vapor = build_vapor_eq(evap, transpiration, cond_term)
        eq_temp  = build_temp_eq(Q_bio, Q_rad, cond_term, ceiling_mask, P)
        eq_O2    = build_O2_eq(photosynthesis, respiration, O2_cons_eff)
        eq_CO2   = build_CO2_eq(photosynthesis, respiration, CO2_prod_eff)
        eq_S     = build_S_eq(S, M, source_S, lambda_S_eff)
        eq_M     = build_M_eq(M, S, soil_mask, P)
        eq_bio   = build_bio_eq(biomass, photosynthesis, respiration, cooperation_term, P)

        # Solve
        eq_theta.solve(var=theta, dt=cfg.DT, solver=tight_solver)
        eq_vapor.solve(var=Cv, dt=cfg.DT, solver=tight_solver)
        eq_temp.solve(var=Temp, dt=cfg.DT, solver=tight_solver)
        eq_O2.solve(var=O2, dt=cfg.DT, solver=tight_solver)
        eq_CO2.solve(var=CO2, dt=cfg.DT, solver=tight_solver)
        eq_S.solve(var=S, dt=cfg.DT, solver=tight_solver)
        eq_M.solve(var=M, dt=cfg.DT, solver=tight_solver)
        eq_bio.solve(var=biomass, dt=cfg.DT, solver=tight_solver)

        if step in (0, 1, 10, 100, 500, 720, 1000, 479):
            print(
                f"[M] step={step:4d} "
                f"mean={np.mean(M.value):.12f} "
                f"min={np.min(M.value):.12f} "
                f"max={np.max(M.value):.12f} "
                f"delta={np.mean(M.value - M.old):.3e}"
            )

        #les flux enregistrés correspondent à l’état pré-résolution, tandis que la biomasse enregistrée correspond à l’état post-résolution
        net_biomass = (YB * photosynthesis - respiration + cooperation_term)

        # Record
        if step % cfg.RECORD_STEP == 0:
            hist['time_h'].append(t / 3600.0)
            hist['CO2'].append(np.mean(CO2.value))
            hist['O2'].append(np.mean(O2.value))
            hist['Temp'].append(np.mean(Temp.value))
            hist['theta'].append(np.mean(theta.value))
            hist['biomass'].append(np.mean(biomass.value))
            hist['S'].append(np.mean(S.value))
            hist['M'].append(np.mean(M.value))
            hist['P_photo'].append(np.mean(photosynthesis.value))
            hist['R_plant'].append(np.mean(respiration.value))
            hist['R_crew'].append(CO2_prod_eff)
            hist['Q_bio'].append(np.mean(Q_bio.value))
            hist['Q_rad'].append(np.mean(Q_rad.value))
            hist['Cv'].append(np.mean(Cv.value))
            hist['C_coop'].append(np.mean(cooperation_term))
            hist['net_gas'].append(np.mean(net_gas.value))
            hist['net_biomass'].append(np.mean(net_biomass))

        # Progress
        if step % 500 == 0 or step == cfg.N_STEPS - 1:
            pct = step / cfg.N_STEPS * 100
            print(f"[crew ×{crew_mult}] Step {step}/{cfg.N_STEPS} ({pct:.1f}%) | "
                  f"CO2 = {np.mean(CO2.value):.6f} | "
                  f"O2 = {np.mean(O2.value):.6f} | "
                  f"T = {np.mean(Temp.value):.2f} | "
                  f"P_photo = {np.mean(photosynthesis.value):.3e}")

    # --- Diagnostic summary (72 h physical-time window) ---
    print_diagnostic_summary(hist, window_hours=72.0, crew_mult=crew_mult)

    # --- Save ---
    # Trajectoire A — C2 reference run (30 days)
    output_dir = f"TrajA_C2_{cfg.DURATION_DAYS}d_crew_{crew_mult:.3f}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    save_timeseries_csv(hist, output_dir)
    save_figure(hist, output_dir, crew_mult, cfg.PMAX_FACTOR)

    print(f"\n📁 Results saved in: {output_dir}")
    print("=" * 72)

    return hist, output_dir


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    # --- Load base parameters ---
    try:
        with open('params_real.json', 'r') as f:
            P_base = json.load(f)
        print("✅ Base parameters loaded from params_real.json")
    except FileNotFoundError:
        print("❌ params_real.json not found.")
        sys.exit(1)

    print(f"Pmax (base)   = {P_base['Pmax']:.6e} s^-1")
    print(f"Pmax (test)   = {P_base['Pmax'] * cfg.PMAX_FACTOR:.6e} s^-1  (×{cfg.PMAX_FACTOR})")
    print(f"R_maint(base) = {P_base['R_maint']:.6e} s^-1")
    print(f"R_maint(test) = {P_base['R_maint'] / cfg.RMAINT_DIV:.6e} s^-1  (÷{cfg.RMAINT_DIV})")

    scale_ratio = cfg.BASELINE_SPEED / cfg.SPEED
    print(f"scale_ratio = {scale_ratio:.4f}")

    print("\n--- Corrected parameters ---")
    print(
        f"k_CO2 = {P_base['k_CO2']:.6e} "
        f"(expected {cfg.K_CO2_EXPECTED:.6e})"
    )
    print(f"Y_B    = {P_base['Y_B']:.6e}")
    print(f"P_stress_threshold= {P_base['P_stress_threshold']:.6e}  (expected ~2.78e-8)")

    # --- Mesh ---
    mesh = Grid3D(nx=cfg.NX, ny=cfg.NY, nz=cfg.NZ,
                  dx=cfg.DX, dy=cfg.DY, dz=cfg.DZ)

    # --- Masks (built once) ---
    geometry_params = {
        'x_split': P_base['x_split'],
        'z_ceil': P_base['z_ceil'],
        'z_sol_continental': P_base['z_sol_continental'],
        'z_plant_max_continental': P_base['z_plant_max_continental'],
        'z_sol_marsh': P_base['z_sol_marsh'],
        'z_plant_max_marsh': P_base['z_plant_max_marsh'],
    }
    masks = build_masks(mesh, geometry_params)
    M_ceil, M_photo, M_resp, M_transp, M_uptake = masks

    report_masks(mesh, {
        'M_ceil': M_ceil,
        'M_photo': M_photo,
        'M_resp': M_resp,
        'M_transp': M_transp,
        'M_uptake': M_uptake,
    })


    # --- Thermal solver (built once, reused across crews) ---
    tight_solver = DefaultSolver(tolerance=1e-12, iterations=5000)

    # --- Timestamp ---
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    # --- Crew sweep ---
    summary = []

    print("\n" + "=" * 72)
    print(f"SWEEP CREW_MULTIPLIER — Trajectoire A ({cfg.DURATION_DAYS} jours)")
    print("=" * 72)
    print(f"CREW_VALUES = {cfg.CREW_VALUES}")
    print("=" * 72)

    for crew in cfg.CREW_VALUES:
        print(f"\n{'='*60}")
        print(f"  RUN crew_mult = {crew}")
        print(f"{'='*60}")

        hist, out_dir = run_simulation(
            crew, P_base, mesh, masks, tight_solver, timestamp
        )

        # --- Sweep summary: use the same 72 h physical-time window ---
        mask_72h, _, _ = get_window_mask(hist, window_hours=72.0)

        co2_72h = np.asarray(hist['CO2'])[mask_72h]
        o2_72h  = np.asarray(hist['O2'])[mask_72h]

        summary.append({
            'duration_days': cfg.DURATION_DAYS,
            'crew': crew,
            'CO2_final': float(hist['CO2'][-1]),
            'CO2_max_72h': float(co2_72h.max()),
            'O2_final': float(hist['O2'][-1]),
            'O2_min_72h': float(o2_72h.min()),
            'theta_final': float(hist['theta'][-1]),
            'biomass_final': float(hist['biomass'][-1]),
            'P_photo_final': float(hist['P_photo'][-1]),
        })

    # --- Final summary ---
    print("\n" + "=" * 72)
    print("SWEEP CREW_MULTIPLIER — RÉSUMÉ")
    print("=" * 72)
    print(f"{'crew':>6s}  {'dur':>4s}  {'CO2_max_72h':>12s}  {'O2_min_72h':>12s}  "
          f"{'theta_f':>8s}  {'B_f':>8s}  {'P_photo_f':>11s}  gas_ok")
    for s in summary:
        gas_ok = (s['CO2_max_72h'] < 0.005) and (s['O2_min_72h'] > 0.15)
        gas_ok_str = "YES" if gas_ok else "NO"
        print(f"{s['crew']:>6.2f}  {s['duration_days']:>4d}  "
              f"{s['CO2_max_72h']:>12.6f}  {s['O2_min_72h']:>12.6f}  "
              f"{s['theta_final']:>8.4f}  {s['biomass_final']:>8.5f}  "
              f"{s['P_photo_final']:>11.3e}  {gas_ok_str}")

    # --- Summary CSV ---
    csv_summary = f"sweep_crew_{timestamp}.csv"
    with open(csv_summary, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n📁 Summary saved: {csv_summary}")
    print("=" * 72)


if __name__ == "__main__":
    main()