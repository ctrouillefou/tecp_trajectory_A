#!/usr/bin/env python3
# =====================================================================
# test_porte1bis_matter.py
#
# Porte 1bis — Matter balance test (carbon, oxygen, water) for C2.
#
# Changes vs previous version:
#   * S_INIT and M_INIT read from cfg (S_INIT = 1e-5).
#   * hist['C_coop'] recorded and passed to the carbon balance.
#   * net_photo = mean(P) - mean(R)  (was sum, inconsistent with P_photo).
#   * RECORD_STEP = 1 (720 records over 3 days).
#   * Water balance: primary test cond_flux=None (strict conservation);
#     secondary diagnostic cond_flux=cond_mean (leak hypothesis).
#   * Publication-grade provenance header extended.
# =====================================================================

import json, os, csv, time
import numpy as np
from fipy import CellVariable, Grid3D
from fipy.solvers import DefaultSolver
import matplotlib; matplotlib.use('Agg')

import config as cfg
from biology import *
from equations import *
from build_masks import build_masks
from diagnostics import (
    compute_carbon_balance_from_hist,
    compute_oxygen_balance_from_hist,
    compute_water_balance_from_hist,
    print_matter_balance_summary,
)
from logging_setup import log_run_header, log_run_footer

# --- C2 reference (all read from cfg) ---
K_CO2          = cfg.K_CO2_EXPECTED
PMAX_FACTOR    = cfg.PMAX_FACTOR
RMAINT_DIV     = cfg.RMAINT_DIV
RQ             = cfg.RQ
CREW           = cfg.CREW_VALUES[0]
GAMMA          = cfg.GAMMA
B_ROOT_FRAC    = cfg.B_ROOT_FRACTION
S_INIT         = cfg.S_INIT
M_INIT         = cfg.M_INIT
DURATION_DAYS  = 3       # Porte 1bis specific
RECORD_STEP    = 1       # 720 records over 3 days

MONITORED = {'theta': 0.0, 'Temp': 250.0, 'Cv': 0.0, 'O2': 0.0,
             'CO2': 0.0, 'bio': 0.0, 'S': 0.0, 'M': 0.0}


def main():
    with open('params_real.json') as f:
        P_base = json.load(f)
    Y_B = P_base.get('Y_B', 1.0)

    # --- Local parameter set ---
    P = P_base.copy()
    P['k_CO2']   = K_CO2
    P['Pmax']    = P_base['Pmax'] * PMAX_FACTOR
    P['R_maint'] = P_base['R_maint'] / RMAINT_DIV
    P['gamma']   = GAMMA
    P['transpiration_coeff'] = P_base['uptake_coeff'] * 0.9
    P['evap_coeff']          = P_base['evap_coeff'] / 10.0

    O2_cons_eff  = P_base['O2_cons'] * CREW
    CO2_prod_eff = RQ * O2_cons_eff

    scale_ratio  = cfg.BASELINE_SPEED / cfg.SPEED
    lambda_S_eff = P['lambda_S'] * scale_ratio
    k_prod_eff   = P['k_prod']   * scale_ratio
    k_basal_eff  = P['k_basal']  * scale_ratio

    # --- Provenance header (all effective params traced) ---
    log_run_header(
        "TECP V10.3-Audit — Porte 1bis: Matter balance test (C2, 3 days)",
        config_lines=[
            f"K_CO2           = {K_CO2:.4e}",
            f"PMAX_FACTOR     = {PMAX_FACTOR}",
            f"RMAINT_DIV      = {RMAINT_DIV}",
            f"RQ              = {RQ}",
            f"CREW            = {CREW}",
            f"GAMMA           = {GAMMA}",
            f"B_ROOT_FRAC     = {B_ROOT_FRAC}",
            f"DURATION_DAYS   = {DURATION_DAYS}",
            f"RECORD_STEP     = {RECORD_STEP}",
            f"DT              = {cfg.DT:.0f} s",
            f"STROBE_STEPS    = {cfg.STROBE_STEPS}",
            f"NX, NY, NZ      = {cfg.NX}, {cfg.NY}, {cfg.NZ}",
            f"DX, DY, DZ      = {cfg.DX}, {cfg.DY}, {cfg.DZ}",
            f"SPEED           = {cfg.SPEED}",
            f"BASELINE_SPEED  = {cfg.BASELINE_SPEED}",
            f"scale_ratio     = {scale_ratio:.6f}",
            f"lambda_S_eff    = {lambda_S_eff:.6e}",
            f"k_prod_eff      = {k_prod_eff:.6e}",
            f"k_basal_eff     = {k_basal_eff:.6e}",
            f"Y_B             = {Y_B}",
            f"O2_cons_eff     = {O2_cons_eff:.4e}",
            f"CO2_prod_eff    = {CO2_prod_eff:.4e}",
            f"Solver          = DefaultSolver(tolerance=1e-12, iterations=5000)",
        ],
        extra_lines=[
            f"Initial state   : theta=0.30, Temp=298.0 K, Cv=0.02, "
            f"O2=0.21, CO2=4.0e-4",
            f"                  bio=0.1*(0.8*M_photo+0.2*M_uptake), "
            f"S={S_INIT:.4e}, M={M_INIT:.4f}",
        ],
    )

    # --- Mesh and masks ---
    mesh = Grid3D(nx=cfg.NX, ny=cfg.NY, nz=cfg.NZ,
                  dx=cfg.DX, dy=cfg.DY, dz=cfg.DZ)
    geometry = {k: P_base[k] for k in [
        'x_split', 'z_ceil', 'z_sol_continental', 'z_plant_max_continental',
        'z_sol_marsh', 'z_plant_max_marsh']}
    masks = build_masks(mesh, geometry, extend_photo_to_plant_max=False)
    M_ceil, M_photo, M_resp, M_transp, M_uptake = masks
    soil_mask, ceiling_mask = M_uptake, M_ceil
    V_total = float(np.sum(mesh.cellVolumes))

    # --- Initial state ---
    M_photo_arr  = np.asarray(M_photo.value)
    M_uptake_arr = np.asarray(M_uptake.value)
    bio_init = 0.1 * ((1 - B_ROOT_FRAC) * M_photo_arr
                      + B_ROOT_FRAC * M_uptake_arr)

    theta = CellVariable(mesh=mesh, value=0.3,      hasOld=True)
    Temp  = CellVariable(mesh=mesh, value=298.0,    hasOld=True)
    Cv    = CellVariable(mesh=mesh, value=0.02,     hasOld=True)
    O2    = CellVariable(mesh=mesh, value=0.21,     hasOld=True)
    CO2   = CellVariable(mesh=mesh, value=0.0004,   hasOld=True)
    bio   = CellVariable(mesh=mesh, value=bio_init, hasOld=True)
    S     = CellVariable(mesh=mesh, value=S_INIT,   hasOld=True)   # CHANGE
    M     = CellVariable(mesh=mesh, value=M_INIT,   hasOld=True)   # CHANGE
    vars_dict = {'theta': theta, 'Temp': Temp, 'Cv': Cv, 'O2': O2,
                 'CO2': CO2, 'bio': bio, 'S': S, 'M': M}

    tight_solver = DefaultSolver(tolerance=1e-12, iterations=5000)

    hist = {k: [] for k in ['time_h', 'CO2', 'O2', 'bio', 'M', 'theta',
                            'S', 'Cv', 'Temp', 'P_photo', 'R_plant',
                            'C_coop', 'net_photo', 'cond', 'evap',
                            'transp']}
    viol = {k: 0 for k in MONITORED}

    N = DURATION_DAYS * cfg.STROBE_STEPS
    t_start = time.time()
    print(f"\nLaunching C2 — {DURATION_DAYS} days, {N} steps, "
          f"RECORD_STEP={RECORD_STEP}")
    print(f"  Pmax x{PMAX_FACTOR}, k_CO2={K_CO2}, RQ={RQ}, crew={CREW}")
    print(f"  S_INIT={S_INIT:.4e}, M_INIT={M_INIT:.4f}")

    for step in range(N):
        t = step * cfg.DT
        for v in vars_dict.values():
            v.updateOld()

        day    = 0.5 + 0.5 * np.sin(2 * np.pi * t / cfg.FORCING_PERIOD)
        cond   = compute_condensation(P, Cv, mesh)
        evap   = compute_evap(P, theta, Temp, day)
        transp = compute_transpiration(P, bio, day, M_transp)
        bio_active = np.maximum(0, bio.value)

        photo = compute_photosynthesis(P, CO2, Temp, theta,
                                       bio_active, M_photo, day)
        resp  = compute_respiration(P, Temp, bio_active, M_resp)
        Q_bio, Q_rad = compute_thermal_sources(P, photo, resp)

        stress   = compute_stress(P, photo)
        source_S = (k_prod_eff * stress + k_basal_eff) * soil_mask
        hill     = compute_hill_term(P, S)
        coop     = compute_cooperation(P, hill, bio, soil_mask, cfg.TAU_BASE)

        eq_theta = build_theta_eq(transp, evap, cond)
        eq_vapor = build_vapor_eq(evap, transp, cond)
        eq_temp  = build_temp_eq(Q_bio, Q_rad, cond, ceiling_mask, P)
        eq_O2    = build_O2_eq(photo, resp, O2_cons_eff)
        eq_CO2   = build_CO2_eq(photo, resp, CO2_prod_eff)
        eq_S     = build_S_eq(S, M, source_S, lambda_S_eff)
        eq_M     = build_M_eq(M, S, soil_mask, P)
        eq_bio   = build_bio_eq(bio, photo, resp, coop, P)

        eq_theta.solve(var=theta, dt=cfg.DT, solver=tight_solver)
        eq_vapor.solve(var=Cv,    dt=cfg.DT, solver=tight_solver)
        eq_temp.solve(var=Temp,   dt=cfg.DT, solver=tight_solver)
        eq_O2.solve(var=O2,       dt=cfg.DT, solver=tight_solver)
        eq_CO2.solve(var=CO2,     dt=cfg.DT, solver=tight_solver)
        eq_S.solve(var=S,         dt=cfg.DT, solver=tight_solver)
        eq_M.solve(var=M,         dt=cfg.DT, solver=tight_solver)
        eq_bio.solve(var=bio,     dt=cfg.DT, solver=tight_solver)

        for name, obj in vars_dict.items():
            if float(np.min(obj.value)) < MONITORED[name]:
                viol[name] += 1

        if step % RECORD_STEP == 0:
            hist['time_h'].append(t / 3600.0)
            hist['CO2'].append(float(np.mean(CO2.value)))
            hist['O2'].append(float(np.mean(O2.value)))
            hist['bio'].append(float(np.mean(bio.value)))
            hist['M'].append(float(np.mean(M.value)))
            hist['theta'].append(float(np.mean(theta.value)))
            hist['S'].append(float(np.mean(S.value)))
            hist['Cv'].append(float(np.mean(Cv.value)))
            hist['Temp'].append(float(np.mean(Temp.value)))
            hist['P_photo'].append(float(np.mean(photo.value)))
            hist['R_plant'].append(float(np.mean(resp.value)))
            hist['C_coop'].append(float(np.mean(coop)))            # CHANGE
            hist['net_photo'].append(                              # CHANGE
                float(np.mean(photo.value) - np.mean(resp.value)))
            hist['cond'].append(float(np.mean(cond.value)))
            hist['evap'].append(float(np.mean(evap.value)))
            hist['transp'].append(float(np.mean(transp.value)))

        if step % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  step {step}/{N} ({step / N * 100:.0f}%) | "
                  f"CO2={np.mean(CO2.value):.4e} | "
                  f"O2={np.mean(O2.value):.4f} | "
                  f"bio={np.mean(bio.value):.4f} | {elapsed:.0f}s")

    elapsed = time.time() - t_start
    print(f"\nRun finished in {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # --- Save timeseries ---
    os.makedirs("Porte1bis", exist_ok=True)
    keys = list(hist.keys())
    with open("Porte1bis/C2_3days_matter.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(len(hist['time_h'])):
            w.writerow([hist[k][i] for k in keys])

    # --- Balances ---
    cb = compute_carbon_balance_from_hist(
        hist, Y_B=Y_B, V_total=V_total,
        crew_CO2_prod=CO2_prod_eff, Y_B_resp=1.0)   # C_coop read from hist

    ob = compute_oxygen_balance_from_hist(
        hist, V_total=V_total,
        O2_cons_eff=O2_cons_eff, Y_O2_resp=1.0)

    # PRIMARY: strict conservation
    wb_primary = compute_water_balance_from_hist(
        hist, V_total=V_total, cond_flux=None)
    # SECONDARY: leak hypothesis (diagnostic only)
    wb_secondary = compute_water_balance_from_hist(
        hist, V_total=V_total, cond_flux=float(np.mean(hist['cond'])))

    print("\n--- Water balance: PRIMARY test (cond_flux=None) ---")
    print_matter_balance_summary(cb, ob, wb_primary)

    print("\n--- Water balance: SECONDARY diagnostic (cond_flux=cond_mean) ---")
    print(f"  mode = {wb_secondary['mode']}")
    print(f"  max_rel = {wb_secondary['max_relative_residual']:.3e}")
    print(f"  bilan_closes = {wb_secondary['bilan_closes']}")

    # --- Summary CSV ---
    summary_rows = [
        {'balance': 'carbon',
         'verdict': 'PASS' if cb.get('bilan_closes') else 'FAIL',
         'max_relative_residual': cb.get('max_relative_residual', np.nan),
         'note': cb.get('note', '')},
        {'balance': 'oxygen',
         'verdict': 'PASS' if ob.get('bilan_closes') else 'FAIL',
         'max_relative_residual': ob.get('max_relative_residual', np.nan),
         'note': ob.get('note', '')},
        {'balance': 'water_primary',
         'verdict': 'PASS' if wb_primary.get('bilan_closes') else 'FAIL',
         'max_relative_residual': wb_primary.get('max_relative_residual',
                                                np.nan),
         'note': wb_primary.get('mode', '')},
        {'balance': 'water_secondary',
         'verdict': 'PASS' if wb_secondary.get('bilan_closes') else 'FAIL',
         'max_relative_residual': wb_secondary.get('max_relative_residual',
                                                  np.nan),
         'note': wb_secondary.get('mode', '')},
    ]
    with open("Porte1bis/matter_summary.csv", 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['balance', 'verdict',
                                          'max_relative_residual', 'note'])
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\nSaved to Porte1bis/")
    print(f"  - C2_3days_matter.csv  ({len(hist['time_h'])} records)")
    print(f"  - matter_summary.csv   (4 verdicts: 3 primary + 1 secondary)")
    print(f"  Positivity violations: {sum(viol.values())}")

    log_run_footer(
        elapsed_s=elapsed,
        verdicts={
            'Carbon'         : 'PASS' if cb.get('bilan_closes') else 'FAIL',
            'Oxygen'         : 'PASS' if ob.get('bilan_closes') else 'FAIL',
            'Water_primary'  : 'PASS' if wb_primary.get('bilan_closes') else 'FAIL',
            'Water_secondary': 'PASS' if wb_secondary.get('bilan_closes') else 'FAIL',
        },
        violations=sum(viol.values()),
        files=[
            "Porte1bis/C2_3days_matter.csv",
            "Porte1bis/matter_summary.csv",
        ],
        extra_lines=[
            f"Carbon max_rel  = {cb.get('max_relative_residual', float('nan')):.3e}",
            f"Oxygen max_rel  = {ob.get('max_relative_residual', float('nan')):.3e}",
            f"Water max_rel (primary)   = "
            f"{wb_primary.get('max_relative_residual', float('nan')):.3e}",
            f"Water max_rel (secondary) = "
            f"{wb_secondary.get('max_relative_residual', float('nan')):.3e}",
        ],
    )


if __name__ == "__main__":
    main()