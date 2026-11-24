#!/usr/bin/env python3
# =====================================================================
# test_porte2_dt_convergence.py
#
# Porte 2 — Temporal convergence test for C2.
#
# Runs the same C2 configuration twice:
#   - Reference : DT = cfg.DT       (360 s)
#   - Refined   : DT = cfg.DT / 2   (180 s)
#
# Compares equilibrium values at t = 10 days:
#   CO2, O2, bio, theta, S, Temp
# PASS if max relative difference < 5 % on all variables.
#
# Configuration C2:
#   Pmax x2, k_CO2 = 0.001, RQ = 0.85, crew = 0.10
#   R_maint x1, gamma = 0.0020, B_root_fraction = 0.20
#   Option 2, tight solver (tol = 1e-12)
#   Duration = 10 days, RECORD_STEP = 20 (2 h)
#
# Estimated wall time: ~4.5 h total (1.4 h + 2.8 h).
# Launch with: PYTHONUNBUFFERED=1 nohup python3 -u test_porte2_dt_convergence.py > porte2.log 2>&1 &
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
from logging_setup import log_run_header, log_run_footer

# --- C2 reference configuration ---
K_CO2         = 0.0010
PMAX_FACTOR   = 2.0
RMAINT_DIV    = 1.0
RQ            = 0.85
CREW          = 0.10
GAMMA         = 0.0020
B_ROOT_FRAC   = 0.20
DURATION_DAYS = 30
RECORD_STEP   = 20
TOL_REL       = 0.05   # 5 % convergence threshold

MONITORED = {'theta': 0.0, 'Temp': 250.0, 'Cv': 0.0, 'O2': 0.0,
             'CO2': 0.0, 'S': 0.0, 'M': 0.0}


def run_one(dt_value, label, P_base, mesh, masks, V_total, tight_solver):
    """Run C2 for DURATION_DAYS with the given dt. Returns hist + summary."""

    M_ceil, M_photo, M_resp, M_transp, M_uptake = masks
    soil_mask, ceiling_mask = M_uptake, M_ceil

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
    S     = CellVariable(mesh=mesh, value=cfg.S_INIT,     hasOld=True)
    M     = CellVariable(mesh=mesh, value=0.1,      hasOld=True)
    vars_dict = {'theta': theta, 'Temp': Temp, 'Cv': Cv, 'O2': O2,
                 'CO2': CO2, 'S': S, 'M': M}

    # Steps per day = STROBE_STEPS (at DT=360), doubled for DT/2
    steps_per_day = int(round(cfg.STROBE_STEPS * cfg.DT / dt_value))
    N = DURATION_DAYS * steps_per_day

    hist = {k: [] for k in ['time_h', 'CO2', 'O2', 'bio', 'M',
                            'theta', 'S', 'Cv', 'Temp']}

    print(f"\n=== {label} : DT = {dt_value:.0f} s ===")
    print(f"  steps/day = {steps_per_day}, total N = {N}")
    t_start = time.time()

    for step in range(N):
        t = step * dt_value
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

        eq_theta.solve(var=theta, dt=dt_value, solver=tight_solver)
        eq_vapor.solve(var=Cv,    dt=dt_value, solver=tight_solver)
        eq_temp.solve(var=Temp,   dt=dt_value, solver=tight_solver)
        eq_O2.solve(var=O2,       dt=dt_value, solver=tight_solver)
        eq_CO2.solve(var=CO2,     dt=dt_value, solver=tight_solver)
        eq_S.solve(var=S,         dt=dt_value, solver=tight_solver)
        eq_M.solve(var=M,         dt=dt_value, solver=tight_solver)
        eq_bio.solve(var=bio,     dt=dt_value, solver=tight_solver)

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

        if step % 5000 == 0:
            elapsed = time.time() - t_start
            print(f"  [{label}] {step}/{N} ({step / N * 100:.0f}%) | "
                  f"CO2={np.mean(CO2.value):.4e} | "
                  f"O2={np.mean(O2.value):.4f} | {elapsed:.0f}s")

    elapsed = time.time() - t_start
    print(f"  [{label}] finished in {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # Equilibrium means over last 3 days (36 records at 2 h)
    n_last = min(36, len(hist['time_h']))
    summary = {
        'label': label,
        'dt_s': dt_value,
        'CO2_eq':  float(np.mean(hist['CO2'][-n_last:])),
        'O2_eq':   float(np.mean(hist['O2'][-n_last:])),
        'bio_eq':  float(np.mean(hist['bio'][-n_last:])),
        'theta_eq':float(np.mean(hist['theta'][-n_last:])),
        'S_eq':    float(np.mean(hist['S'][-n_last:])),
        'Temp_eq': float(np.mean(hist['Temp'][-n_last:])),
        'elapsed_s': elapsed,
    }
    return hist, summary


def main():
    with open('params_real.json') as f:
        P_base = json.load(f)

    mesh = Grid3D(nx=cfg.NX, ny=cfg.NY, nz=cfg.NZ,
                  dx=cfg.DX, dy=cfg.DY, dz=cfg.DZ)
    geometry = {k: P_base[k] for k in [
        'x_split', 'z_ceil', 'z_sol_continental', 'z_plant_max_continental',
        'z_sol_marsh', 'z_plant_max_marsh']}
    masks = build_masks(mesh, geometry, extend_photo_to_plant_max=False)
    V_total = float(np.sum(mesh.cellVolumes))

    tight_solver = DefaultSolver(tolerance=1e-12, iterations=5000)

    log_run_header(
        "TECP V10.3-Audit — Porte 2: Temporal convergence (C2, 10 days)",
        config_lines=[
            f"K_CO2         = {K_CO2:.4e}",
            f"PMAX_FACTOR   = {PMAX_FACTOR}",
            f"RMAINT_DIV    = {RMAINT_DIV}",
            f"RQ            = {RQ}",
            f"CREW          = {CREW}",
            f"GAMMA         = {GAMMA}",
            f"B_ROOT_FRAC   = {B_ROOT_FRAC}",
            f"DURATION_DAYS = {DURATION_DAYS}",
            f"RECORD_STEP   = {RECORD_STEP}",
            f"DT_ref        = {cfg.DT:.0f} s",
            f"DT_refined    = {cfg.DT/2:.0f} s",
            f"Solver        : DefaultSolver(tolerance=1e-12, iterations=5000)",
            f"V_total       = {V_total:.4e} m3",
        ],
    )

    os.makedirs("Porte2", exist_ok=True)

    # Reference run (DT = cfg.DT)
    hist_ref, summary_ref = run_one(cfg.DT, "REF_DT360", P_base, mesh, masks,
                                    V_total, tight_solver)
    with open("Porte2/REF_DT360.csv", 'w', newline='') as f:
        keys = list(hist_ref.keys())
        w = csv.writer(f); w.writerow(keys)
        for i in range(len(hist_ref['time_h'])):
            w.writerow([hist_ref[k][i] for k in keys])

    # Refined run (DT/2)
    hist_fin, summary_fin = run_one(cfg.DT / 2, "FIN_DT180", P_base, mesh, masks,
                                    V_total, tight_solver)
    with open("Porte2/FIN_DT180.csv", 'w', newline='') as f:
        keys = list(hist_fin.keys())
        w = csv.writer(f); w.writerow(keys)
        for i in range(len(hist_fin['time_h'])):
            w.writerow([hist_fin[k][i] for k in keys])

    # --- Compare equilibrium values ---
    print("\n" + "=" * 78)
    print("PORTE 2 — TEMPORAL CONVERGENCE (DT vs DT/2, t = 10 days)")
    print("=" * 78)
    print(f"{'variable':>10s}  {'DT=360s':>14s}  {'DT=180s':>14s}  "
          f"{'rel_diff':>10s}  {'status':>8s}")
    print("-" * 78)

    variables = ['CO2_eq', 'O2_eq', 'bio_eq', 'theta_eq', 'S_eq', 'Temp_eq']
    verdicts = {}
    max_rel = 0.0
    for var in variables:
        v_ref = summary_ref[var]
        v_fin = summary_fin[var]
        denom = max(abs(v_ref), abs(v_fin), 1e-30)
        rel = abs(v_fin - v_ref) / denom
        status = "PASS" if rel < TOL_REL else "FAIL"
        verdicts[var] = status
        max_rel = max(max_rel, rel)
        print(f"{var:>10s}  {v_ref:>14.6e}  {v_fin:>14.6e}  "
              f"{rel:>10.2e}  {status:>8s}")

    overall = "PASS" if max_rel < TOL_REL else "FAIL"
    print("-" * 78)
    print(f"Max relative difference : {max_rel:.3e}")
    print(f"Overall verdict         : {overall}  (threshold {TOL_REL*100:.0f} %)")
    print("=" * 78)

    # Save convergence summary CSV
    with open("Porte2/convergence_summary.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variable', 'DT_360s', 'DT_180s', 'rel_diff', 'verdict'])
        for var in variables:
            v_ref = summary_ref[var]
            v_fin = summary_fin[var]
            denom = max(abs(v_ref), abs(v_fin), 1e-30)
            rel = abs(v_fin - v_ref) / denom
            w.writerow([var, v_ref, v_fin, rel, verdicts[var]])

    total_elapsed = summary_ref['elapsed_s'] + summary_fin['elapsed_s']
    log_run_footer(
        elapsed_s=total_elapsed,
        verdicts={k.replace('_eq', ''): v for k, v in verdicts.items()},
        files=[
            "Porte2/REF_DT360.csv",
            "Porte2/FIN_DT180.csv",
            "Porte2/convergence_summary.csv",
        ],
        extra_lines=[
            f"Max relative diff = {max_rel:.3e}",
            f"Overall           = {overall}",
        ],
    )


if __name__ == "__main__":
    main()