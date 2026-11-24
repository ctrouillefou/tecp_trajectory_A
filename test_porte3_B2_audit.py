#!/usr/bin/env python3
# =====================================================================
# test_porte3_B2_audit.py
#
# Porte 3 — B.2 parameter sensitivity audit for C2.
#
# Tests +/-20% variations on four parameters:
#   - k_CO2                  (Michaelis constant for CO2)
#   - Pmax                   (maximum photosynthetic capacity)
#   - R_maint                (maintenance respiration)
#   - P_stress_threshold     (WWW signalling chain — see WARNING below)
#
# *** WARNING ***
# P_stress_threshold is a PHENOMENOLOGICAL threshold on the
# photosynthesis RATE. It acts via compute_stress -> source_S -> S ->
# Hill -> cooperation. It is NOT the C3 CO2 compensation point
# (30-50 ppm, in concentration units). Testing it probes whether the
# WWW signalling chain is functionally active in C2. Expected outcome:
# effect ~ 0 on CO2/bio, consistent with C_coop/P_photo ~ 3e-7.
# Do NOT describe this result as "sensitivity to C3 compensation point".
#
# Changes vs previous version:
#   * S_INIT, M_INIT read from cfg (S_INIT = 1e-5).
#   * 'P_stress_threshold' added to P_SEUIL_ALIASES; display name
#     clarified as "P_stress_threshold (WWW chain)".
#   * Provenance header extended with effective factors and S_INIT.
#
# Configuration C2 (all from cfg):
#   Pmax x2, k_CO2 = 0.001, RQ = 0.85, crew = 0.10
#   R_maint x1, gamma = 0.0020, B_root_fraction = 0.20
#   S_INIT = 1e-5, M_INIT = 0.1
#   Duration = 3 days, RECORD_STEP = 20 (2 h)
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

# --- C2 reference (all from cfg) ---
K_CO2_REF      = cfg.K_CO2_EXPECTED
PMAX_FACTOR    = cfg.PMAX_FACTOR
RMAINT_DIV     = cfg.RMAINT_DIV
RQ             = cfg.RQ
CREW           = cfg.CREW_VALUES[0]
GAMMA          = cfg.GAMMA
B_ROOT_FRAC    = cfg.B_ROOT_FRACTION
S_INIT         = cfg.S_INIT
M_INIT         = cfg.M_INIT
DURATION_DAYS  = 3
RECORD_STEP    = 20
N_LAST_24H     = 12

SENSITIVITY_THRESHOLDS = {'robust': 0.5, 'acceptable': 1.0}

# Aliases (CHANGE: P_stress_threshold added)
P_SEUIL_ALIASES = ['P_seuil', 'P_threshold', 'CO2_comp', 'CO2_compensation',
                   'CO2_compensation_point', 'Gamma_star',
                   'P_stress_threshold']

MONITORED = {'theta': 0.0, 'Temp': 250.0, 'Cv': 0.0, 'O2': 0.0,
             'CO2': 0.0, 'S': 0.0, 'M': 0.0}


def resolve_p_seuil_key(P_base):
    for alias in P_SEUIL_ALIASES:
        if alias in P_base:
            return alias
    return None


def effective_value(P_base, key, factor):
    if key == 'k_CO2':
        return K_CO2_REF * factor
    elif key == 'Pmax':
        return P_base['Pmax'] * PMAX_FACTOR * factor
    elif key == 'R_maint':
        return P_base['R_maint'] / RMAINT_DIV * factor
    else:
        return P_base[key] * factor


def apply_c2_with_perturbation(P_base, key=None, factor=1.0):
    P = P_base.copy()
    P['k_CO2']   = K_CO2_REF
    P['Pmax']    = P_base['Pmax'] * PMAX_FACTOR
    P['R_maint'] = P_base['R_maint'] / RMAINT_DIV
    P['gamma']   = GAMMA
    P['transpiration_coeff'] = P_base['uptake_coeff'] * 0.9
    P['evap_coeff']          = P_base['evap_coeff'] / 10.0

    if key is not None:
        if key == 'k_CO2':
            P['k_CO2'] = K_CO2_REF * factor
        elif key == 'Pmax':
            P['Pmax'] = P_base['Pmax'] * PMAX_FACTOR * factor
        elif key == 'R_maint':
            P['R_maint'] = P_base['R_maint'] / RMAINT_DIV * factor
        else:
            if key not in P_base:
                raise KeyError(f"Parameter '{key}' not found in params_real.json")
            P[key] = P_base[key] * factor
    return P


HIST_KEYS = ['time_h', 'CO2', 'O2', 'bio', 'M', 'theta', 'S', 'Cv', 'Temp']


def load_hist_from_csv(path):
    data = np.genfromtxt(path, delimiter=',', names=True, dtype=float)
    hist = {}
    for name in data.dtype.names:
        hist[name] = list(np.atleast_1d(data[name]))
    return hist


def run_one(label, P, P_base, mesh, masks, tight_solver, resume=True):
    csv_path = f"Porte3/{label}.csv"
    if resume and os.path.exists(csv_path):
        try:
            hist = load_hist_from_csv(csv_path)
            if len(hist['time_h']) >= 2:
                print(f"  [{label}] loaded from {csv_path} "
                      f"({len(hist['time_h'])} records)")
                return hist
        except Exception as e:
            print(f"  [{label}] failed to reload {csv_path}: {e}")
            print(f"  [{label}] will re-run instead")

    M_ceil, M_photo, M_resp, M_transp, M_uptake = masks
    soil_mask, ceiling_mask = M_uptake, M_ceil

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
    S     = CellVariable(mesh=mesh, value=S_INIT,   hasOld=True)   # CHANGE
    M     = CellVariable(mesh=mesh, value=M_INIT,   hasOld=True)   # CHANGE
    vars_dict = {'theta': theta, 'Temp': Temp, 'Cv': Cv, 'O2': O2,
                 'CO2': CO2, 'S': S, 'M': M}

    hist = {k: [] for k in HIST_KEYS}

    N = DURATION_DAYS * cfg.STROBE_STEPS
    t_start = time.time()
    print(f"\n=== {label} : {N} steps ({DURATION_DAYS} days) ===")
    print(f"  k_CO2={P['k_CO2']:.4e}  Pmax={P['Pmax']:.4e}  "
          f"R_maint={P['R_maint']:.4e}  S_INIT={S_INIT:.4e}")

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

        if step % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  [{label}] {step}/{N} ({step / N * 100:.0f}%) | "
                  f"CO2={np.mean(CO2.value):.4e} | "
                  f"O2={np.mean(O2.value):.4f} | {elapsed:.0f}s")

    elapsed = time.time() - t_start
    print(f"  [{label}] finished in {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(HIST_KEYS)
        for i in range(len(hist['time_h'])):
            w.writerow([hist[k][i] for k in HIST_KEYS])

    return hist


def equilibrium_from_hist(hist, n_last=N_LAST_24H):
    n = min(n_last, len(hist['time_h']))
    return {
        'CO2_eq':   float(np.mean(hist['CO2'][-n:])),
        'O2_eq':    float(np.mean(hist['O2'][-n:])),
        'bio_eq':   float(np.mean(hist['bio'][-n:])),
        'theta_eq': float(np.mean(hist['theta'][-n:])),
        'S_eq':     float(np.mean(hist['S'][-n:])),
        'Temp_eq':  float(np.mean(hist['Temp'][-n:])),
        'CO2_final': float(hist['CO2'][-1]),
        'O2_final':  float(hist['O2'][-1]),
    }


def sensitivity(y_up, y_down, y_ref):
    if abs(y_ref) < 1e-30:
        return float('nan')
    return 2.5 * (y_up - y_down) / y_ref


def verdict_from_S(S, thresholds=SENSITIVITY_THRESHOLDS):
    if not np.isfinite(S):
        return 'n/a'
    aS = abs(S)
    if aS < thresholds['robust']:
        return 'robust'
    elif aS < thresholds['acceptable']:
        return 'acceptable'
    else:
        return 'fragile'


def main():
    with open('params_real.json') as f:
        P_base = json.load(f)

    p_seuil_key = resolve_p_seuil_key(P_base)

    params_to_test = [
        {'name': 'k_CO2 (Michaelis)', 'key': 'k_CO2',
         'lit': '10-30 (Rubisco) / 300-1000 (canopy) ppm',
         'lit_ref': 'Farquhar 1980; Smith & Dukes 2013'},
        {'name': 'Pmax', 'key': 'Pmax',
         'lit': '10-40 umol CO2 m^-2 s^-1 (lettuce-wheat)',
         'lit_ref': 'Long 1991; Sage & Pearcy 1987'},
        {'name': 'R_maint', 'key': 'R_maint',
         'lit': '1-5% of daily Pmax',
         'lit_ref': 'Penning de Vries 1975; Amthor 2000'},
    ]
    if p_seuil_key is not None:
        params_to_test.append({
            'name': 'P_stress_threshold (WWW chain)',  # CHANGE
            'key': p_seuil_key,
            'lit': 'N/A — phenomenological rate threshold, NOT C3 compensation point',
            'lit_ref': 'See params_real.json changelog 2026-10-24',
        })
    else:
        print("WARNING: no P_seuil / P_stress_threshold key found "
              "(checked aliases: " + ", ".join(P_SEUIL_ALIASES) + "). "
              "Skipping WWW-chain audit.")

    total_runs = 1 + 2 * len(params_to_test)

    log_run_header(
        "TECP V10.3-Audit — Porte 3: B.2 parameter sensitivity audit",
        config_lines=[
            f"K_CO2_REF       = {K_CO2_REF:.4e}",
            f"PMAX_FACTOR     = {PMAX_FACTOR}",
            f"RMAINT_DIV      = {RMAINT_DIV}",
            f"RQ              = {RQ}",
            f"CREW            = {CREW}",
            f"GAMMA           = {GAMMA}",
            f"B_ROOT_FRAC     = {B_ROOT_FRAC}",
            f"S_INIT          = {S_INIT:.4e}",
            f"M_INIT          = {M_INIT:.4f}",
            f"DURATION_DAYS   = {DURATION_DAYS}",
            f"RECORD_STEP     = {RECORD_STEP}",
            f"DT              = {cfg.DT:.0f} s",
            f"STROBE_STEPS    = {cfg.STROBE_STEPS}",
            f"Perturbation    : +/-20%",
            f"Total runs      : {total_runs}",
            f"Parameters      : {[p['name'] for p in params_to_test]}",
            f"Solver          = DefaultSolver(tolerance=1e-12, iterations=5000)",
        ],
        extra_lines=[
            "Sensitivity metric: S = 2.5 * (Y_+20 - Y_-20) / Y_ref",
            "Verdicts: |S|<0.5 robust, 0.5-1.0 acceptable, >=1.0 fragile",
            "Equilibrium: mean of last 24h (12 records at 2h intervals)",
            "NOTE: P_stress_threshold tests the WWW signalling chain, "
            "NOT the C3 CO2 compensation point.",
        ],
    )

    mesh = Grid3D(nx=cfg.NX, ny=cfg.NY, nz=cfg.NZ,
                  dx=cfg.DX, dy=cfg.DY, dz=cfg.DZ)
    geometry = {k: P_base[k] for k in [
        'x_split', 'z_ceil', 'z_sol_continental', 'z_plant_max_continental',
        'z_sol_marsh', 'z_plant_max_marsh']}
    masks = build_masks(mesh, geometry, extend_photo_to_plant_max=False)

    tight_solver = DefaultSolver(tolerance=1e-12, iterations=5000)

    os.makedirs("Porte3", exist_ok=True)
    t_global_start = time.time()

    # --- Reference ---
    print("\n" + "=" * 72)
    print("REFERENCE (factor = 1.0)")
    print("=" * 72)
    P_ref = apply_c2_with_perturbation(P_base, key=None)
    hist_ref = run_one("reference", P_ref, P_base, mesh, masks,
                       tight_solver, resume=False)
    eq_ref = equilibrium_from_hist(hist_ref)

    results = {}
    for param in params_to_test:
        key  = param['key']
        name = param['name']
        print("\n" + "=" * 72)
        print(f"PARAMETER: {name} (key='{key}')")
        print("=" * 72)

        label_m = f"{key}_minus20".replace(' ', '_')
        label_p = f"{key}_plus20".replace(' ', '_')

        P_m = apply_c2_with_perturbation(P_base, key=key, factor=0.8)
        P_p = apply_c2_with_perturbation(P_base, key=key, factor=1.2)

        hist_m = run_one(label_m, P_m, P_base, mesh, masks,
                         tight_solver, resume=False)
        eq_m = equilibrium_from_hist(hist_m)

        hist_p = run_one(label_p, P_p, P_base, mesh, masks,
                         tight_solver, resume=False)
        eq_p = equilibrium_from_hist(hist_p)

        results[name] = {
            'key': key,
            'lit': param['lit'],
            'lit_ref': param['lit_ref'],
            'value_ref': effective_value(P_base, key, 1.0),
            'value_m': effective_value(P_base, key, 0.8),
            'value_p': effective_value(P_base, key, 1.2),
            'eq_ref': eq_ref, 'eq_m': eq_m, 'eq_p': eq_p,
            'S_CO2': sensitivity(eq_p['CO2_eq'], eq_m['CO2_eq'],
                                 eq_ref['CO2_eq']),
            'S_O2':  sensitivity(eq_p['O2_eq'],  eq_m['O2_eq'],
                                 eq_ref['O2_eq']),
            'S_bio': sensitivity(eq_p['bio_eq'], eq_m['bio_eq'],
                                 eq_ref['bio_eq']),
        }

    print("\n" + "=" * 78)
    print("PORTE 3 — B.2 PARAMETER SENSITIVITY SUMMARY")
    print("=" * 78)
    print(f"{'parameter':<28s} {'value_ref':>11s} "
          f"{'CO2_ref':>11s} {'CO2_-20':>11s} {'CO2_+20':>11s} "
          f"{'S_CO2':>8s} {'verdict':>11s}")
    print("-" * 78)
    for name, r in results.items():
        S = r['S_CO2']
        v = verdict_from_S(S)
        print(f"{name:<28s} {r['value_ref']:>11.4e} "
              f"{r['eq_ref']['CO2_eq']:>11.4e} "
              f"{r['eq_m']['CO2_eq']:>11.4e} "
              f"{r['eq_p']['CO2_eq']:>11.4e} "
              f"{S:>+8.3f} {v:>11s}")
    print("-" * 78)

    print("\nSecondary sensitivities:")
    print(f"{'parameter':<28s} {'S_O2':>8s} {'verdict':>11s} "
          f"{'S_bio':>8s} {'verdict':>11s}")
    print("-" * 72)
    for name, r in results.items():
        print(f"{name:<28s} {r['S_O2']:>+8.3f} "
              f"{verdict_from_S(r['S_O2']):>11s} "
              f"{r['S_bio']:>+8.3f} "
              f"{verdict_from_S(r['S_bio']):>11s}")
    print("=" * 78)

    summary_path = "Porte3/sensitivity_summary.csv"
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'parameter', 'key', 'lit_range', 'lit_ref',
            'value_ref', 'value_minus20', 'value_plus20',
            'CO2_ref', 'CO2_minus20', 'CO2_plus20', 'S_CO2', 'verdict_CO2',
            'O2_ref',  'O2_minus20',  'O2_plus20',  'S_O2',  'verdict_O2',
            'bio_ref', 'bio_minus20', 'bio_plus20', 'S_bio', 'verdict_bio',
        ])
        for name, r in results.items():
            w.writerow([
                name, r['key'], r['lit'], r['lit_ref'],
                r['value_ref'], r['value_m'], r['value_p'],
                r['eq_ref']['CO2_eq'], r['eq_m']['CO2_eq'], r['eq_p']['CO2_eq'],
                r['S_CO2'], verdict_from_S(r['S_CO2']),
                r['eq_ref']['O2_eq'],  r['eq_m']['O2_eq'],  r['eq_p']['O2_eq'],
                r['S_O2'], verdict_from_S(r['S_O2']),
                r['eq_ref']['bio_eq'], r['eq_m']['bio_eq'], r['eq_p']['bio_eq'],
                r['S_bio'], verdict_from_S(r['S_bio']),
            ])

    n_fragile = sum(1 for r in results.values()
                    if verdict_from_S(r['S_CO2']) == 'fragile')
    overall = "PASS" if n_fragile == 0 else f"INVESTIGATE ({n_fragile} fragile)"

    print("\nRecommendations:")
    for name, r in results.items():
        v = verdict_from_S(r['S_CO2'])
        if v == 'fragile':
            print(f"  [FRAGILE] {name}: S_CO2 = {r['S_CO2']:+.3f} "
                  f"- investigate before manuscript")
        elif v == 'acceptable':
            print(f"  [ACCEPTABLE] {name}: S_CO2 = {r['S_CO2']:+.3f}")
        else:
            print(f"  [ROBUST] {name}: S_CO2 = {r['S_CO2']:+.3f}")

    total_elapsed = time.time() - t_global_start
    log_run_footer(
        elapsed_s=total_elapsed,
        verdicts={name: verdict_from_S(r['S_CO2'])
                  for name, r in results.items()},
        files=[
            "Porte3/reference.csv",
            "Porte3/sensitivity_summary.csv",
            *[f"Porte3/{r['key']}_minus20.csv" for r in results.values()],
            *[f"Porte3/{r['key']}_plus20.csv"  for r in results.values()],
        ],
        extra_lines=[
            f"Overall: {overall}",
            f"Fragile parameters: {n_fragile}",
        ],
    )


if __name__ == "__main__":
    main()