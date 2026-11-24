#!/usr/bin/env python3
"""run_spatial_convergence.py -- 12^3 vs 17^3 spatial convergence check.

Purpose
-------
Before freezing F2 (parameter sensitivity at 17^3), verify that the four
key runs (baseline, Pmax x0.80, Pmax x1.20, Y_B x0.80) give metrics that
do not depend significantly on spatial resolution.

Protocol
--------
* Mesh 12^3, dx = 0.85/12 m  (domain kept at 0.85 m, V_total preserved)
* DT = 360 s, 30 days (N_STEPS = 7200)
* Solver: DefaultSolver(tolerance=1e-12, iterations=5000)
* Append to spatial_convergence_12cubed.csv AFTER EACH run.
* Print [run N/4] label + metrics after each run.
* Compare vs 17^3 metrics from sensitivity_all.csv at the end.

Verdict thresholds
------------------
max |rel diff| < 2 %  -> PASS, 17^3 asymptotically converged
2 - 5 %               -> WARN, discuss before 25^3
> 5 %                 -> ALARM, do not freeze F2

Launch
------
    nohup python3 -u run_spatial_convergence.py > spatial_convergence/nohup.log 2>&1 &
"""

import os
import sys
import time
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

# --- Patch config BEFORE importing run_simulation ---
import config as cfg
cfg.NX = cfg.NY = cfg.NZ = 12
cfg.DX = cfg.DY = cfg.DZ = 0.85 / 12
cfg.DURATION_DAYS = 30
cfg.N_PERIODS = 30
cfg.N_STEPS = 30 * cfg.STROBE_STEPS      # = 7200
cfg.CREW_VALUES = [0.10]

from fipy import Grid3D
from fipy.solvers import DefaultSolver
from run_simulation import run_simulation
from build_masks import build_masks

# --- Metric extraction (inline copy of run_sensitivity.extract_metrics) ---
def extract_metrics(hist):
    t_h = np.asarray(hist['time_h'])
    CO2 = np.asarray(hist['CO2'])
    P   = np.asarray(hist['P_photo'])
    Rp  = np.asarray(hist['R_plant'])
    Rc  = np.asarray(hist['R_crew'])
    QPR = P / np.maximum(Rp + Rc, 1e-30)

    i_pk = int(np.argmax(CO2))
    t_peak = float(t_h[i_pk])
    co2_peak = float(CO2[i_pk])

    d = QPR - 1.0
    s = np.sign(d)
    idx = np.where(np.diff(s) != 0)[0]
    if len(idx):
        i0 = int(idx[0])
        t_cross = float(t_h[i0] + (-d[i0]) * (t_h[i0+1] - t_h[i0])
                                  / (d[i0+1] - d[i0]))
    else:
        t_cross = np.nan

    m72 = t_h >= t_h[-1] - 72.0
    return dict(
        t_co2_peak_h=t_peak,
        t_qpr_cross_h=t_cross,
        co2_max=co2_peak,
        co2_final=float(CO2[-1]),
        delta_B=float(hist['biomass'][-1] - hist['biomass'][0]),
        bio_init=float(hist['biomass'][0]),
        bio_final=float(hist['biomass'][-1]),
        T_final=float(hist['Temp'][-1]),
        qpr_72h_mean=float(np.mean(QPR[m72])),
    )

# --- Paths ---
OUT = 'spatial_convergence'
os.makedirs(OUT, exist_ok=True)
CSV_OUT = f'{OUT}/spatial_convergence_12cubed.csv'

# --- Load params ---
with open('params_real.json') as f:
    P_BASE = json.load(f)
GEOM = {k: P_BASE[k] for k in [
    'x_split', 'z_ceil', 'z_sol_continental', 'z_plant_max_continental',
    'z_sol_marsh', 'z_plant_max_marsh']}

# --- Build 12^3 mesh + masks + solver once ---
print(f"Building mesh 12^3 (dx={cfg.DX:.6f} m, domain={12*cfg.DX:.4f} m)",
      flush=True)
MESH = Grid3D(nx=12, ny=12, nz=12, dx=cfg.DX, dy=cfg.DY, dz=cfg.DZ)
MASKS = build_masks(MESH, GEOM)
SOLVER = DefaultSolver(tolerance=1e-12, iterations=5000)
print(f"V_total = {float(np.sum(MESH.cellVolumes)):.6e} m^3", flush=True)

# --- 4 variations (labels match sensitivity_all.csv) ---
BASE_PF = 2.0
BASE_RD = 1.0
VARIATIONS = [
    ('baseline',   'baseline', 1.0),
    ('Pmax_x0.80', 'pmax',     0.80),
    ('Pmax_x1.20', 'pmax',     1.20),
    ('Y_B_x0.80',  'yb',       0.80),
]

# --- CSV header ---
CSV_COLS = ['label', 'mesh', 'crew', 'pmax_factor', 'y_b',
            'co2_max', 't_co2_peak_h', 't_qpr_cross_h', 'delta_B',
            'bio_init', 'bio_final', 'delta_B_normalized',
            'T_final', 'qpr_72h_mean', 'duration_s', 'status']
if not os.path.exists(CSV_OUT):
    pd.DataFrame(columns=CSV_COLS).to_csv(CSV_OUT, index=False)

def append_row(row_dict):
    row = {k: row_dict.get(k, np.nan) for k in CSV_COLS}
    pd.DataFrame([row]).to_csv(CSV_OUT, mode='a', header=False, index=False)
    print(f"  -> row appended to {CSV_OUT}", flush=True)

# --- Main loop ---
print("=" * 72, flush=True)
print(f"SPATIAL CONVERGENCE 12^3 -- {len(VARIATIONS)} runs x 30 d", flush=True)
print("=" * 72, flush=True)

for i, (label, kind, pert) in enumerate(VARIATIONS, start=1):
    print(f"\n{'='*72}\n[run {i}/{len(VARIATIONS)}] {label}\n{'='*72}",
          flush=True)

    Pb = P_BASE.copy()
    pf, rd = BASE_PF, BASE_RD
    if kind == 'pmax':
        pf = BASE_PF * pert
    elif kind == 'yb':
        Pb['Y_B'] = P_BASE['Y_B'] * pert

    saved = (cfg.PMAX_FACTOR, cfg.RMAINT_DIV)
    try:
        cfg.PMAX_FACTOR = pf
        cfg.RMAINT_DIV = rd
        ts = time.strftime('%Y%m%d_%H%M%S')
        t0 = time.time()
        hist, out_dir = run_simulation(0.10, Pb, MESH, MASKS, SOLVER, ts)
        duration_s = time.time() - t0
        metrics = extract_metrics(hist)

        # Rename output dir to mark 12^3 + label
        new_dir = f'TrajA_C2_12cubed_30d_crew_0.100_{label}_{ts}'
        if os.path.exists(out_dir) and not os.path.exists(new_dir):
            shutil.move(out_dir, new_dir)
            print(f"  -> renamed {out_dir} -> {new_dir}", flush=True)

        dBN = (metrics['bio_final'] - metrics['bio_init']) / metrics['bio_init']
        append_row(dict(
            label=label, mesh='12^3', crew=0.10,
            pmax_factor=pf, y_b=Pb['Y_B'],
            co2_max=metrics['co2_max'],
            t_co2_peak_h=metrics['t_co2_peak_h'],
            t_qpr_cross_h=metrics['t_qpr_cross_h'],
            delta_B=metrics['delta_B'],
            bio_init=metrics['bio_init'],
            bio_final=metrics['bio_final'],
            delta_B_normalized=dBN,
            T_final=metrics['T_final'],
            qpr_72h_mean=metrics['qpr_72h_mean'],
            duration_s=duration_s, status='OK',
        ))
        print(f"\n[run {i}/{len(VARIATIONS)}] {label} DONE in "
              f"{duration_s/60:.1f} min", flush=True)
        print(f"  co2_max = {metrics['co2_max']:.6e}  "
              f"t_peak = {metrics['t_co2_peak_h']:.1f} h  "
              f"t_QPR = {metrics['t_qpr_cross_h']:.1f} h  "
              f"dB = {metrics['delta_B']:+.6e}", flush=True)
    except Exception as e:
        import traceback
        print(f"\n[run {i}/{len(VARIATIONS)}] {label} FAILED: {e}", flush=True)
        traceback.print_exc()
        append_row({'label': label, 'mesh': '12^3',
                    'status': f'FAILED: {e}'})
    finally:
        cfg.PMAX_FACTOR, cfg.RMAINT_DIV = saved

# --- Final comparison 12^3 vs 17^3 ---
print("\n" + "=" * 72, flush=True)
print("COMPARISON 12^3 vs 17^3", flush=True)
print("=" * 72, flush=True)

df12 = pd.read_csv(CSV_OUT)
df17 = pd.read_csv('sensitivity_all.csv')
df17 = df17[df17['label'].isin([v[0] for v in VARIATIONS])].copy()

METRICS = ['co2_max', 't_co2_peak_h', 't_qpr_cross_h', 'delta_B']
rows = []
for label in [v[0] for v in VARIATIONS]:
    r12 = df12[df12['label'] == label]
    r17 = df17[df17['label'] == label]
    if r12.empty or r17.empty:
        print(f"  WARN {label}: missing data", flush=True)
        continue
    r12, r17 = r12.iloc[-1], r17.iloc[0]
    for m in METRICS:
        v12, v17 = float(r12[m]), float(r17[m])
        rel = np.nan if abs(v17) < 1e-30 else 100.0 * (v12 - v17) / v17
        rows.append(dict(label=label, metric=m,
                         val_12cubed=v12, val_17cubed=v17,
                         rel_diff_pct=rel))

cmp_df = pd.DataFrame(rows)
cmp_df.to_csv(f'{OUT}/comparison_12cubed_vs_17cubed.csv', index=False)
piv = cmp_df.pivot_table(index='label', columns='metric',
                         values='rel_diff_pct')
print(piv.round(3).to_string(na_rep='NaN'), flush=True)

max_abs = float(np.nanmax(np.abs(piv.values)))
print("\n" + "-" * 72, flush=True)
if max_abs < 2.0:
    print(f"VERDICT : PASS -- max |diff| = {max_abs:.3f} % < 2 %", flush=True)
    print("  -> 17^3 asymptotically converged; F2 publishable.", flush=True)
elif max_abs < 5.0:
    print(f"VERDICT : WARN -- max |diff| = {max_abs:.3f} % (2-5 %)", flush=True)
    print("  -> Discuss; consider 25^3 before freezing F2.", flush=True)
else:
    print(f"VERDICT : ALARM -- max |diff| = {max_abs:.3f} % > 5 %", flush=True)
    print("  -> Do NOT freeze F2; investigate.", flush=True)
print("=" * 72, flush=True)