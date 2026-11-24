#!/usr/bin/env python3
"""Sensitivity sweep: 5 params × ±20 % around the C2 baseline.

Each run is a fresh 30-day simulation. Config is patched in memory
(not on disk), restored after each call. Outputs:
    sensitivity_all.csv   (one row per run, 7 metrics + metadata)
    TrajA_C2_30d_crew_*/  (one directory per run, as usual)
"""

import time
import json
import numpy as np
import pandas as pd
from fipy import Grid3D
from fipy.solvers import DefaultSolver
import matplotlib
matplotlib.use('Agg')

import config as cfg
from run_simulation import run_simulation
from build_masks import build_masks

# --------------------------------------------------------------------------
# Shared mesh / masks / solver (built once, reused across all runs)
# --------------------------------------------------------------------------
with open('params_real.json') as f:
    P_BASE = json.load(f)

MESH = Grid3D(nx=cfg.NX, ny=cfg.NY, nz=cfg.NZ,
              dx=cfg.DX, dy=cfg.DY, dz=cfg.DZ)
GEOM = {k: P_BASE[k] for k in [
    'x_split', 'z_ceil', 'z_sol_continental', 'z_plant_max_continental',
    'z_sol_marsh', 'z_plant_max_marsh']}
MASKS  = build_masks(MESH, GEOM)
SOLVER = DefaultSolver(tolerance=1e-12, iterations=5000)


# --------------------------------------------------------------------------
# Metric extraction (the 7 metrics requested by the protocol)
# --------------------------------------------------------------------------
def extract_metrics(hist):
    t_h  = np.asarray(hist['time_h'])
    CO2  = np.asarray(hist['CO2'])
    P    = np.asarray(hist['P_photo'])
    Rp   = np.asarray(hist['R_plant'])
    Rc   = np.asarray(hist['R_crew'])
    QPR  = P / np.maximum(Rp + Rc, 1e-30)

    i_pk     = int(np.argmax(CO2))
    t_peak   = float(t_h[i_pk])
    co2_peak = float(CO2[i_pk])

    # First crossing Q_PR = 1 (linear interpolation)
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
        t_co2_peak_h   = t_peak,
        t_qpr_cross_h  = t_cross,
        co2_max        = co2_peak,
        co2_final      = float(CO2[-1]),
        delta_B        = float(hist['biomass'][-1] - hist['biomass'][0]),
        bio_init       = float(hist['biomass'][0]),
        bio_final      = float(hist['biomass'][-1]),
        T_final        = float(hist['Temp'][-1]),
        qpr_72h_mean   = float(np.mean(QPR[m72])),
        qpr_72h_minus1 = qpr_72h_mean - 1.0
    )


# --------------------------------------------------------------------------
# One run, with cfg patching and cleanup
# --------------------------------------------------------------------------
def run_one(label, crew_mult, Pb, pf, rd):
    """Patch cfg, run, restore. Returns metrics dict or None on failure."""
    saved = (cfg.PMAX_FACTOR, cfg.RMAINT_DIV, cfg.K_CO2_EXPECTED, cfg.CREW_VALUES)
    try:
        cfg.PMAX_FACTOR   = pf
        cfg.RMAINT_DIV    = rd
        cfg.K_CO2_EXPECTED = Pb['k_CO2']   # skip k_CO2 validation
        cfg.CREW_VALUES    = [crew_mult]
        ts = time.strftime('%Y%m%d_%H%M%S')
        Pmax_eff  = Pb['Pmax'] * pf
        Rmaint_eff = Pb['R_maint'] / rd
        print(f'  [effective] Pmax={Pmax_eff:.4e}, R_maint={Rmaint_eff:.4e}, '
              f'k_CO2={Pb["k_CO2"]:.4e}, Y_B={Pb["Y_B"]:.4e}, '
              f'O2_cons={P_BASE["O2_cons"]*crew_mult:.3e}, '
              f'CO2_prod={cfg.RQ*P_BASE["O2_cons"]*crew_mult:.3e}')
        hist, _ = run_simulation(crew_mult, Pb, MESH, MASKS, SOLVER, ts)
        m = extract_metrics(hist)
        m.update(label=label, crew=crew_mult, pmax_factor=pf,
                 rmaint_div=rd, k_co2=Pb['k_CO2'], y_b=Pb['Y_B'])
        return m
    except Exception as e:
        print(f'  ✗ {label}: {e}')
        return None
    finally:
        cfg.PMAX_FACTOR, cfg.RMAINT_DIV, cfg.K_CO2_EXPECTED, cfg.CREW_VALUES = saved


# --------------------------------------------------------------------------
# Main sweep
# --------------------------------------------------------------------------
def main():
    delta = 0.20
    BASE_CREW = 0.10
    BASE_PF   = cfg.PMAX_FACTOR
    BASE_RD   = cfg.RMAINT_DIV

    # (label, kind, perturbation)
    variations = []
    for pert in (1.0 - delta, 1.0 + delta):
        variations += [
            (f'Pmax_x{pert:.2f}',    'pmax',   pert),
            (f'R_maint_x{pert:.2f}', 'rmaint', pert),
            (f'k_CO2_x{pert:.2f}',   'kco2',   pert),
            (f'R_crew_x{pert:.2f}',  'crew',   pert),
            (f'Y_B_x{pert:.2f}',     'yb',     pert),
        ]
    # + baseline for self-contained comparison
    variations.append(('baseline', 'baseline', 1.0))

    results = []
    for i, (label, kind, pert) in enumerate(variations, start=1):
        print(f'\n{"="*72}\n  [{i}/{len(variations)}]  {label}\n{"="*72}')
        Pb   = P_BASE.copy()
        pf, rd, crew = BASE_PF, BASE_RD, BASE_CREW

        if   kind == 'pmax'   : pf = BASE_PF * pert
        elif kind == 'rmaint' : rd = BASE_RD / pert          # R_maint = base/div
        elif kind == 'kco2'   : Pb['k_CO2'] = P_BASE['k_CO2'] * pert
        elif kind == 'crew'   : crew = BASE_CREW * pert
        elif kind == 'yb'     : Pb['Y_B'] = P_BASE['Y_B'] * pert

        m = run_one(label, crew, Pb, pf, rd)
        if m is not None:
            results.append(m)
            print(f'  → peak @ {m["t_co2_peak_h"]:5.1f} h, '
                  f'Q_PR@ {m["t_qpr_cross_h"]:5.1f} h, '
                  f'CO2_max = {m["co2_max"]:.4e}')

    df = pd.DataFrame(results)
    df.to_csv('sensitivity_all.csv', index=False)
    print('\n' + '=' * 72)
    print('SWEEP COMPLETE — saved sensitivity_all.csv')
    print('=' * 72)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4g}'))


if __name__ == '__main__':
    main()