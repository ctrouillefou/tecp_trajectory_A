#!/usr/bin/env python3
"""F - Robustness: convergence dt + sensitivity (reads existing data).

F1: dt = 360 s vs 180 s (Porte2/REF_DT360.csv, Porte2/FIN_DT180.csv)
F2/F3/F4: placeholders — skipped if data files absent, with clear warnings.

Never triggers a new simulation: only loads existing CSVs and produces
a summary table + comparison figure.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis_common import C, style, save_fig, _trapz

OUT = 'analysis_F'
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------
# F1 — Temporal convergence (dt = 360 s vs 180 s)
# ---------------------------------------------------------------
def load_csv_cols(path, cols):
    df = pd.read_csv(path)
    return {c: df[c].to_numpy() for c in cols}


def compare_dt(path_ref, path_fin, vars_to_check, out):
    """Interpolate FIN onto REF time-base, compute max relative difference."""
    d_ref = load_csv_cols(path_ref, ['time_h'] + vars_to_check)
    d_fin = load_csv_cols(path_fin, ['time_h'] + vars_to_check)
    t_ref = d_ref['time_h']
    t_fin = d_fin['time_h']

    rows = []
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for v in vars_to_check:
        y_ref = d_ref[v]
        y_fin = np.interp(t_ref, t_fin, d_fin[v])
        scale = np.maximum(np.abs(y_ref), 1e-30)
        rel = 100.0 * np.abs(y_ref - y_fin) / scale
        i_pk = int(np.argmax(rel))
        rows.append(dict(variable=v,
                         max_rel_pct=float(rel[i_pk]),
                         t_of_max_h=float(t_ref[i_pk]),
                         mean_rel_pct=float(np.mean(rel))))
        ax.plot(t_ref / 24.0, rel, lw=1.0, label=v)

    ax.set_xlabel('Time (days)')
    ax.set_ylabel('|REF − FIN| / REF (%)')
    ax.set_title('F1 — dt convergence (360 s vs 180 s)')
    ax.legend(fontsize=7, ncol=2)
    style(ax)
    save_fig(fig, out, 'F1_dt_convergence')

    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# F2 — Parameter sensitivity (placeholder, reads *_summary.csv)
# ---------------------------------------------------------------
def collect_sensitivity(pattern='sensitivity_*_summary.csv'):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f'  [F2] no sensitivity data found ({pattern}) — skipped')
        return None
    rows = []
    for f in files:
        df = pd.read_csv(f)
        # Convention assumed: columns = ['param', 'value', 'CO2_final', 'biomass_final', 'P_photo_72h']
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------
# F3 — Solver tolerance (placeholder)
# ---------------------------------------------------------------
def check_tolerance_data(pattern='tolerance_*.csv'):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f'  [F3] no tolerance data found ({pattern}) — skipped')
        return None
    # If files exist, could compare columns as in compare_dt
    return files


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    print('=' * 72)
    print('ANALYSIS F — ROBUSTNESS (read-only, no simulation)')
    print('=' * 72)

    # F1 — dt convergence
    ref = 'Porte2/REF_DT360.csv'
    fin = 'Porte2/FIN_DT180.csv'
    if os.path.exists(ref) and os.path.exists(fin):
        df1 = compare_dt(ref, fin,
                         ['CO2', 'O2', 'Temp', 'bio', 'S'],
                         OUT)
        df1.to_csv(f'{OUT}/F1_dt_summary.csv', index=False)
        print('\n--- F1 — dt convergence summary ---')
        print(df1.to_string(index=False, float_format=lambda x: f'{x:.4g}'))
        print('\n   PASS if all max_rel_pct < 5 % ; watch S (was 2.87 % at 10 d).')
    else:
        print(f'  [F1] {ref} or {fin} missing — cannot assess dt convergence.')

    # F2 — sensitivity
    df2 = collect_sensitivity()
    if df2 is not None:
        df2.to_csv(f'{OUT}/F2_sensitivity_consolidated.csv', index=False)
        print('\n--- F2 — sensitivity summary ---')
        print(df2.to_string(index=False))

    # F3 — tolerance
    files3 = check_tolerance_data()
    if files3:
        print(f'\n  [F3] tolerance files: {len(files3)} found (aggregation TODO)')

    # F4 — spatial convergence: typically needs new run, do nothing
    print('\n  [F4] spatial convergence: requires a dedicated finer-mesh run.')

    print('\n' + '=' * 72)
    print('F complete. Figures + CSVs in:', OUT)
    print('=' * 72)


if __name__ == '__main__':
    main()