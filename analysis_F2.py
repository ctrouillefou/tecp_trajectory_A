#!/usr/bin/env python3

"""F2 — Normalized parameter sensitivity from sensitivity_all.csv.

    S_p = (ΔX / X_baseline) / (Δp / p_baseline)

Interpretation:
    |S_p| > 1    → amplified response
    |S_p| ≈ 1    → proportional response
    |S_p| < 0.3  → negligible response

Robustness notes:
  * qpr_72h_mean is replaced by (qpr_72h_mean − 1). The threshold of
    interest is Q_PR = 1, so we measure the deviation from that
    threshold rather than from 1.0 itself.
  * t_qpr_cross_h may be NaN if Q_PR never crosses 1. These cases are
    reported explicitly, and the corresponding S_p is NaN.
  * delta_B can be near zero or change sign; if |ΔB_baseline| < 1e-5
    the normalized sensitivity is flagged as NaN to avoid spurious
    amplification. For a proper biomass log-ratio sensitivity, add
    bio_init + bio_final to run_sensitivity.py's extract_metrics()
    (see note at bottom).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis_common import C, style, save_fig   # C, style unused here but harmless

OUT = 'analysis_F'
os.makedirs(OUT, exist_ok=True)
DELTA  = 0.20
CSV_IN = 'sensitivity_all.csv'

# ======================================================================
# Load & sanity
# ======================================================================
if not os.path.exists(CSV_IN):
    raise SystemExit(f'Missing {CSV_IN} — run run_sensitivity.py first')

df = pd.read_csv(CSV_IN)

# Exactly one baseline
n_base = (df['label'] == 'baseline').sum()
assert n_base == 1, f'Expected 1 baseline, got {n_base}'
n_runs = len(df)
print(f'  Loaded {n_runs} runs (1 baseline + {n_runs-1} variations)')



# NaN report on Q_PR crossing
nan_count = int(df['t_qpr_cross_h'].isna().sum())
if nan_count > 0:
    labels_nan = df.loc[df['t_qpr_cross_h'].isna(), 'label'].tolist()
    print(f'\n  ⚠ {nan_count} run(s) have NaN t_qpr_cross_h '
          f'(Q_PR never crossed 1): {labels_nan}')
    print('    → these S_p values will be reported as NaN')

# ======================================================================
# Derived / robust metrics
# ======================================================================
df['qpr_72h_minus1'] = df['qpr_72h_mean'] - 1.0
base = df[df['label'] == 'baseline'].iloc[0]
base_qpr_m1 = float(base['qpr_72h_minus1'])

# Guard on delta_B: if |ΔB_baseline| is tiny, S_p is meaningless
DELTA_B_MIN = 1e-5
delta_B_too_small = abs(float(base['delta_B'])) < DELTA_B_MIN
if delta_B_too_small:
    print(f'\n  ⚠ |delta_B_baseline| = {float(base["delta_B"]):.2e} '
          f'< {DELTA_B_MIN:.0e}; S_p on delta_B will be NaN.')
    print('    For a robust biomass sensitivity, add bio_init + bio_final '
          'to run_sensitivity.py and use log(B_final/B_init).')

if abs(base_qpr_m1) < 1e-4:
    print(f'\n  ⚠ |qpr_72h_minus1_baseline| = {base_qpr_m1:.2e} '
          f'is very small; S_p on qpr_72h_minus1 may be numerically noisy.')

# ======================================================================
# Build sensitivity table
# ======================================================================
OUTPUTS = ['t_co2_peak_h', 't_qpr_cross_h', 'co2_max', 'co2_final',
           'delta_B','log_biomass_ratio', 'T_final', 'qpr_72h_minus1']
PARAMS  = ['Pmax', 'R_maint', 'k_CO2', 'R_crew', 'Y_B']

rows = []
for param in PARAMS:
    for sign, tag in [(-1, f'{1 - DELTA:.2f}'), (+1, f'{1 + DELTA:.2f}')]:
        label = f'{param}_x{tag}'
        row = df[df['label'] == label]
        if row.empty:
            print(f'  ⚠ missing run: {label}')
            continue
        row = row.iloc[0]
        for v in OUTPUTS:
            if (v == 'log_biomass_ratio'
                    and 'bio_init' in df.columns
                    and 'bio_final' in df.columns
                    and base['bio_init'] > 0 and row['bio_init'] > 0):
                x0 = float(np.log(base['bio_final'] / base['bio_init']))
                x1 = float(np.log(row['bio_final'] / row['bio_init']))
            else:
                x0 = float(base[v])
                x1 = float(row[v])

            if np.isnan(x0) or np.isnan(x1):
                S = np.nan
            elif abs(x0) < 1e-30:
                S = np.nan
            else:
                S = ((x1 - x0) / x0) / (sign * DELTA)

            rows.append(dict(
                param=param,
                pert=sign * DELTA,
                variable=v,
                S=S,
                x_baseline=x0,
                x_pert=x1,
                x_rel_change=((x1 - x0) / x0) if abs(x0) > 1e-30 else np.nan,
            ))

sens = pd.DataFrame(rows)
sens.to_csv(f'{OUT}/F2_sensitivity.csv', index=False)

# ======================================================================
# Console pivot
# ======================================================================
piv = sens.pivot_table(index='param', columns='variable', values='S')
piv = piv.reindex(index=PARAMS, columns=OUTPUTS)

print('\n' + '=' * 72)
print('F2 — Normalized sensitivity S_p = (ΔX/X_baseline) / (Δp/p_baseline)')
print('     ±20 % perturbations, C2 baseline')
print('=' * 72)
print(piv.round(3).to_string(na_rep='NaN'))

# ======================================================================
# Heatmap
# ======================================================================
fig, ax = plt.subplots(figsize=(9, 3.6))
data = piv.values.astype(float)
im = ax.imshow(data, aspect='auto', cmap='RdBu_r',
               vmin=-2, vmax=2,
               extent=[-0.5, len(OUTPUTS) - 0.5, len(PARAMS) - 0.5, -0.5])

ax.set_xticks(range(len(OUTPUTS)))
ax.set_xticklabels(OUTPUTS, rotation=25, ha='right', fontsize=8)
ax.set_yticks(range(len(PARAMS)))
ax.set_yticklabels(PARAMS, fontsize=9)

for i in range(len(PARAMS)):
    for j in range(len(OUTPUTS)):
        val = data[i, j]
        if np.isnan(val):
            ax.text(j, i, 'NaN', ha='center', va='center',
                    fontsize=7, color='grey')
        else:
            color = 'white' if abs(val) > 1.2 else 'black'
            ax.text(j, i, f'{val:+.2f}', ha='center', va='center',
                    fontsize=7, color=color)

plt.colorbar(im, ax=ax, label='S_p', shrink=0.85)
ax.set_title('F2 — Parameter sensitivity (C2 baseline, ±20 %)')
fig.tight_layout()
fig.savefig(f'{OUT}/F2_sensitivity_heatmap.png', dpi=180)
fig.savefig(f'{OUT}/F2_sensitivity_heatmap.pdf')
plt.close(fig)
print(f'\n  saved: {OUT}/F2_sensitivity_heatmap.png')

# ======================================================================
# Compact textual summary
# ======================================================================
print('\n' + '=' * 72)
print('SUMMARY — dominant effects (|S_p| > 0.3)')
print('=' * 72)
for param in PARAMS:
    strong = piv.loc[param].dropna()
    strong = strong[strong.abs() > 0.3]
    if len(strong) == 0:
        print(f'  {param:8s}: no significant effect (all |S_p| ≤ 0.3)')
    else:
        parts = [f'{v}={S:+.2f}' for v, S in strong.items()]
        print(f'  {param:8s}: ' + ', '.join(parts))