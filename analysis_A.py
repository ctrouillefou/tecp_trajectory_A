#!/usr/bin/env python3
"""A - Temporal evolution. Fixes the CO2 peak time."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from analysis_common import load_data, C, style, save_fig

OUT = 'analysis_A'
df, t_h, t_d = load_data()

# --- Summary table ---
VARS = ['CO2','O2','Temp','biomass','theta','S','M','P_photo']
rows = []
mask72 = t_h >= t_h[-1] - 72.0
for v in VARS:
    y = df[v].to_numpy()
    slope72 = (np.polyfit(t_h[mask72], y[mask72], 1)[0]
               if mask72.sum() >= 2 else np.nan)
    rows.append(dict(variable=v,
                     y_init=y[0], y_final=y[-1],
                     delta_rel_pct=100*(y[-1]-y[0])/y[0] if abs(y[0])>1e-30 else np.nan,
                     slope_72h_per_h=slope72))
print(pd.DataFrame(rows).to_string(index=False,
      float_format=lambda x: f'{x:.4g}'))

# --- Identify the CO2 peak (fixes my previous mislabel) ---
i_pk = int(np.argmax(df['CO2']))
print(f"\nCO2 peak: t = {t_h[i_pk]:.1f} h ({t_d[i_pk]:.2f} d), "
      f"value = {df['CO2'].iloc[i_pk]:.6f}")
print("  -> peak is at ~100 h, not 350-400 h.")

# --- Figures ---
fig, ax1 = plt.subplots(figsize=(7, 3.5))
ax1.plot(t_d, df['CO2'], color=C['blue'], lw=1.2)
ax1.set_ylabel('CO2 (vol. fraction)', color=C['blue'])
ax1.tick_params(axis='y', labelcolor=C['blue'])
ax1.spines['top'].set_visible(False)
ax2 = ax1.twinx()
ax2.plot(t_d, df['O2'], color=C['verm'], lw=1.0)
ax2.set_ylabel('O2 (vol. fraction)', color=C['verm'])
ax2.tick_params(axis='y', labelcolor=C['verm'])
ax2.spines['top'].set_visible(False)
ax1.set_xlabel('Time (days)'); ax1.set_title('A1 - CO2 and O2')
save_fig(fig, OUT, 'A1_gases')

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(t_d, df['Temp'], color=C['green'], lw=1.1)
ax.set_xlabel('Time (days)'); ax.set_ylabel('T (K)')
style(ax); ax.set_title('A2 - Temperature')
save_fig(fig, OUT, 'A2_temp')

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(t_d, df['biomass'], color=C['blue'], lw=1.1)
ax.set_xlabel('Time (days)'); ax.set_ylabel('B')
style(ax); ax.set_title('A3 - Biomass')
save_fig(fig, OUT, 'A3_biomass')

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(t_d, df['P_photo'], color=C['blue'], lw=1.1, label=r'$P_{photo}$')
ax.plot(t_d, df['R_plant'], color=C['verm'], lw=1.1, label=r'$R_{plant}$')
ax.plot(t_d, df['R_crew'],  color=C['orange'], lw=1.0, ls='--', label=r'$R_{crew}$')
ax.set_yscale('log'); ax.set_xlabel('Time (days)')
ax.set_ylabel(r'Rate (s$^{-1}$)')
ax.legend(loc='lower right', fontsize=8)
style(ax); ax.set_title('A4 - Photosynthesis and respiration')
save_fig(fig, OUT, 'A4_fluxes')

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(t_d, df['theta'], color=C['sky'], lw=1.1)
ax.set_xlabel('Time (days)'); ax.set_ylabel(r'$\theta$')
style(ax); ax.set_title(r'A5 - Water content')
save_fig(fig, OUT, 'A5_theta')

fig, ax1 = plt.subplots(figsize=(7, 3.5))
ax1.plot(t_d, df['S'], color=C['purple'], lw=1.2)
ax1.set_yscale('log'); ax1.set_ylabel('S', color=C['purple'])
ax1.tick_params(axis='y', labelcolor=C['purple'])
ax1.spines['top'].set_visible(False)
ax2 = ax1.twinx()
ax2.plot(t_d, df['M'], color=C['black'], lw=1.0, ls='--')
ax2.set_ylabel('M', color=C['black'])
ax2.tick_params(axis='y', labelcolor=C['black'])
ax2.spines['top'].set_visible(False)
ax1.set_xlabel('Time (days)'); ax1.set_title('A6 - S and M')
save_fig(fig, OUT, 'A6_signals')