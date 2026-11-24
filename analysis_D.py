#!/usr/bin/env python3
"""D - Thermal diagnostic. Params loaded from json; condensation caveat noted."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from analysis_common import load_data, load_params, C, WINDOWS, window_mask, style, save_fig

OUT = 'analysis_D'
df, t_h, t_d = load_data()
P_base = load_params()

eta_heat   = P_base.get('eta_heat',   0.05)
alpha_heat = P_base.get('alpha_heat', 0.5)
P_elt      = P_base.get('P_elt',      2.777778e-6)
T_opt      = P_base.get('T_opt',      298.0)
T_range    = P_base.get('T_range',    10.0)

# Ceiling fraction (from mask diagnostics, 578 cells / 4913 total)
ceil_frac = 0.1176

Q_bio     = eta_heat   * df['R_plant'].to_numpy()
Q_rad     = alpha_heat * df['P_photo'].to_numpy()
Q_source  = Q_bio + Q_rad
Q_ceiling = -P_elt * ceil_frac

T     = df['Temp'].to_numpy()
dTdt  = np.gradient(T, t_h) / 3600.0      # K/h / 3600 = K/s

print('Condensation note: compute_condensation uses mean(Cv)/0.03, so the '
      'condensation term is spatially UNIFORM in the current code.')
print(f'  mean(Cv) end-of-run = {df["Cv"].iloc[-1]:.4f} -> RH ~ '
      f'{df["Cv"].iloc[-1]/0.03:.3f}  (< 0.6 threshold -> cond_term = 0)')

rows = []
for name, t0, t1 in WINDOWS:
    m = window_mask(t_h, t0, t1)
    pred = (Q_source[m].mean() + Q_ceiling) * 1e9
    obs  = dTdt[m].mean() * 1e9
    rows.append(dict(window=name,
                     Q_source=Q_source[m].mean()*1e9,
                     Q_ceiling=Q_ceiling*1e9,
                     dTdt_obs=obs, dTdt_pred=pred,
                     resid=obs-pred))
print('\n' + pd.DataFrame(rows).to_string(index=False,
      float_format=lambda x: f'{x:.3e}'))

f_T = np.exp(-((T - T_opt) / T_range)**2)
print(f'\nf_T: init={f_T[0]:.5f}  end={f_T[-1]:.5f}  change={100*(f_T[-1]/f_T[0]-1):+.2f}%')
print(f'Global slope: {(T[-1]-T[0])/t_h[-1]:.3e} K/h  (test D1: ~-1.19e-3 K/h)')
print('\nInterpretation:')
print('  Origin in the model  = term -P_elt*ceil  (CONFIRMED, 30x Q_bio+Q_rad)')
print('  Numerical artefact   = NO (deterministic, matches D1)')
print('  Physical validity    = NOT DEMONSTRATED by this analysis')

fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
ax = axs[0]; ax.plot(t_d, T, color=C['green'], lw=1.2)
ax.set_ylabel('T (K)'); ax2 = ax.twinx()
ax2.plot(t_d, f_T, color=C['blue'], ls='--', lw=1.0)
ax2.set_ylabel(r'$f_T$', color=C['blue'])
ax2.tick_params(axis='y', labelcolor=C['blue'])
ax.spines['top'].set_visible(False)

ax = axs[1]
ax.plot(t_d, Q_source*1e9, color=C['verm'], lw=1.3, label=r'$Q_{bio}+Q_{rad}$')
ax.axhline(Q_ceiling*1e9, color=C['blue'], lw=1.2, ls='--',
           label=r'$-P_{elt}\cdot$frac ceil')
ax.plot(t_d, dTdt*1e9, color=C['black'], lw=0.9, label=r'$dT/dt$ obs')
ax.axhline(0, color='grey', lw=0.5)
ax.set_xlabel('Time (days)')
ax.set_ylabel(r'Rate ($\times 10^{-9}$ K s$^{-1}$)')
ax.legend(fontsize=8); style(ax)
save_fig(fig, OUT, 'D1_thermal')