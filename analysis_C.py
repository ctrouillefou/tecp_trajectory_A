#!/usr/bin/env python3
"""C - Biomass decomposition. Notes the code-identity and time-offset caveats."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from analysis_common import load_data, C, WINDOWS, window_mask, trapz_s, style, save_fig

OUT = 'analysis_C'
df, t_h, t_d = load_data()

YB      = 1.0
J_photo =  YB * df['P_photo'].to_numpy()
J_resp  = -df['R_plant'].to_numpy()
J_coop  =  df['C_coop'].to_numpy()
J_total = J_photo + J_resp + J_coop

err = np.max(np.abs(df['net_biomass'].to_numpy() - J_total))
print(f'Check: max|net_biomass - (YB*P - R + Ccoop)| = {err:.2e}')
print('  NB: this is a CODE IDENTITY, not independent validation.')

print('\n  NB: sources (P_photo, R_plant, C_coop) are recorded from the '
      'pre-solve state; biomass is recorded post-solve (one step later). '
      'Time-integral is still accurate to O(dt) over 30 days.')

f_resp = -J_resp / np.maximum(J_photo, 1e-30)
f_coop = J_coop / np.maximum(J_photo, 1e-30)

rows = []
for name, t0, t1 in WINDOWS:
    m = window_mask(t_h, t0, t1)
    rows.append(dict(window=name,
                     Jphoto=J_photo[m].mean(),
                     Jresp=J_resp[m].mean(),
                     Jcoop=J_coop[m].mean(),
                     Jtot=J_total[m].mean(),
                     f_resp=f_resp[m].mean(),
                     f_coop=f_coop[m].mean()))
print('\n' + pd.DataFrame(rows).to_string(index=False,
      float_format=lambda x: f'{x:.3e}'))

I_photo = trapz_s(J_photo, t_h)
I_resp  = trapz_s(J_resp,  t_h)
I_coop  = trapz_s(J_coop,  t_h)
I_total = I_photo + I_resp + I_coop
dB      = df['biomass'].iloc[-1] - df['biomass'].iloc[0]
print(f'\nIntegrals over 0-{t_h[-1]:.0f} h:')
print(f'  I_photo = {I_photo:+.4e}')
print(f'  I_resp  = {I_resp:+.4e}')
print(f'  I_coop  = {I_coop:+.4e}')
print(f'  I_total = {I_total:+.4e}')
print(f'  dB obs  = {dB:+.4e}')
print(f'  ratio   = {I_total/dB:.3f}')

imin = int(np.argmin(df['biomass'].to_numpy()))
print(f'\nBiomass minimum: t={t_h[imin]:.1f} h, B={df["biomass"].iloc[imin]:.6f}')

fig, axs = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
ax = axs[0]
ax.plot(t_d, J_photo*1e9, color=C['blue'], lw=1.3, label=r'$Y_B P_{photo}$')
ax.plot(t_d, J_resp*1e9,  color=C['verm'], lw=1.3, label=r'$-R_{plant}$')
ax.plot(t_d, J_total*1e9, color=C['black'], lw=1.0, ls='--', label=r'$J_B$')
ax.axhline(0, color='grey', lw=0.5)
ax.set_ylabel(r'Rate ($\times 10^{-9}$ s$^{-1}$)')
ax.legend(fontsize=8); style(ax)

ax = axs[1]
ax.plot(t_d, f_resp, color=C['verm'], lw=1.3, label=r'$f_{resp}$')
ax.axhline(1.0, color=C['black'], lw=0.5, alpha=0.5, label=r'seuil $1/Y_B$')
ax.set_xlabel('Time (days)'); ax.set_ylabel('Ratio')
ax.legend(fontsize=8); style(ax)

save_fig(fig, OUT, 'C1_biomass')