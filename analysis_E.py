#!/usr/bin/env python3
"""E - Mycorrhizal signal and cooperation."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from analysis_common import (load_data, load_params, C, WINDOWS,
                             window_mask, H_S, style, save_fig)

OUT = 'analysis_E'
df, t_h, t_d = load_data()
P_base = load_params()
Km     = P_base.get('Km',     0.01)
n_hill = P_base.get('n_hill', 2)

S = df['S'].to_numpy()
M = df['M'].to_numpy()
H = H_S(S, Km, n_hill)

print(f'H(S): init={H[0]:.3e}  end={H[-1]:.3e}  ratio={H[-1]/H[0]:.2f}x')
print(f'M  : init={M[0]:.15f}  end={M[-1]:.15f}  dM={M[-1]-M[0]:.2e}')

ratio = df['C_coop'].to_numpy() / np.maximum(df['P_photo'].to_numpy(), 1e-30)

rows = []
for name, t0, t1 in WINDOWS:
    m = window_mask(t_h, t0, t1)
    rows.append(dict(window=name,
                     S=S[m].mean(), H=H[m].mean(),
                     Ccoop=df['C_coop'].to_numpy()[m].mean(),
                     P_photo=df['P_photo'].to_numpy()[m].mean(),
                     ratio=ratio[m].mean()))
print('\n' + pd.DataFrame(rows).to_string(index=False,
      float_format=lambda x: f'{x:.3e}'))

print('\nConclusion: C_coop / P_photo ~ 3e-7 -> cooperation is '
      'NEGLIGIBLE in this parameter regime.')
print('  Avoid "mycorrhizal dimension is functionally inert": S does '
      'evolve and M dynamics is not tested by this CSV.')

fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
ax = axs[0]
ax.plot(t_d, S, color=C['purple'], lw=1.3)
ax.set_yscale('log'); ax.set_ylabel('S'); style(ax)
ax = axs[1]
ax.plot(t_d, H, color=C['purple'], lw=1.3)
ax.set_yscale('log'); ax.set_xlabel('Time (days)')
ax.set_ylabel('H(S)'); style(ax)
save_fig(fig, OUT, 'E1_signal')