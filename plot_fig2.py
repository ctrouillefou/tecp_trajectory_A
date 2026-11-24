#!/usr/bin/env python3
"""Manuscript Figure 2 — C2 reference (T, B, S, M), 4 panels.

Reads audit_timeseries.csv (produced by run_simulation.py at repo root)
and applies the Elsevier/LSSR style from plot_style.py.

Panels:
    (a) Temperature T
    (b) Biomass B  (with the t ≈ 46 h minimum marked — J_plant = 0)
    (c) Mycorrhizal signal S
    (d) Mycelium density M

NOTE — caption discrepancy:
    The manuscript caption (as of the version received) describes three
    panels (a, b, c with S and M on twin axes). This script produces
    FOUR panels. If you use the 4-panel version, update the Fig. 2
    caption so that (c) = S and (d) = M.

Usage:
    python3 plot_fig2.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_style import setup_style, panel_label, save, OKABE_ITO, MM

CSV_PATH = 'audit_timeseries.csv'
OUTDIR   = 'figures'

setup_style()

# --- Load ---
df = pd.read_csv(CSV_PATH)
t_h = df['time_h'].to_numpy()
t_d = t_h / 24.0
T   = df['Temp'].to_numpy()
B   = df['biomass'].to_numpy()
S   = df['S'].to_numpy()
M   = df['M'].to_numpy()

# --- Figure (2 × 2) ---
fig, axes = plt.subplots(
    2, 2, figsize=(190 * MM, 130 * MM),
    sharex=True, constrained_layout=True,
)
axes = axes.ravel()

# (a) Temperature
ax = axes[0]
ax.plot(t_d, T, color=OKABE_ITO['vermillon'], lw=1.2)
ax.set_ylabel('T (K)')
drift_K_per_h = (T[-1] - T[0]) / t_h[-1]
ax.text(
    0.04, 0.05,
    f'drift ≈ {drift_K_per_h:.2e} K/h\n'
    f'$T_0$ = {T[0]:.2f} K, $T_f$ = {T[-1]:.2f} K',
    transform=ax.transAxes, fontsize=7, va='bottom', ha='left',
)
panel_label(ax, '(a)')

# (b) Biomass
ax = axes[1]
ax.plot(t_d, B, color=OKABE_ITO['green'], lw=1.2)
ax.set_ylabel(r'Biomass $B$ (dimensionless)')
i_min = int(np.argmin(B))
ax.plot(t_d[i_min], B[i_min], marker='o', ms=3,
        color=OKABE_ITO['green'], mfc='white', mew=0.8)
ax.annotate(
    f'min $B$ = {B[i_min]:.5f}\nat t ≈ {t_h[i_min]:.0f} h\n'
    r'($J_\mathrm{plant} = 0$)',
    xy=(t_d[i_min], B[i_min]),
    xytext=(t_d[i_min] + 7.0, B[i_min] + 1.5e-4),
    fontsize=7, ha='left', va='bottom',
    arrowprops=dict(arrowstyle='-', lw=0.5, color='0.4'),
)
panel_label(ax, '(b)')

# (c) Signal S (plotted in units of 1e-5 for readability)
ax = axes[2]
ax.plot(t_d, S * 1e5, color=OKABE_ITO['purple'], lw=1.2)
ax.set_ylabel(r'$S$ ($\times 10^{-5}$)')
ax.set_xlabel('Time (days)')
panel_label(ax, '(c)')

# (d) Mycelium M
ax = axes[3]
ax.plot(t_d, M, color=OKABE_ITO['orange'], lw=1.2)
ax.set_ylabel(r'$M$ (dimensionless)')
ax.set_xlabel('Time (days)')
panel_label(ax, '(d)')

# --- Save ---
save(fig, 'fig2', outdir=OUTDIR)