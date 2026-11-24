#!/usr/bin/env python3
"""Manuscript Figure 1 — C2 reference gas dynamics (CO2, O2), 2 panels.

Reads audit_timeseries.csv (produced by run_simulation.py at repo root)
and applies the Elsevier/LSSR style from plot_style.py.

Panels:
    (a) atmospheric CO2 volume fraction (30 d)
    (b) atmospheric O2 volume fraction  (30 d)

Usage:
    python3 plot_fig1.py
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
CO2 = df['CO2'].to_numpy()
O2  = df['O2'].to_numpy()

# --- Figure ---
fig, axes = plt.subplots(
    2, 1, figsize=(90 * MM, 130 * MM),
    sharex=True, constrained_layout=True,
)

# (a) CO2
ax = axes[0]
ax.plot(t_d, CO2, color=OKABE_ITO['vermillon'], lw=1.2)
ax.set_ylabel(r'CO$_2$ volume fraction')
ax.set_ylim(3.5e-4, 1.10e-3)

i_max = int(np.argmax(CO2))
ax.plot(t_d[i_max], CO2[i_max], marker='o', ms=3,
        color=OKABE_ITO['vermillon'], mfc='white', mew=0.8)
ax.text(
    t_d[i_max] + 1.2, CO2[i_max] + 0.5e-4,
    f'max = {CO2[i_max]:.3e}\n(t ≈ {t_h[i_max]:.0f} h)',
    fontsize=7, va='bottom', ha='left',
)
panel_label(ax, '(a)')

# (b) O2
ax = axes[1]
ax.plot(t_d, O2, color=OKABE_ITO['blue'], lw=1.2)
ax.set_ylabel(r'O$_2$ volume fraction')
ax.set_xlabel('Time (days)')
ax.set_ylim(0.2090, 0.2105)
panel_label(ax, '(b)')

# --- Save ---
save(fig, 'fig1', outdir=OUTDIR)