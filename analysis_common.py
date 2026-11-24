#!/usr/bin/env python3
"""Shared infrastructure for TECP analyses A-E."""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# NumPy compatibility: np.trapz deprecated in 2.0, removed in 2.4
try:
    _trapz = np.trapezoid          # NumPy >= 2.0
except AttributeError:
    _trapz = np.trapz              # NumPy < 2.0

CSV_PATH = 'audit_timeseries.csv'
PARAMS_PATH = 'params_real.json'

# Standard analysis windows (hours)
WINDOWS = [
    ('early (0-72 h)',     0.0,   72.0),
    ('middle (324-396 h)', 324.0, 396.0),
    ('late (646-718 h)',   646.0, 718.0),
]

# Okabe-Ito palette (single source of truth)
C = dict(
    blue='#0072B2', verm='#D55E00', green='#009E73',
    sky='#56B4E9', yellow='#F0E442', purple='#CC79A7',
    orange='#E69F00', black='#000000',
)

# --------------------------------------------------------------- loaders
def load_data(path=CSV_PATH):
    """Return (df, t_h, t_d)."""
    df = pd.read_csv(path)
    t_h = df['time_h'].to_numpy()
    return df, t_h, t_h / 24.0


def load_params(path=PARAMS_PATH):
    """Return base parameters dict (or {} if absent)."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

# --------------------------------------------------------------- numerics
def trapz_s(y, t_h):
    """Time-integral of a rate y (s^-1) over t_h (hours) -> dimensionless.

    Works on NumPy >= 2.0 (np.trapezoid) and < 2.0 (np.trapz).
    Uses keyword `x=` so the axis is unambiguous.
    """
    return float(_trapz(np.asarray(y), x=np.asarray(t_h)) * 3600.0)

def window_mask(t_h, t0, t1):
    return (t_h >= t0) & (t_h <= t1)


def window_stats(y, t_h, t0, t1):
    """Mean / terminal slope (per hour) / time-integral for a window."""
    y = np.asarray(y); t_h = np.asarray(t_h)
    m = window_mask(t_h, t0, t1)
    if int(m.sum()) < 2:
        return dict(mean=np.nan, slope_per_h=np.nan,
                    integral_s=np.nan, n=int(m.sum()))
    yw, tw = y[m], t_h[m]
    return dict(
        mean         = float(np.mean(yw)),
        slope_per_h  = float(np.polyfit(tw, yw, 1)[0]),
        integral_s   = trapz_s(yw, tw),
        n            = int(m.sum()),
    )

# --------------------------------------------------------------- physics
def H_S(S, Km=0.01, n=2):
    """Hill activation H(S) = S^n / (Km^n + S^n + eps)."""
    S = np.asarray(S, dtype=float)
    return S**n / (Km**n + S**n + 1e-8)


def Q_PR(P_photo, R_plant, R_crew):
    """Q_PR = P_photo / (R_plant + R_crew)."""
    den = np.asarray(R_plant) + np.asarray(R_crew)
    return np.asarray(P_photo) / np.maximum(den, 1e-30)


def Q_PoverR(P_photo, R_plant):
    return np.asarray(P_photo) / np.maximum(np.asarray(R_plant), 1e-30)

# --------------------------------------------------------------- plotting
def style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def save_fig(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f'{outdir}/{name}.png', dpi=180)
    fig.savefig(f'{outdir}/{name}.pdf')
    plt.close(fig)
    print(f'  saved: {outdir}/{name}.png')