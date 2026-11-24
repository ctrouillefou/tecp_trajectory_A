#!/usr/bin/env python3
# =====================================================================
# check_csvs.py
#
# Empirical verification of DS10's claims (I2, I3, C1) from CSV files.
#
# Checks:
#   (a) S(t) shape in Porte 2 REF vs FIN (oscillation vs trend)
#   (b) Equilibrium status at 3 days for all Porte 3 runs
#   (c) S(t) comparison in the three P_seuil runs (detect dead WWW chain)
#   (d) Recompute Porte 2 with corrected 24h windows (fix I2)
# =====================================================================

import numpy as np
import csv
from pathlib import Path


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def col(data, name):
    return np.array([float(r[name]) for r in data])


def last_n_mean(arr, n):
    return float(np.mean(arr[-n:]))


# --- (d) Porte 2: same physical time window ---
print("=" * 70)
print("(d) Porte 2 — Comparison over the same 24h interval")
print("=" * 70)

t_start, t_end = 694.0, 718.0

def window_mean(data, var, t_start, t_end):
    t = col(data, 'time_h')
    x = col(data, var)
    mask = (t >= t_start) & (t <= t_end)
    return float(np.mean(x[mask])), int(np.sum(mask))

ref = load_csv("Porte2/REF_DT360.csv")
fin = load_csv("Porte2/FIN_DT180.csv")

for var in ['CO2', 'O2', 'bio', 'S']:
    eq_ref, n_ref = window_mean(ref, var, t_start, t_end)
    eq_fin, n_fin = window_mean(fin, var, t_start, t_end)

    rel = abs(eq_fin - eq_ref) / max(
        abs(eq_ref), abs(eq_fin), 1e-30
    )

    print(
        f"{var:4s}: REF={eq_ref:.6e} (n={n_ref})  "
        f"FIN={eq_fin:.6e} (n={n_fin})  "
        f"rel={rel:.2e}  "
        f"{'PASS' if rel < 0.05 else 'FAIL'}"
    )

# --- (a) S(t) shape in Porte 2 ---
print("\n" + "=" * 70)
print("(a) Porte 2 — S(t) shape (oscillation vs trend)")
print("=" * 70)
for name, data in [("REF", ref), ("FIN", fin)]:
    s = col(data, 'S')
    diffs = np.diff(s)
    print(f"  {name}: min={np.min(s):.4e} max={np.max(s):.4e} "
          f"mean={np.mean(s):.4e}")
    print(f"        std(diff)={np.std(diffs):.2e}  "
          f"mean(diff)={np.mean(diffs):+.2e}")

# --- (b) Equilibrium status at 3 days for Porte 3 ---
print("\n" + "=" * 70)
print("(b) Porte 3 — Equilibrium status at 3 days")
print("=" * 70)
runs = [
    "Porte3/reference.csv",
    "Porte3/k_CO2_minus20.csv", "Porte3/k_CO2_plus20.csv",
    "Porte3/Pmax_minus20.csv",  "Porte3/Pmax_plus20.csv",
    "Porte3/R_maint_minus20.csv", "Porte3/R_maint_plus20.csv",
    "Porte3/P_seuil_minus20.csv", "Porte3/P_seuil_plus20.csv",
]
print(f"  {'run':<28s} {'CO2_end':>11s} {'CO2_prev':>11s} {'rel_drift':>10s}")
for path in runs:
    if not Path(path).exists():
        print(f"  {Path(path).stem:<28s} MISSING")
        continue
    co2 = col(load_csv(path), 'CO2')
    drift = abs(co2[-1] - co2[-2]) / max(abs(co2[-1]), abs(co2[-2]), 1e-30)
    print(f"  {Path(path).stem:<28s} {co2[-1]:>11.4e} "
          f"{co2[-2]:>11.4e} {drift:>10.2e}")

# --- (c) S(t) in P_seuil runs ---
print("\n" + "=" * 70)
print("(c) Porte 3 — S(t) in P_seuil runs (dead WWW chain test)")
print("=" * 70)
S_data = {}
for label, path in [("reference", "Porte3/reference.csv"),
                    ("minus20",   "Porte3/P_seuil_minus20.csv"),
                    ("plus20",    "Porte3/P_seuil_plus20.csv")]:
    if not Path(path).exists():
        print(f"  {label}: MISSING")
        continue
    s = col(load_csv(path), 'S')
    S_data[label] = s
    print(f"  {label:9s}: S(0)={s[0]:.6e} S(end)={s[-1]:.6e} "
          f"Delta={s[-1]-s[0]:+.2e}")

if 'minus20' in S_data and 'plus20' in S_data:
    diff = S_data['plus20'] - S_data['minus20']
    max_diff = float(np.max(np.abs(diff)))
    print(f"\n  max |S(plus20) - S(minus20)| = {max_diff:.2e}")
    if max_diff < 1e-15:
        print("  VERDICT: S(t) IDENTICAL to machine precision")
        print("           -> WWW chain is DEAD (C1 confirmed)")
    elif max_diff < 1e-10:
        print("  VERDICT: S(t) differs very weakly")
        print("           -> WWW chain exists but contributes < 1e-5 %")
    else:
        print("  VERDICT: S(t) differs substantially")
        print("           -> WWW chain is active; C1 needs refinement")

print("\n" + "=" * 70)