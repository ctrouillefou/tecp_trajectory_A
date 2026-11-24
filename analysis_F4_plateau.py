#!/usr/bin/env python3
# =====================================================================
# analysis_F4_plateau.py
#
# Spatial-resolution dependence diagnostic: 12³ vs 17³.
#
# For each of 4 configurations (baseline, Pmax×0.80, Pmax×1.20,
# Y_B×0.80) and each of 2 resolutions (12³, 17³), computes:
#   - peak CO2
#   - argmax time (reported for reference, NOT used as primary metric)
#   - contiguous plateau at 99% and 99.9% of max CO2, containing the
#     argmax: [t_start, t_end], centre, width, n_points
#   - ΔB = biomass_final - biomass_initial
#
# Comparison 12³ vs 17³ (per passation #2 §3.6):
#   symmetric relative difference  |a - b| / mean(a,b) × 100%
#
# Primary metric (passation #2 §1): centre of the 99% plateau.
# The discrete argmax is noise-sensitive on broad plateaus — this is
# the entire point of the plateau method (see passation #1 §6.1).
#
# Methodological note (passation #1 §6.2, #2 §5):
#   This is a *resolution dependence* test, NOT a formal proof of
#   spatial convergence. Do not write "spatial convergence demonstrated".
#
# Outputs:
#   spatial_convergence/plateau_analysis.csv
#   spatial_convergence/comparison_12cubed_vs_17cubed.csv
#   spatial_convergence/F4_plateau_diagnostic.png / .pdf
#
# Usage:
#   python3 analysis_F4_plateau.py
#
# If a glob pattern does not match, edit INPUT_PATTERNS below.
# A diagnostic scan of candidate directories is printed at startup.
# =====================================================================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis_common import C, style, save_fig, load_data

OUT = 'spatial_convergence'
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------
# Directory mapping — EDIT HERE if conventions differ
# ---------------------------------------------------------------
# One glob pattern per (config, resolution). If multiple CSVs match,
# the most recent is used (a warning is printed).
INPUT_PATTERNS = {
    'baseline': {
        '12cubed': '../spatial_convergence/TrajA_C2_12cubed_30d_crew_0.100_baseline_20261105_221520/audit_timeseries.csv',
        '17cubed': '../TrajA_C2_30d_crew_0.100_20261105_080226/audit_timeseries.csv',
    },
    'Pmax_x0.80': {
        '12cubed': '../spatial_convergence/TrajA_C2_12cubed_30d_crew_0.100_Pmax_x0.80_20261106_012728/audit_timeseries.csv',
        '17cubed': '../TrajA_C2_30d_crew_0.100_20261102_000819/audit_timeseries.csv',
    },
    'Pmax_x1.20': {
        '12cubed': '../spatial_convergence/TrajA_C2_12cubed_30d_crew_0.100_Pmax_x1.20_20261106_044644/audit_timeseries.csv',
        '17cubed': '../TrajA_C2_30d_crew_0.100_20261103_145757/audit_timeseries.csv',
    },
    'Y_B_x0.80': {
        '12cubed': '../spatial_convergence/TrajA_C2_12cubed_30d_crew_0.100_Y_B_x0.80_20261106_080915/audit_timeseries.csv',
        '17cubed': '../TrajA_C2_30d_crew_0.100_20261103_071431/audit_timeseries.csv',
    },
}

THRESHOLDS = [0.99, 0.999]   # plateau fractions


# ---------------------------------------------------------------
# Diagnostic scan (helps user spot mismatched patterns)
# ---------------------------------------------------------------
def scan_candidates():
    print("--- Diagnostic: candidate directories ---")
    for pattern in ['TrajA_C2_*', 'spatial_convergence/TrajA_C2_*']:
        matches = sorted(glob.glob(pattern))
        if matches:
            print(f"  {pattern}: {len(matches)} match(es), showing up to 10:")
            for m in matches[:10]:
                print(f"    {m}")
            if len(matches) > 10:
                print(f"    ... and {len(matches) - 10} more")
        else:
            print(f"  {pattern}: none")
    print()


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def find_csv(pattern, label, verbose=True):
    """Return the most recent CSV matching `pattern`; warn on multiple."""
    matches = sorted(glob.glob(pattern))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No CSV found for [{label}]:\n  pattern = {pattern}\n"
            f"  Edit INPUT_PATTERNS at the top of this script."
        )
    if len(matches) > 1 and verbose:
        print(f"  NOTE [{label}]: {len(matches)} matches for {pattern}")
        for m in matches:
            print(f"    {m}")
        print(f"  → using most recent: {matches[-1]}")
    return matches[-1]


def plateau_interval(t_h, y, frac):
    """Contiguous interval around argmax where y >= frac * y_max."""
    t = np.asarray(t_h, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return None
    i_peak = int(np.argmax(y))
    y_max = float(y[i_peak])
    if y_max <= 0:
        return None
    thresh = frac * y_max

    i_left = i_peak
    while i_left > 0 and y[i_left - 1] >= thresh:
        i_left -= 1
    i_right = i_peak
    while i_right < len(y) - 1 and y[i_right + 1] >= thresh:
        i_right += 1

    return {
        'i_left': i_left, 'i_right': i_right,
        't_start': float(t[i_left]), 't_end': float(t[i_right]),
        'center': float((t[i_left] + t[i_right]) / 2),
        'width': float(t[i_right] - t[i_left]),
        'n_points': int(i_right - i_left + 1),
    }


def symmetric_rel_pct(a, b):
    """|a - b| / mean(|a|, |b|) × 100, with safe division."""
    denom = (abs(a) + abs(b)) / 2.0
    if denom < 1e-30:
        return np.nan
    return abs(a - b) / denom * 100.0


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    print("=" * 72)
    print("F4 — Spatial-resolution dependence diagnostic (12³ vs 17³)")
    print("=" * 72)
    scan_candidates()

    # ---- Load + compute plateau metrics ----
    loaded = {}       # (cfg, res) -> (t_h, co2, delta_B, peak, t_argmax)
    rows = []
    for cfg_label, paths in INPUT_PATTERNS.items():
        for resolution in ('12cubed', '17cubed'):
            try:
                path = find_csv(paths[resolution], f'{cfg_label}/{resolution}')
            except FileNotFoundError as e:
                print(f"\n  ERROR: {e}")
                continue
            df, t_h, _ = load_data(path)
            co2 = df['CO2'].to_numpy()
            bio = df['biomass'].to_numpy()

            delta_B = float(bio[-1] - bio[0])
            peak = float(co2.max())
            t_argmax = float(t_h[int(np.argmax(co2))])

            loaded[(cfg_label, resolution)] = (t_h, co2, delta_B, peak, t_argmax)

            for thr in THRESHOLDS:
                pl = plateau_interval(t_h, co2, thr)
                row = dict(
                    config=cfg_label, resolution=resolution,
                    threshold_pct=round(thr * 100, 1),
                    peak_co2=peak, t_argmax_h=t_argmax,
                    delta_B=delta_B, source_path=path,
                )
                if pl is None:
                    row.update(dict(t_start_h=np.nan, t_end_h=np.nan,
                                    center_h=np.nan, width_h=np.nan,
                                    n_points=0))
                else:
                    row.update(dict(t_start_h=pl['t_start'],
                                    t_end_h=pl['t_end'],
                                    center_h=pl['center'],
                                    width_h=pl['width'],
                                    n_points=pl['n_points']))
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/plateau_analysis.csv', index=False)
    print(f"\n  saved: {OUT}/plateau_analysis.csv  ({len(df)} rows)")

    # ---- Comparison 12³ vs 17³ ----
    print("\n" + "=" * 72)
    print("COMPARISON 12³ vs 17³  (symmetric relative difference, %)")
    print("=" * 72)
    comp_rows = []
    for cfg_label in INPUT_PATTERNS:
        sub = df[df['config'] == cfg_label]
        if len(sub) == 0:
            print(f"  [{cfg_label}] no data — skipped")
            continue
        r12 = sub[sub['resolution'] == '12cubed']
        r17 = sub[sub['resolution'] == '17cubed']
        if len(r12) == 0 or len(r17) == 0:
            print(f"  [{cfg_label}] missing resolution — skipped")
            continue

        # Peak & ΔB are per (config, resolution) — not threshold-dependent
        peak12 = r12.iloc[0]['peak_co2'];  peak17 = r17.iloc[0]['peak_co2']
        dB12   = r12.iloc[0]['delta_B'];   dB17   = r17.iloc[0]['delta_B']
        ta12   = r12.iloc[0]['t_argmax_h']; ta17  = r17.iloc[0]['t_argmax_h']

        centers = {}
        for thr_pct in [99.0, 99.9]:
            r12t = r12[r12['threshold_pct'] == thr_pct]
            r17t = r17[r17['threshold_pct'] == thr_pct]
            if len(r12t) and len(r17t):
                centers[thr_pct] = (
                    r12t.iloc[0]['center_h'],
                    r17t.iloc[0]['center_h'],
                    symmetric_rel_pct(r12t.iloc[0]['center_h'],
                                      r17t.iloc[0]['center_h']),
                )
            else:
                centers[thr_pct] = (np.nan, np.nan, np.nan)

        comp = dict(
            config=cfg_label,
            rel_diff_peak_pct=symmetric_rel_pct(peak12, peak17),
            rel_diff_delta_B_pct=symmetric_rel_pct(dB12, dB17),
            rel_diff_center_99_pct=centers[99.0][2],
            rel_diff_center_999_pct=centers[99.9][2],
            t_argmax_12_h=ta12, t_argmax_17_h=ta17,
            center_99_12_h=centers[99.0][0], center_99_17_h=centers[99.0][1],
            center_999_12_h=centers[99.9][0], center_999_17_h=centers[99.9][1],
        )
        comp_rows.append(comp)

        print(f"\n  [{cfg_label}]")
        print(f"    peak CO₂       : {peak12:.4e}  vs  {peak17:.4e}"
              f"   →  {comp['rel_diff_peak_pct']:.2f} %")
        print(f"    ΔB             : {dB12:+.6e}  vs  {dB17:+.6e}"
              f"   →  {comp['rel_diff_delta_B_pct']:.2f} %")
        print(f"    argmax         : {ta12:.1f} h  vs  {ta17:.1f} h"
              f"   (raw; not primary)")
        for thr_pct in [99.0, 99.9]:
            c12, c17, rel = centers[thr_pct]
            print(f"    centre {thr_pct:4.1f}%    : {c12:.1f} h  vs  {c17:.1f} h"
                  f"   →  {rel:.2f} %")

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(f'{OUT}/comparison_12cubed_vs_17cubed.csv', index=False)
    print(f"\n  saved: {OUT}/comparison_12cubed_vs_17cubed.csv  "
          f"({len(comp_df)} rows)")

    # ---- Figure (2×2 panels) ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axes = axes.ravel()
    for ax, cfg_label in zip(axes, INPUT_PATTERNS):
        plotted_any = False
        for resolution, color, ls in [('12cubed', C['blue'], '--'),
                                       ('17cubed', C['verm'], '-')]:
            if (cfg_label, resolution) not in loaded:
                continue
            t_h, co2, _, _, _ = loaded[(cfg_label, resolution)]
            ax.plot(t_h / 24.0, co2, color=color, ls=ls, lw=1.2,
                    label=resolution)
            plotted_any = True

        # Shade the 17³ 99% plateau
        row17 = df[(df['config'] == cfg_label)
                   & (df['resolution'] == '17cubed')
                   & (df['threshold_pct'] == 99.0)]
        if len(row17) and not np.isnan(row17.iloc[0]['center_h']):
            r = row17.iloc[0]
            ax.axvspan(r['t_start_h'] / 24.0, r['t_end_h'] / 24.0,
                       color='gray', alpha=0.15, label='99% plateau (17³)')
            ax.axvline(r['center_h'] / 24.0, color='gray', lw=0.8, ls=':')

        ax.set_title(cfg_label)
        ax.set_ylabel('CO₂ fraction')
        style(ax)
        if plotted_any:
            ax.legend(fontsize=7)

    for ax in axes[-2:]:
        ax.set_xlabel('Time (days)')
    fig.suptitle('F4 — CO₂ trajectories, 12³ vs 17³ (4 configurations)')
    save_fig(fig, OUT, 'F4_plateau_diagnostic')

    # ---- Verdict summary ----
    print("\n" + "=" * 72)
    print("VERDICT (per passation #2 §3.6 thresholds)")
    print("=" * 72)
    if len(comp_df) > 0:
        print(f"  max |Δ peak|      : "
              f"{comp_df['rel_diff_peak_pct'].max():.2f} %")
        print(f"  max |ΔB|          : "
              f"{comp_df['rel_diff_delta_B_pct'].max():.2f} %")
        print(f"  max |Δ centre 99%| : "
              f"{comp_df['rel_diff_center_99_pct'].max():.2f} %")
        print(f"  max |Δ centre 99.9%| : "
              f"{comp_df['rel_diff_center_999_pct'].max():.2f} %")
        ok = (comp_df['rel_diff_center_99_pct'].max() < 2.0
              and comp_df['rel_diff_peak_pct'].max() < 1.0
              and comp_df['rel_diff_delta_B_pct'].max() < 1.0)
        print(f"\n  → {'PASS' if ok else 'REVIEW'}  "
              f"(passation #2 reports 0.54% / 0.72% / 1.9%)")
    print("\n  NOTE: resolution-dependence test, not spatial convergence.")
    print("=" * 72)


if __name__ == '__main__':
    main()