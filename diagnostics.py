# diagnostics.py
# =====================================================================
# Diagnostic and reporting functions for the TECP V10.3 audit.
# =====================================================================

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def report_masks(mesh, masks_dict):
    """Print active-cell count and volume fraction for each mask."""
    total_volume = float(np.sum(mesh.cellVolumes))
    print("\n=== MASK DIAGNOSTIC ===")
    print(f"Total domain volume: {total_volume:.6e}")
    for name, mask in masks_dict.items():
        active = np.asarray(mask.value) > 0.5
        n_active = int(np.count_nonzero(active))
        vol_active = float(np.sum(np.asarray(mesh.cellVolumes)[active]))
        frac = vol_active / total_volume if total_volume > 0 else np.nan
        print(f"{name:10s} cells={n_active:5d}  "
              f"volume={vol_active:.4e}  fraction={frac:.4f}")


# =====================================================================
# Porte 1 — Carbon balance (CHANGE: C_coop term added)
# =====================================================================
def compute_carbon_balance_from_hist(hist, Y_B, V_total, crew_CO2_prod,
                                     C_coop=None, Y_B_resp=1.0):
    """
    Carbon balance from a finished simulation (offline).

    Pools (volume-integrated, uniform grid assumed):
        C_total(t) = [CO2(t) + bio(t)/Y_B] * V_total

    Per-cell equations:
        dCO2/dt = -P + R + CO2_prod_eff
        dbio/dt = Y_B*P - R + C_coop

    Therefore:
        dC_total/dt = crew_CO2_prod * V_total
                      + R_mean * V_total * (Y_B_resp - 1/Y_B)
                      + C_coop_mean * V_total / Y_B

    The second term is the carbon "loss" due to Y_B != 1 (photorespiration,
    root exudates, etc.). The third term is the cooperation contribution
    (WWW chain). With Y_B = 1 and Y_B_resp = 1, the second term is zero,
    and the third term is numerically ~3e-7 of P_photo in C2 — but it MUST
    be included to make the identity exact.

    Parameters
    ----------
    hist : dict
        Must contain 'time_h', 'CO2', 'bio', 'P_photo', 'R_plant'.
        May contain 'C_coop'; otherwise C_coop must be supplied.
    Y_B : float
        Biomass/CO2 yield coefficient (from params_real.json).
    V_total : float
        Total domain volume (m^3).
    crew_CO2_prod : float
        Crew CO2 production (per unit volume, 1/s), including crew_mult
        and RQ.
    C_coop : float, array, or None
        Cooperation term. If None (default), read from hist['C_coop'].
        If absent from hist too, treated as 0 (with warning printed
        externally).
    Y_B_resp : float, optional
        CO2:biomass stoichiometry for plant respiration. Default 1.0.

    Returns
    -------
    dict with 'time_h', 'C_total', 'dC_dt', 'flux_expected', 'residual',
    'relative_residual', 'max_relative_residual', 'bilan_closes'.
    """
    times = np.array(hist['time_h'])
    CO2   = np.array(hist['CO2'])
    bio   = np.array(hist['bio'])
    R     = np.array(hist['R_plant'])

    if len(times) < 2:
        return {'bilan_closes': False, 'max_relative_residual': np.inf,
                'note': 'history too short'}

    # --- Resolve C_coop term -------------------------------------------
    if C_coop is None:
        if 'C_coop' in hist:
            C_coop_arr = np.array(hist['C_coop'], dtype=float)
        else:
            C_coop_arr = np.zeros_like(CO2)
    elif np.isscalar(C_coop):
        C_coop_arr = np.full_like(CO2, float(C_coop))
    else:
        C_coop_arr = np.asarray(C_coop, dtype=float)

    # Pool carbone total (CO2-equivalent x volume)
    C_total = (CO2 + bio / Y_B) * V_total

    dt_s = (times[1] - times[0]) * 3600.0
    dC_dt = np.full_like(C_total, np.nan)
    dC_dt[1:] = np.diff(C_total) / dt_s

    # Flux attendu (CHANGE: include C_coop term)
    crew_total = crew_CO2_prod * V_total
    resp_term  = R * V_total * (Y_B_resp - 1.0 / Y_B)
    coop_term  = C_coop_arr * V_total / Y_B
    flux_expected = crew_total + resp_term + coop_term

    residual = dC_dt - flux_expected
    rel      = np.abs(residual) / np.maximum(np.abs(C_total), 1e-30)
    valid    = ~np.isnan(rel)
    max_rel  = float(np.max(rel[valid])) if np.any(valid) else np.inf

    return {
        'time_h': times,
        'C_total': C_total,
        'dC_dt': dC_dt,
        'flux_expected': flux_expected,
        'residual': residual,
        'relative_residual': rel,
        'max_relative_residual': max_rel,
        'bilan_closes': max_rel < 1e-3,
    }


def daily_integrated_net_photo(hist, window_hours=24):
    """Rolling 24-h integral of net_photo = P - R."""
    times = np.array(hist['time_h'])
    if 'net_photo' in hist:
        net = np.array(hist['net_photo'])
    else:
        net = np.array(hist['P_photo']) - np.array(hist['R_plant'])

    if len(times) < 2:
        return {'time_h': np.array([]), 'integral': np.array([]),
                'min_integral': np.nan, 'mean_integral': np.nan,
                'all_positive': True}

    dt_h = times[1] - times[0]
    n_window = max(1, int(round(window_hours / dt_h)))

    integrals, mid_times = [], []
    for i in range(n_window, len(times)):
        t_win   = times[i-n_window:i]
        net_win = net[i-n_window:i]
        integrals.append(np.trapz(net_win, t_win))
        mid_times.append(times[i] - window_hours / 2.0)

    integrals = np.array(integrals)
    mid_times = np.array(mid_times)
    if len(integrals) == 0:
        return {'time_h': np.array([]), 'integral': np.array([]),
                'min_integral': np.nan, 'mean_integral': np.nan,
                'all_positive': True}

    return {
        'time_h': mid_times,
        'integral': integrals,
        'min_integral': float(np.min(integrals)),
        'mean_integral': float(np.mean(integrals)),
        'all_positive': bool(np.all(integrals > 0)),
    }


# =====================================================================
# Porte 1bis — Oxygen balance
# =====================================================================
def compute_oxygen_balance_from_hist(hist, V_total, O2_cons_eff,
                                     Y_O2_resp=1.0):
    """O2 balance (unchanged from original)."""
    times = np.array(hist['time_h'])
    O2_arr = np.array(hist['O2'])
    P      = np.array(hist['P_photo'])
    R      = np.array(hist['R_plant'])

    if len(times) < 2:
        return {'bilan_closes': False, 'max_relative_residual': np.inf,
                'note': 'history too short'}

    O2_total = O2_arr * V_total
    dt_s = (times[1] - times[0]) * 3600.0
    dO2_dt = np.full_like(O2_total, np.nan)
    dO2_dt[1:] = np.diff(O2_total) / dt_s

    flux_expected = (P - R * Y_O2_resp - O2_cons_eff) * V_total

    residual = dO2_dt - flux_expected
    rel      = np.abs(residual) / np.maximum(np.abs(O2_total), 1e-30)
    valid    = ~np.isnan(rel)
    max_rel  = float(np.max(rel[valid])) if np.any(valid) else np.inf

    return {
        'time_h': times,
        'O2_total': O2_total,
        'dO2_dt': dO2_dt,
        'flux_expected': flux_expected,
        'residual': residual,
        'relative_residual': rel,
        'max_relative_residual': max_rel,
        'bilan_closes': max_rel < 1e-3,
    }


# =====================================================================
# Porte 1bis — Water balance (CHANGE: docstring clarification)
# =====================================================================
def compute_water_balance_from_hist(hist, V_total, cond_flux=None):
    """
    Water balance from a finished simulation (offline).

    Pool (volume-integrated):
        H2O_total(t) = (theta_mean + Cv_mean) * V_total

    Model behavior (from build_theta_eq + build_vapor_eq):
        d(theta + Cv)/dt = div(D_theta * grad(theta))
                          + div(D_vapor * grad(Cv))
        The condensation term appears with OPPOSITE signs in the two
        equations, so it cancels exactly in the sum. It is a pure
        redistribution (vapor -> liquid), NOT a leak.

    Test modes
    ----------
    cond_flux = None (RECOMMENDED for C2):
        Strict conservation. dH2O_total/dt is expected to vanish up
        to numerical diffusion discretization. This is the primary
        test for Porte 1bis.

    cond_flux = float (secondary diagnostic only):
        Tests the hypothesis that condensation leaves the (theta+Cv)
        system. Included for completeness; NOT the expected model
        behavior for C2.

    Parameters
    ----------
    hist : dict
        Must contain 'time_h', 'theta', 'Cv'.
    V_total : float
        Total domain volume (m^3).
    cond_flux : float or None, optional
        Volume-averaged condensation rate (1/s). See above.

    Returns
    -------
    dict with 'bilan_closes', 'max_relative_residual', 'mode', ...
    """
    if 'Cv' not in hist:
        return {'bilan_closes': False,
                'max_relative_residual': np.inf,
                'note': "missing 'Cv' in hist; cannot check water balance"}

    times = np.array(hist['time_h'])
    theta = np.array(hist['theta'])
    Cv    = np.array(hist['Cv'])

    if len(times) < 2:
        return {'bilan_closes': False, 'max_relative_residual': np.inf,
                'note': 'history too short'}

    H2O_total = (theta + Cv) * V_total
    dt_s = (times[1] - times[0]) * 3600.0
    dH2O_dt = np.full_like(H2O_total, np.nan)
    dH2O_dt[1:] = np.diff(H2O_total) / dt_s

    if cond_flux is not None:
        flux_expected = -float(cond_flux) * V_total * np.ones_like(H2O_total)
        mode = 'condensation leak (untracked condensate)'
    else:
        flux_expected = np.zeros_like(H2O_total)
        mode = 'strict conservation (theta <-> Cv only)'

    residual = dH2O_dt - flux_expected
    rel      = np.abs(residual) / np.maximum(np.abs(H2O_total), 1e-30)
    valid    = ~np.isnan(rel)
    max_rel  = float(np.max(rel[valid])) if np.any(valid) else np.inf

    return {
        'time_h': times,
        'H2O_total': H2O_total,
        'dH2O_dt': dH2O_dt,
        'flux_expected': flux_expected,
        'residual': residual,
        'relative_residual': rel,
        'max_relative_residual': max_rel,
        'bilan_closes': max_rel < 1e-3,
        'mode': mode,
    }


# =====================================================================
# Matter balance summary (unchanged)
# =====================================================================
def print_matter_balance_summary(carbon, oxygen, water):
    print("\n" + "=" * 72)
    print("PORTE 1bis — MATTER BALANCE SUMMARY")
    print("=" * 72)

    def _status(bal, name):
        if bal.get('note') and not bal.get('bilan_closes', False):
            return f"  {name:<8s}: SKIP ({bal['note']})"
        verdict = "PASS" if bal['bilan_closes'] else "FAIL"
        return (f"  {name:<8s}: {verdict}  "
                f"(max_rel = {bal['max_relative_residual']:.3e})")

    print("\n[Carbon]  C_total = (CO2 + bio/Y_B) * V_total")
    if 'C_total' in carbon and len(carbon['C_total']) >= 2:
        print(f"  C_total start = {carbon['C_total'][0]:.6e} CO2-eq.m3")
        print(f"  C_total end   = {carbon['C_total'][-1]:.6e} CO2-eq.m3")
        print(f"  Delta C       = {carbon['C_total'][-1]-carbon['C_total'][0]:+.6e}")
    print(_status(carbon, 'Carbon'))

    print("\n[Oxygen]  O2_total = O2 * V_total")
    if 'O2_total' in oxygen and len(oxygen['O2_total']) >= 2:
        print(f"  O2_total start = {oxygen['O2_total'][0]:.6e} m3")
        print(f"  O2_total end   = {oxygen['O2_total'][-1]:.6e} m3")
        print(f"  Delta O2       = {oxygen['O2_total'][-1]-oxygen['O2_total'][0]:+.6e}")
    print(_status(oxygen, 'Oxygen'))

    print("\n[Water]   H2O_total = (theta + Cv) * V_total")
    if 'H2O_total' in water and len(water['H2O_total']) >= 2:
        print(f"  H2O_total start = {water['H2O_total'][0]:.6e} m3")
        print(f"  H2O_total end   = {water['H2O_total'][-1]:.6e} m3")
        print(f"  Delta H2O       = {water['H2O_total'][-1]-water['H2O_total'][0]:+.6e}")
    if 'mode' in water:
        print(f"  Mode            = {water['mode']}")
    print(_status(water, 'Water'))

    n_pass = sum([
        bool(carbon.get('bilan_closes', False)),
        bool(oxygen.get('bilan_closes', False)),
        bool(water.get('bilan_closes', False)),
    ])
    print("\n" + "-" * 72)
    print(f"OVERALL: {n_pass}/3 balances PASS")
    if n_pass < 3:
        print("Investigation checklist (in order):")
        if not carbon.get('bilan_closes', False):
            print("  [C]  Y_B propagation, C_coop term, sync flux/post-solve")
        if not oxygen.get('bilan_closes', False):
            print("  [O2] Y_O2_resp, O2_cons_eff uniformity")
        if not water.get('bilan_closes', False):
            print("  [H2O] Condensation fate, transpiration/evap masks")
    print("=" * 72)


def get_window_mask(hist, window_hours=72.0):
    """Return a boolean mask selecting records in the last `window_hours`."""
    times = np.asarray(hist['time_h'], dtype=float)
    if len(times) == 0:
        raise ValueError("hist['time_h'] is empty")
    if not np.all(np.isfinite(times)):
        raise ValueError("Non-finite values in hist['time_h']")
    if len(times) > 1 and not np.all(np.diff(times) >= 0):
        raise ValueError("hist['time_h'] is not sorted")

    expected_len = len(times)
    for key in hist:
        if len(hist[key]) != expected_len:
            raise ValueError(
                f"Length mismatch: '{key}' has {len(hist[key])} records, "
                f"expected {expected_len}"
            )

    t_end = times[-1]
    t_start = max(0.0, t_end - window_hours)
    mask = times >= t_start
    return mask, t_start, t_end


def print_diagnostic_summary(hist, window_hours=72.0, crew_mult=None):
    """Carbon fluxes, gas stats, basic threshold checks."""
    times = np.asarray(hist['time_h'], dtype=float)
    mask, t_start, t_end = get_window_mask(hist, window_hours=window_hours)
    n_selected = int(np.sum(mask))
    actual_window = (t_end - times[mask][0]) if n_selected > 0 else 0.0

    P_photo_mean = np.mean(np.asarray(hist['P_photo'])[mask])
    R_plant_mean = np.mean(np.asarray(hist['R_plant'])[mask])
    R_crew_mean  = np.mean(np.asarray(hist['R_crew'])[mask])
    net_flux = R_plant_mean + R_crew_mean - P_photo_mean

    title = (f"DIAGNOSTIC SUMMARY (crew x{crew_mult})"
             if crew_mult is not None else "DIAGNOSTIC SUMMARY")
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"\nTime window: t = [{t_start:.1f} h, {t_end:.1f} h] "
          f"({n_selected} records, actual span = {actual_window:.1f} h)")

    print(f"\nCarbon fluxes (mean over last {actual_window:.1f} h):")
    print(f"  Photosynthesis   P_photo : {P_photo_mean:+.6e} s^-1")
    print(f"  Plant respiration R_plant: {R_plant_mean:+.6e} s^-1")
    print(f"  Crew production   R_crew : {R_crew_mean:+.6e} s^-1")
    print(f"  Net flux                 : {net_flux:+.6e} s^-1")

    CO2_arr  = np.asarray(hist['CO2'])[mask]
    O2_arr   = np.asarray(hist['O2'])[mask]
    Temp_arr = np.asarray(hist['Temp'])[mask]
    print(f"\nCO2 / O2 / T statistics (last {actual_window:.1f} h):")
    print(f"  CO2 mean = {CO2_arr.mean():.6f}  min = {CO2_arr.min():.6f}  "
          f"max = {CO2_arr.max():.6f}")
    print(f"  O2  mean = {O2_arr.mean():.6f}  min = {O2_arr.min():.6f}  "
          f"max = {O2_arr.max():.6f}")
    print(f"  T   mean = {Temp_arr.mean():.2f} K  min = {Temp_arr.min():.2f}  "
          f"max = {Temp_arr.max():.2f}")
    print(f"\nBasic gas threshold checks:")
    print(f"  CO2 < 0.005 ?  -> "
          f"{'YES' if CO2_arr.max() < 0.005 else 'NO'}  "
          f"(max = {CO2_arr.max():.6f})")
    print(f"  O2  > 0.150 ?  -> "
          f"{'YES' if O2_arr.min() > 0.150 else 'NO'}  "
          f"(min = {O2_arr.min():.6f})")


def save_timeseries_csv(hist, output_dir):
    """Save the time-series CSV (unchanged)."""
    csv_file = os.path.join(output_dir, "audit_timeseries.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_h', 'CO2', 'O2', 'Temp', 'theta', 'biomass',
                         'S', 'M', 'P_photo', 'R_plant', 'R_crew',
                         'Q_bio', 'Q_rad', 'Cv', 'C_coop', 'net_gas',
                         'net_biomass'])
        for i in range(len(hist['time_h'])):
            writer.writerow([hist[k][i] for k in [
                'time_h', 'CO2', 'O2', 'Temp', 'theta', 'biomass',
                'S', 'M', 'P_photo', 'R_plant', 'R_crew', 'Q_bio',
                'Q_rad', 'Cv', 'C_coop', 'net_gas', 'net_biomass'
            ]])


def save_figure(hist, output_dir, crew_mult, Pmax_factor):
    """Save CO2/O2 figure (unchanged)."""
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(hist['time_h'], hist['CO2'], 'r-', label='CO2')
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('CO2 fraction', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.axhline(y=0.005, color='r', linestyle='--', alpha=0.5,
                label='CO2 limit')
    ax2 = ax1.twinx()
    ax2.plot(hist['time_h'], hist['O2'], 'b-', label='O2')
    ax2.set_ylabel('O2 fraction', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.axhline(y=0.15, color='b', linestyle='--', alpha=0.5,
                label='O2 limit')
    plt.title(f'V10.3 audit — Pmax x{Pmax_factor}, crew x{crew_mult}')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "audit.png"), dpi=150)
    plt.close(fig)