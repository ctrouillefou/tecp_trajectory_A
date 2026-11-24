#!/usr/bin/env python3
"""B - CO2 flux budgets. Fixes the missing x3600 and the color key."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from analysis_common import (load_data, C, WINDOWS, window_mask,
                             trapz_s, Q_PR, Q_PoverR, style, save_fig)

OUT = 'analysis_B'
df, t_h, t_d = load_data()

P     = df['P_photo'].to_numpy()
R_p   = df['R_plant'].to_numpy()
R_c   = df['R_crew'].to_numpy()
R_tot = R_p + R_c
J_net = P - R_tot             # >0 = CO2 sink
J_bio = P - R_p               # == CSV 'net_gas'

err = np.max(np.abs(df['net_gas'].to_numpy() - J_bio))
print(f'Check: max|net_gas - (P - R_plant)| = {err:.2e}  (should be << 1e-9)')

q_pr  = Q_PR(P, R_p, R_c)
q_p_r = Q_PoverR(P, R_p)

rows = []
for name, t0, t1 in WINDOWS:
    m = window_mask(t_h, t0, t1)
    Pw, Rtw, Jw = P[m], R_tot[m], J_net[m]
    tw = t_h[m]
    IP = trapz_s(Pw, tw)
    IR = trapz_s(Rtw, tw)
    IN = IP - IR
    CO2w = df['CO2'].to_numpy()[m]
    dCO2 = CO2w[-1] - CO2w[0]
    pred = -IN                              # expected ΔCO2 from fluxes
    rows.append(dict(window=name,
                     P_mean=Pw.mean(), Rtot_mean=Rtw.mean(),
                     Jnet_mean=Jw.mean(),
                     Q_PR=Pw.mean()/Rtw.mean(),
                     Q_PoverR=Pw.mean()/R_p[m].mean(),
                     int_P=IP, int_Rtot=IR, int_net=IN,
                     dCO2_obs=dCO2, dCO2_pred=pred,
                     ratio=pred/dCO2 if abs(dCO2) > 1e-30 else np.nan))

print('\n' + pd.DataFrame(rows).to_string(index=False,
      float_format=lambda x: f'{x:.4g}'))

def find_crossings(t, y, target=1.0):
    """Return list of t where y(t) = target (linear interpolation, all sign changes)."""
    d = np.asarray(y) - target
    s = np.sign(d)
    idx = np.where(np.diff(s) != 0)[0]
    out = []
    for i in idx:
        t0, t1 = t[i], t[i+1]
        y0, y1 = y[i], y[i+1]
        if y1 == y0:
            continue
        out.append(t0 + (target - y0) * (t1 - t0) / (y1 - y0))
    return out

# --- B1 : P, R_tot, J_net ---
fig, ax1 = plt.subplots(figsize=(7, 3.6))
ax1.plot(t_d, P*1e9,     color=C['blue'], lw=1.3, label=r'$P_{photo}$')
ax1.plot(t_d, R_tot*1e9, color=C['verm'], lw=1.3, label=r'$R_{plant}+R_{crew}$')
ax1.set_ylabel(r'Rate ($\times 10^{-9}$ s$^{-1}$)')
ax1.set_xlabel('Time (days)')
ax1.spines['top'].set_visible(False)
ax1.legend(loc='lower right', fontsize=8)
ax2 = ax1.twinx()
ax2.plot(t_d, J_net*1e9, color=C['black'], lw=1.0, ls='--', label=r'$J_{net}$')
ax2.axhline(0, color=C['black'], lw=0.4, alpha=0.5)
ax2.set_ylabel(r'$J_{net}$ ($\times 10^{-9}$ s$^{-1}$)')
ax2.spines['top'].set_visible(False)
ax2.legend(loc='upper right', fontsize=8)
ax1.set_title('B1 - Fluxes and net CO2 budget')
save_fig(fig, OUT, 'B1_fluxes')

# --- B2 : Q_PR and Q_PoverR ---
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(t_d, q_pr,  color=C['verm'], lw=1.3, label=r'$Q_{PR}$')
ax.plot(t_d, q_p_r, color=C['blue'], lw=1.0, ls='--', label=r'$Q_{P/R}$')
ax.axhline(1, color=C['black'], lw=0.6, alpha=0.6)
ax.set_ylim(0.5, 1.15)
ax.set_xlabel('Time (days)'); ax.set_ylabel('Ratio')
ax.set_title('B2 - Compensation ratios')
ax.legend(loc='lower right', fontsize=8)
style(ax)
save_fig(fig, OUT, 'B2_ratios')

# --- Exact crossings (fixes the incorrect 'Q_PR = 1 @ 46 h') ---
t_qpr   = find_crossings(t_h, q_pr,  1.0)
t_jpl   = find_crossings(t_h, J_bio, 0.0)   # bilan VÉGÉTAL (net_gas)
t_jtot  = find_crossings(t_h, J_net, 0.0)   # bilan GLOBAL (avec équipage)

print('\n--- Exact crossings (linear interpolation) ---')
print(f'  J_plant = P - R_plant crosses 0 : t = {t_jpl[0]:.2f} h' if t_jpl else '  J_plant: no crossing')
print(f'  Q_PR    = P / (R_plant+R_crew) crosses 1 : t = {t_qpr[0]:.2f} h' if t_qpr else '  Q_PR: no crossing')
print(f'  J_total = P - R_plant - R_crew crosses 0 : t = {t_jtot[0]:.2f} h' if t_jtot else '  J_total: no crossing')
print('  (NB: 46 h is J_plant, not Q_PR. CO2 peak should be near the J_total/Q_PR crossing.)')

# --- B3 : zoom Q_PR crossing ---
fig, axs = plt.subplots(2, 1, figsize=(7, 5.5), sharex=False)
ax = axs[0]
ax.plot(t_d, q_pr, color=C['verm'], lw=1.3, label=r'$Q_{PR}$')
ax.axhline(1.0, color=C['black'], lw=0.6, alpha=0.6)
if t_qpr:
    ax.axvline(t_qpr[0]/24.0, color=C['black'], ls=':', lw=0.9)
    ax.annotate(f'crossing @ {t_qpr[0]:.1f} h',
                xy=(t_qpr[0]/24.0, 1.0), xytext=(t_qpr[0]/24.0 + 3, 0.85),
                fontsize=8, arrowprops=dict(arrowstyle='->', color='grey', lw=0.7))
ax.set_ylabel(r'$Q_{PR}$'); ax.set_xlabel('Time (days)')
ax.set_title('B3a — Q_PR(t) over 30 days'); style(ax)

ax = axs[1]
ax.plot(t_d, J_bio*1e9,  color=C['blue'], lw=1.3, label=r'$J_{plant}=P-R_{plant}$')
ax.plot(t_d, J_net*1e9,  color=C['verm'], lw=1.3, label=r'$J_{total}=P-R_{plant}-R_{crew}$')
ax.axhline(0, color='grey', lw=0.5)
for t_evt, col, lbl in [(t_jpl[0]  if t_jpl  else None, C['blue'], 'J_plant=0'),
                        (t_jtot[0] if t_jtot else None, C['verm'], 'J_total=0')]:
    if t_evt:
        ax.axvline(t_evt/24.0, color=col, ls=':', lw=0.9)
        ax.text(t_evt/24.0 + 0.3, ax.get_ylim()[1]*0.85 if ax.get_ylim()[1] > 0 else 5,
                f'{lbl} @ {t_evt:.1f} h', color=col, fontsize=7)
ax.set_ylabel(r'Rate ($\times 10^{-9}$ s$^{-1}$)')
ax.set_xlabel('Time (days)')
ax.set_title('B3b — Vegetative vs global budget')
ax.legend(loc='lower right', fontsize=8); style(ax)

save_fig(fig, OUT, 'B3_crossings')