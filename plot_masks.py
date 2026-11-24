# =====================================================================
# PLOT_MASKS — 2D VISUALISATION OF THE BOOLEAN MASKS
# 
# Description:
#   Creates a 2D cross‑section of the TECP reactor showing the vertical
#   profiles of the five Boolean masks for both biomes. The plot
#   includes labels and a legend.
#
# Version: 1.0
# Reference: Trouillefou, C.M. et al. (2026). Parametric resilience
#            in a closed ecological reactor: The Wood Wide Web as a
#            distributed PID controller for life support.
#            npj Microgravity (submitted).
#
# Key features:
#   1. Two side‑by‑side panels (continental and marsh).
#   2. Coloured horizontal bands for each mask.
#   3. Soil surface line and mask labels.
#   4. Saves output as PNG and SVG.
# =====================================================================

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(10, 8), sharey=True)

biomes = ['Continental (left)', 'Marsh (right)']
z_sol = [0.4, 0.2]          # soil surface (normalised)
z_max = [0.8, 0.6]          # maximum foliage height
z_resp = [0.6, 0.4]         # mid‑height of the foliage (z_sol + z_max)/2

# Horizontal text displacement
x_positions = {
    'M_ceil': 0.15,
    'M_photo': 0.15,
    'M_resp': 0.85,
    'M_transp': 0.50,
    'M_uptake': 0.85
}

for i, biome in enumerate(biomes):
    ax[i].set_xlim(0, 1)
    ax[i].set_ylim(0, 1)
    ax[i].set_title(biome, fontsize=14, fontweight='bold')
    ax[i].set_xticks([])
    ax[i].set_ylabel('z (normalised)', fontsize=11)

    # --- Soil surface line ---
    ax[i].axhline(y=z_sol[i], color='brown', linestyle='--', linewidth=2, label='Soil surface')

    # --- M_ceil: Peltier ceiling (z > 0.9) ---
    ax[i].axhspan(0.9, 1.0, color='lightblue', alpha=0.3)
    ax[i].text(x_positions['M_ceil'], 0.95, "M_ceil = 1", ha='center', va='center', fontsize=10, color='navy')

    # --- M_uptake: root zone (0 to z_sol) ---
    ax[i].axhspan(0, z_sol[i], color='brown', alpha=0.2)
    ax[i].text(x_positions['M_uptake'], z_sol[i] / 2, "M_uptake = 1", ha='center', va='center', fontsize=10, color='saddlebrown')

    # --- M_resp: up to mid‑height (0 to z_resp) ---
    ax[i].axhspan(0, z_resp[i], color='orange', alpha=0.15)
    ax[i].text(x_positions['M_resp'], z_resp[i] / 2, "M_resp = 1", ha='center', va='center', fontsize=10, color='darkorange')

    # --- M_photo: lower foliage (z_sol to z_resp) ---
    ax[i].axhspan(z_sol[i], z_resp[i], color='green', alpha=0.15)
    ax[i].text(x_positions['M_photo'], (z_sol[i] + z_resp[i]) / 2, "M_photo = 1", ha='center', va='center', fontsize=10, color='darkgreen')

    # --- M_transp: lower canopy (z_sol to z_resp) ---
    # M_photo et M_transp occupent la même zone verticale, donc on décale le texte horizontalement
    ax[i].axhspan(z_sol[i], z_resp[i], color='purple', alpha=0.1)
    ax[i].text(x_positions['M_transp'], (z_sol[i] + z_resp[i]) / 2, "M_transp = 1", ha='center', va='center', fontsize=10, color='darkmagenta')

    # --- Vertical line to separate texts ---
    ax[i].axvline(x=0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)

    # --- Legende ---
    ax[i].legend(loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('masks_TECP.png', dpi=300)
plt.savefig('masks_TECP.svg', format='svg')
plt.show()