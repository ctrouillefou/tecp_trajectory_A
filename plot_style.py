# =====================================================================
# plot_style.py
# Shared style for TECP manuscript figures (Elsevier / LSSR).
#
# - Sans-serif, 9 pt body, no top/right spines
# - Okabe-Ito palette (colorblind-safe)
# - 300 dpi, PDF vector output + PNG preview
# - mm-based figure sizes (Elsevier: 90 mm single, 190 mm double)
# =====================================================================

import os
import matplotlib.pyplot as plt

OKABE_ITO = {
    'black':     '#000000',
    'orange':    '#E69F00',
    'sky':       '#56B4E9',
    'green':     '#009E73',
    'yellow':    '#F0E442',
    'blue':      '#0072B2',
    'vermillon': '#D55E00',
    'purple':    '#CC79A7',
}

MM = 1.0 / 25.04    # mm -> inches


def setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        # 'savefig.bbox': 'tight',  # <-- problem : conflict with constrained_layout
        'savefig.format': 'pdf',
        'savefig.pad_inches': 0.04,
        'legend.frameon': False,
    })


def panel_label(ax, label, x=0.02, y=0.98):
    """Bold (a)/(b)/(c) at top-left; clip_on=False so it is never cropped."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontweight='bold', va='top', ha='left',
            fontsize=10, clip_on=False)


def save(fig, name, outdir='figures'):
    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(outdir, f'{name}.pdf')
    png = os.path.join(outdir, f'{name}.png')
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    plt.close(fig)
    print(f'  saved: {pdf}')
    print(f'  saved: {png}')
