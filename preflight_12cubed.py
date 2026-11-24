#!/usr/bin/env python3
"""Preflight 12³ — vérifie versions, fichiers, biomass_init avant les 4 runs."""
import os, sys, json
import numpy as np

SEP = "=" * 72
print(SEP); print("PREFLIGHT 12³ — TECP V10.3"); print(SEP)

# --- 1. Versions ---
print(f"Python : {sys.version.split()[0]}")
try:
    import fipy; print(f"FiPy   : {fipy.__version__}")
except Exception as e: print(f"FiPy   : MISSING ({e})"); sys.exit(1)
import numpy; print(f"NumPy  : {numpy.__version__}")

# --- 2. Fichiers requis ---
required = ['params_real.json', 'config.py', 'biology.py', 'equations.py',
            'build_masks.py', 'run_simulation.py', 'sensitivity_all.csv']
missing = [f for f in required if not os.path.exists(f)]
if missing:
    print(f"\n✗ MANQUANTS : {missing}"); sys.exit(1)
print(f"Fichiers : OK ({len(required)} présents)")

# --- 3. Config 12³ (domaine physique identique) ---
import config as cfg
cfg.NX = cfg.NY = cfg.NZ = 12
cfg.DX = cfg.DY = cfg.DZ = 0.85 / 12
print(f"\nMesh 12³ : {cfg.NX}³, dx = {cfg.DX:.6f} m, "
      f"domaine = {12*cfg.DX:.4f} m (vs 0.85 m à 17³)")

# --- 4. Masques + biomass_init ---
from fipy import Grid3D
from build_masks import build_masks
with open('params_real.json') as f: P_BASE = json.load(f)
geom = {k: P_BASE[k] for k in ['x_split','z_ceil','z_sol_continental',
        'z_plant_max_continental','z_sol_marsh','z_plant_max_marsh']}
mesh = Grid3D(nx=12, ny=12, nz=12, dx=cfg.DX, dy=cfg.DX, dz=cfg.DX)
M_ceil, M_photo, M_resp, M_transp, M_uptake = build_masks(mesh, geom)

V_12 = float(np.sum(mesh.cellVolumes))
V_17 = 0.61413  # du log d'origine
print(f"\nV_total : 12³ = {V_12:.6e} m³  |  17³ = {V_17:.6e} m³  "
      f"|  Δ = {100*(V_12-V_17)/V_17:+.3f} %")

bio_init_12 = float(np.mean(cfg.BIOMASS_INIT_SCALE * (
    (1.0 - cfg.B_ROOT_FRACTION) * np.asarray(M_photo.value)
    + cfg.B_ROOT_FRACTION * np.asarray(M_uptake.value))))
BIO_REF = 0.0223513588130274  # baseline t=0 sensitivity_all.csv
delta_pct = 100.0 * (bio_init_12 - BIO_REF) / BIO_REF

print(f"\nM_photo mean (12³)  = {float(np.mean(M_photo.value)):.6f}")
print(f"M_uptake mean (12³) = {float(np.mean(M_uptake.value)):.6f}")
print(f"bio_init 12³        = {bio_init_12:.8f}")
print(f"bio_init 17³ (ref)  = {BIO_REF:.8f}")
print(f"Δ bio_init          = {delta_pct:+.3f} %")

# --- 5. Verdict ---
print("\n" + "-" * 72)
if abs(delta_pct) < 2.0:
    print(f"VERDICT : PASS (|Δ| = {abs(delta_pct):.3f} % < 2 %)")
    print("  → GO pour les 4 runs (attendre confirmation utilisateur)")
elif abs(delta_pct) < 5.0:
    print(f"VERDICT : WARN (|Δ| = {abs(delta_pct):.3f} % entre 2 et 5 %)")
    print("  → Discuter normalisation avant GO")
else:
    print(f"VERDICT : ALERT (|Δ| = {abs(delta_pct):.3f} % > 5 %)")
    print("  → NE PAS LANCER, discuter la stratégie de normalisation")
print("-" * 72)