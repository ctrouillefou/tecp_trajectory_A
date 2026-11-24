# config.py
# =====================================================================
# Global constants for the TECP V10.3 audit.
# All physics is in biology.py / equations.py.
# This file only centralises parameters and defaults.
# =====================================================================

# --- Time scale ---
TAU_BASE = 3600.0                       # seconds (1 hour)

# --- Signal speed ---
BASELINE_SPEED = 168.0
SPEED = 64.0

# --- Simulation duration ---
DURATION_DAYS = 30
DT = 360.0                              # 6 min
FORCING_PERIOD = 24.0 * 3600.0          # 24 h
N_PERIODS = DURATION_DAYS
STROBE_STEPS = int(FORCING_PERIOD / DT)
N_STEPS = N_PERIODS * STROBE_STEPS

# --- Parameter modifiers (audit) ---
PMAX_FACTOR = 2.0
RMAINT_DIV = 1.0
GAMMA = 0.002

# --- Reference configuration: C2 ---
CONFIG_NAME = "C2"

PMAX_FACTOR = 2.0
RMAINT_DIV = 1.0
GAMMA = 0.002

RQ = 0.85
CREW_VALUES = [0.10]

K_CO2_EXPECTED = 0.001

B_ROOT_FRACTION = 0.20
BIOMASS_INIT_SCALE = 0.10

S_INIT = 1e-5
M_INIT = 0.1

# --- Crew sweep ---
CREW_VALUES = [0.1] ##[0.05, 0.1, 0.2, 0.3, 0.4]

# --- Mesh ---
NX = NY = NZ = 17
DX = DY = DZ = 0.05

# --- Diffusivities (SI units) ---
K_THETA   = 1e-6    # hydraulic diffusivity
D_VAPOR   = 1e-5    # vapor diffusivity
KT        = 2e-5    # thermal diffusivity (corrected from 0.5 in V10.2)
D_GAS     = 2e-5    # gas diffusivity
D_MYCO    = 1e-4    # signal diffusivity
D_M       = 1e-6    # mycelium diffusivity
CHI       = 1e-3    # signal convection coefficient

# --- Other constants ---
M_MAX = 1.0         # maximum mycelium capacity
RECORD_STEP = 20   # record every 20 steps (2 h)

# --- Corrections from V10.2 to V10.3 (for traceability) ---
# 1. Y_B read from params (no hard-coded 0.6).
# 2. Thermal equation includes Q_bio and Q_rad.
# 3. Biomass initialised only inside M_photo.
# 4. TAU_BASE defined as a global constant.
# 5. Mask diagnostics added (report_masks).
# 6. Cumulative mutation of P avoided (P_run copy).
# 7. Crew-multiplier sweep: run_simulation(crew_mult) + main loop.
# 8. kT corrected to 2e-5 (physical air thermal diffusivity).