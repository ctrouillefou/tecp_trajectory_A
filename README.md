
# TECP Trajectory A — C2 Reference Configuration

## Description

This repository contains the **complete code, parameters, datasets, and analysis outputs** supporting the manuscript:

**Carbon-budget dynamics of a closed ecological reactor model. I. Reference configuration and functionally negligible mycorrhizal cooperation (TECP Trajectory A)**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24438100.svg)](https://doi.org/10.5281/zenodo.24438100)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Submitted to *Life Sciences in Space Research* (Elsevier).

> **DOI note.** `10.5281/zenodo.24438100` is real but *in progress* at the time of writing. It will be finalised upon Zenodo deposition. Until then, treat it as a placeholder.

The study presents a numerical investigation of **Trajectory A** of the **Toroidal Ecological Coupled Platform (TECP)** model — a three-dimensional reaction–diffusion model of a closed ecological reactor with two biomes (continental and marsh), a mycelium compartment, a mycorrhizal signalling variable, and a crew respiratory load.

The reference configuration **C2** is integrated for 30 days on a 17³ grid with a 360 s timestep. This repository accompanies the manuscript and provides everything needed to reproduce the figures, tables, and numerical controls reported in the paper.

---

## Reference configuration C2

- **Model:** TECP, Trajectory A
- **Grid:** 17³ = 4913 cells, Δx = Δy = Δz = 0.05 m, V = 0.614 m³
- **Timestep:** Δt = 360 s (7200 steps for 30 d)
- **Duration:** 30 d (720 h); output every 2 h → 360 records
- **Solver:** FiPy `DefaultSolver`, tolerance 1e-12, max 5000 iterations (applied to **every** equation — required because O₂, biomass, θ vary by < 10⁻⁶ per step)
- **Key parameters (Table S1):**
  - `P_max = 5.5556 × 10⁻⁶ s⁻¹` (base × 2)
  - `R_maint = 5.5556 × 10⁻⁷ s⁻¹` (base ÷ 1)
  - `k_CO₂ = 1 × 10⁻³` (dimensionless)
  - `Y_B = 1.0` (dimensionless coupling coefficient)
  - `RQ = 0.85`
  - `crew_mult = 0.10` → `U_crew = 9.722 × 10⁻¹⁰ s⁻¹`, `Q_crew = 8.263 × 10⁻¹⁰ s⁻¹`
  - `S_INIT = 1 × 10⁻⁵`, `M_INIT = 0.1`
  - `B_0 = 0.1 · (0.8 χ_photo + 0.2 χ_soil)`
  - `τ_base = 3600 s`
  - `scale factor 2.625 = 168/64` applied to `k_prod`, `k_basal`, `λ_S`

See `params_real.json` and `config.py` for the complete parameter set; Table S1 of the supplementary material for the published version.

---

## Environment

- Python 3.10.12
- FiPy 4.0.3
- NumPy 2.2.6
- pandas, matplotlib

**Installation:**
```bash
pip install fipy==4.0.3 numpy==2.2.6 pandas matplotlib
```

---

## Repository structure

```text
.
├── README.md                            # This file
├── LICENSE                              # GPL-v3 full text
├── params_real.json                     # C2 parameter set (base + metadata)
│
├── config.py                            # Global constants (mesh, DT, solver, crew)
├── biology.py                           # Reaction terms (P, R, S, M, C_coop, etc.)
├── equations.py                         # FiPy equation builders
├── build_masks.py                       # Boolean biome masks
├── diagnostics.py                       # Time-series CSV writer, plot helper, balance diagnostics
├── logging_setup.py                     # Provenance logger (used by test_porte1bis and test_porte3)
│
├── run_simulation.py                    # Main simulation entry point (Step 1)
├── run_sensitivity.py                   # F2 sensitivity sweep (Step 5a)
├── run_spatial_convergence.py           # F4 12³ runs (Step 6a)
│
├── test_porte1bis_matter.py             # 3-day matter balance (Step 3)
├── test_porte2_dt_convergence.py        # F1 temporal convergence (Step 2)
├── test_porte3_B2_audit.py              # 3-day sensitivity sweep (Step 4)
│
├── analysis_common.py                   # Shared loaders, palette, helpers
├── analysis_F2.py                       # F2 normalized sensitivities (Step 5b)
├── analysis_F4_plateau.py               # F4 plateau-centre analysis (Step 6b)
├── check_csvs.py                        # Offline verifier (Step 7) — does NOT produce CSVs
│
├── plot_style.py                        # Elsevier/LSSR figure style (Okabe-Ito, 300 dpi)
├── plot_fig1.py                         # manuscript Fig. 1 (CO₂ / O₂, 2 panels)
├── plot_fig2.py                         # manuscript Fig. 2 (T, B, S, M, 4 panels)
│
├── audit_timeseries.csv                 # Canonical C2 reference (360 records × 17 cols)
├── sensitivity_all.csv                  # F2 raw metrics (11 runs, produced by run_sensitivity.py)
│
├── analysis_F/                          # ← produced by analysis_F2.py (OUT = 'analysis_F')
│   ├── F2_sensitivity.csv
│   └── F2_sensitivity_heatmap.png / .pdf
│
├── Porte1bis/                           # ← produced by test_porte1bis_matter.py
│   ├── C2_3days_matter.csv              # 3-day matter-balance time series
│   └── matter_summary.csv               # Verdicts (C, O₂, H₂O)
│
├── Porte2/                              # ← produced by test_porte2_dt_convergence.py
│   ├── REF_DT360.csv                    # Δt = 360 s
│   ├── FIN_DT180.csv                    # Δt = 180 s
│   └── convergence_summary.csv          # Per-variable relative differences (Eq. 13)
│
├── Porte3/                              # ← produced by test_porte3_B2_audit.py
│   ├── reference.csv
│   ├── {k_CO2, Pmax, R_maint, P_stress_threshold}_{minus20,plus20}.csv   # 8 files
│   └── sensitivity_summary.csv
│
├── spatial_convergence/                 # ← produced by run_spatial_convergence.py + analysis_F4_plateau.py
│   ├── TrajA_C2_12cubed_30d_crew_0.100_{baseline,Pmax_x0.80,Pmax_x1.20,Y_B_x0.80}_*/
│   ├── spatial_convergence_12cubed.csv  # raw 12³ metrics (from run_spatial_convergence.py)
│   ├── comparison_12cubed_vs_17cubed.csv  # CANONICAL (from analysis_F4_plateau.py — see Step 6)
│   ├── plateau_analysis.csv
│   └── F4_plateau_diagnostic.png / .pdf
│
├── TrajA_C2_30d_crew_0.100_<timestamp>/ # One folder per 30-day run (17³)
│   ├── audit_timeseries.csv
│   └── audit.png                        # quick diagnostic plot (not a manuscript figure)
│
└── *.log                                # Run logs (provenance)
```

**Notes on the tree:**

- `analysis_A.py … analysis_F.py`, `plot_masks.py`, `preflight_12cubed.py`, `compute_volume_fractions.py` are **not** part of the published delivery. They were exploratory diagnostics from earlier sessions and are not required to reproduce the manuscript.
- `F2_sensitivity.csv` lives in `analysis_F/` (not at the repository root). The manuscript's Data availability section currently lists it as a root-level file; either move it or update the manuscript.
- `comparison_12cubed_vs_17cubed.csv` is written by **two** scripts: `run_spatial_convergence.py` (flat format, at the end of the script) **and** `analysis_F4_plateau.py` (rich format). Running them in the order given in Step 6 below makes `analysis_F4_plateau.py` the **last writer** — this is the canonical version referenced in Table S3.

---

## Reproduction order

Each step is independent, except for the ordering constraints noted. Total sequential wall-clock time on a single core is **~23 h** (see per-step estimates).

### 1. Reference simulation (C2, 30 d, 17³)
```bash
nohup python3 -u run_simulation.py
```
- Produces `TrajA_C2_30d_crew_0.100_<timestamp>/audit_timeseries.csv` and `audit.png`.
- The canonical reference is **copied** at the repository root as `audit_timeseries.csv`.
- Wall-clock: **~45 min** (single 30-day 17³ run; extrapolated from F2: 11 runs in 8.5 h).

### 2. Temporal convergence (F1, Δt = 360 vs 180 s)
```bash
nohup python3 -u test_porte2_dt_convergence.py > porte2_30d.log 2>&1 &
```
- Produces `Porte2/REF_DT360.csv`, `Porte2/FIN_DT180.csv`, `Porte2/convergence_summary.csv`.
- Wall-clock: **≈ 2.25 h**.

### 3. Matter balance (Porte 1bis, 3 d)
```bash
nohup python3 -u test_porte1bis_matter.py > porte1bis.log 2>&1 &
```
- Produces `Porte1bis/C2_3days_matter.csv`, `Porte1bis/matter_summary.csv`.
- Wall-clock: **≈ 45 min**.

### 4. Sensitivity sweep, 3 d (Porte 3)
```bash
nohup python3 -u test_porte3_B2_audit.py > porte3_audit.log 2>&1 &
```
- Produces `Porte3/reference.csv`, 8 perturbation CSVs, `Porte3/sensitivity_summary.csv`.
- Wall-clock: **≈ 7 h**.

### 5. Sensitivity sweep, 30 d (F2)
```bash
nohup python3 -u run_sensitivity.py        # produces sensitivity_all.csv at root
nohup python3 -u analysis_F2.py            # produces analysis_F/F2_sensitivity.csv + heatmap
```
- Wall-clock: **≈ 8.5 h** (sweep) + seconds (analysis).

### 6. Spatial resolution dependence (F4)
The 17³ counterparts are **already included** in `TrajA_C2_30d_crew_0.100_<timestamp>/`. This step generates the 12³ runs and compares them.

> **Important — script order.** `analysis_F4_plateau.py` reads **hardcoded** paths into timestamped folders (e.g. `../TrajA_C2_30d_crew_0.100_20261105_080226/audit_timeseries.csv`). If you regenerate the runs, update `INPUT_PATTERNS` at the top of `analysis_F4_plateau.py`. Running `run_spatial_convergence.py` **then** `analysis_F4_plateau.py` ensures the `comparison_12cubed_vs_17cubed.csv` written last is the canonical (rich-format) version used in Table S3.

```bash
nohup python3 -u run_spatial_convergence.py      # produces 12³ runs + spatial_convergence_12cubed.csv
nohup python3 -u analysis_F4_plateau.py          # produces plateau_analysis.csv + canonical comparison_*.csv
```
- Wall-clock: **≈ 2 h** (12³ runs) + seconds (analysis).

### 7. Auxiliary verification (optional)
```bash
python3 check_csvs.py
```
`check_csvs.py` is an **offline verifier**: it re-reads `Porte2/REF_DT360.csv`, `Porte2/FIN_DT180.csv`, `Porte3/*.csv` and recomputes the PASS/FAIL comparisons. **It does not produce any CSV.** It must be run after Steps 2–4.

> **Note on FiPy solver settings.** All simulations use a tight solver configuration `DefaultSolver(tolerance=1e-12, iterations=5000)` applied to every equation. This is necessary because O₂, biomass, and θ vary by less than 10⁻⁶ per time step; the default FiPy tolerance (~10⁻⁵) leaves them frozen at their initial values.

---

## Figure generation

The two manuscript figures are produced by dedicated scripts that read `audit_timeseries.csv` and apply the Elsevier/LSSR style in `plot_style.py`:

| Manuscript figure | Script | Contents |
|---|---|---|
| **Fig. 1** | `plot_fig1.py` | (a) CO₂ fraction, (b) O₂ fraction — 2 panels, 30 d |
| **Fig. 2** | `plot_fig2.py` | (a) T, (b) B, (c) S, (d) M — 4 panels, 30 d |

---

## Manuscript item → script → output mapping

| Manuscript item | Script | Output file |
|---|---|---|
| Table 1 (C2 initial / final, §3.1) | `run_simulation.py` | `audit_timeseries.csv` |
| Table 2 (ΔB raw, Y_B ±20 %, §3.4) | `run_sensitivity.py` | `sensitivity_all.csv` (Y_B rows) |
| Table 3 (Porte 3 sensitivity, §3.5) | `test_porte3_B2_audit.py` | `Porte3/sensitivity_summary.csv` |
| Table S1 (full parameter set) | — | `params_real.json` |
| Table S2 (F2 normalized sensitivities) | `analysis_F2.py` | `analysis_F/F2_sensitivity.csv` |
| Table S3 (F4 spatial comparison) | `analysis_F4_plateau.py` | `spatial_convergence/comparison_12cubed_vs_17cubed.csv` |
| Table S4 (controls summary) | `test_porte2_dt_convergence.py` + `test_porte1bis_matter.py` | `Porte2/convergence_summary.csv` + `Porte1bis/matter_summary.csv` |
| Eq. (1)–(12) | `equations.py` + `biology.py` | — |
| Eq. (13) ε_X | `test_porte2_dt_convergence.py` | `Porte2/convergence_summary.csv` |
| Eq. (14) carbon balance | `test_porte1bis_matter.py` | `Porte1bis/C2_3days_matter.csv` |
| Eq. (15) Q_PR | `run_sensitivity.py` (`extract_metrics()`) | `sensitivity_all.csv` |
| Eq. (16) S_p | `analysis_F2.py` | `analysis_F/F2_sensitivity.csv` |
| Porte 1bis balances (§3.2) | `test_porte1bis_matter.py` | `Porte1bis/matter_summary.csv` |
| F1 temporal convergence (§3.3) | `test_porte2_dt_convergence.py` | `Porte2/convergence_summary.csv` |
| F2 sensitivity, 30 d (§3.4) | `run_sensitivity.py` → `analysis_F2.py` | `sensitivity_all.csv` → `analysis_F/F2_sensitivity.csv` |
| Porte 3 sensitivity, 3 d (§3.5) | `test_porte3_B2_audit.py` | `Porte3/sensitivity_summary.csv` |
| F4 spatial dependence (§3.6) | `run_spatial_convergence.py` → `analysis_F4_plateau.py` | `spatial_convergence/comparison_12cubed_vs_17cubed.csv` |
| Summary of controls (§3.7) | `test_porte2_dt_convergence.py` + `test_porte1bis_matter.py` + `analysis_F2.py` + `analysis_F4_plateau.py` | aggregate of above |
| Fig. 1 (CO₂ / O₂) | `plot_fig1.py`| manuscript figure |
| Fig. 2 (T, B, S, M) | `plot_fig2.py`| manuscript figure |

---

## Numerical controls — summary

See §2.3 and §3.7 of the manuscript for details.

- **F1@30 d** (`test_porte2_dt_convergence.py`): max relative difference = 8.25 × 10⁻³ (on signal S), below the 5 % threshold.
- **Porte 1bis (3 d)** (`test_porte1bis_matter.py`): carbon 1.95 × 10⁻⁷, oxygen 1.58 × 10⁻¹⁰, water 5.88 × 10⁻¹⁰. Positivity violations: 0.
- **F2 (30 d, ±20 %)** (`analysis_F2.py`): 11 runs. Key sensitivities — `P_max` −2.44 on `t_peak`, `R_maint` +1.58 on `CO₂,max`, `crew` +1.07 on `ΔB`, `Y_B` sign reversal at ×0.80.
- **Porte 3 (3 d, ±20 %)** (`test_porte3_B2_audit.py`): `k_CO₂` 0.76, `P_max` −1.35, `R_maint` +1.41, `P_stress_threshold` ~10⁻¹⁵ (negligible).
- **F4 (12³ vs 17³)** (`analysis_F4_plateau.py`): max |Δ centre 99 %| = 1.89 %.

> These are **numerical code identities**, not independent physical validations.

---

## Known limitations

1. **`analysis_F4_plateau.py` contains hardcoded paths** to timestamped run folders. Regenerating the 12³ or 17³ runs requires editing `INPUT_PATTERNS` at the top of the script.
2. The thermal drift is dominated by a phenomenological ceiling-cooling term (`P_elt`), not a validated physical prediction.
3. The mycorrhizal cooperation term is functionally negligible in the C2 regime. This is a regime-specific finding, not a universal statement.
4. The 12³–17³ comparison is a **resolution-dependence assessment**, not a formal proof of spatial convergence.
5. `Y_B` is a dimensionless model parameter, not a physiological yield.

---

## License

GNU General Public License, version 3 (GPL-v3). See `LICENSE`.

## Citation

Trouillefou, C. M. (2026). *TECP Trajectory A — C2 reference: simulation code, data, and analysis scripts* [Software/Data]. Zenodo. https://doi.org/10.5281/zenodo.24438100 *(DOI pending)*

Companion manuscript submitted to *Life Sciences in Space Research*:
```bibtex
@article{Trouillefou2026TECP,
  title   = {Carbon-budget dynamics of a closed ecological reactor model. {I}. Reference configuration and functionally negligible mycorrhizal cooperation ({TECP} Trajectory A)},
  author  = {Trouillefou, Christophe Marcel},
  journal = {Life Sciences in Space Research},
  year    = {2026},
  note    = {Submitted},
  doi     = {10.5281/zenodo.24438100}
}
```

## Contact

Christophe Marcel Trouillefou
Lycée Blaise Pascal, Lubumbashi, République démocratique du Congo
Email: christophe.trouillefou@gmail.com
GitHub: https://github.com/ctrouillefou/tecp

---

## Acknowledgements

- Dr. Claude Plassard (INRAE, Université de Montpellier) for discussions on mycorrhizal ecology.
- Computational resources provided by a Google Cloud VM instance.

## Companion study

A companion manuscript (**Trajectory C**, TECP V10.5) extends the present model with organic-matter and phosphorus cycles and tests the nutrient-limited regime in which the mycorrhizal cooperation pathway may become dynamically relevant. The present Trajectory A provides the nutrient-replete reference baseline against which Trajectory C is compared.

