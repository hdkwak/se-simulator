# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Python interpreter: `/opt/anaconda3/envs/ocdenv/bin/python` (3.12)
- `uv` is **not** installed — use pip and the interpreter above directly

## Commands

```bash
# Run all tests
/opt/anaconda3/envs/ocdenv/bin/pytest tests/ -v

# Run a single test file
/opt/anaconda3/envs/ocdenv/bin/pytest tests/test_rcwa_engine.py -v

# Run a single test by name
/opt/anaconda3/envs/ocdenv/bin/pytest tests/test_rcwa_engine.py::test_energy_conservation_lossless -v

# Lint
/opt/anaconda3/envs/ocdenv/bin/ruff check src/
/opt/anaconda3/envs/ocdenv/bin/ruff check src/ --fix

# Run the GUI application
/opt/anaconda3/envs/ocdenv/bin/python -m se_simulator
```

## Architecture

The project is a Spectroscopic Ellipsometry simulator using RCWA (Rigorous Coupled-Wave Analysis / Fourier Modal Method). It is built in 5 planned steps; Steps 1 and 2 are complete.

### Data flow

```
YAML configs → Pydantic schemas → RCWAEngine.run()
                                       ↓
MaterialDatabase (optical constants)
modes.py (Kx, Ky, kz wavevectors)
fourier.py (ε Fourier decomposition + Toeplitz/Li matrices)
eigensolver.py (P·Q eigenvalue → W, V, kz per layer)
smatrix.py (Redheffer star product → global S-matrix)
                                       ↓
                                 RCWAResult
                    (jones_reflection, jones_transmission,
                     r_orders, t_orders, energy_conservation)
```

### Key physics conventions

- **kz branch cut**: `Im(kz) ≥ 0` enforced via `np.where(np.imag(kz) < 0, -kz, kz)`
- **Phase propagation**: `exp(+1j * kz * k0 * d)` — positive sign so that `|phase| < 1` for lossy/evanescent modes when `Im(kz) > 0`
- **Field vector layout**: `[Ex_0..Ex_{n-1}, Ey_0..Ey_{n-1}]`; p-index = `zero_order_idx`, s-index = `zero_order_idx + n_modes`
- **Jones matrix indexing**: `jones_r[1,1]` = Rpp, `jones_r[0,0]` = Rss
- **Li factorization**: `E_xx = [Toeplitz(1/ε)]⁻¹` (inverse rule for normal component), `E_yy = Toeplitz(ε)` (direct rule). In `assemble_coupled_wave_matrix(e_xx, e_yx, e_yx, e_yx, ...)` the 4th arg is always `e_yx` (Toeplitz(ε)), **not** `e_xx`.
- **Order indexing**: row-major; inner loop over m (x), outer loop over n (y): `[(m, n) for n in range(-Ny, Ny+1) for m in range(-Nx, Nx+1)]`

### Module map

| Module | Purpose |
|---|---|
| `config/schemas.py` | Pydantic v2 models: `SystemConfig`, `SampleConfig`, `SimConditions`, `GratingLayer`, `MaterialSpec`, `WavelengthSpec`, `FittingConditions` |
| `config/manager.py` | `ConfigManager` — load/save YAML configs, `get_wavelengths()`, `ConfigValidationError` |
| `materials/database.py` | `MaterialDatabase` — resolve `MaterialSpec`, return `ε(λ)` arrays |
| `materials/interpolator.py` | `MaterialEntry`, `load_csv_library` (header: `wavelength_nm,n,k`) |
| `materials/models.py` | Dispersion models: `cauchy`, `sellmeier`, `drude`, `tauc_lorentz` |
| `rcwa/modes.py` | `make_order_indices`, `make_kx/ky_matrix`, `make_kz_array`, `free_space_matrices` (W0=I, V0 from Q@diag(1/kz0)) |
| `rcwa/fourier.py` | `rasterize_layer`, `compute_epsilon_fourier_2d`, `build_toeplitz_matrix`, `build_li_matrices` |
| `rcwa/eigensolver.py` | `assemble_coupled_wave_matrix` → `(PQ, Q)`, `solve_eigenproblem` → `(W, V, kz)`, `solve_uniform_layer`, `is_uniform_layer` |
| `rcwa/smatrix.py` | `build_layer_smatrix`, `build_semiinfinite_smatrix`, `redheffer_star_product`, `propagate_global_smatrix`, `extract_jones_matrices`, `compute_diffraction_efficiencies` |
| `rcwa/engine.py` | `RCWAEngine.run()` (serial/parallel via `ProcessPoolExecutor`), `run_single()`, `convergence_test()` (requires Step 3) |
| `rcwa/results.py` | `RCWAResult` dataclass |
| `ui/main_window.py` | PySide6 `MainWindow` skeleton (3-panel splitter) |

### Config system

Three independent YAML files validated by Pydantic:
- `system_config.yaml` → `SystemConfig` (instrument optics, calibration)
- `sample_config.yaml` → `SampleConfig` (layers, materials, unit cell)
- `sim_conditions.yaml` → `SimConditions` (AOI, wavelengths, harmonics, fitting)

Defaults live in `src/se_simulator/config/defaults/`. `ConfigManager` writes to `configs/` at the repo root.

### Ruff rules

`select = ["E", "F", "W", "I", "N", "UP", "B", "SIM"]`, `ignore = ["E501"]`, line length 100. Notable suppressions:
- `CalibrationErrors` fields (`delta_P_deg` etc.) carry `# noqa: N815`
- Physics-domain uppercase variables must be renamed to lowercase (N806)

### Steps completed / pending

- **Step 1** ✅ — Project scaffold, config system, materials engine, UI skeleton
- **Step 2** ✅ — Full RCWA engine (modes, Fourier, eigensolver, S-matrix, orchestrator), 24 tests pass
- **Step 3** 🔲 — Ellipsometer model (`se_simulator/ellipsometer/`), psi/delta computation (`prcsa.py`)
- **Step 4** 🔲 — Fitting engine (`se_simulator/fitting/`)
- **Step 5** 🔲 — Complete PySide6 UI
