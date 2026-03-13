"""RCWAEngine: top-level orchestrator for the RCWA simulation."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from se_simulator.config.schemas import SampleConfig, SimConditions
from se_simulator.materials.database import MaterialDatabase
from se_simulator.rcwa.modes import (
    free_space_matrices,
    make_kx_matrix,
    make_ky_matrix,
    make_kz_array,
    make_order_indices,
)
from se_simulator.rcwa.results import RCWAResult
from se_simulator.rcwa.smatrix import (
    compute_diffraction_efficiencies,
    extract_jones_matrices,
    propagate_global_smatrix,
)

logger = logging.getLogger(__name__)


def _resolve_sample_materials(sample: SampleConfig, materials_db: MaterialDatabase) -> None:
    """Pre-register all materials referenced in the sample into the database."""
    for spec in sample.materials.values():
        if spec.name not in materials_db._entries:
            materials_db.resolve(spec)


def _compute_single(
    sample: SampleConfig,
    sim: SimConditions,
    materials_db: MaterialDatabase,
    wavelength_nm: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Core computation for one wavelength. Returns (jones_r, jones_t, R_total, T_total)."""
    k0 = 2.0 * np.pi / wavelength_nm
    wl_arr = np.array([wavelength_nm])

    eps_sup = materials_db.get_epsilon(sample.superstrate_material, wl_arr)[0]
    eps_sub = materials_db.get_epsilon(sample.substrate_material, wl_arr)[0]

    n_sup = np.sqrt(np.real(eps_sup))
    aoi_rad = np.radians(sim.aoi_deg)
    az_rad = np.radians(sim.azimuth_deg)
    kx_inc = float(n_sup * np.sin(aoi_rad) * np.cos(az_rad))
    ky_inc = float(n_sup * np.sin(aoi_rad) * np.sin(az_rad))

    order_indices = make_order_indices(sim.n_harmonics_x, sim.n_harmonics_y)
    n_modes = len(order_indices)

    kx = make_kx_matrix(order_indices, kx_inc, sample.Lx_nm, k0)
    ky = make_ky_matrix(order_indices, ky_inc, sample.Ly_nm, k0)
    w0, v0 = free_space_matrices(kx, ky)

    s_global = propagate_global_smatrix(
        sample.layers, sample, sim, materials_db,
        wavelength_nm, kx, ky, w0, v0, order_indices,
    )

    zero_order_idx = order_indices.index((0, 0))
    kx_d = np.diag(kx)
    ky_d = np.diag(ky)
    kz_inc_arr = make_kz_array(kx_d, ky_d, eps_sup)
    kz_sub_arr = make_kz_array(kx_d, ky_d, eps_sub)

    jones_r, jones_t = extract_jones_matrices(
        s_global, order_indices, zero_order_idx,
        kx_inc, ky_inc,
        kz_inc_arr[zero_order_idx], kz_sub_arr[zero_order_idx],
        eps_sup, eps_sub, n_modes,
    )

    _, _, r_total, t_total = compute_diffraction_efficiencies(
        s_global, order_indices, kx_d, ky_d,
        kz_inc_arr, kz_sub_arr, eps_sup, eps_sub,
    )

    return jones_r, jones_t, r_total, t_total


# ---------------------------------------------------------------------------
# Module-level worker for ProcessPoolExecutor (must be picklable)
# ---------------------------------------------------------------------------

def _rcwa_worker(args: tuple) -> tuple:
    """Worker function for parallel RCWA computation.

    args = (i, sample_dict, sim_dict, wavelength_nm)
    Returns (i, jones_r, jones_t, R_total, T_total).
    """
    i, sample_dict, sim_dict, wavelength_nm = args

    sample = SampleConfig.model_validate(sample_dict)
    sim = SimConditions.model_validate(sim_dict)

    db = MaterialDatabase()
    _resolve_sample_materials(sample, db)

    jones_r, jones_t, r_total, t_total = _compute_single(sample, sim, db, wavelength_nm)
    return (i, jones_r, jones_t, r_total, t_total)


# ---------------------------------------------------------------------------
# RCWAEngine
# ---------------------------------------------------------------------------

class RCWAEngine:
    """Top-level RCWA simulation orchestrator."""

    def __init__(self, materials_db: MaterialDatabase) -> None:
        self.materials_db = materials_db
        self._per_wavelength_time: float | None = None

    def run(
        self,
        sample: SampleConfig,
        sim: SimConditions,
        wavelengths_nm: np.ndarray | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> RCWAResult:
        """Run the full simulation across all wavelengths.

        Automatically selects TMM for uniform stacks and RCWA for periodic
        structures, unless overridden by ``sim.engine_override``.
        """
        from se_simulator.config.manager import ConfigManager
        from se_simulator.rcwa.dispatcher import select_engine
        from se_simulator.rcwa.tmm import compute_tmm

        if wavelengths_nm is None:
            wavelengths_nm = ConfigManager().get_wavelengths(sim.wavelengths)

        engine_override = getattr(sim, "engine_override", "auto")
        engine = select_engine(sample, engine_override)
        logger.info("[Engine] Selected: %s", engine.upper())

        _resolve_sample_materials(sample, self.materials_db)

        if engine == "tmm":
            return self._run_tmm(sample, sim, wavelengths_nm, progress_callback, compute_tmm)

        # --- RCWA path (original code below) ---
        n_wl = len(wavelengths_nm)
        order_indices = make_order_indices(sim.n_harmonics_x, sim.n_harmonics_y)
        n_orders = len(order_indices)

        jones_r_all = np.zeros((n_wl, 2, 2), dtype=complex)
        jones_t_all = np.zeros((n_wl, 2, 2), dtype=complex)
        r_total_all = np.zeros(n_wl)
        t_total_all = np.zeros(n_wl)
        r_orders_all: np.ndarray | None = None
        t_orders_all: np.ndarray | None = None

        if sim.parallel_wavelengths and n_wl > 1:
            sample_dict = sample.model_dump()
            sim_dict = sim.model_dump()
            all_args = [
                (i, sample_dict, sim_dict, float(wl))
                for i, wl in enumerate(wavelengths_nm)
            ]
            completed = 0
            with ProcessPoolExecutor(max_workers=None) as executor:
                future_map = {executor.submit(_rcwa_worker, a): a[0] for a in all_args}
                for future in as_completed(future_map):
                    idx, jr, jt, rt, tt = future.result()
                    jones_r_all[idx] = jr
                    jones_t_all[idx] = jt
                    r_total_all[idx] = rt
                    t_total_all[idx] = tt
                    completed += 1
                    if progress_callback is not None:
                        progress_callback(completed / n_wl)
        else:
            for i, wl in enumerate(wavelengths_nm):
                jr, jt, rt, tt = _compute_single(sample, sim, self.materials_db, float(wl))
                jones_r_all[i] = jr
                jones_t_all[i] = jt
                r_total_all[i] = rt
                t_total_all[i] = tt
                if progress_callback is not None:
                    progress_callback((i + 1) / n_wl)

        if sim.output_orders:
            r_orders_all = np.zeros((n_wl, n_orders))
            t_orders_all = np.zeros((n_wl, n_orders))
            # Recompute per-order efficiencies (cheap reuse of existing results)
            for i, wl in enumerate(wavelengths_nm):
                wl_arr = np.array([float(wl)])
                eps_sup = self.materials_db.get_epsilon(sample.superstrate_material, wl_arr)[0]
                eps_sub = self.materials_db.get_epsilon(sample.substrate_material, wl_arr)[0]
                k0 = 2.0 * np.pi / float(wl)
                n_sup = np.sqrt(np.real(eps_sup))
                aoi_rad = np.radians(sim.aoi_deg)
                az_rad = np.radians(sim.azimuth_deg)
                kx_inc = float(n_sup * np.sin(aoi_rad) * np.cos(az_rad))
                ky_inc = float(n_sup * np.sin(aoi_rad) * np.sin(az_rad))
                kx = make_kx_matrix(order_indices, kx_inc, sample.Lx_nm, k0)
                ky = make_ky_matrix(order_indices, ky_inc, sample.Ly_nm, k0)
                w0, v0 = free_space_matrices(kx, ky)
                s_global = propagate_global_smatrix(
                    sample.layers, sample, sim, self.materials_db,
                    float(wl), kx, ky, w0, v0, order_indices,
                )
                kx_d = np.diag(kx)
                ky_d = np.diag(ky)
                kz_inc_arr = make_kz_array(kx_d, ky_d, eps_sup)
                kz_sub_arr = make_kz_array(kx_d, ky_d, eps_sub)
                ro, to, _, _ = compute_diffraction_efficiencies(
                    s_global, order_indices, kx_d, ky_d,
                    kz_inc_arr, kz_sub_arr, eps_sup, eps_sub,
                )
                r_orders_all[i] = ro
                t_orders_all[i] = to

        return RCWAResult(
            wavelengths_nm=wavelengths_nm,
            jones_reflection=jones_r_all,
            jones_transmission=jones_t_all,
            r_orders=r_orders_all,
            t_orders=t_orders_all,
            order_indices=order_indices,
            energy_conservation=r_total_all + t_total_all,
        )

    def _run_tmm(
        self,
        sample: SampleConfig,
        sim: SimConditions,
        wavelengths_nm: np.ndarray,
        progress_callback: Callable[[float], None] | None,
        compute_tmm: Callable,
    ) -> RCWAResult:
        """Run TMM and return an RCWAResult-compatible object."""
        jones_r, jones_t = compute_tmm(
            sample, self.materials_db, wavelengths_nm, sim.aoi_deg, sim.azimuth_deg
        )
        if progress_callback is not None:
            progress_callback(1.0)

        # Energy conservation from jones matrices
        rss = jones_r[:, 0, 0]
        rpp = jones_r[:, 1, 1]
        tss = jones_t[:, 0, 0]
        tpp = jones_t[:, 1, 1]
        r_total = 0.5 * (np.abs(rss) ** 2 + np.abs(rpp) ** 2)
        t_total = 0.5 * (np.abs(tss) ** 2 + np.abs(tpp) ** 2)

        return RCWAResult(
            wavelengths_nm=wavelengths_nm,
            jones_reflection=jones_r,
            jones_transmission=jones_t,
            r_orders=None,
            t_orders=None,
            order_indices=None,
            energy_conservation=r_total + t_total,
        )

    def run_single(
        self,
        sample: SampleConfig,
        sim: SimConditions,
        wavelength_nm: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run RCWA for a single wavelength. Returns (jones_r, jones_t) each (2,2)."""
        _resolve_sample_materials(sample, self.materials_db)
        jr, jt, _rt, _tt = _compute_single(sample, sim, self.materials_db, wavelength_nm)
        return jr, jt

    def convergence_test(
        self,
        sample: SampleConfig,
        sim: SimConditions,
        wavelength_nm: float,
        n_range: range = range(1, 10),
    ) -> dict[str, np.ndarray]:
        """Sweep n_harmonics and return {'N', 'psi', 'delta'} arrays."""
        try:
            from se_simulator.ellipsometer.prcsa import compute_psi_delta
        except ImportError as exc:
            raise NotImplementedError(
                "Ellipsometer model not yet implemented — complete Step 3 first."
            ) from exc


        ns = list(n_range)
        psis = []
        deltas = []
        for n in ns:
            sim_n = sim.model_copy(update={"n_harmonics_x": n, "n_harmonics_y": n})
            jr, _ = self.run_single(sample, sim_n, wavelength_nm)
            psi, delta = compute_psi_delta(jr)
            psis.append(psi)
            deltas.append(delta)

        return {
            "N": np.array(ns),
            "psi": np.array(psis),
            "delta": np.array(deltas),
        }

    def estimate_time(self, sample: SampleConfig, sim: SimConditions) -> float:
        """Estimate total simulation time in seconds based on a single-wavelength timing."""
        from se_simulator.config.manager import ConfigManager

        if self._per_wavelength_time is None:
            wls = ConfigManager().get_wavelengths(sim.wavelengths)
            first_wl = float(wls[0])
            _resolve_sample_materials(sample, self.materials_db)
            t0 = time.perf_counter()
            _compute_single(sample, sim, self.materials_db, first_wl)
            self._per_wavelength_time = time.perf_counter() - t0

        wls = ConfigManager().get_wavelengths(sim.wavelengths)
        return self._per_wavelength_time * len(wls)
