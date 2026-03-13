"""Library generation, storage, and loading for the SE-RCWA fitting engine."""

from __future__ import annotations

import itertools
import json
import re
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np

import se_simulator
from se_simulator.config.schemas import SampleConfig, SimConditions, SystemConfig
from se_simulator.ellipsometer.signals import EllipsometryResult


@dataclass
class ParameterSpec:
    """Defines one floating parameter in the library grid."""

    name: str
    min_value: float
    max_value: float
    n_points: int

    @property
    def values(self) -> np.ndarray:
        return np.linspace(self.min_value, self.max_value, self.n_points)


@dataclass
class LibrarySpec:
    """Complete specification for a library."""

    parameters: list[ParameterSpec]
    system_config_snapshot: dict
    sample_config_snapshot: dict
    sim_conditions_snapshot: dict
    created_at: str
    rcwa_version: str
    n_wavelengths: int
    wavelengths_nm: list[float]
    signals: list[str]


def _parse_path(path_str: str) -> list[str | int]:
    """Parse a dot-path string into a list of attribute names and indices."""
    parts: list[str | int] = []
    for segment in path_str.split("."):
        m = re.match(r"^(\w+)\[(\d+)\]$", segment)
        if m:
            parts.append(m.group(1))
            parts.append(int(m.group(2)))
        else:
            m2 = re.match(r"^(\w+)\[(\w+)\]$", segment)
            if m2:
                parts.append(m2.group(1))
                parts.append(m2.group(2))  # string dict key
            else:
                parts.append(segment)
    return parts


def apply_params(
    base_sample: SampleConfig,
    param_specs: list[ParameterSpec],
    values: tuple[float, ...],
) -> SampleConfig:
    """Apply parameter values to a deep-copied SampleConfig.

    Uses dot-path notation: ``layers[0].thickness_nm`` sets
    ``sample.layers[0].thickness_nm = value``.
    """
    clone = base_sample.model_copy(deep=True)
    for spec, val in zip(param_specs, values, strict=True):
        parts = _parse_path(spec.name)
        obj: object = clone
        for part in parts[:-1]:
            if isinstance(part, int):
                obj = obj[part]  # type: ignore[index]
            else:
                try:
                    obj = getattr(obj, part)
                except AttributeError:
                    obj = obj[part]  # type: ignore[index]
        last = parts[-1]
        if isinstance(last, int):
            obj[last] = val  # type: ignore[index]
        else:
            try:
                setattr(obj, last, val)
            except AttributeError:
                obj[last] = val  # type: ignore[index]
    return clone


def _spec_to_dict(spec: LibrarySpec) -> dict:
    return {
        "parameters": [
            {
                "name": p.name,
                "min_value": p.min_value,
                "max_value": p.max_value,
                "n_points": p.n_points,
            }
            for p in spec.parameters
        ],
        "system_config_snapshot": spec.system_config_snapshot,
        "sample_config_snapshot": spec.sample_config_snapshot,
        "sim_conditions_snapshot": spec.sim_conditions_snapshot,
        "created_at": spec.created_at,
        "rcwa_version": spec.rcwa_version,
        "n_wavelengths": spec.n_wavelengths,
        "wavelengths_nm": spec.wavelengths_nm,
        "signals": spec.signals,
    }


def _dict_to_spec(d: dict) -> LibrarySpec:
    params = [
        ParameterSpec(
            name=p["name"],
            min_value=p["min_value"],
            max_value=p["max_value"],
            n_points=p["n_points"],
        )
        for p in d["parameters"]
    ]
    return LibrarySpec(
        parameters=params,
        system_config_snapshot=d["system_config_snapshot"],
        sample_config_snapshot=d["sample_config_snapshot"],
        sim_conditions_snapshot=d["sim_conditions_snapshot"],
        created_at=d["created_at"],
        rcwa_version=d["rcwa_version"],
        n_wavelengths=d["n_wavelengths"],
        wavelengths_nm=d["wavelengths_nm"],
        signals=d["signals"],
    )


class LibraryStore:
    """HDF5-backed storage for a pre-computed spectral library."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def create(self, spec: LibrarySpec, overwrite: bool = False) -> None:
        """Create a new HDF5 library file with pre-allocated datasets."""
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"Library already exists at {self.path}")

        n_entries = 1
        for p in spec.parameters:
            n_entries *= p.n_points
        n_wl = spec.n_wavelengths
        n_params = len(spec.parameters)
        chunk_rows = min(100, n_entries)

        with h5py.File(self.path, "w") as f:
            f.attrs["spec"] = json.dumps(_spec_to_dict(spec))

            params_ds = f.create_dataset(
                "parameters",
                shape=(n_entries, n_params),
                dtype="float64",
                chunks=(chunk_rows, max(1, n_params)),
                compression="gzip",
                compression_opts=4,
            )
            params_ds[:] = np.nan

            grp = f.create_group("spectra")
            for sig in spec.signals:
                grp.create_dataset(
                    sig,
                    shape=(n_entries, n_wl),
                    dtype="float32",
                    chunks=(chunk_rows, n_wl),
                    compression="gzip",
                    compression_opts=4,
                    fillvalue=np.nan,
                )

    def write_entry(
        self, index: int, params: np.ndarray, result: EllipsometryResult
    ) -> None:
        """Write one library entry at the given flat index."""
        sig_map = {
            "psi": result.psi_deg,
            "delta": result.delta_deg,
            "alpha": result.alpha,
            "beta": result.beta,
            "chi": result.chi,
            "xi": result.xi,
        }
        with h5py.File(self.path, "a") as f:
            f["parameters"][index] = params
            for sig_name, sig_data in sig_map.items():
                key = f"spectra/{sig_name}"
                if key in f:
                    f[key][index] = sig_data.astype("float32")

    def read_all(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Load the full library into memory.

        Returns ``(parameters, spectra_dict)`` where ``parameters`` has shape
        ``(N, N_params)`` and each spectra array has shape ``(N, N_wavelengths)``.
        """
        with h5py.File(self.path, "r") as f:
            parameters = f["parameters"][:]
            spectra: dict[str, np.ndarray] = {}
            for sig_name in f["spectra"]:
                spectra[sig_name] = f[f"spectra/{sig_name}"][:]
        return parameters, spectra

    def get_spec(self) -> LibrarySpec:
        """Load the LibrarySpec from the HDF5 file attributes."""
        with h5py.File(self.path, "r") as f:
            spec_dict = json.loads(f.attrs["spec"])
        return _dict_to_spec(spec_dict)

    def validate(self, sample_fraction: float = 0.01) -> list[str]:
        """Validate a random sample of entries. Returns a list of warning strings."""
        warnings: list[str] = []
        with h5py.File(self.path, "r") as f:
            n = f["parameters"].shape[0]
            n_sample = max(1, int(n * sample_fraction))
            indices = np.random.choice(n, size=n_sample, replace=False)
            params_all = f["parameters"][:]
            has_alpha = "spectra/alpha" in f
            has_beta = "spectra/beta" in f
            for idx in indices:
                if np.any(np.isnan(params_all[idx])):
                    continue
                if has_alpha and has_beta:
                    alpha = f["spectra/alpha"][idx]
                    beta = f["spectra/beta"][idx]
                    energy_proxy = float(np.mean(alpha**2 + beta**2))
                    if energy_proxy > 1.005:
                        warnings.append(
                            f"Entry {idx}: energy proxy {energy_proxy:.4f} > 1.005"
                        )
        return warnings

    @property
    def n_entries(self) -> int:
        """Total number of pre-allocated entries."""
        with h5py.File(self.path, "r") as f:
            return int(f["parameters"].shape[0])

    @property
    def is_complete(self) -> bool:
        """True if every entry has been written (no NaN rows in /parameters)."""
        with h5py.File(self.path, "r") as f:
            params = f["parameters"][:]
        return not bool(np.any(np.isnan(params)))


# ---------------------------------------------------------------------------
# Module-level worker — must be picklable (top-level function)
# ---------------------------------------------------------------------------


def _library_worker(args: tuple) -> tuple[int, list[float], dict[str, list[float]]]:
    """Worker for one library entry. Runs RCWA + ellipsometer for a parameter set.

    ``args = (index, param_values, sample_dict, system_dict, sim_dict, signals, param_specs_data)``
    Returns ``(index, param_values, spectra_dict)``.
    """
    (
        index,
        param_values,
        sample_dict,
        system_dict,
        sim_dict,
        signals,
        param_specs_data,
    ) = args

    from se_simulator.config.schemas import SampleConfig, SimConditions, SystemConfig
    from se_simulator.ellipsometer.prcsa import compute_spectrum
    from se_simulator.fitting.library import ParameterSpec, apply_params
    from se_simulator.materials.database import MaterialDatabase
    from se_simulator.rcwa.engine import RCWAEngine

    sample = SampleConfig.model_validate(sample_dict)
    system = SystemConfig.model_validate(system_dict)
    sim = SimConditions.model_validate(sim_dict)

    param_specs = [
        ParameterSpec(
            name=p["name"],
            min_value=p["min_value"],
            max_value=p["max_value"],
            n_points=p["n_points"],
        )
        for p in param_specs_data
    ]

    sample_mod = apply_params(sample, param_specs, tuple(param_values))
    db = MaterialDatabase()
    engine = RCWAEngine(db)
    rcwa_result = engine.run(sample_mod, sim)
    ell = compute_spectrum(rcwa_result, system)

    full_sig = {
        "psi": ell.psi_deg.tolist(),
        "delta": ell.delta_deg.tolist(),
        "alpha": ell.alpha.tolist(),
        "beta": ell.beta.tolist(),
        "chi": ell.chi.tolist(),
        "xi": ell.xi.tolist(),
    }
    sig_data = {k: v for k, v in full_sig.items() if k in signals}
    return (index, list(param_values), sig_data)


def _reconstruct_result(
    sig_data: dict[str, list[float]], wavelengths_nm: list[float]
) -> EllipsometryResult:
    """Reconstruct an EllipsometryResult from serialised signal data."""
    wl = np.array(wavelengths_nm)
    n = len(wl)
    zeros = np.zeros(n)
    return EllipsometryResult(
        wavelengths_nm=wl,
        psi_deg=np.array(sig_data.get("psi", zeros)),
        delta_deg=np.array(sig_data.get("delta", zeros)),
        alpha=np.array(sig_data.get("alpha", zeros)),
        beta=np.array(sig_data.get("beta", zeros)),
        chi=np.array(sig_data.get("chi", zeros)),
        xi=np.array(sig_data.get("xi", zeros)),
        jones_reflection=np.zeros((n, 2, 2), dtype=complex),
        energy_conservation=np.zeros(n),
    )


class LibraryGenerator:
    """Generates all library entries, optionally in parallel."""

    def __init__(
        self,
        spec: LibrarySpec,
        store: LibraryStore,
        n_workers: int | None = None,
    ) -> None:
        self.spec = spec
        self.store = store
        self.n_workers = n_workers

    def generate(
        self,
        progress_callback: Callable[[int, int], None] | None = None,
        resume: bool = True,
    ) -> None:
        """Generate all library entries.

        If ``resume=True``, previously completed entries are skipped.
        """
        grids = [p.values for p in self.spec.parameters]
        all_combinations = list(itertools.product(*grids))
        total = len(all_combinations)

        if resume and self.store.path.exists():
            with h5py.File(self.store.path, "r") as f:
                existing = f["parameters"][:]
            done_mask = ~np.any(np.isnan(existing), axis=1)
        else:
            done_mask = np.zeros(total, dtype=bool)

        param_specs_data = [
            {
                "name": p.name,
                "min_value": p.min_value,
                "max_value": p.max_value,
                "n_points": p.n_points,
            }
            for p in self.spec.parameters
        ]

        pending = [
            (
                idx,
                list(combo),
                self.spec.sample_config_snapshot,
                self.spec.system_config_snapshot,
                self.spec.sim_conditions_snapshot,
                self.spec.signals,
                param_specs_data,
            )
            for idx, combo in enumerate(all_combinations)
            if not done_mask[idx]
        ]

        completed = int(np.sum(done_mask))

        if not pending:
            if progress_callback:
                progress_callback(total, total)
            return

        use_serial = self.n_workers == 1 or total == 1

        if use_serial:
            for args in pending:
                idx, param_vals, sig_data = _library_worker(args)
                result = _reconstruct_result(sig_data, self.spec.wavelengths_nm)
                self.store.write_entry(idx, np.array(param_vals), result)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
        else:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                future_map = {
                    executor.submit(_library_worker, args): args[0] for args in pending
                }
                for future in as_completed(future_map):
                    idx, param_vals, sig_data = future.result()
                    result = _reconstruct_result(sig_data, self.spec.wavelengths_nm)
                    self.store.write_entry(idx, np.array(param_vals), result)
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)

    def _apply_params(
        self,
        base_sample: SampleConfig,
        param_specs: list[ParameterSpec],
        values: tuple[float, ...],
    ) -> SampleConfig:
        """Thin wrapper around the module-level ``apply_params``."""
        return apply_params(base_sample, param_specs, values)

    def estimate_time(self) -> float:
        """Estimate total generation time by timing a single entry."""
        sample = SampleConfig.model_validate(self.spec.sample_config_snapshot)
        system = SystemConfig.model_validate(self.spec.system_config_snapshot)
        sim = SimConditions.model_validate(self.spec.sim_conditions_snapshot)

        from se_simulator.ellipsometer.prcsa import compute_spectrum
        from se_simulator.materials.database import MaterialDatabase
        from se_simulator.rcwa.engine import RCWAEngine

        db = MaterialDatabase()
        engine = RCWAEngine(db)
        grids = [p.values for p in self.spec.parameters]
        first_combo = tuple(g[0] for g in grids)
        sample_mod = apply_params(sample, self.spec.parameters, first_combo)

        t0 = time.perf_counter()
        rcwa_result = engine.run(sample_mod, sim)
        compute_spectrum(rcwa_result, system)
        elapsed = time.perf_counter() - t0

        total = 1
        for p in self.spec.parameters:
            total *= p.n_points
        return elapsed * total


def build_library_spec(
    parameters: list[ParameterSpec],
    sample: SampleConfig,
    system: SystemConfig,
    sim: SimConditions,
    signals: list[str] | None = None,
) -> LibrarySpec:
    """Convenience helper: build a ``LibrarySpec`` from config objects.

    Resolves wavelengths using the ``sim.wavelengths`` spec.
    """
    from se_simulator.config.manager import ConfigManager

    wls = ConfigManager().get_wavelengths(sim.wavelengths)
    if signals is None:
        signals = list(sim.fitting.fit_signals)

    return LibrarySpec(
        parameters=parameters,
        system_config_snapshot=system.model_dump(),
        sample_config_snapshot=sample.model_dump(),
        sim_conditions_snapshot=sim.model_dump(),
        created_at=datetime.now(UTC).isoformat(),
        rcwa_version=se_simulator.__version__,
        n_wavelengths=len(wls),
        wavelengths_nm=wls.tolist(),
        signals=signals,
    )
