"""MaterialDatabase: resolve MaterialSpec objects and cache optical constants."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from se_simulator.config.schemas import MaterialSpec
    from se_simulator.materials.interpolator import MaterialEntry

_LIBRARY_DIR = Path(__file__).parent / "library"


class MaterialDatabase:
    """Resolve and cache optical-constant entries for named materials."""

    def __init__(self) -> None:
        self._cache: dict[str, "MaterialEntry"] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, spec: "MaterialSpec") -> "MaterialEntry":
        """Register a MaterialSpec and return the corresponding MaterialEntry."""
        from se_simulator.materials.interpolator import MaterialEntry, load_csv_library
        from se_simulator.materials.models import cauchy, drude, sellmeier, tauc_lorentz

        name = spec.name
        if name in self._cache:
            return self._cache[name]

        if spec.source == "library":
            lib_name = spec.library_name or name
            csv_path = _LIBRARY_DIR / f"{lib_name}.csv"
            entry = load_csv_library(csv_path, name=name)

        elif spec.source == "constant_nk":
            n_val = float(spec.n if spec.n is not None else 1.0)
            k_val = float(spec.k if spec.k is not None else 0.0)
            # Use a two-point wavelength range (100–3000 nm) so interpolation always works
            wl = np.array([100.0, 3000.0])
            n_arr = np.full(2, n_val)
            k_arr = np.full(2, k_val)
            entry = MaterialEntry(name=name, wavelengths_nm=wl, n=n_arr, k=k_arr)

        elif spec.source == "cauchy":
            wl = np.linspace(200.0, 3000.0, 500)
            n_arr, k_arr = cauchy(wl, spec.coefficients)
            entry = MaterialEntry(name=name, wavelengths_nm=wl, n=n_arr, k=k_arr)

        elif spec.source == "sellmeier":
            wl = np.linspace(200.0, 3000.0, 500)
            n_arr, k_arr = sellmeier(wl, spec.coefficients)
            entry = MaterialEntry(name=name, wavelengths_nm=wl, n=n_arr, k=k_arr)

        elif spec.source == "drude":
            wl = np.linspace(200.0, 3000.0, 500)
            n_arr, k_arr = drude(wl, spec.coefficients)
            entry = MaterialEntry(name=name, wavelengths_nm=wl, n=n_arr, k=k_arr)

        elif spec.source == "tauc_lorentz":
            wl = np.linspace(200.0, 3000.0, 500)
            n_arr, k_arr = tauc_lorentz(wl, spec.coefficients)
            entry = MaterialEntry(name=name, wavelengths_nm=wl, n=n_arr, k=k_arr)

        else:
            msg = f"Unknown material source: {spec.source!r}"
            raise ValueError(msg)

        self._cache[name] = entry
        return entry

    def get_epsilon(
        self, name: str, wavelengths_nm: np.ndarray
    ) -> np.ndarray:
        """Return complex permittivity array at the requested wavelengths."""
        entry = self._cache[name]
        n = entry.n_interp(wavelengths_nm)
        k = entry.k_interp(wavelengths_nm)
        return (n + 1j * k) ** 2

    def check_extrapolation(
        self, entry: "MaterialEntry", wavelengths_nm: np.ndarray
    ) -> list[str]:
        """Return warning strings for wavelengths outside the tabulated range."""
        wl_min = entry.wavelengths_nm.min()
        wl_max = entry.wavelengths_nm.max()
        warnings: list[str] = []
        outside = wavelengths_nm[(wavelengths_nm < wl_min) | (wavelengths_nm > wl_max)]
        if len(outside) > 0:
            warnings.append(
                f"{entry.name}: wavelengths {outside} nm are outside tabulated range "
                f"[{wl_min}, {wl_max}] nm"
            )
        return warnings
