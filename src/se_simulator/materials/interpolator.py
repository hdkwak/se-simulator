"""MaterialEntry dataclass and CSV library loader."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass
class MaterialEntry:
    """Tabulated optical constants with PCHIP interpolation."""

    name: str
    wavelengths_nm: np.ndarray  # (N,), sorted ascending
    n: np.ndarray               # (N,), real refractive index
    k: np.ndarray               # (N,), extinction coefficient

    def __post_init__(self) -> None:
        self._n_interp = PchipInterpolator(self.wavelengths_nm, self.n, extrapolate=True)
        self._k_interp = PchipInterpolator(self.wavelengths_nm, self.k, extrapolate=True)

    def n_interp(self, wavelengths_nm: np.ndarray) -> np.ndarray:
        """Return interpolated n at the given wavelengths."""
        return np.clip(self._n_interp(wavelengths_nm).real, 0.0, None)

    def k_interp(self, wavelengths_nm: np.ndarray) -> np.ndarray:
        """Return interpolated k at the given wavelengths (clipped to >= 0)."""
        return np.clip(self._k_interp(wavelengths_nm).real, 0.0, None)


def load_csv_library(path: Path, name: str | None = None) -> MaterialEntry:
    """Load a CSV optical-constants file with header 'wavelength_nm,n,k'."""
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    wl = data[:, 0]
    n = data[:, 1]
    k = data[:, 2]
    # Sort by wavelength
    order = np.argsort(wl)
    return MaterialEntry(
        name=name or path.stem,
        wavelengths_nm=wl[order],
        n=n[order],
        k=k[order],
    )
