"""Stage 1: χ² nearest-neighbor search over the spectral library."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from se_simulator.config.schemas import FittingConditions
from se_simulator.ellipsometer.signals import EllipsometryResult
from se_simulator.fitting.library import LibraryStore


def chi_squared(
    target: dict[str, np.ndarray],
    library: dict[str, np.ndarray],
    sigma: dict[str, float | np.ndarray],
    wavelength_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute χ² between a target spectrum and all N library entries.

    χ²_i = Σ_λ [ (target_sig(λ) - lib_sig_i(λ))² / σ_sig² + ... ]

    Parameters
    ----------
    target:
        Dict mapping signal name → array of shape ``(Nλ,)``.
    library:
        Dict mapping signal name → array of shape ``(N, Nλ)``.
    sigma:
        Dict mapping signal name → scalar uncertainty or array of shape ``(Nλ,)``.
    wavelength_mask:
        Boolean array of shape ``(Nλ,)``; ``True`` = include in sum.

    Returns
    -------
    np.ndarray of shape ``(N,)``.
    """
    chi2: np.ndarray | None = None

    for sig_name, target_arr in target.items():
        if sig_name not in library:
            continue
        lib_arr = library[sig_name].astype(float)  # (N, Nλ)
        sig_val = sigma.get(sig_name, 1.0)

        t = target_arr
        la = lib_arr
        if wavelength_mask is not None:
            t = t[wavelength_mask]
            la = la[:, wavelength_mask]

        # Fully vectorised: no Python loops over N
        residuals = la - t[np.newaxis, :]  # (N, Nλ)
        contribution = np.sum(residuals**2 / (sig_val**2), axis=1)  # (N,)

        chi2 = contribution if chi2 is None else chi2 + contribution

    if chi2 is None:
        n = next(iter(library.values())).shape[0]
        return np.zeros(n)
    return chi2


@dataclass
class SearchResult:
    """Output from Stage 1 nearest-neighbour search."""

    top_k_indices: np.ndarray
    top_k_chi2: np.ndarray
    top_k_params: np.ndarray
    parameter_names: list[str]
    best_fit_params: np.ndarray
    best_fit_chi2: float
    best_fit_spectrum: EllipsometryResult


class NearestNeighborSearch:
    """Stage 1: brute-force χ² search over the full library."""

    def __init__(self, store: LibraryStore, fitting_config: FittingConditions) -> None:
        self.store = store
        self.fitting_config = fitting_config
        self._parameters: np.ndarray | None = None
        self._spectra: dict[str, np.ndarray] | None = None

    def _load(self) -> None:
        if self._parameters is None:
            self._parameters, self._spectra = self.store.read_all()

    def search(
        self,
        target: EllipsometryResult,
        top_k: int | None = None,
    ) -> SearchResult:
        """Search the library and return the top-k best-matching entries."""
        self._load()
        assert self._parameters is not None and self._spectra is not None

        k = top_k if top_k is not None else self.fitting_config.top_k_candidates
        spec = self.store.get_spec()

        sig_map = {
            "psi": target.psi_deg,
            "delta": target.delta_deg,
            "alpha": target.alpha,
            "beta": target.beta,
            "chi": target.chi,
        }
        sigma_defaults: dict[str, float] = {
            "psi": self.fitting_config.sigma_psi,
            "delta": self.fitting_config.sigma_delta,
            "alpha": 0.01,
            "beta": 0.01,
            "chi": 0.01,
        }

        target_dict = {
            s: sig_map[s] for s in self.fitting_config.fit_signals if s in sig_map
        }
        sigma_dict: dict[str, float | np.ndarray] = {
            s: sigma_defaults.get(s, 1.0) for s in target_dict
        }

        wl_mask: np.ndarray | None = None
        if self.fitting_config.wavelength_mask is not None:
            wl_lo, wl_hi = self.fitting_config.wavelength_mask
            wl = np.array(spec.wavelengths_nm)
            wl_mask = (wl >= wl_lo) & (wl <= wl_hi)

        chi2 = chi_squared(target_dict, self._spectra, sigma_dict, wl_mask)

        k_actual = min(k, len(chi2))
        top_indices = np.argsort(chi2)[:k_actual]

        best_idx = int(top_indices[0])
        best_params = self._parameters[best_idx]
        best_spectrum = self._spectrum_from_library(best_idx, spec)

        return SearchResult(
            top_k_indices=top_indices,
            top_k_chi2=chi2[top_indices],
            top_k_params=self._parameters[top_indices],
            parameter_names=[p.name for p in spec.parameters],
            best_fit_params=best_params,
            best_fit_chi2=float(chi2[best_idx]),
            best_fit_spectrum=best_spectrum,
        )

    def _spectrum_from_library(self, index: int, spec) -> EllipsometryResult:
        """Build an EllipsometryResult from the cached library data."""
        assert self._spectra is not None
        wl = np.array(spec.wavelengths_nm)
        n = len(wl)
        zeros = np.zeros(n)

        def _get(name: str) -> np.ndarray:
            if name in self._spectra:
                return self._spectra[name][index].astype(float)
            return zeros.copy()

        return EllipsometryResult(
            wavelengths_nm=wl,
            psi_deg=_get("psi"),
            delta_deg=_get("delta"),
            alpha=_get("alpha"),
            beta=_get("beta"),
            chi=_get("chi"),
            xi=_get("xi"),
            jones_reflection=np.zeros((n, 2, 2), dtype=complex),
            energy_conservation=np.zeros(n),
        )
