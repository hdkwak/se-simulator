"""Dispersion models for optical constants."""
from __future__ import annotations

import numpy as np


def cauchy(
    wavelengths_nm: np.ndarray,
    coefficients: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Cauchy model: n = A + B/lambda^2 + C/lambda^4 + ...

    coefficients = [A, B, C, ...] in ascending order (B in nm^2, C in nm^4, ...).
    Returns (n, k) where k = 0.
    """
    lam = np.asarray(wavelengths_nm, dtype=float)
    n = np.zeros_like(lam)
    for i, c in enumerate(coefficients):
        n += c / lam ** (2 * i) if i > 0 else np.full_like(lam, c)
    return n, np.zeros_like(lam)


def sellmeier(
    wavelengths_nm: np.ndarray,
    coefficients: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Sellmeier model: n^2 = 1 + sum_i B_i*lambda^2 / (lambda^2 - C_i).

    coefficients = [B1, C1, B2, C2, ...] where C_i is in nm^2.
    Returns (n, k) where k = 0.
    """
    lam = np.asarray(wavelengths_nm, dtype=float)
    lam2 = lam ** 2
    n2 = np.ones_like(lam)
    pairs = len(coefficients) // 2
    for i in range(pairs):
        b = coefficients[2 * i]
        c = coefficients[2 * i + 1]
        n2 += b * lam2 / (lam2 - c)
    n2 = np.clip(n2, 1.0, None)
    return np.sqrt(n2), np.zeros_like(lam)


def drude(
    wavelengths_nm: np.ndarray,
    coefficients: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Drude free-electron model.

    coefficients = [eps_inf, omega_p_eV, gamma_eV].
    Returns (n, k).
    """
    lam = np.asarray(wavelengths_nm, dtype=float)
    # Convert wavelength to eV
    hc_eV_nm = 1239.8419  # eV*nm
    energy_ev = hc_eV_nm / lam

    eps_inf, omega_p, gamma = float(coefficients[0]), float(coefficients[1]), float(coefficients[2])
    eps = eps_inf - omega_p ** 2 / (energy_ev ** 2 + 1j * gamma * energy_ev)
    n_complex = np.sqrt(eps)
    return np.abs(n_complex.real), np.abs(n_complex.imag)


def tauc_lorentz(
    wavelengths_nm: np.ndarray,
    coefficients: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Tauc-Lorentz model (single oscillator).

    coefficients = [eps_inf, A, E0_eV, C_eV, Eg_eV].
    Returns (n, k).
    """
    lam = np.asarray(wavelengths_nm, dtype=float)
    hc_eV_nm = 1239.8419
    energy_ev = hc_eV_nm / lam

    eps_inf = float(coefficients[0])
    amp = float(coefficients[1])
    e0 = float(coefficients[2])
    c_damp = float(coefficients[3])
    eg = float(coefficients[4])

    e = energy_ev
    eps2 = np.where(
        e > eg,
        amp * e0 * c_damp * (e - eg) ** 2 / ((e ** 2 - e0 ** 2) ** 2 + c_damp ** 2 * e ** 2) / e,
        0.0,
    )

    # Kramers-Kronig to get eps1 (numerical integration)
    from scipy.integrate import trapezoid

    e_grid = np.linspace(eg + 1e-6, 50.0, 4096)
    hc_eV_nm2 = 1239.8419
    lam_grid = hc_eV_nm2 / e_grid
    eps2_grid = np.where(
        e_grid > eg,
        amp * e0 * c_damp * (e_grid - eg) ** 2
        / ((e_grid ** 2 - e0 ** 2) ** 2 + c_damp ** 2 * e_grid ** 2)
        / e_grid,
        0.0,
    )

    eps1 = np.empty_like(e)
    for i, ei in enumerate(e):
        integrand = e_grid * eps2_grid / (e_grid ** 2 - ei ** 2 + 1e-30j)
        eps1[i] = eps_inf + 2.0 / np.pi * trapezoid(integrand.real, e_grid)

    del lam_grid  # unused but computed for reference
    eps = eps1 + 1j * eps2
    n_complex = np.sqrt(eps)
    return np.clip(n_complex.real, 0.0, None), np.clip(n_complex.imag, 0.0, None)
