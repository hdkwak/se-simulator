"""Layer eigenvalue problem for RCWA: assemble PQ matrix and solve for modal kz."""

from __future__ import annotations

import numpy as np


def assemble_coupled_wave_matrix(
    e_xx: np.ndarray,
    e_yx: np.ndarray,
    e_xy: np.ndarray,
    e_yy: np.ndarray,
    kx: np.ndarray,  # (n, n) diagonal
    ky: np.ndarray,  # (n, n) diagonal
) -> np.ndarray:
    """Assemble the 2n×2n matrix P·Q whose eigenvalues are kz².

    P = [[Kx·Ky,       E_yy - Kx²],
         [Ky² - E_xx,  -Kx·Ky   ]]

    Q = [[Kx·Ky,       E_xx - Kx²],
         [Ky² - E_yy,  -Kx·Ky   ]]

    Returns P @ Q, shape (2n, 2n), complex128.
    """
    p11 = kx @ ky
    p12 = e_yy - kx @ kx
    p21 = ky @ ky - e_xx
    p22 = -(kx @ ky)

    q11 = kx @ ky
    q12 = e_xx - kx @ kx
    q21 = ky @ ky - e_yy
    q22 = -(kx @ ky)

    p_mat = np.block([[p11, p12], [p21, p22]])
    q_mat = np.block([[q11, q12], [q21, q22]])

    return p_mat @ q_mat, q_mat


def solve_eigenproblem(
    pq: np.ndarray,
    q_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve PQ·W = W·diag(kz²) using numpy.linalg.eig.

    Post-processing:
      - kz = sqrt(eigenvalues), branch Im(kz) >= 0
      - V = Q · W · diag(1/kz)

    Returns (W, V, kz).
    """
    kz_sq, w_mat = np.linalg.eig(pq)

    kz_sqrt = np.sqrt(kz_sq.astype(complex))
    kz = np.where(np.imag(kz_sqrt) < 0, -kz_sqrt, kz_sqrt)

    # Guard against near-zero kz (grazing/degenerate modes)
    kz_safe = np.where(np.abs(kz) < 1e-10, 1e-10, kz)

    v_mat = q_mat @ w_mat @ np.diag(1.0 / kz_safe)

    return w_mat, v_mat, kz


def solve_uniform_layer(
    eps: complex,
    kx: np.ndarray,   # (n, n) diagonal
    ky: np.ndarray,   # (n, n) diagonal
    n_modes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast path for a uniform (non-patterned) layer with scalar permittivity.

    W = identity(2n)
    kz_i = sqrt(eps - Kx_i² - Ky_i²)  (branch Im >= 0)
    V = Q_uniform · W · diag(1/kz)

    Returns (W, V, kz) matching shapes of solve_eigenproblem.
    """
    kx_d = np.diag(kx).astype(complex)
    ky_d = np.diag(ky).astype(complex)

    # Per-mode kz
    kz_per_mode = np.sqrt((eps - kx_d**2 - ky_d**2).astype(complex))
    kz_per_mode = np.where(np.imag(kz_per_mode) < 0, -kz_per_mode, kz_per_mode)

    # kz for the full 2n eigenvectors (same values repeated twice)
    kz_full = np.concatenate([kz_per_mode, kz_per_mode])
    kz_safe = np.where(np.abs(kz_full) < 1e-10, 1e-10 + 0j, kz_full)

    w_mat = np.eye(2 * n_modes, dtype=complex)

    # Q for uniform medium: E_xx = E_yy = eps*I, E_xy = E_yx = eps*I
    e_eps = eps * np.eye(n_modes, dtype=complex)
    _pq, q_mat = assemble_coupled_wave_matrix(e_eps, e_eps, e_eps, e_eps, kx, ky)

    v_mat = q_mat @ w_mat @ np.diag(1.0 / kz_safe)

    return w_mat, v_mat, kz_full


def is_uniform_layer(layer: object) -> bool:
    """Return True if layer has no patterned shapes (uniform medium)."""
    return (
        getattr(layer, "type", None) == "uniform"
        or len(getattr(layer, "shapes", [])) == 0  # noqa: SIM300
    )
