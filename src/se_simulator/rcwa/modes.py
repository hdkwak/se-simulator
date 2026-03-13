"""Wavevector and mode index utilities for RCWA."""

from __future__ import annotations

import numpy as np


def make_order_indices(nx: int, ny: int) -> list[tuple[int, int]]:
    """Return (m, n) diffraction order pairs in standard row-major RCWA ordering.

    Ordering: for n from -Ny to +Ny, for m from -Nx to +Nx.
    Total length: (2*Nx+1) * (2*Ny+1).

    Example: Nx=Ny=1 → [(-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1)]
    """
    return [
        (m, n)
        for n in range(-ny, ny + 1)
        for m in range(-nx, nx + 1)
    ]


def make_kx_matrix(
    order_indices: list[tuple[int, int]],
    kx_inc: float,
    lx_nm: float,
    k0: float,
) -> np.ndarray:
    """Return diagonal matrix Kx of shape (n, n), complex128.

    Kx[i,i] = kx_inc + m_i * (2π / lx_nm) / k0
    All values normalized by k0.
    """
    diag = np.array(
        [kx_inc + m * (2.0 * np.pi / lx_nm) / k0 for m, _ in order_indices],
        dtype=complex,
    )
    return np.diag(diag)


def make_ky_matrix(
    order_indices: list[tuple[int, int]],
    ky_inc: float,
    ly_nm: float,
    k0: float,
) -> np.ndarray:
    """Return diagonal matrix Ky of shape (n, n), analogous to make_kx_matrix."""
    diag = np.array(
        [ky_inc + n_idx * (2.0 * np.pi / ly_nm) / k0 for _, n_idx in order_indices],
        dtype=complex,
    )
    return np.diag(diag)


def make_kz_array(
    kx_diag: np.ndarray,
    ky_diag: np.ndarray,
    eps: complex,
) -> np.ndarray:
    """Compute kz for each mode: kz_i = sqrt(eps - Kx_i² - Ky_i²).

    Branch chosen so Im(kz) >= 0 (physically correct for forward propagation).
    Returns shape (n,) complex128.
    """
    kz_sq = eps - kx_diag**2 - ky_diag**2
    kz = np.sqrt(kz_sq.astype(complex))
    # Enforce Im(kz) >= 0 (outgoing / decaying wave convention)
    kz = np.where(np.imag(kz) < 0, -kz, kz)
    return kz


def free_space_matrices(
    kx: np.ndarray,  # (n, n) diagonal
    ky: np.ndarray,  # (n, n) diagonal
) -> tuple[np.ndarray, np.ndarray]:
    """Compute gap-medium (ε=μ=1) eigenvector matrices W0 and V0.

    W0 = identity(2n)
    V0 encodes the magnetic field coupling for the free-space gap layer.

    The gap medium Q matrix is:
        Q = [[Kx·Ky,     I - Kx²],
             [Ky² - I,   -Kx·Ky ]]
    V0 = Q·W0·diag(1/kz0), where kz0 = sqrt(1 - Kx²_diag - Ky²_diag).

    Returns (W0, V0) each shape (2n, 2n).
    """
    n = kx.shape[0]
    kx_d = np.diag(kx).real.astype(complex)  # (n,) diagonal entries
    ky_d = np.diag(ky).real.astype(complex)

    kz0 = make_kz_array(kx_d, ky_d, 1.0 + 0j)  # free-space kz, shape (n,)

    # Q matrix blocks for gap medium (eps=1)
    kx_mat = kx
    ky_mat = ky
    i_n = np.eye(n, dtype=complex)

    q11 = kx_mat @ ky_mat
    q12 = i_n - kx_mat @ kx_mat
    q21 = ky_mat @ ky_mat - i_n
    q22 = -kx_mat @ ky_mat

    q = np.block([[q11, q12], [q21, q22]])  # (2n, 2n)

    w0 = np.eye(2 * n, dtype=complex)

    # V0 = Q @ W0 @ diag(1/kz0)  — kz0 repeated twice for the 2n eigenvectors
    kz0_full = np.concatenate([kz0, kz0])  # (2n,)
    v0 = q @ w0 @ np.diag(1.0 / kz0_full)

    return w0, v0
