"""Tests for the RCWA core modules (Step 2a)."""

from __future__ import annotations

import numpy as np
import pytest

from se_simulator.rcwa.modes import (
    free_space_matrices,
    make_kx_matrix,
    make_ky_matrix,
    make_kz_array,
    make_order_indices,
)
from se_simulator.rcwa.fourier import (
    build_li_matrices,
    build_toeplitz_matrix,
    compute_epsilon_fourier_2d,
    rasterize_layer,
)
from se_simulator.rcwa.eigensolver import solve_uniform_layer
from se_simulator.rcwa.smatrix import (
    build_layer_smatrix,
    identity_smatrix,
    redheffer_star_product,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_simple_layer(
    shape_type: str = "rectangle",
    cx: float = 250.0,
    cy: float = 250.0,
    width: float = 250.0,
    height: float = 500.0,
    fill_material: str = "Si",
    background_material: str = "Air",
    lx: float = 500.0,
    ly: float = 500.0,
):
    """Build a minimal duck-typed GratingLayer-like object for testing."""
    from types import SimpleNamespace
    geom = SimpleNamespace(
        type=shape_type,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        sidewall_angle_deg=90.0,
        vertices=[],
    )
    shape_region = SimpleNamespace(geometry=geom, material=fill_material)
    layer = SimpleNamespace(
        type="grating_1d",
        shapes=[shape_region],
        background_material=background_material,
        Lx_nm=lx,
        Ly_nm=ly,
    )
    return layer


class _SimpleMaterialsDB:
    """Minimal materials database that returns fixed epsilon values."""

    def __init__(self, eps_map: dict[str, complex]):
        self._eps_map = eps_map

    def get_epsilon(self, name: str, wavelengths: np.ndarray) -> np.ndarray:
        eps = self._eps_map[name]
        return np.full(len(wavelengths), eps, dtype=complex)


# ---------------------------------------------------------------------------
# modes.py tests
# ---------------------------------------------------------------------------

def test_make_order_indices_count():
    assert len(make_order_indices(1, 1)) == 9
    assert len(make_order_indices(2, 3)) == 35


def test_make_order_indices_contains_zero():
    for nx, ny in [(0, 0), (1, 1), (3, 5), (5, 2)]:
        assert (0, 0) in make_order_indices(nx, ny)


def test_kx_matrix_diagonal():
    """Normal incidence, Lx=λ=500nm: (0,0) entry = 0, (1,0) entry = 1."""
    lx = 500.0
    lam = 500.0
    k0 = 2 * np.pi / lam
    orders = make_order_indices(1, 0)  # m=-1,0,1 at ny=0 → 3 modes
    kx = make_kx_matrix(orders, kx_inc=0.0, lx_nm=lx, k0=k0)

    # find index of (0,0)
    idx_00 = orders.index((0, 0))
    idx_10 = orders.index((1, 0))

    assert abs(kx[idx_00, idx_00]) < 1e-12
    assert abs(kx[idx_10, idx_10] - 1.0) < 1e-10


def test_kz_branch_cut():
    """All Im(kz) must be >= 0, including evanescent modes (kx > 1)."""
    kx_vals = np.array([0.0, 0.5, 0.9, 1.0, 1.5, 2.0, -1.5])
    ky_vals = np.zeros_like(kx_vals)
    kz = make_kz_array(kx_vals, ky_vals, eps=1.0 + 0j)
    assert np.all(np.imag(kz) >= -1e-14), f"Branch cut violated: {kz}"


# ---------------------------------------------------------------------------
# fourier.py tests
# ---------------------------------------------------------------------------

def test_rasterize_rectangle():
    """Rectangle with ff=0.5 centred in cell: ~50% of pixels should be 'Si'."""
    db = _SimpleMaterialsDB({"Si": 11.9 + 0j, "Air": 1.0 + 0j})
    layer = _make_simple_layer(
        shape_type="rectangle",
        cx=250.0, cy=250.0, width=250.0, height=500.0,
    )
    grid = rasterize_layer(layer, db, wavelength_nm=633.0, grid_size=512)

    fill_frac = np.sum(np.abs(grid - 1.0) > 1.0) / grid.size
    assert abs(fill_frac - 0.5) < 0.02, f"Fill fraction {fill_frac:.4f} not near 0.5"


def test_rasterize_ellipse():
    """Circle inscribed in full unit cell: area fraction ≈ π/4 ≈ 0.785."""
    db = _SimpleMaterialsDB({"Si": 11.9 + 0j, "Air": 1.0 + 0j})
    layer = _make_simple_layer(
        shape_type="ellipse",
        cx=250.0, cy=250.0, width=500.0, height=500.0,
    )
    grid = rasterize_layer(layer, db, wavelength_nm=633.0, grid_size=512)

    fill_frac = np.sum(np.abs(grid - 1.0) > 1.0) / grid.size
    expected = np.pi / 4
    assert abs(fill_frac - expected) < 0.02, f"Fill fraction {fill_frac:.4f}, expected {expected:.4f}"


def test_fourier_coefficients_dc_term():
    """Uniform grid with eps=4.0 → (0,0) coefficient = 4.0."""
    eps_grid = np.full((128, 128), 4.0 + 0j)
    coeffs = compute_epsilon_fourier_2d(eps_grid, nx=3, ny=3)
    dc = coeffs[3, 3]  # index [Nx, Ny] = [3, 3] for the (0,0) order
    assert abs(dc - 4.0) < 1e-10, f"DC term {dc} ≠ 4.0"


def test_fourier_coefficients_fill_factor():
    """Binary grating ff=0.5, eps1=4, eps2=1 → DC ≈ 0.5*4 + 0.5*1 = 2.5."""
    n = 256
    grid = np.ones((n, n), dtype=complex)
    grid[: n // 2, :] = 4.0  # first half of x-axis has eps=4

    coeffs = compute_epsilon_fourier_2d(grid, nx=5, ny=5)
    dc = coeffs[5, 5].real  # (0,0) order
    assert abs(dc - 2.5) < 0.05, f"DC term {dc:.4f}, expected 2.5"


def test_toeplitz_matrix_shape():
    """Toeplitz matrix shape for Nx=Ny=2 → (25,25); Nx=3,Ny=1 → (21,21)."""
    eps_fourier_1 = np.ones((5, 5), dtype=complex)
    e1 = build_toeplitz_matrix(eps_fourier_1, nx=2, ny=2)
    assert e1.shape == (25, 25)

    eps_fourier_2 = np.ones((7, 3), dtype=complex)
    e2 = build_toeplitz_matrix(eps_fourier_2, nx=3, ny=1)
    assert e2.shape == (21, 21)


def test_toeplitz_matrix_symmetry():
    """For real symmetric eps, E[i,j] = conj(E[j,i]) (Hermitian)."""
    # Centred rectangle with real eps → real Fourier coeffs → Hermitian Toeplitz
    db = _SimpleMaterialsDB({"Si": 11.9 + 0j, "Air": 1.0 + 0j})
    layer = _make_simple_layer(
        shape_type="rectangle", cx=250.0, cy=250.0, width=250.0, height=500.0
    )
    grid = rasterize_layer(layer, db, wavelength_nm=633.0, grid_size=256)
    coeffs = compute_epsilon_fourier_2d(grid, nx=3, ny=3)
    e_mat = build_toeplitz_matrix(coeffs, nx=3, ny=3)

    residual = np.max(np.abs(e_mat - e_mat.conj().T))
    assert residual < 1e-8, f"Toeplitz not Hermitian: max residual {residual:.2e}"


def test_li_matrices_shape():
    """build_li_matrices for Nx=Ny=3 returns two (49, 49) matrices."""
    eps_f = np.ones((7, 7), dtype=complex)
    eps_inv_f = np.ones((7, 7), dtype=complex)
    e_yx, e_xx = build_li_matrices(eps_f, eps_inv_f, nx=3, ny=3)
    assert e_yx.shape == (49, 49)
    assert e_xx.shape == (49, 49)


# ---------------------------------------------------------------------------
# eigensolver.py tests
# ---------------------------------------------------------------------------

def test_solve_uniform_layer_kz():
    """Uniform air layer, normal incidence, 1-mode: kz should be 1.0."""
    kx = make_kx_matrix([(0, 0)], kx_inc=0.0, lx_nm=500.0, k0=2 * np.pi / 500.0)
    ky = make_ky_matrix([(0, 0)], ky_inc=0.0, ly_nm=500.0, k0=2 * np.pi / 500.0)
    _w, _v, kz = solve_uniform_layer(eps=1.0 + 0j, kx=kx, ky=ky, n_modes=1)

    # kz for the 0th order (and its duplicate) should be 1.0
    assert abs(kz[0] - 1.0) < 1e-10, f"kz[0] = {kz[0]}"


# ---------------------------------------------------------------------------
# smatrix.py tests
# ---------------------------------------------------------------------------

def test_identity_smatrix_neutral_element():
    """redheffer_star_product(S, identity(n)) should equal S."""
    n = 3
    size = 2 * n
    rng = np.random.default_rng(42)

    # Build a simple diagonal S-matrix (doesn't need to be physical)
    s_mat = {
        "S11": np.diag(rng.uniform(-0.1, 0.1, size) + 1j * rng.uniform(-0.1, 0.1, size)),
        "S12": np.diag(rng.uniform(0.8, 1.0, size) + 1j * rng.uniform(-0.1, 0.1, size)),
        "S21": np.diag(rng.uniform(0.8, 1.0, size) + 1j * rng.uniform(-0.1, 0.1, size)),
        "S22": np.diag(rng.uniform(-0.1, 0.1, size) + 1j * rng.uniform(-0.1, 0.1, size)),
    }
    identity = identity_smatrix(n)
    result = redheffer_star_product(s_mat, identity)

    for key in ("S11", "S12", "S21", "S22"):
        diff = np.max(np.abs(result[key] - s_mat[key]))
        assert diff < 1e-10, f"Block {key}: max diff = {diff:.2e}"


def test_build_layer_smatrix_lossless_phase():
    """|S12[0,0]| ≈ 1 for propagating 0th order in air layer at normal incidence."""
    lam = 500.0  # nm
    k0 = 2 * np.pi / lam
    thickness = 100.0  # nm

    orders = make_order_indices(0, 0)  # single (0,0) mode
    kx = make_kx_matrix(orders, kx_inc=0.0, lx_nm=500.0, k0=k0)
    ky = make_ky_matrix(orders, ky_inc=0.0, ly_nm=500.0, k0=k0)

    w_mat, v_mat, kz = solve_uniform_layer(eps=1.0 + 0j, kx=kx, ky=ky, n_modes=1)
    w0, v0 = free_space_matrices(kx, ky)

    smat = build_layer_smatrix(w_mat, v_mat, kz, thickness, k0, w0, v0)

    # The diagonal of S12 for the propagating mode should have |·| = 1
    s12_00 = smat["S12"][0, 0]
    assert abs(abs(s12_00) - 1.0) < 1e-8, f"|S12[0,0]| = {abs(s12_00):.6f}, expected 1.0"
