"""Permittivity Fourier decomposition and Toeplitz matrix builder for RCWA."""

from __future__ import annotations

import numpy as np

from se_simulator.rcwa.modes import make_order_indices


def rasterize_layer(
    layer: object,
    materials_db: object,
    wavelength_nm: float,
    grid_size: int = 512,
) -> np.ndarray:
    """Rasterize permittivity ε(x,y) of a layer onto a (grid_size, grid_size) grid.

    Returns complex128 array of shape (grid_size, grid_size).
    Pixel (i, j) → physical position x = (i+0.5)/N * Lx_nm, y = (j+0.5)/N * Ly_nm.
    """
    lx = layer.Lx_nm  # type: ignore[attr-defined]
    ly = layer.Ly_nm  # type: ignore[attr-defined]

    wl_arr = np.array([wavelength_nm])
    eps_bg = materials_db.get_epsilon(layer.background_material, wl_arr)[0]  # type: ignore[attr-defined]

    grid = np.full((grid_size, grid_size), eps_bg, dtype=complex)

    # Pixel centre coordinates
    i_idx = np.arange(grid_size)
    j_idx = np.arange(grid_size)
    x_coords = (i_idx + 0.5) / grid_size * lx  # shape (grid_size,)
    y_coords = (j_idx + 0.5) / grid_size * ly   # shape (grid_size,)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing="ij")  # (Nx, Ny)

    for shape_region in layer.shapes:  # type: ignore[attr-defined]
        geom = shape_region.geometry
        mat_name = shape_region.material
        eps_shape = materials_db.get_epsilon(mat_name, wl_arr)[0]  # type: ignore[attr-defined]

        mask = _rasterize_shape(geom, xx, yy, lx, ly, grid_size)
        grid[mask] = eps_shape

    return grid


def rasterize_inverse_layer(
    layer: object,
    materials_db: object,
    wavelength_nm: float,
    grid_size: int = 512,
) -> np.ndarray:
    """Rasterize 1/ε(x,y) for Li factorization inverse rule.

    Same as rasterize_layer but stores 1/eps at each pixel.
    Guards against division by zero (|eps| < 1e-10 → stores 0).
    """
    eps_grid = rasterize_layer(layer, materials_db, wavelength_nm, grid_size)
    inv_grid = np.zeros_like(eps_grid)
    nonzero = np.abs(eps_grid) >= 1e-10
    inv_grid[nonzero] = 1.0 / eps_grid[nonzero]
    return inv_grid


def _rasterize_shape(
    geom: object,
    xx: np.ndarray,
    yy: np.ndarray,
    lx: float,
    ly: float,
    grid_size: int,
) -> np.ndarray:
    """Return a boolean mask for pixels inside the given ShapeGeometry."""
    shape_type = geom.type  # type: ignore[attr-defined]
    cx, cy = geom.cx, geom.cy  # type: ignore[attr-defined]
    width, height = geom.width, geom.height  # type: ignore[attr-defined]

    if shape_type == "rectangle":
        mask = (np.abs(xx - cx) <= width / 2) & (np.abs(yy - cy) <= height / 2)

    elif shape_type == "trapezoid":
        sidewall = geom.sidewall_angle_deg  # type: ignore[attr-defined]
        taper = np.tan(np.radians(90.0 - sidewall))
        y_bot = cy - height / 2
        # Effective half-width at each y: decreases from width/2 at bottom toward top
        dy = yy - y_bot
        hw = width / 2 - dy * taper
        hw = np.maximum(hw, 0.0)
        in_y = (yy >= y_bot) & (yy <= cy + height / 2)
        mask = in_y & (np.abs(xx - cx) <= hw)

    elif shape_type == "ellipse":
        mask = ((xx - cx) / (width / 2)) ** 2 + ((yy - cy) / (height / 2)) ** 2 <= 1.0

    elif shape_type == "polygon":
        vertices = geom.vertices  # type: ignore[attr-defined]
        mask = _polygon_mask(vertices, xx, yy)

    else:
        mask = np.zeros_like(xx, dtype=bool)

    return mask


def _polygon_mask(
    vertices: list[tuple[float, float]],
    xx: np.ndarray,
    yy: np.ndarray,
) -> np.ndarray:
    """Ray-casting point-in-polygon test for arbitrary polygon vertices."""
    n_vert = len(vertices)
    mask = np.zeros(xx.shape, dtype=bool)
    if n_vert < 3:
        return mask

    xs = np.array([v[0] for v in vertices])
    ys = np.array([v[1] for v in vertices])

    # Vectorized ray-casting (ray goes in +x direction from each point)
    for idx in range(n_vert):
        xi, yi = xs[idx], ys[idx]
        xj, yj = xs[(idx + 1) % n_vert], ys[(idx + 1) % n_vert]

        # Edge from (xi,yi) to (xj,yj)
        cond1 = ((yi > yy) != (yj > yy))
        with np.errstate(divide="ignore", invalid="ignore"):
            x_intersect = (xj - xi) * (yy - yi) / (yj - yi + 1e-300) + xi
        cond2 = xx < x_intersect
        mask ^= cond1 & cond2

    return mask


def compute_epsilon_fourier_2d(
    eps_grid: np.ndarray,
    nx: int,
    ny: int,
) -> np.ndarray:
    """Compute 2D Fourier coefficients of the permittivity map.

    Returns array of shape (2*Nx+1, 2*Ny+1).
    Index [Nx+m, Ny+n] corresponds to Fourier order (m, n).

    Normalised: eps_hat[0,0] = spatial average of eps_grid.
    """
    grid_size = eps_grid.shape[0]
    # FFT2 then normalize
    ft = np.fft.fft2(eps_grid) / (grid_size**2)
    # Shift so that zero-frequency is at centre
    ft_shifted = np.fft.fftshift(ft)

    # Centre index in the shifted array
    cx = grid_size // 2
    cy = grid_size // 2

    # Extract orders -Nx..Nx, -Ny..Ny
    return ft_shifted[cx - nx : cx + nx + 1, cy - ny : cy + ny + 1].copy()


def build_toeplitz_matrix(
    eps_fourier: np.ndarray,
    nx: int,
    ny: int,
) -> np.ndarray:
    """Build the (n×n) block-Toeplitz convolution matrix E from Fourier coefficients.

    E[p, q] = eps_hat[mp - mq, np - nq]
    where (mp, np) and (mq, nq) are the order pairs from make_order_indices.

    Returns shape (n, n), complex128.
    """
    orders = make_order_indices(nx, ny)
    n_modes = len(orders)

    e_mat = np.zeros((n_modes, n_modes), dtype=complex)
    for p, (mp, np_p) in enumerate(orders):
        for q, (mq, nq) in enumerate(orders):
            dm = mp - mq
            dn = np_p - nq
            if -nx <= dm <= nx and -ny <= dn <= ny:
                e_mat[p, q] = eps_fourier[nx + dm, ny + dn]
            # Outside range → 0 (should not occur for properly truncated series)

    return e_mat


def build_li_matrices(
    eps_fourier: np.ndarray,
    eps_inv_fourier: np.ndarray,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build Li factorization matrices (E_yx, E_xx).

    E_yx = Toeplitz of ε  (Laurent rule — used for y-field component)
    E_xx = inv(Toeplitz of 1/ε)  (inverse rule — used for x-field component)

    Returns (E_yx, E_xx) each of shape (n, n), complex128.
    """
    e_yx = build_toeplitz_matrix(eps_fourier, nx, ny)
    e_inv = build_toeplitz_matrix(eps_inv_fourier, nx, ny)
    e_xx = np.linalg.solve(e_inv, np.eye(e_inv.shape[0], dtype=complex))
    return e_yx, e_xx
