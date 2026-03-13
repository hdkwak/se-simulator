"""S-matrix (Redheffer star product) for RCWA layer propagation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from se_simulator.config.schemas import SampleConfig, SimConditions
    from se_simulator.materials.database import MaterialDatabase


def identity_smatrix(n: int) -> dict[str, np.ndarray]:
    """Return the identity S-matrix of block size 2n.

    S11 = S22 = zeros(2n, 2n)
    S12 = S21 = eye(2n)
    Neutral element for the Redheffer star product.
    """
    size = 2 * n
    return {
        "S11": np.zeros((size, size), dtype=complex),
        "S12": np.eye(size, dtype=complex),
        "S21": np.eye(size, dtype=complex),
        "S22": np.zeros((size, size), dtype=complex),
    }


def redheffer_star_product(
    sa: dict[str, np.ndarray],
    sb: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Combine two S-matrices using the Redheffer star product.

    Uses np.linalg.solve for numerical stability (avoids explicit inversion).

    Formulas (Whittaker & Culshaw 1999):
        D = (I - SB11 @ SA22)^-1
        F = (I - SA22 @ SB11)^-1
        S11 = SA11 + SA12 @ D @ SB11 @ SA21
        S12 = SA12 @ D @ SB12
        S21 = SB21 @ F @ SA21
        S22 = SB22 + SB21 @ F @ SA22 @ SB12
    """
    size = sa["S11"].shape[0]
    i_n = np.eye(size, dtype=complex)

    # Compute D*X = solve(I - SB11 @ SA22, X)
    lhs_d = i_n - sb["S11"] @ sa["S22"]
    lhs_f = i_n - sa["S22"] @ sb["S11"]

    # S11 = SA11 + SA12 @ solve(lhs_d, SB11 @ SA21)
    s11 = sa["S11"] + sa["S12"] @ np.linalg.solve(lhs_d, sb["S11"] @ sa["S21"])
    # S12 = SA12 @ solve(lhs_d, SB12)
    s12 = sa["S12"] @ np.linalg.solve(lhs_d, sb["S12"])
    # S21 = SB21 @ solve(lhs_f, SA21)
    s21 = sb["S21"] @ np.linalg.solve(lhs_f, sa["S21"])
    # S22 = SB22 + SB21 @ solve(lhs_f, SA22 @ SB12)
    s22 = sb["S22"] + sb["S21"] @ np.linalg.solve(lhs_f, sa["S22"] @ sb["S12"])

    return {"S11": s11, "S12": s12, "S21": s21, "S22": s22}


def build_layer_smatrix(
    w_mat: np.ndarray,
    v_mat: np.ndarray,
    kz: np.ndarray,
    thickness_nm: float,
    k0: float,
    w0: np.ndarray,
    v0: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the S-matrix for a single layer (Moharam 1995 stable formulation).

    Algorithm:
      1. Phase matrix X = diag(exp(-i*kz*k0*d))
      2. A = solve(W, W0) + solve(V, V0)
         B = solve(W, W0) - solve(V, V0)
      3. D = A - X @ B @ solve(A, X @ B)
         S11 = solve(D, X @ B @ solve(A, X @ A) - B)
         S12 = solve(D, X) @ (A - B @ solve(A, B))
         S21 = S12,  S22 = S11

    Returns dict with keys 'S11', 'S12', 'S21', 'S22', each shape (2n, 2n).
    """
    # Phase matrix
    phase = np.exp(1j * kz * k0 * thickness_nm)
    x_mat = np.diag(phase)

    # Interface matrices using solve for stability
    # A = W^-1 W0 + V^-1 V0,  B = W^-1 W0 - V^-1 V0
    ww0 = np.linalg.solve(w_mat, w0)   # W^-1 W0
    vv0 = np.linalg.solve(v_mat, v0)   # V^-1 V0

    a_mat = ww0 + vv0
    b_mat = ww0 - vv0

    # D = A - X @ B @ A^-1 @ X @ B
    xb = x_mat @ b_mat
    a_inv_xb = np.linalg.solve(a_mat, xb)
    d_mat = a_mat - x_mat @ b_mat @ a_inv_xb  # noqa: SIM300 (clearer explicit)

    # S11 = D^-1 @ (X @ B @ A^-1 @ X @ A - B)
    xa = x_mat @ a_mat
    a_inv_xa = np.linalg.solve(a_mat, xa)
    rhs_s11 = x_mat @ b_mat @ a_inv_xa - b_mat
    s11 = np.linalg.solve(d_mat, rhs_s11)

    # S12 = D^-1 @ X @ (A - B @ A^-1 @ B)
    a_inv_b = np.linalg.solve(a_mat, b_mat)
    inner_s12 = a_mat - b_mat @ a_inv_b
    s12 = np.linalg.solve(d_mat, x_mat @ inner_s12)

    s21 = s12
    s22 = s11

    return {"S11": s11, "S12": s12, "S21": s21, "S22": s22}


def build_semiinfinite_smatrix(
    w_mat: np.ndarray,
    v_mat: np.ndarray,
    w0: np.ndarray,
    v0: np.ndarray,
    side: str,
) -> dict[str, np.ndarray]:
    """Build the S-matrix for a semi-infinite medium (superstrate or substrate).

    For INPUT side:
        A = W^-1 W0 + V^-1 V0,  B = W^-1 W0 - V^-1 V0
        S11 = -solve(A, B),   S12 = 2*solve(A, I)
        S21 = 0.5*(A - B*solve(A,B)),  S22 = -S11

    For OUTPUT side: same A, B; S11 = -solve(A,B),
        S12 = S21 = 0.5*(A - B*solve(A,B)),  S22 = S11
    """
    ww0 = np.linalg.solve(w_mat, w0)
    vv0 = np.linalg.solve(v_mat, v0)
    a_mat = ww0 + vv0
    b_mat = ww0 - vv0

    size = a_mat.shape[0]
    i_n = np.eye(size, dtype=complex)
    a_inv_b = np.linalg.solve(a_mat, b_mat)
    a_inv_i = np.linalg.solve(a_mat, i_n)

    s11 = -a_inv_b
    s_tran = 0.5 * (a_mat - b_mat @ a_inv_b)

    if side == "input":
        s12 = 2.0 * a_inv_i
        s21 = s_tran
        s22 = -s11
    else:  # "output"
        s12 = s_tran
        s21 = s_tran.copy()
        s22 = s11.copy()

    return {"S11": s11, "S12": s12, "S21": s21, "S22": s22}


def propagate_global_smatrix(
    layers: list,
    sample: SampleConfig,
    sim: SimConditions,
    materials_db: MaterialDatabase,
    wavelength_nm: float,
    kx: np.ndarray,
    ky: np.ndarray,
    w0: np.ndarray,
    v0: np.ndarray,
    order_indices: list[tuple[int, int]],
) -> dict[str, np.ndarray]:
    """Propagate the global S-matrix through the full layer stack at one wavelength."""
    from se_simulator.rcwa.eigensolver import (
        assemble_coupled_wave_matrix,
        is_uniform_layer,
        solve_eigenproblem,
        solve_uniform_layer,
    )
    from se_simulator.rcwa.fourier import (
        build_li_matrices,
        build_toeplitz_matrix,
        compute_epsilon_fourier_2d,
        rasterize_inverse_layer,
        rasterize_layer,
    )
    from se_simulator.rcwa.modes import make_kz_array  # noqa: F401 (used below)

    nx = sim.n_harmonics_x
    ny = sim.n_harmonics_y
    n_modes = len(order_indices)
    wl_arr = np.array([wavelength_nm])

    # Superstrate
    eps_sup = materials_db.get_epsilon(sample.superstrate_material, wl_arr)[0]
    w_sup, v_sup, _ = solve_uniform_layer(eps_sup, kx, ky, n_modes)
    s_global = build_semiinfinite_smatrix(w_sup, v_sup, w0, v0, "input")

    # Layer stack
    for layer in layers:
        if is_uniform_layer(layer):
            eps_l = materials_db.get_epsilon(layer.background_material, wl_arr)[0]
            w_l, v_l, kz_l = solve_uniform_layer(eps_l, kx, ky, n_modes)
        else:
            eps_grid = rasterize_layer(layer, materials_db, wavelength_nm)
            eps_inv_grid = rasterize_inverse_layer(layer, materials_db, wavelength_nm)
            eps_f = compute_epsilon_fourier_2d(eps_grid, nx, ny)
            eps_inv_f = compute_epsilon_fourier_2d(eps_inv_grid, nx, ny)

            if sim.li_factorization:
                e_yx, e_xx = build_li_matrices(eps_f, eps_inv_f, nx, ny)
            else:
                e_yx = build_toeplitz_matrix(eps_f, nx, ny)
                e_xx = e_yx

            pq, q_mat = assemble_coupled_wave_matrix(e_xx, e_yx, e_yx, e_yx, kx, ky)
            w_l, v_l, kz_l = solve_eigenproblem(pq, q_mat)

        k0 = 2.0 * np.pi / wavelength_nm
        s_layer = build_layer_smatrix(w_l, v_l, kz_l, layer.thickness_nm, k0, w0, v0)
        s_global = redheffer_star_product(s_global, s_layer)

    # Substrate
    eps_sub = materials_db.get_epsilon(sample.substrate_material, wl_arr)[0]
    w_sub, v_sub, _ = solve_uniform_layer(eps_sub, kx, ky, n_modes)
    s_sub = build_semiinfinite_smatrix(w_sub, v_sub, w0, v0, "output")
    s_global = redheffer_star_product(s_global, s_sub)

    return s_global


def extract_jones_matrices(
    s_global: dict[str, np.ndarray],
    order_indices: list[tuple[int, int]],
    zero_order_idx: int,
    kx_inc: float,
    ky_inc: float,
    kz_inc: complex,
    kz_sub: complex,
    eps_sup: complex,
    eps_sub: complex,
    n_modes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract 2x2 Jones reflection and transmission matrices for the zeroth order.

    Field vector layout: [Ex_0..Ex_{n-1}, Ey_0..Ey_{n-1}]
      p-component (TM/x) at index: zero_order_idx
      s-component (TE/y) at index: zero_order_idx + n_modes

    jones_r = [[Rss, Rsp], [Rps, Rpp]],  jones_r[1,1] = Rpp
    """
    p_idx = zero_order_idx
    s_idx = zero_order_idx + n_modes

    s11 = s_global["S11"]
    s21 = s_global["S21"]

    # Each column of S11/S21 is the response to a unit incident amplitude
    jones_r = np.array(
        [
            [s11[s_idx, s_idx], s11[s_idx, p_idx]],  # [Rss, Rsp]
            [s11[p_idx, s_idx], s11[p_idx, p_idx]],  # [Rps, Rpp]
        ],
        dtype=complex,
    )
    jones_t = np.array(
        [
            [s21[s_idx, s_idx], s21[s_idx, p_idx]],  # [Tss, Tsp]
            [s21[p_idx, s_idx], s21[p_idx, p_idx]],  # [Tps, Tpp]
        ],
        dtype=complex,
    )
    return jones_r, jones_t


def compute_diffraction_efficiencies(
    s_global: dict[str, np.ndarray],
    order_indices: list[tuple[int, int]],
    kx_diag: np.ndarray,
    ky_diag: np.ndarray,
    kz_inc: np.ndarray,
    kz_sub: np.ndarray,
    eps_sup: complex,
    eps_sub: complex,
    inc_polarization: str = "s",
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute per-order diffraction efficiencies and totals.

    For each order i:
      R_i = Re(kz_inc[i]) / Re(kz_inc[0]) * (|r_x[i]|^2 + |r_y[i]|^2)
      T_i = Re(kz_sub[i]) * Re(eps_sup) / (Re(kz_inc[0]) * Re(eps_sub)) * (...)
    Only propagating orders (Re(kz) > 1e-6) contribute.

    Returns (R_orders, T_orders, R_total, T_total).
    """
    n_modes = len(order_indices)
    zero_order_idx = order_indices.index((0, 0))

    # Incident field vector: s or p polarization at the zero order
    size = 2 * n_modes
    e_inc = np.zeros(size, dtype=complex)
    if inc_polarization == "s":
        e_inc[zero_order_idx + n_modes] = 1.0
    else:
        e_inc[zero_order_idx] = 1.0

    # Reflected and transmitted amplitudes for all orders
    r_all = s_global["S11"] @ e_inc   # (2n,)
    t_all = s_global["S21"] @ e_inc   # (2n,)

    kz0_re = np.real(kz_inc[zero_order_idx])
    eps_sup_re = np.real(eps_sup)
    eps_sub_re = np.real(eps_sub)

    r_orders = np.zeros(n_modes)
    t_orders = np.zeros(n_modes)

    for i in range(n_modes):
        # Only propagating orders
        if np.real(kz_inc[i]) <= 1e-6:
            continue
        r_sq = abs(r_all[i]) ** 2 + abs(r_all[i + n_modes]) ** 2
        r_orders[i] = np.real(kz_inc[i]) / kz0_re * r_sq

        if np.real(kz_sub[i]) <= 1e-6:
            continue
        t_sq = abs(t_all[i]) ** 2 + abs(t_all[i + n_modes]) ** 2
        t_orders[i] = np.real(kz_sub[i]) * eps_sup_re / (kz0_re * eps_sub_re) * t_sq

    r_total = float(np.sum(r_orders))
    t_total = float(np.sum(t_orders))
    return r_orders, t_orders, r_total, t_total
