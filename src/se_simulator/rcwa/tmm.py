"""Transfer Matrix Method (TMM) for isotropic uniform film stacks.

All computation is fully vectorized over wavelengths using NumPy broadcasting
and batched matrix multiplication — no Python loop over wavelengths.

Performance target: 100 wavelengths, 5-layer stack < 50 ms (single CPU core).
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from se_simulator.config.schemas import GratingLayer, SampleConfig
    from se_simulator.materials.database import MaterialDatabase

logger = logging.getLogger(__name__)


def _admittance(n_complex: np.ndarray, cos_theta: np.ndarray, pol: str) -> np.ndarray:
    """Return the optical admittance for the given polarization.

    Parameters
    ----------
    n_complex:  shape (N_lambda,) complex refractive index
    cos_theta:  shape (N_lambda,) complex cosine of refraction angle
    pol:        's' or 'p'
    """
    if pol == "s":
        return n_complex * cos_theta
    # p-polarization
    return n_complex / cos_theta


def _char_matrix_batch(
    delta: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    """Build batch of 2x2 characteristic matrices.

    Parameters
    ----------
    delta : shape (N_lambda,) — phase thickness
    eta   : shape (N_lambda,) — optical admittance

    Returns
    -------
    M : shape (N_lambda, 2, 2) complex
    """
    c = np.cos(delta)
    s = np.sin(delta)
    m = np.empty((len(delta), 2, 2), dtype=complex)
    m[:, 0, 0] = c
    m[:, 0, 1] = -1j * s / eta
    m[:, 1, 0] = -1j * eta * s
    m[:, 1, 1] = c
    return m


def _matmul_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batch matrix multiply: (N, 2, 2) @ (N, 2, 2) -> (N, 2, 2)."""
    return np.einsum("...ij,...jk->...ik", a, b)


def _fresnel_from_system_matrix(
    m: np.ndarray,
    eta0: np.ndarray,
    eta_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract Fresnel r and t from the 2x2 system matrix M.

    Parameters
    ----------
    m      : shape (N_lambda, 2, 2)
    eta0   : shape (N_lambda,) — superstrate admittance
    eta_s  : shape (N_lambda,) — substrate admittance

    Returns
    -------
    r, t   : each shape (N_lambda,) complex
    """
    m00 = m[:, 0, 0]
    m01 = m[:, 0, 1]
    m10 = m[:, 1, 0]
    m11 = m[:, 1, 1]

    denom = m00 * eta0 + m01 * eta0 * eta_s + m10 + m11 * eta_s
    r = (m00 * eta0 + m01 * eta0 * eta_s - m10 - m11 * eta_s) / denom
    t = 2.0 * eta0 / denom
    return r, t


def _snell_cos(n0: np.ndarray, n_j: np.ndarray, cos_theta0: np.ndarray) -> np.ndarray:
    """Return cos(theta_j) in layer j via Snell's law.

    sin(theta_j) = (n0/n_j) * sin(theta0)
    cos(theta_j) = sqrt(1 - sin^2(theta_j))   [branch: Im(cos) >= 0]
    """
    sin_theta_j = (n0 / n_j) * np.sqrt(1.0 - cos_theta0**2)
    cos_theta_j = np.sqrt(1.0 - sin_theta_j**2 + 0j)
    # Enforce Im(cos) >= 0 (evanescent / absorbing branch)
    cos_theta_j = np.where(np.imag(cos_theta_j) < 0, -cos_theta_j, cos_theta_j)
    return cos_theta_j


def _coherent_rt(
    layers: list[GratingLayer],
    n_sup: np.ndarray,
    n_sub: np.ndarray,
    cos_theta0: np.ndarray,
    wavelengths_nm: np.ndarray,
    materials_db: MaterialDatabase,
    pol: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute coherent Fresnel (r, t) for a list of layers using TMM.

    Returns r, t each shape (N_lambda,).
    """
    n_wl = len(wavelengths_nm)

    # Superstrate admittance
    cos_theta_sup = cos_theta0.copy()
    eta0 = _admittance(n_sup, cos_theta_sup, pol)
    # Substrate admittance
    cos_theta_sub = _snell_cos(n_sup, n_sub, cos_theta0)
    eta_s = _admittance(n_sub, cos_theta_sub, pol)

    # Accumulate product matrix — start with identity
    m_total = np.zeros((n_wl, 2, 2), dtype=complex)
    m_total[:, 0, 0] = 1.0
    m_total[:, 1, 1] = 1.0

    cos_prev = cos_theta_sup
    for layer in layers:
        d = layer.thickness_nm
        mat_name = layer.background_material
        eps_j = materials_db.get_epsilon(mat_name, wavelengths_nm)
        n_j = np.sqrt(eps_j)
        cos_j = _snell_cos(n_sup, n_j, cos_theta0)
        delta_j = 2.0 * np.pi * n_j * d * cos_j / wavelengths_nm
        eta_j = _admittance(n_j, cos_j, pol)
        m_j = _char_matrix_batch(delta_j, eta_j)
        m_total = _matmul_batch(m_total, m_j)
        cos_prev = cos_j  # noqa: F841  (kept for clarity)

    r, t = _fresnel_from_system_matrix(m_total, eta0, eta_s)
    return r, t


def _intensity_rt(
    r: np.ndarray,
    t: np.ndarray,
    eta0: np.ndarray,
    eta_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Fresnel amplitudes to intensity R and T."""
    r_int = np.abs(r) ** 2
    t_int = (np.real(eta_s) / np.real(eta0)) * np.abs(t) ** 2
    return r_int, t_int


def compute_tmm(
    sample: SampleConfig,
    materials_db: MaterialDatabase,
    wavelengths_nm: np.ndarray,
    aoi_degrees: float,
    azimuth_degrees: float = 0.0,  # noqa: ARG001 — reserved for future anisotropic support
) -> tuple[np.ndarray, np.ndarray]:
    """Compute reflection Jones matrix via TMM.

    Returns
    -------
    jones_r : shape (N_lambda, 2, 2) complex — [[rpp, rps], [rsp, rss]]
    jones_t : shape (N_lambda, 2, 2) complex — [[tpp, tps], [tsp, tss]]

    Convention matches RCWAResult:
        jones_r[i, 1, 1] = rpp,  jones_r[i, 0, 0] = rss
        cross-pol terms are zero for isotropic media
    """
    n_wl = len(wavelengths_nm)
    aoi_rad = np.radians(aoi_degrees)
    cos_theta0 = np.full(n_wl, np.cos(aoi_rad), dtype=complex)

    eps_sup = materials_db.get_epsilon(sample.superstrate_material, wavelengths_nm)
    eps_sub = materials_db.get_epsilon(sample.substrate_material, wavelengths_nm)
    n_sup = np.sqrt(eps_sup)
    n_sub = np.sqrt(eps_sub)

    # Split stack at incoherent boundaries
    # Build contiguous coherent sub-stacks
    coherent_groups: list[list[GratingLayer]] = []
    current_group: list[GratingLayer] = []
    incoherent_layers: list[tuple[int, GratingLayer]] = []  # (group_after_idx, layer)

    for layer in sample.layers:
        if getattr(layer, "incoherent", False):
            if current_group:
                coherent_groups.append(current_group)
                current_group = []
            incoherent_layers.append((len(coherent_groups), layer))
            coherent_groups.append([])  # placeholder for this incoherent layer
        else:
            current_group.append(layer)
    if current_group:
        coherent_groups.append(current_group)

    def _group_rt(
        group: list[GratingLayer],
        n_sup_g: np.ndarray,
        n_sub_g: np.ndarray,
        pol: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return r, t, R, T for a coherent group."""
        if not group:
            # Identity — no layers: r=0, t=1
            eta0_g = _admittance(n_sup_g, cos_theta0, pol)
            eta_s_g = _admittance(
                n_sub_g, _snell_cos(n_sup_g, n_sub_g, cos_theta0), pol
            )
            r_g = np.zeros(n_wl, dtype=complex)
            t_g = np.ones(n_wl, dtype=complex)
            r_int_g, t_int_g = _intensity_rt(r_g, t_g, eta0_g, eta_s_g)
            return r_g, t_g, r_int_g, t_int_g

        r_g, t_g = _coherent_rt(group, n_sup_g, n_sub_g, cos_theta0, wavelengths_nm, materials_db, pol)
        eta0_g = _admittance(n_sup_g, cos_theta0, pol)
        cos_sub_g = _snell_cos(n_sup_g, n_sub_g, cos_theta0)
        eta_s_g = _admittance(n_sub_g, cos_sub_g, pol)
        r_int_g, t_int_g = _intensity_rt(r_g, t_g, eta0_g, eta_s_g)
        return r_g, t_g, r_int_g, t_int_g

    def _combine_incoherent(
        r_front: np.ndarray,
        t_front: np.ndarray,
        r_front_int: np.ndarray,
        t_front_int: np.ndarray,
        r_back_int: np.ndarray,
        t_back_int: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Combine front and back with intensity equations."""
        denom = 1.0 - r_front_int * r_back_int
        r_total_int = r_front_int + t_front_int**2 * r_back_int / denom
        t_total_int = t_front_int * t_back_int / denom
        # Phase taken from front stack only
        phase = np.angle(r_front)
        r_eff = np.sqrt(np.abs(r_total_int)) * np.exp(1j * phase)
        t_eff = np.sqrt(np.abs(t_total_int))
        return r_eff, t_eff

    def _full_stack_rt(pol: str) -> tuple[np.ndarray, np.ndarray]:
        """Compute effective (r, t) for the entire stack for one polarization."""
        if not incoherent_layers and len(coherent_groups) <= 1:
            # Simple all-coherent stack
            layers_flat = coherent_groups[0] if coherent_groups else []
            r, t = _coherent_rt(layers_flat, n_sup, n_sub, cos_theta0, wavelengths_nm, materials_db, pol)
            return r, t

        # Process groups sequentially; combine at incoherent boundaries
        # Effective n_sup / n_sub tracking is complex for multi-boundary —
        # use simplified: all coherent groups share the same n_sup/n_sub of the stack.
        # (Correct for single incoherent layer; approximation for multiple.)
        r_acc, t_acc, r_acc_int, t_acc_int = _group_rt(
            coherent_groups[0] if coherent_groups else [], n_sup, n_sub, pol
        )

        for g_idx in range(1, len(coherent_groups)):
            r_back, _t_back, r_back_int, t_back_int = _group_rt(
                coherent_groups[g_idx], n_sup, n_sub, pol
            )
            r_acc, t_acc = _combine_incoherent(
                r_acc, t_acc, r_acc_int, t_acc_int, r_back_int, t_back_int
            )
            eta0 = _admittance(n_sup, cos_theta0, pol)
            cos_sub = _snell_cos(n_sup, n_sub, cos_theta0)
            eta_s = _admittance(n_sub, cos_sub, pol)
            r_acc_int, t_acc_int = _intensity_rt(r_acc, t_acc, eta0, eta_s)

        return r_acc, t_acc

    rss, tss = _full_stack_rt("s")
    rpp, tpp = _full_stack_rt("p")

    # Check energy conservation (warn, don't raise)
    eta0_s = _admittance(n_sup, cos_theta0, "s")
    cos_sub0 = _snell_cos(n_sup, n_sub, cos_theta0)
    eta_s_s = _admittance(n_sub, cos_sub0, "s")
    eta0_p = _admittance(n_sup, cos_theta0, "p")
    eta_s_p = _admittance(n_sub, cos_sub0, "p")

    r_s_int, t_s_int = _intensity_rt(rss, tss, eta0_s, eta_s_s)
    r_p_int, t_p_int = _intensity_rt(rpp, tpp, eta0_p, eta_s_p)
    ec_s = r_s_int + t_s_int
    ec_p = r_p_int + t_p_int
    if np.any(ec_s < 0.998) or np.any(ec_p < 0.998):
        bad_s = np.sum(ec_s < 0.998)
        bad_p = np.sum(ec_p < 0.998)
        warnings.warn(
            f"TMM energy conservation: {bad_s} s-pol and {bad_p} p-pol wavelengths "
            f"have R+T < 0.998 (min s={ec_s.min():.4f}, min p={ec_p.min():.4f})",
            stacklevel=2,
        )

    # Build Jones matrices matching RCWAResult convention:
    #   jones_r[i, 1, 1] = rpp,  jones_r[i, 0, 0] = rss
    jones_r = np.zeros((n_wl, 2, 2), dtype=complex)
    jones_t = np.zeros((n_wl, 2, 2), dtype=complex)
    jones_r[:, 0, 0] = rss
    jones_r[:, 1, 1] = rpp
    jones_t[:, 0, 0] = tss
    jones_t[:, 1, 1] = tpp

    return jones_r, jones_t
