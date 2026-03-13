"""Jones-to-Mueller matrix transformation.

Converts complex Jones reflection coefficients to a real 4x4 normalized
Mueller matrix via the Kronecker product and the A-matrix similarity
transformation.

Phase convention
----------------
Follows the project convention: jones_r[1,1] = Rpp, jones_r[0,0] = Rss.
The Mueller matrix is normalized so that M[0,0] = 1.0.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def jones_to_mueller(
    rpp: np.ndarray,
    rps: np.ndarray,
    rsp: np.ndarray,
    rss: np.ndarray,
) -> np.ndarray:
    """Compute 4x4 real normalized Mueller matrix from Jones reflection coefficients.

    Uses the relation M = A (J otimes J*) A^{-1} where A is the 4x4
    similarity matrix connecting coherency and Stokes vectors.

    Parameters
    ----------
    rpp, rps, rsp, rss : np.ndarray, shape (N_lambda,), complex
        Jones reflection matrix elements at each wavelength.
        Convention: jones_r = [[Rss, Rsp], [Rps, Rpp]] so
        rpp = jones_r[:, 1, 1], rss = jones_r[:, 0, 0],
        rps = jones_r[:, 1, 0], rsp = jones_r[:, 0, 1].

    Returns
    -------
    m_normalized : np.ndarray, shape (N_lambda, 4, 4), real float
        Mueller matrix normalized by M[:,0,0] so that m_11 = 1.0 everywhere.

    Complexity
    ----------
    O(N_lambda) time and space — dominated by element-wise complex arithmetic.
    """
    rpp = np.asarray(rpp, dtype=complex)
    rps = np.asarray(rps, dtype=complex)
    rsp = np.asarray(rsp, dtype=complex)
    rss = np.asarray(rss, dtype=complex)

    n_lambda = rpp.shape[0]

    # Build Jones matrix: J[i] = [[rpp[i], rps[i]], [rsp[i], rss[i]]]
    # Shape: (N_lambda, 2, 2)
    j = np.stack([  # noqa: N806
        np.stack([rpp, rps], axis=-1),
        np.stack([rsp, rss], axis=-1),
    ], axis=-2)

    jc = np.conj(j)  # noqa: N806

    # Kronecker product K = J ⊗ J*, shape (N_lambda, 4, 4)
    k = np.einsum('...ij,...kl->...ikjl', j, jc).reshape(n_lambda, 4, 4)  # noqa: N806

    # Similarity matrix A and its inverse
    a = np.array([  # noqa: N806
        [1,  0,   0,  1],
        [1,  0,   0, -1],
        [0,  1,   1,  0],
        [0, 1j, -1j,  0],
    ], dtype=complex)
    a_inv = np.linalg.inv(a)  # noqa: N806

    # M_complex = A @ K @ A_inv, shape (N_lambda, 4, 4)
    m_complex = (a @ k) @ a_inv  # noqa: N806

    # Imaginary residual check
    max_imag = np.max(np.abs(np.imag(m_complex)))
    if max_imag > 1e-10:
        logger.warning(
            "jones_to_mueller: imaginary residual %.3e exceeds 1e-10. "
            "Possible numerical issue.", max_imag
        )

    m = np.real(m_complex)  # noqa: N806

    # Normalize by M11
    m11 = m[:, 0, 0][:, np.newaxis, np.newaxis]  # noqa: N806
    m_normalized = m / m11  # noqa: N806

    return m_normalized
