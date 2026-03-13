"""Elementary Jones matrices for PRCSA ellipsometer optical components."""

from __future__ import annotations

import numpy as np


def rotation_matrix(angle_deg: float) -> np.ndarray:
    """2D rotation matrix [[cos θ, -sin θ], [sin θ, cos θ]]."""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=complex)


def linear_polarizer(angle_deg: float) -> np.ndarray:
    """Jones matrix for a linear polarizer with transmission axis at angle_deg from x-axis."""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c * c, c * s], [c * s, s * s]], dtype=complex)


def wave_plate(fast_axis_deg: float, retardance_deg: float) -> np.ndarray:
    """Jones matrix for a wave plate.

    fast_axis_deg: angle of fast axis from x-axis.
    retardance_deg: phase retardance Γ — fast axis phase leads slow axis by Γ.
    """
    gamma = np.radians(retardance_deg)
    r = rotation_matrix(-fast_axis_deg)
    r_inv = rotation_matrix(fast_axis_deg)
    # In fast-axis frame: fast gets 0 phase, slow gets +Γ
    waveplate_diag = np.diag(np.array([1.0, np.exp(1j * gamma)], dtype=complex))
    return r_inv @ waveplate_diag @ r


def rotating_compensator(
    c_deg: float,
    delta_deg: float,
    retardance_deg: float,
) -> np.ndarray:
    """Wave plate with fast axis at (c_deg + delta_deg)."""
    return wave_plate(c_deg + delta_deg, retardance_deg)
