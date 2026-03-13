"""PRCSA Ellipsometer optical model for SE-RCWA Simulator."""

from se_simulator.ellipsometer.mueller import jones_to_mueller
from se_simulator.ellipsometer.prcsa import (
    compute_fourier_coefficients,
    compute_psi_delta,
    compute_spectrum,
    resolve_retardance,
)
from se_simulator.ellipsometer.signals import EllipsometryResult

__all__ = [
    "EllipsometryResult",
    "compute_fourier_coefficients",
    "compute_psi_delta",
    "compute_spectrum",
    "jones_to_mueller",
    "resolve_retardance",
]
