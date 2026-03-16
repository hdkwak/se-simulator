"""Calibration error injection and sensitivity analysis for PRCSA ellipsometer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from se_simulator.config.schemas import CompensatorRetardanceModel, DataCollectionConfig, SystemConfig
from se_simulator.ellipsometer.jones import rotation_matrix, wave_plate

if TYPE_CHECKING:
    from se_simulator.rcwa.results import RCWAResult


def apply_calibration_errors(
    jones_r: np.ndarray,
    system: SystemConfig,
    wavelength_nm: float,
    data_collection: DataCollectionConfig | None = None,
) -> np.ndarray:
    """Apply instrument angle offset errors to the effective Jones matrix.

    Models mis-calibration by applying rotation corrections for P, A, C offsets
    and a wave-plate correction for retardance error ΔΓ:

        J_eff = R(ΔA) @ J_r @ R(-ΔC) @ R(-ΔP) [@ wave_plate(C, ΔΓ)]

    Returns jones_r.copy() when all offsets are zero (< 1e-30).
    """
    ce = system.calibration_errors
    delta_p = ce.delta_P_deg
    delta_a = ce.delta_A_deg
    delta_c = ce.delta_C_deg

    # Evaluate retardance error at this wavelength
    delta_ret = 0.0
    if ce.delta_retardance_model is not None:
        from se_simulator.ellipsometer.prcsa import resolve_retardance

        tmp = system.model_copy(update={"compensator_retardance": ce.delta_retardance_model})
        delta_ret = float(resolve_retardance(tmp, np.array([wavelength_nm]))[0])

    # Early exit — identity transform
    if delta_p == 0.0 and delta_a == 0.0 and delta_c == 0.0 and delta_ret == 0.0:
        return jones_r.copy()

    j_eff = jones_r.astype(complex, copy=True)

    # Analyzer offset: rotate output frame by ΔA
    if delta_a != 0.0:
        j_eff = rotation_matrix(delta_a) @ j_eff

    # Compensator axis offset: rotate pre-compensator frame by -ΔC
    if delta_c != 0.0:
        j_eff = j_eff @ rotation_matrix(-delta_c)

    # Polarizer offset: rotate input frame by -ΔP
    if delta_p != 0.0:
        j_eff = j_eff @ rotation_matrix(-delta_p)

    # Retardance error: apply additional wave plate at nominal compensator angle
    if delta_ret != 0.0:
        c_deg = data_collection.compensator_angle_deg if data_collection is not None else 0.0
        j_eff = j_eff @ wave_plate(c_deg, delta_ret)

    return j_eff


def compute_sensitivity(
    jones_r: np.ndarray,
    system: SystemConfig,
    wavelength_nm: float,
    parameter: Literal["delta_P", "delta_A", "delta_C", "delta_retardance"],
    delta: float = 0.1,
) -> tuple[float, float]:
    """Compute numerical sensitivity of (Ψ, Δ) to a calibration parameter.

    Uses central finite difference: dΨ/dp ≈ [Ψ(p+δ) - Ψ(p-δ)] / (2δ).
    Returns (dPsi_dp, dDelta_dp) in degrees/degree.
    """
    from se_simulator.ellipsometer.prcsa import compute_psi_delta

    def psi_delta_at(offset: float) -> tuple[float, float]:
        ce = system.calibration_errors
        if parameter == "delta_P":
            new_ce = ce.model_copy(update={"delta_P_deg": ce.delta_P_deg + offset})
        elif parameter == "delta_A":
            new_ce = ce.model_copy(update={"delta_A_deg": ce.delta_A_deg + offset})
        elif parameter == "delta_C":
            new_ce = ce.model_copy(update={"delta_C_deg": ce.delta_C_deg + offset})
        else:  # delta_retardance
            cur = 0.0
            if ce.delta_retardance_model is not None:
                from se_simulator.ellipsometer.prcsa import resolve_retardance

                tmp = system.model_copy(
                    update={"compensator_retardance": ce.delta_retardance_model}
                )
                cur = float(resolve_retardance(tmp, np.array([wavelength_nm]))[0])
            new_model = CompensatorRetardanceModel(type="constant", value=cur + offset)
            new_ce = ce.model_copy(update={"delta_retardance_model": new_model})

        new_system = system.model_copy(update={"calibration_errors": new_ce})
        j_eff = apply_calibration_errors(jones_r, new_system, wavelength_nm)
        return compute_psi_delta(j_eff)

    psi_p, delta_p = psi_delta_at(+delta)
    psi_m, delta_m = psi_delta_at(-delta)
    return (psi_p - psi_m) / (2.0 * delta), (delta_p - delta_m) / (2.0 * delta)


def sensitivity_spectrum(
    rcwa_result: RCWAResult,
    system: SystemConfig,
    parameters: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Compute sensitivity spectra for all (or specified) calibration parameters.

    Returns dict with keys '<param>_psi' and '<param>_delta', each shape (Nλ,).
    Valid parameter names: 'delta_P', 'delta_A', 'delta_C', 'delta_retardance'.
    """
    if parameters is None:
        parameters = ["delta_P", "delta_A", "delta_C", "delta_retardance"]

    wavelengths_nm = np.asarray(rcwa_result.wavelengths_nm)
    n_wl = len(wavelengths_nm)

    result: dict[str, np.ndarray] = {}
    for param in parameters:
        psi_sens = np.empty(n_wl)
        delta_sens = np.empty(n_wl)
        for i, wl in enumerate(wavelengths_nm):
            jr = rcwa_result.jones_reflection[i]
            dp, dd = compute_sensitivity(  # type: ignore[arg-type]
                jr, system, float(wl), parameter=param
            )
            psi_sens[i] = dp
            delta_sens[i] = dd
        result[f"{param}_psi"] = psi_sens
        result[f"{param}_delta"] = delta_sens

    return result
