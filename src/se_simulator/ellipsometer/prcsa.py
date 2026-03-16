"""PRCSA ellipsometer signal computation: Fourier coefficients, psi/delta, spectrum."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

from se_simulator.config.schemas import DataCollectionConfig, SystemConfig
from se_simulator.ellipsometer.jones import linear_polarizer, rotating_compensator
from se_simulator.ellipsometer.signals import EllipsometryResult
from se_simulator.rcwa.results import RCWAResult

if TYPE_CHECKING:
    from se_simulator.config.schemas import SampleConfig, SimConditions
    from se_simulator.rcwa.engine import RCWAEngine


def _angles_from(
    system: SystemConfig,
    data_collection: DataCollectionConfig | None,
) -> tuple[float, float, float]:
    """Return (polarizer_deg, analyzer_deg, compensator_deg) from DataCollectionConfig.

    Falls back to 45 / 45 / 0 defaults when no DataCollectionConfig is available.
    """
    if data_collection is not None:
        return (
            data_collection.polarizer_angle_deg,
            data_collection.analyzer_angle_deg,
            data_collection.compensator_angle_deg,
        )
    # Graceful fallback — PSA geometry defaults
    return 45.0, 45.0, 0.0

_M = 1024  # Number of compensator rotation samples


def resolve_retardance(
    system: SystemConfig,
    wavelengths_nm: np.ndarray,
) -> np.ndarray:
    """Evaluate compensator retardance Γ(λ) in degrees for all wavelengths."""
    model = system.compensator_retardance
    n = len(wavelengths_nm)

    if model.type == "constant":
        return np.full(n, float(model.value))

    if model.type == "polynomial":
        # coefficients = [c0, c1, c2, ...] where poly = c0 + c1*λ + c2*λ² + ...
        # numpy.polyval uses descending order, so reverse
        coeffs = list(reversed(model.coefficients))
        return np.polyval(coeffs, wavelengths_nm).astype(float)

    if model.type == "tabulated":
        raw = np.genfromtxt(model.file_path, delimiter=",", comments="#")
        if raw.ndim == 1:
            raw = raw[np.newaxis, :]
        wl_tab = raw[:, 0]
        ret_tab = raw[:, 1]
        cs = CubicSpline(wl_tab, ret_tab)
        return cs(wavelengths_nm).astype(float)

    raise ValueError(f"Unknown retardance model type: {model.type!r}")


def compute_fourier_coefficients(
    jones_r: np.ndarray,
    system: SystemConfig,
    retardance_deg: float,
    data_collection: DataCollectionConfig | None = None,
) -> dict[str, float]:
    """Compute normalized PRCSA Fourier coefficients via numerical FFT.

    I(δ) = I₀ [1 + α cos(2δ) + β sin(2δ) + χ cos(4δ) + ξ sin(4δ)]

    Optical angles (polarizer, analyzer, compensator) are taken from
    *data_collection* when provided; otherwise defaults (45°, 45°, 0°) are used.
    """
    p_deg, a_deg, c_deg = _angles_from(system, data_collection)

    j_p = linear_polarizer(p_deg)
    j_a = linear_polarizer(a_deg)

    e_inc = np.array([1.0, 0.0], dtype=complex)
    e_after_p = j_p @ e_inc  # fixed; precompute

    delta_vals = np.linspace(0.0, 360.0, _M, endpoint=False)  # degrees

    intensity = np.empty(_M)
    for idx, d in enumerate(delta_vals):
        j_c = rotating_compensator(c_deg, d, retardance_deg)
        e_out = j_a @ jones_r @ j_c @ e_after_p
        intensity[idx] = float(np.real(np.dot(e_out.conj(), e_out)))

    fft = np.fft.rfft(intensity)
    i0 = fft[0].real / _M  # DC (mean intensity × M / M)

    alpha = 2.0 * fft[2].real / (_M * i0)
    beta = -2.0 * fft[2].imag / (_M * i0)
    chi = 2.0 * fft[4].real / (_M * i0)
    xi = -2.0 * fft[4].imag / (_M * i0)

    return {"alpha": float(alpha), "beta": float(beta),
            "chi": float(chi), "xi": float(xi), "I0": float(i0)}


def compute_psi_delta(
    jones_r: np.ndarray,
    unwrap: bool = False,
) -> tuple[float, float]:
    """Compute ellipsometric angles Ψ, Δ from the Jones reflection matrix.

    ρ = Rpp / Rss,  Ψ = arctan(|ρ|) in degrees,  Δ = arg(ρ) in degrees.
    jones_r layout: [[Rpp, Rps], [Rsp, Rss]]  (row 0 = p-out, row 1 = s-out)
    Wait — check RCWA convention: jones_r[1,1]=Rpp, jones_r[0,0]=Rss.
    Consistent with RCWA engine: [[Rss, Rsp],[Rps, Rpp]] → jones_r[0,0]=Rss, jones_r[1,1]=Rpp.
    """
    rpp = jones_r[1, 1]
    rss = jones_r[0, 0]

    if abs(rss) < 1e-30:
        rss = 1e-30 + 0j

    rho = rpp / rss
    psi_deg = float(np.degrees(np.arctan(abs(rho))))
    delta_deg = float(np.degrees(np.angle(rho)))
    return psi_deg, delta_deg


def apply_depolarization(
    jones_r: np.ndarray,
    system: SystemConfig,
    wavelength_nm: float,
    rcwa_engine: RCWAEngine,
    sample: SampleConfig,
    sim: SimConditions,
) -> np.ndarray:
    """Apply incoherent averaging over AOI/wavelength spread (v1.0 simplified).

    When aoi_spread_deg > 0, runs RCWA at 5 Gaussian sample points (±2σ) and
    averages the intensity signals I(δ) incoherently, then returns the effective
    Jones matrix that reproduces the averaged Ψ/Δ.

    Currently a stub: returns jones_r unchanged when no RCWA averaging is needed.
    Full multi-AOI averaging is a v2.0 feature.
    """  # noqa: E501
    depol = system.depolarization
    if depol.aoi_spread_deg == 0.0 and depol.wavelength_bandwidth_nm == 0.0:
        return jones_r.copy()

    # v1.0: return unchanged — full Mueller-matrix averaging deferred to v2.0
    return jones_r.copy()


def compute_spectrum(
    rcwa_result: RCWAResult,
    system: SystemConfig,
    data_collection: DataCollectionConfig | None = None,
    include_calibration_errors: bool = False,
    rcwa_engine: RCWAEngine | None = None,
    sample: SampleConfig | None = None,
    sim: SimConditions | None = None,
) -> EllipsometryResult:
    """Compute the full ellipsometric spectrum from RCWA results.

    Optical angles (polarizer, analyzer, compensator) come from *data_collection*
    when provided; otherwise defaults (45°/45°/0°) are used.
    Instrument calibration settings come from *system*.
    """
    from se_simulator.ellipsometer.calibration import apply_calibration_errors

    wavelengths_nm = np.asarray(rcwa_result.wavelengths_nm)
    n_wl = len(wavelengths_nm)

    retardances = resolve_retardance(system, wavelengths_nm)

    depol = system.depolarization
    use_depolarization = (
        (depol.aoi_spread_deg > 0.0 or depol.wavelength_bandwidth_nm > 0.0)
        and rcwa_engine is not None
        and sample is not None
        and sim is not None
    )

    psi_arr = np.empty(n_wl)
    delta_arr = np.empty(n_wl)
    alpha_arr = np.empty(n_wl)
    beta_arr = np.empty(n_wl)
    chi_arr = np.empty(n_wl)
    xi_arr = np.empty(n_wl)

    for i, wl in enumerate(wavelengths_nm):
        jr = rcwa_result.jones_reflection[i]
        if include_calibration_errors:
            jr = apply_calibration_errors(
                jr, system, float(wl), data_collection=data_collection
            )
        if use_depolarization:
            jr = apply_depolarization(jr, system, float(wl), rcwa_engine, sample, sim)

        psi_arr[i], delta_arr[i] = compute_psi_delta(jr)

        coeffs = compute_fourier_coefficients(
            jr, system, float(retardances[i]), data_collection=data_collection
        )
        alpha_arr[i] = coeffs["alpha"]
        beta_arr[i] = coeffs["beta"]
        chi_arr[i] = coeffs["chi"]
        xi_arr[i] = coeffs["xi"]

    from se_simulator.ellipsometer.mueller import jones_to_mueller

    jones_r = rcwa_result.jones_reflection  # (N_lambda, 2, 2)
    mueller = jones_to_mueller(
        rpp=jones_r[:, 1, 1],
        rps=jones_r[:, 1, 0],
        rsp=jones_r[:, 0, 1],
        rss=jones_r[:, 0, 0],
    )

    return EllipsometryResult(
        wavelengths_nm=wavelengths_nm,
        psi_deg=psi_arr,
        delta_deg=delta_arr,
        alpha=alpha_arr,
        beta=beta_arr,
        chi=chi_arr,
        xi=xi_arr,
        jones_reflection=rcwa_result.jones_reflection,
        energy_conservation=rcwa_result.energy_conservation,
        mueller_matrix=mueller,
    )
