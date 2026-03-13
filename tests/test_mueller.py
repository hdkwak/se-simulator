"""Tests for Jones-to-Mueller matrix transformation."""

from __future__ import annotations

import time

import numpy as np
import pytest

from se_simulator.ellipsometer.mueller import jones_to_mueller
from se_simulator.ellipsometer.signals import EllipsometryResult

# ---------------------------------------------------------------------------
# Helper: Fresnel reflection for isotropic substrate
# ---------------------------------------------------------------------------

def _fresnel_isotropic(n_sub: complex, aoi_deg: float) -> tuple[complex, complex]:
    """Return (rpp, rss) for air/substrate interface via Fresnel equations."""
    theta_i = np.radians(aoi_deg)
    cos_i = np.cos(theta_i)
    sin_i = np.sin(theta_i)
    cos_t = np.sqrt(1.0 - (sin_i / n_sub) ** 2)

    rss = (cos_i - n_sub * cos_t) / (cos_i + n_sub * cos_t)
    rpp = (n_sub * cos_i - cos_t) / (n_sub * cos_i + cos_t)
    return rpp, rss


# Parametrized (rpp, rss) pairs covering various psi/delta regimes
_ISOTROPIC_CASES = [
    # (n_sub, aoi_deg, label)
    (4.3 + 0.07j, 65.0, "Si_65deg"),
    (1.5 + 0.0j,  45.0, "Glass_45deg"),
    (3.0 + 1.0j,  70.0, "Metal_70deg"),
    (2.0 + 0.5j,  55.0, "Absorber_55deg"),
]


def _make_isotropic_jones(n_sub: complex, aoi_deg: float, n_wl: int = 1):
    """Return (rpp, rps, rsp, rss) arrays of length n_wl for isotropic."""
    rpp, rss = _fresnel_isotropic(n_sub, aoi_deg)
    return (
        np.full(n_wl, rpp),
        np.zeros(n_wl, dtype=complex),
        np.zeros(n_wl, dtype=complex),
        np.full(n_wl, rss),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNormalization:
    """M[:,0,0] must be 1.0 after normalization."""

    def test_normalization_isotropic(self):
        rpp, rps, rsp, rss = _make_isotropic_jones(4.3 + 0.07j, 65.0, n_wl=5)
        m = jones_to_mueller(rpp, rps, rsp, rss)
        np.testing.assert_allclose(m[:, 0, 0], 1.0, atol=np.finfo(float).eps * 10)

    def test_normalization_multiple_wavelengths(self):
        """Different Jones values at each wavelength still yield m11=1."""
        n_wl = 10
        rng = np.random.default_rng(42)
        rpp = rng.standard_normal(n_wl) + 1j * rng.standard_normal(n_wl)
        rss = rng.standard_normal(n_wl) + 1j * rng.standard_normal(n_wl)
        rps = np.zeros(n_wl, dtype=complex)
        rsp = np.zeros(n_wl, dtype=complex)
        m = jones_to_mueller(rpp, rps, rsp, rss)
        np.testing.assert_allclose(m[:, 0, 0], 1.0, atol=np.finfo(float).eps * 10)


class TestImaginaryResidual:
    """For pure isotropic Jones, imaginary part of pre-real Mueller must be tiny."""

    def test_residual_below_threshold(self):
        rpp, rps, rsp, rss = _make_isotropic_jones(4.3 + 0.07j, 65.0, n_wl=3)

        # Manually compute to check imaginary part
        j = np.stack([
            np.stack([rpp, rps], axis=-1),
            np.stack([rsp, rss], axis=-1),
        ], axis=-2)
        jc = np.conj(j)
        n = rpp.shape[0]
        k = np.einsum('...ij,...kl->...ikjl', j, jc).reshape(n, 4, 4)
        a = np.array([
            [1, 0, 0, 1], [1, 0, 0, -1],
            [0, 1, 1, 0], [0, 1j, -1j, 0],
        ], dtype=complex)
        a_inv = np.linalg.inv(a)
        m_complex = (a @ k) @ a_inv
        assert np.max(np.abs(np.imag(m_complex))) < 1e-10


class TestIsotropicBlockStructure:
    """For rps=rsp=0, off-block-diagonal elements must vanish."""

    @pytest.mark.parametrize("n_sub,aoi,label", _ISOTROPIC_CASES)
    def test_off_block_zeros(self, n_sub, aoi, label):
        rpp, rps, rsp, rss = _make_isotropic_jones(n_sub, aoi, n_wl=1)
        m = jones_to_mueller(rpp, rps, rsp, rss)

        # Off-block-diagonal indices: rows 0-1 cols 2-3, rows 2-3 cols 0-1
        off_block = np.concatenate([
            m[:, 0, 2:4].ravel(),
            m[:, 1, 2:4].ravel(),
            m[:, 2, 0:2].ravel(),
            m[:, 3, 0:2].ravel(),
        ])
        np.testing.assert_allclose(off_block, 0.0, atol=1e-6)


class TestPsiDeltaConsistency:
    """Mueller elements must satisfy analytical relations for isotropic samples.

    m12 = m21 = -cos(2*psi)
    m33 = m44 = sin(2*psi)*cos(delta)
    m34 = -m43 = sin(2*psi)*sin(delta)
    """

    @pytest.mark.parametrize("n_sub,aoi,label", _ISOTROPIC_CASES)
    def test_mueller_vs_psi_delta(self, n_sub, aoi, label):
        rpp_val, rss_val = _fresnel_isotropic(n_sub, aoi)
        rpp = np.array([rpp_val])
        rss = np.array([rss_val])
        rps = np.zeros(1, dtype=complex)
        rsp = np.zeros(1, dtype=complex)

        m = jones_to_mueller(rpp, rps, rsp, rss)

        rho = rpp_val / rss_val
        psi = np.arctan(abs(rho))
        delta = np.angle(rho)

        expected_m12 = -np.cos(2 * psi)
        expected_m33 = np.sin(2 * psi) * np.cos(delta)
        expected_m34 = np.sin(2 * psi) * np.sin(delta)

        np.testing.assert_allclose(m[0, 0, 1], expected_m12, atol=1e-10,
                                   err_msg=f"m12 failed for {label}")
        np.testing.assert_allclose(m[0, 1, 0], expected_m12, atol=1e-10,
                                   err_msg=f"m21 failed for {label}")
        np.testing.assert_allclose(m[0, 2, 2], expected_m33, atol=1e-10,
                                   err_msg=f"m33 failed for {label}")
        np.testing.assert_allclose(m[0, 3, 3], expected_m33, atol=1e-10,
                                   err_msg=f"m44 failed for {label}")
        np.testing.assert_allclose(m[0, 2, 3], expected_m34, atol=1e-10,
                                   err_msg=f"m34 failed for {label}")
        np.testing.assert_allclose(m[0, 3, 2], -expected_m34, atol=1e-10,
                                   err_msg=f"m43 failed for {label}")


class TestIdentityM11:
    """Alias test: m11 = 1.0 everywhere (same as normalization, explicit name)."""

    def test_identity_m11(self):
        rpp, rps, rsp, rss = _make_isotropic_jones(3.0 + 1.0j, 70.0, n_wl=20)
        m = jones_to_mueller(rpp, rps, rsp, rss)
        np.testing.assert_allclose(m[:, 0, 0], 1.0, atol=np.finfo(float).eps * 10)


class TestVectorizationPerformance:
    """Vectorized Mueller computation for 1000 wavelengths must run < 50 ms."""

    def test_performance(self):
        n_wl = 1000
        rng = np.random.default_rng(123)
        rpp = rng.standard_normal(n_wl) + 1j * rng.standard_normal(n_wl)
        rss = rng.standard_normal(n_wl) + 1j * rng.standard_normal(n_wl)
        rps = np.zeros(n_wl, dtype=complex)
        rsp = np.zeros(n_wl, dtype=complex)

        # Warm up
        jones_to_mueller(rpp, rps, rsp, rss)

        t0 = time.perf_counter()
        jones_to_mueller(rpp, rps, rsp, rss)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert elapsed_ms < 50.0, f"Took {elapsed_ms:.1f} ms, expected < 50 ms"


class TestCsvExportFormat:
    """EllipsometryResult.export_mueller_csv produces correct format."""

    def test_csv_columns_and_values(self, tmp_path):
        n_wl = 3
        rpp, rps, rsp, rss = _make_isotropic_jones(4.3 + 0.07j, 65.0, n_wl=n_wl)
        m = jones_to_mueller(rpp, rps, rsp, rss)

        result = EllipsometryResult(
            wavelengths_nm=np.linspace(400, 800, n_wl),
            psi_deg=np.zeros(n_wl),
            delta_deg=np.zeros(n_wl),
            alpha=np.zeros(n_wl),
            beta=np.zeros(n_wl),
            chi=np.zeros(n_wl),
            xi=np.zeros(n_wl),
            jones_reflection=np.zeros((n_wl, 2, 2), dtype=complex),
            energy_conservation=np.ones(n_wl),
            mueller_matrix=m,
        )

        csv_path = tmp_path / "mueller.csv"
        result.export_mueller_csv(csv_path)

        text = csv_path.read_text()
        lines = text.strip().split("\n")

        # Header + n_wl data rows
        assert len(lines) == n_wl + 1

        # 17 columns: wavelength + 16 Mueller elements
        header_cols = lines[0].split(",")
        assert len(header_cols) == 17
        assert header_cols[0] == "wavelength_nm"
        assert header_cols[1] == "m11"
        assert header_cols[-1] == "m44"

        # Data rows: 17 columns, m11=1.0, all finite
        for line in lines[1:]:
            vals = line.split(",")
            assert len(vals) == 17
            floats = [float(v) for v in vals]
            assert all(np.isfinite(f) for f in floats)
            assert float(vals[1]) == pytest.approx(1.0, abs=1e-10)

    def test_no_mueller_raises(self, tmp_path):
        n_wl = 2
        result = EllipsometryResult(
            wavelengths_nm=np.linspace(400, 800, n_wl),
            psi_deg=np.zeros(n_wl),
            delta_deg=np.zeros(n_wl),
            alpha=np.zeros(n_wl),
            beta=np.zeros(n_wl),
            chi=np.zeros(n_wl),
            xi=np.zeros(n_wl),
            jones_reflection=np.zeros((n_wl, 2, 2), dtype=complex),
            energy_conservation=np.ones(n_wl),
        )
        with pytest.raises(ValueError, match="No Mueller matrix"):
            result.export_mueller_csv(tmp_path / "fail.csv")


@pytest.mark.skip(reason="Requires full RCWA pipeline — TODO: integration test")
class TestBothEnginePaths:
    """Test Mueller matrix via full compute_spectrum pipeline."""

    def test_compute_spectrum_populates_mueller(self):
        pass
