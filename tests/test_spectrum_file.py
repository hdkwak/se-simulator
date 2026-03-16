"""Tests for the .sespec self-contained spectrum file format."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from se_simulator.spectrum import (
    EncodedArray,
    MeasuredProvenance,
    SimulatedProvenance,
    SpectrumData,
    SpectrumFile,
    load_spectrum,
    save_spectrum,
)
from se_simulator.spectrum.migrations import migrate
from se_simulator.spectrum.schema import CURRENT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spectrum(n: int = 5) -> SpectrumData:
    wl = np.linspace(400.0, 800.0, n)
    psi = np.linspace(20.0, 30.0, n)
    delta = np.linspace(170.0, 180.0, n)
    return SpectrumData(
        wavelengths_nm=EncodedArray.from_ndarray(wl),
        psi_deg=EncodedArray.from_ndarray(psi),
        delta_deg=EncodedArray.from_ndarray(delta),
    )


def _make_simulated_provenance() -> SimulatedProvenance:
    return SimulatedProvenance(
        stack_snapshot={"layers": [{"material": "SiO2", "thickness_nm": 100.0}]},
        simulation_conditions={"n_harmonics_x": 5, "n_harmonics_y": 5},
        simulated_at="2026-03-16T00:00:00Z",
        simulator_version="test",
    )


def _make_measured_provenance() -> MeasuredProvenance:
    return MeasuredProvenance(
        data_collection={
            "aoi_deg": 65.0,
            "wavelength_start_nm": 400.0,
            "wavelength_end_nm": 800.0,
            "wavelength_step_nm": 10.0,
        },
        system_config_snapshot={"instrument_name": "SE Simulator"},
        instrument_id="SN-001",
        measured_at="2026-03-16T08:00:00Z",
    )


# ---------------------------------------------------------------------------
# EncodedArray round-trip
# ---------------------------------------------------------------------------

class TestEncodedArray:
    def test_text_roundtrip_1d(self):
        arr = np.array([1.0, 2.5, np.pi], dtype="float64")
        enc = EncodedArray.from_ndarray(arr, encoding="text")
        assert enc.encoding == "text"
        np.testing.assert_array_equal(enc.to_ndarray(), arr)

    def test_text_full_precision(self):
        # Value with many significant digits — must survive round-trip exactly
        arr = np.array([1.0 / 3.0], dtype="float64")
        enc = EncodedArray.from_ndarray(arr, encoding="text")
        rt = enc.to_ndarray()
        assert rt[0] == arr[0], "Full float64 precision not preserved in text encoding"

    def test_base64_roundtrip_2d(self):
        arr = np.random.default_rng(0).standard_normal((3, 4))
        enc = EncodedArray.from_ndarray(arr, encoding="base64")
        assert enc.encoding == "base64"
        np.testing.assert_array_equal(enc.to_ndarray(), arr)

    def test_shape_preserved(self):
        arr = np.ones((2, 2, 2))
        enc = EncodedArray.from_ndarray(arr)
        assert enc.shape == [2, 2, 2]
        assert enc.to_ndarray().shape == (2, 2, 2)

    def test_complex_jones_split(self):
        """Jones matrix round-trip via real/imag split."""
        jr = np.random.default_rng(1).standard_normal((4, 2, 2)) + \
             1j * np.random.default_rng(2).standard_normal((4, 2, 2))
        real_enc = EncodedArray.from_ndarray(jr.real)
        imag_enc = EncodedArray.from_ndarray(jr.imag)
        jr_rt = real_enc.to_ndarray() + 1j * imag_enc.to_ndarray()
        np.testing.assert_array_equal(jr_rt, jr)


# ---------------------------------------------------------------------------
# SpectrumData
# ---------------------------------------------------------------------------

class TestSpectrumData:
    def test_from_ellipsometry_result(self):
        from dataclasses import dataclass

        @dataclass
        class FakeResult:
            wavelengths_nm: np.ndarray
            psi_deg: np.ndarray
            delta_deg: np.ndarray
            alpha: np.ndarray
            beta: np.ndarray
            chi: np.ndarray
            xi: np.ndarray
            jones_reflection: np.ndarray

        n = 4
        jr = np.ones((n, 2, 2), dtype=complex)
        result = FakeResult(
            wavelengths_nm=np.linspace(400, 700, n),
            psi_deg=np.ones(n) * 25.0,
            delta_deg=np.ones(n) * 175.0,
            alpha=np.zeros(n),
            beta=np.zeros(n),
            chi=np.zeros(n),
            xi=np.zeros(n),
            jones_reflection=jr,
        )
        sd = SpectrumData.from_ellipsometry_result(result, include_jones=True)
        assert sd.jones_reflection_real is not None
        assert sd.jones_reflection_imag is not None
        np.testing.assert_array_equal(sd.wavelengths(), result.wavelengths_nm)
        np.testing.assert_array_equal(sd.psi(), result.psi_deg)

    def test_jones_reconstructed(self):
        n = 3
        jr = np.random.default_rng(3).standard_normal((n, 2, 2)) + \
             1j * np.random.default_rng(4).standard_normal((n, 2, 2))
        sd = _make_spectrum(n)
        sd = sd.model_copy(update={
            "jones_reflection_real": EncodedArray.from_ndarray(jr.real),
            "jones_reflection_imag": EncodedArray.from_ndarray(jr.imag),
        })
        np.testing.assert_array_equal(sd.jones(), jr)

    def test_jones_none_when_absent(self):
        assert _make_spectrum().jones() is None


# ---------------------------------------------------------------------------
# SpectrumFile construction and helpers
# ---------------------------------------------------------------------------

class TestSpectrumFile:
    def test_simulated_provenance(self):
        sf = SpectrumFile(
            provenance=_make_simulated_provenance(),
            spectrum=_make_spectrum(),
        )
        assert sf.provenance.origin == "simulated"
        assert sf.schema_version == CURRENT_SCHEMA_VERSION

    def test_measured_provenance(self):
        sf = SpectrumFile(
            provenance=_make_measured_provenance(),
            spectrum=_make_spectrum(),
        )
        assert sf.provenance.origin == "measured"

    def test_attach_recipe_roundtrip(self):
        from se_simulator.config.recipe import MeasurementRecipe

        recipe = MeasurementRecipe.model_validate({
            "metadata": {"recipe_type": "measurement"},
            "floating_parameters": [],
            "forward_model": {
                "data_collection": {},
                "stack": {
                    "inline": {
                        "superstrate": {"library_name": "Air", "source": "library", "name": "Air"},
                        "substrate": {"library_name": "Si", "source": "library", "name": "Si"},
                        "layers": [],
                    }
                },
            },
            "fitting_configuration": {},
        })
        sf = SpectrumFile(
            provenance=_make_simulated_provenance(),
            spectrum=_make_spectrum(),
        ).attach_recipe(recipe)

        assert sf.recipe is not None
        recipe_rt = sf.get_recipe()
        assert recipe_rt is not None

    def test_fit_results_none_by_default(self):
        sf = SpectrumFile(provenance=_make_simulated_provenance(), spectrum=_make_spectrum())
        assert sf.fit_results is None
        assert sf.get_fit_results() is None

    def test_discriminated_union_simulated(self):
        raw = SpectrumFile(
            provenance=_make_simulated_provenance(),
            spectrum=_make_spectrum(),
        ).model_dump(mode="json")
        # Pydantic should discriminate correctly on load
        sf = SpectrumFile.model_validate(raw)
        assert isinstance(sf.provenance, SimulatedProvenance)

    def test_discriminated_union_measured(self):
        raw = SpectrumFile(
            provenance=_make_measured_provenance(),
            spectrum=_make_spectrum(),
        ).model_dump(mode="json")
        sf = SpectrumFile.model_validate(raw)
        assert isinstance(sf.provenance, MeasuredProvenance)


# ---------------------------------------------------------------------------
# I/O round-trip
# ---------------------------------------------------------------------------

class TestIO:
    def test_save_load_simulated(self, tmp_path):
        sf = SpectrumFile(
            provenance=_make_simulated_provenance(),
            spectrum=_make_spectrum(10),
        )
        path = tmp_path / "test.sespec"
        save_spectrum(sf, path)
        assert path.exists()
        sf_rt = load_spectrum(path)
        assert sf_rt.schema_version == CURRENT_SCHEMA_VERSION
        assert sf_rt.provenance.origin == "simulated"
        np.testing.assert_array_equal(sf_rt.spectrum.wavelengths(), sf.spectrum.wavelengths())
        np.testing.assert_array_equal(sf_rt.spectrum.psi(), sf.spectrum.psi())
        np.testing.assert_array_equal(sf_rt.spectrum.delta(), sf.spectrum.delta())

    def test_save_load_measured(self, tmp_path):
        sf = SpectrumFile(
            provenance=_make_measured_provenance(),
            spectrum=_make_spectrum(8),
        )
        path = tmp_path / "measured.sespec"
        save_spectrum(sf, path)
        sf_rt = load_spectrum(path)
        assert sf_rt.provenance.origin == "measured"
        mp = sf_rt.provenance
        assert isinstance(mp, MeasuredProvenance)
        assert mp.instrument_id == "SN-001"

    def test_none_fields_omitted_in_yaml(self, tmp_path):
        sf = SpectrumFile(
            provenance=_make_simulated_provenance(),
            spectrum=_make_spectrum(3),
        )
        path = tmp_path / "minimal.sespec"
        save_spectrum(sf, path)
        content = path.read_text()
        assert "recipe:" not in content
        assert "fit_results:" not in content
        assert "jones_reflection_real:" not in content

    def test_system_config_snapshot_in_measured(self, tmp_path):
        prov = _make_measured_provenance()
        assert prov.system_config_snapshot is not None
        sf = SpectrumFile(provenance=prov, spectrum=_make_spectrum())
        path = tmp_path / "with_system.sespec"
        save_spectrum(sf, path)
        sf_rt = load_spectrum(path)
        assert sf_rt.provenance.system_config_snapshot is not None
        assert sf_rt.provenance.system_config_snapshot["instrument_name"] == "SE Simulator"

    def test_full_precision_survives_io(self, tmp_path):
        arr = np.array([1.0 / 3.0, np.pi, np.e], dtype="float64")
        spectrum = SpectrumData(
            wavelengths_nm=EncodedArray.from_ndarray(np.array([400.0, 500.0, 600.0])),
            psi_deg=EncodedArray.from_ndarray(arr),
            delta_deg=EncodedArray.from_ndarray(arr),
        )
        sf = SpectrumFile(provenance=_make_simulated_provenance(), spectrum=spectrum)
        path = tmp_path / "precision.sespec"
        save_spectrum(sf, path)
        sf_rt = load_spectrum(path)
        np.testing.assert_array_equal(sf_rt.spectrum.psi(), arr)


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

class TestMigrations:
    def test_already_current_version_passes_through(self):
        raw = {"schema_version": CURRENT_SCHEMA_VERSION, "other": "data"}
        assert migrate(raw) is raw

    def test_unknown_version_raises(self):
        with pytest.raises(ValueError, match="No migration path"):
            migrate({"schema_version": "99.0"})
