"""Pydantic models for the self-contained .sespec file format.

A .sespec file bundles:
  - spectrum: measured or simulated psi/delta (+ optional Mueller / Jones)
  - provenance: how the spectrum was obtained (simulated vs measured)
  - recipe: MeasurementRecipe describing how fitting should be done
  - fit_results: written back after a fitting run
"""

from __future__ import annotations

import ast
import base64
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, Field

CURRENT_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Array encoding
# ---------------------------------------------------------------------------

class EncodedArray(BaseModel):
    """A numerical array serialisable as full-precision text or base64 bytes.

    TEXT encoding writes each element with full float64 repr precision
    (``%.17g`` format), making the file human-readable and diff-friendly.
    BASE64 encoding writes the raw little-endian float64 bytes as a base64
    string — useful for Jones matrices (shape N×2×2).
    """

    encoding: Literal["text", "base64"] = "text"
    dtype: str = "float64"
    shape: list[int]
    data: str  # "[1.0, ...]" for text, base64 string for base64

    @classmethod
    def from_ndarray(
        cls,
        arr: np.ndarray,
        encoding: Literal["text", "base64"] = "text",
    ) -> EncodedArray:
        arr64 = np.asarray(arr, dtype="float64")
        if encoding == "base64":
            buf = arr64.tobytes()
            return cls(
                encoding="base64",
                dtype="float64",
                shape=list(arr64.shape),
                data=base64.b64encode(buf).decode("ascii"),
            )
        # TEXT — full float64 precision via %.17g
        flat = arr64.flatten()
        data_str = "[" + ", ".join(f"{v:.17g}" for v in flat) + "]"
        return cls(
            encoding="text",
            dtype=str(arr64.dtype),
            shape=list(arr64.shape),
            data=data_str,
        )

    def to_ndarray(self) -> np.ndarray:
        if self.encoding == "base64":
            buf = base64.b64decode(self.data)
            arr = np.frombuffer(buf, dtype=self.dtype)
        else:
            arr = np.array(ast.literal_eval(self.data), dtype=self.dtype)
        return arr.reshape(self.shape)


# ---------------------------------------------------------------------------
# Spectrum data
# ---------------------------------------------------------------------------

class SpectrumData(BaseModel):
    """Ellipsometric spectra.  psi/delta are always present; others optional."""

    wavelengths_nm: EncodedArray
    psi_deg: EncodedArray
    delta_deg: EncodedArray

    # Mueller-derived — absent when only psi/delta was computed / measured
    alpha: EncodedArray | None = None
    beta: EncodedArray | None = None
    chi: EncodedArray | None = None
    xi: EncodedArray | None = None

    # Jones matrices per wavelength: stored as two float64 arrays of shape (N,2,2)
    jones_reflection_real: EncodedArray | None = None
    jones_reflection_imag: EncodedArray | None = None

    @classmethod
    def from_ellipsometry_result(
        cls,
        result: object,  # EllipsometryResult — avoid circular import
        include_jones: bool = True,
        encoding: Literal["text", "base64"] = "text",
    ) -> SpectrumData:
        def enc(arr: np.ndarray | None) -> EncodedArray | None:
            return EncodedArray.from_ndarray(arr, encoding) if arr is not None else None

        jr = getattr(result, "jones_reflection", None)
        return cls(
            wavelengths_nm=enc(result.wavelengths_nm),  # type: ignore[arg-type]
            psi_deg=enc(result.psi_deg),  # type: ignore[arg-type]
            delta_deg=enc(result.delta_deg),  # type: ignore[arg-type]
            alpha=enc(getattr(result, "alpha", None)),
            beta=enc(getattr(result, "beta", None)),
            chi=enc(getattr(result, "chi", None)),
            xi=enc(getattr(result, "xi", None)),
            jones_reflection_real=enc(jr.real) if include_jones and jr is not None else None,
            jones_reflection_imag=enc(jr.imag) if include_jones and jr is not None else None,
        )

    def wavelengths(self) -> np.ndarray:
        return self.wavelengths_nm.to_ndarray()

    def psi(self) -> np.ndarray:
        return self.psi_deg.to_ndarray()

    def delta(self) -> np.ndarray:
        return self.delta_deg.to_ndarray()

    def jones(self) -> np.ndarray | None:
        """Reconstruct complex Jones matrix array of shape (N, 2, 2), or None."""
        if self.jones_reflection_real is None or self.jones_reflection_imag is None:
            return None
        return self.jones_reflection_real.to_ndarray() + 1j * self.jones_reflection_imag.to_ndarray()


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class SimulatedProvenance(BaseModel):
    """Records how a simulated spectrum was produced."""

    origin: Literal["simulated"] = "simulated"

    # Full stack snapshot — self-contained, not a path reference
    stack_snapshot: Annotated[dict, Field(description="Stack model_dump")]

    # Algorithm knobs used for the simulation
    simulation_conditions: Annotated[dict, Field(description="SimulationConditionsEmbed model_dump")]

    # Instrument model applied; None means ideal (no calibration errors)
    system_config_snapshot: Annotated[dict | None, Field(description="SystemConfig model_dump")] = None

    # ISO-8601 timestamp
    simulated_at: str | None = None

    # App version string for reproducibility
    simulator_version: str | None = None


class MeasuredProvenance(BaseModel):
    """Records the instrument conditions at the time of data acquisition."""

    origin: Literal["measured"] = "measured"

    # Optical angles and wavelength range at acquisition time
    data_collection: Annotated[dict, Field(description="DataCollectionConfig model_dump")]

    # Instrument hardware configuration — enables post-hoc re-calibration
    system_config_snapshot: Annotated[dict | None, Field(description="SystemConfig model_dump")] = None

    # Free-text instrument identifier (serial number, lab name, etc.)
    instrument_id: str | None = None

    # ISO-8601 timestamp
    measured_at: str | None = None

    # Original file path for audit trail
    source_file: str | None = None


# ---------------------------------------------------------------------------
# Root document
# ---------------------------------------------------------------------------

class SpectrumFile(BaseModel):
    """Root model for a .sespec file.

    Fields
    ------
    schema_version:
        Version string for migration. Bump minor for additive changes,
        major for breaking changes requiring a migration function.
    provenance:
        How the spectrum was obtained.  Discriminated on the ``origin`` field.
    spectrum:
        The numerical payload.  Always present.
    recipe:
        ``MeasurementRecipe`` model_dump.  None when no recipe is attached yet.
    fit_results:
        ``FitResults`` model_dump written back after a fitting run.  None before
        fitting.  Lives at the top level (not inside recipe) because the recipe
        is immutable / reusable across multiple fitting runs.
    """

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION)

    provenance: SimulatedProvenance | MeasuredProvenance = Field(discriminator="origin")

    spectrum: SpectrumData

    # Raw dicts rather than the Pydantic types themselves to avoid circular
    # imports and to keep schema.py free of recipe-module dependencies.
    # Callers can round-trip via MeasurementRecipe.model_validate(sf.recipe).
    recipe: dict | None = None

    fit_results: dict | None = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def attach_recipe(self, recipe: object) -> SpectrumFile:
        """Return a new SpectrumFile with *recipe* embedded (model_dump)."""
        return self.model_copy(update={"recipe": recipe.model_dump(mode="json")})  # type: ignore[union-attr]

    def attach_fit_results(self, fit_results: object) -> SpectrumFile:
        """Return a new SpectrumFile with *fit_results* embedded."""
        return self.model_copy(update={"fit_results": fit_results.model_dump(mode="json")})  # type: ignore[union-attr]

    def get_recipe(self):
        """Parse and return the embedded recipe as a MeasurementRecipe, or None."""
        if self.recipe is None:
            return None
        from se_simulator.config.recipe import MeasurementRecipe
        return MeasurementRecipe.model_validate(self.recipe)

    def get_fit_results(self):
        """Parse and return the embedded fit_results as a FitResults, or None."""
        if self.fit_results is None:
            return None
        from se_simulator.config.recipe import FitResults
        return FitResults.model_validate(self.fit_results)
