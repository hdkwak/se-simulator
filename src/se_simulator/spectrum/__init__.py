"""spectrum — self-contained .sespec file format.

Public API
----------
SpectrumFile
    Root Pydantic model for a .sespec file.
SpectrumData
    Numerical payload (psi/delta + optional Mueller/Jones arrays).
SimulatedProvenance
    Provenance for spectra produced by the simulator.
MeasuredProvenance
    Provenance for experimentally acquired spectra.
EncodedArray
    A NumPy array serialised as full-precision text or base64 bytes.
load_spectrum(path) -> SpectrumFile
    Load and validate a .sespec file (with automatic migration).
save_spectrum(spec, path)
    Write a SpectrumFile to a .sespec file.
"""

from se_simulator.spectrum.io import load_spectrum, save_spectrum
from se_simulator.spectrum.schema import (
    CURRENT_SCHEMA_VERSION,
    EncodedArray,
    MeasuredProvenance,
    SimulatedProvenance,
    SpectrumData,
    SpectrumFile,
)

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "EncodedArray",
    "MeasuredProvenance",
    "SimulatedProvenance",
    "SpectrumData",
    "SpectrumFile",
    "load_spectrum",
    "save_spectrum",
]
