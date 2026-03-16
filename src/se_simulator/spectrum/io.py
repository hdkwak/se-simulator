"""Load and save .sespec files."""

from __future__ import annotations

from pathlib import Path

import ruamel.yaml

from se_simulator.spectrum.migrations import migrate
from se_simulator.spectrum.schema import SpectrumFile

_yaml = ruamel.yaml.YAML()
_yaml.default_flow_style = False
_yaml.width = 120
_yaml.best_sequence_indent = 2
_yaml.best_map_flow_style = False


def load_spectrum(path: str | Path) -> SpectrumFile:
    """Load a .sespec file and return a validated :class:`SpectrumFile`.

    Applies any necessary schema migrations before validation so that older
    files are transparently upgraded to the current schema version.
    """
    with open(path, encoding="utf-8") as fh:
        raw: dict = _yaml.load(fh)
    if raw is None:
        raise ValueError(f"File is empty: {path}")
    raw = migrate(raw)
    return SpectrumFile.model_validate(raw)


def save_spectrum(spec: SpectrumFile, path: str | Path) -> None:
    """Serialise *spec* to *path* as a human-readable YAML .sespec file.

    ``None`` fields are omitted to keep the file compact when optional
    sections (recipe, fit_results, Mueller channels, Jones matrices) are
    not yet populated.
    """
    raw = spec.model_dump(mode="json", exclude_none=True)
    with open(path, "w", encoding="utf-8") as fh:
        _yaml.dump(raw, fh)
