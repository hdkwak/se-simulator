"""Engine selection logic for SE-RCWA Simulator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from se_simulator.config.schemas import SampleConfig


def select_engine(
    sample_config: SampleConfig,
    engine_override: str = "auto",
) -> Literal["tmm", "rcwa"]:
    """Return the engine to use for the given sample configuration.

    Returns ``"tmm"`` if every layer has no patterned shapes (i.e. the stack is
    a pure thin-film multilayer).  Returns ``"rcwa"`` if any layer contains
    shapes (periodic structure).

    The *engine_override* parameter (from ``SimConditions.engine_override``)
    takes precedence when set to ``"rcwa"`` or ``"tmm"``.
    """
    if engine_override in ("rcwa", "tmm"):
        return engine_override  # type: ignore[return-value]

    for layer in sample_config.layers:
        if getattr(layer, "shapes", []):
            return "rcwa"
    return "tmm"
