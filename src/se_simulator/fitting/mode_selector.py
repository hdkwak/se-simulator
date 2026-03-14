"""Fitting mode selector: choose between 'library' and 'tmm_direct' strategies."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from se_simulator.config.recipe import MeasurementRecipe
    from se_simulator.config.schemas import SampleConfig


def select_fitting_mode(
    recipe: MeasurementRecipe,
    sample_config: SampleConfig,
) -> Literal["library", "tmm_direct"]:
    """Return the fitting mode to use for the given recipe and sample.

    Decision logic:
      - ``'library'`` is returned when fitting_mode is explicitly ``'library'``,
        or when ``fitting_mode`` is ``'auto'`` and either a library_file is
        provided or any layer contains shapes (patterned).
      - ``'tmm_direct'`` is returned when fitting_mode is explicitly
        ``'tmm_direct'`` (all-uniform stack required) or when ``fitting_mode``
        is ``'auto'``, all layers are uniform, and no library_file is set.

    Raises
    ------
    ValueError
        If ``fitting_mode`` is ``'tmm_direct'`` but the sample has patterned layers.
    """
    mode = recipe.fitting_configuration.fitting_mode
    all_uniform = all(not getattr(layer, "shapes", None) for layer in sample_config.layers)

    if mode == "library":
        return "library"

    if mode == "tmm_direct":
        if not all_uniform:
            raise ValueError(
                "fitting_mode 'tmm_direct' is not valid: sample contains patterned layers."
            )
        return "tmm_direct"

    # mode == "auto"
    has_library = bool(recipe.library_reference.library_file.strip())
    if all_uniform and not has_library:
        return "tmm_direct"
    return "library"
