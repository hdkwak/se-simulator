"""Convenience re-exports of all fitting result types."""

from __future__ import annotations

from se_simulator.fitting.engine import FitResult
from se_simulator.fitting.interpolation import InterpolationResult
from se_simulator.fitting.refinement import RefinementResult
from se_simulator.fitting.search import SearchResult

__all__ = [
    "FitResult",
    "InterpolationResult",
    "RefinementResult",
    "SearchResult",
]
