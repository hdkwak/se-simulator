"""Fitting engine — library generation, search, interpolation, and refinement."""

from se_simulator.fitting.engine import FitResult, FittingEngine
from se_simulator.fitting.interpolation import InterpolationResult, LibraryInterpolator
from se_simulator.fitting.library import (
    LibraryGenerator,
    LibrarySpec,
    LibraryStore,
    ParameterSpec,
    apply_params,
    build_library_spec,
)
from se_simulator.fitting.refinement import GradientRefinement, RefinementResult
from se_simulator.fitting.search import NearestNeighborSearch, SearchResult, chi_squared

__all__ = [
    "FitResult",
    "FittingEngine",
    "GradientRefinement",
    "InterpolationResult",
    "LibraryGenerator",
    "LibraryInterpolator",
    "LibrarySpec",
    "LibraryStore",
    "NearestNeighborSearch",
    "ParameterSpec",
    "RefinementResult",
    "SearchResult",
    "apply_params",
    "build_library_spec",
    "chi_squared",
]
