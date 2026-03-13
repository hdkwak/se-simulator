"""Tests for the optical constants engine."""

import numpy as np
import pytest

from se_simulator.config.schemas import MaterialSpec
from se_simulator.materials.database import MaterialDatabase
from se_simulator.materials.interpolator import MaterialEntry
from se_simulator.materials.models import cauchy


@pytest.fixture()
def db() -> MaterialDatabase:
    return MaterialDatabase()


def test_library_si_loads(db: MaterialDatabase) -> None:
    """Resolve Si from the built-in library and verify wavelength coverage."""
    spec = MaterialSpec(name="Si", source="library", library_name="Si")
    entry = db.resolve(spec)

    assert isinstance(entry, MaterialEntry)
    assert entry.wavelengths_nm.min() <= 300.0
    assert entry.wavelengths_nm.max() >= 800.0


def test_constant_nk(db: MaterialDatabase) -> None:
    """Air resolved as constant n=1, k=0 yields epsilon ≈ 1+0j at 633 nm."""
    spec = MaterialSpec(name="Air", source="constant_nk", n=1.0, k=0.0)
    db.resolve(spec)

    wl = np.array([633.0])
    eps = db.get_epsilon("Air", wl)

    assert eps.shape == (1,)
    assert abs(eps[0] - (1.0 + 0j)) < 1e-10


def test_cauchy_model() -> None:
    """Cauchy model with SiO2-like params returns physically reasonable n values."""
    wavelengths = np.array([500.0, 600.0])
    # Typical SiO2 Cauchy params: A=1.45, B=3560 nm^2, C=0
    n, k = cauchy(wavelengths, [1.45, 3560.0, 0.0])

    # n should be around 1.46–1.47 in this range
    assert np.all(n > 1.44)
    assert np.all(n < 1.50)
    assert np.all(k == 0.0)


def test_interpolation_extrapolation_warning(db: MaterialDatabase) -> None:
    """Requesting wavelengths outside data range triggers a non-empty warning list."""
    spec = MaterialSpec(name="Si", source="library", library_name="Si")
    entry = db.resolve(spec)

    # Request wavelengths well outside the Si data range
    out_of_range = np.array([50.0, 5000.0])
    warnings = db.check_extrapolation(entry, out_of_range)

    assert len(warnings) > 0
    assert "Si" in warnings[0]


def test_epsilon_shape(db: MaterialDatabase) -> None:
    """get_epsilon returns a shape-(50,) complex array with no NaN values for Si."""
    spec = MaterialSpec(name="Si", source="library", library_name="Si")
    db.resolve(spec)

    wavelengths = np.linspace(400.0, 700.0, 50)
    eps = db.get_epsilon("Si", wavelengths)

    assert eps.shape == (50,)
    assert eps.dtype == complex or np.issubdtype(eps.dtype, np.complexfloating)
    assert not np.any(np.isnan(eps))
