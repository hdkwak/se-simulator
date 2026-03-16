"""Schema version migrations for .sespec files.

Each migration is a pure dict → dict function registered with
``@register_migration("from_version")``.  ``migrate()`` walks the chain
from whatever version is in the file to ``CURRENT_SCHEMA_VERSION``.

Adding a new migration
----------------------
1. Define a function that accepts a raw dict and returns an updated dict.
2. Decorate it with ``@register_migration("N.M")`` where N.M is the version
   the function upgrades *from*.
3. Update the version string inside the function to the target version.

Minor bumps (new optional fields with defaults) do NOT need a migration —
Pydantic handles missing optional fields automatically.  Only add a migration
when a field is renamed, removed, or its semantics change.
"""

from __future__ import annotations

from collections.abc import Callable

from se_simulator.spectrum.schema import CURRENT_SCHEMA_VERSION

_MIGRATIONS: dict[str, Callable[[dict], dict]] = {}


def register_migration(from_version: str) -> Callable:
    """Decorator that registers *fn* as the migration from *from_version*."""
    def decorator(fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
        _MIGRATIONS[from_version] = fn
        return fn
    return decorator


def migrate(raw: dict) -> dict:
    """Walk *raw* through all registered migrations to reach CURRENT_SCHEMA_VERSION.

    Raises ``ValueError`` if no migration path exists.
    """
    version = raw.get("schema_version", "0.9")
    while version != CURRENT_SCHEMA_VERSION:
        if version not in _MIGRATIONS:
            raise ValueError(
                f"No migration path from .sespec schema version {version!r} to "
                f"{CURRENT_SCHEMA_VERSION!r}. Update the application."
            )
        raw = _MIGRATIONS[version](raw)
        version = raw.get("schema_version", CURRENT_SCHEMA_VERSION)
    return raw


# ---------------------------------------------------------------------------
# Migrations (none yet — placeholder for future versions)
# ---------------------------------------------------------------------------

# Example of a future migration from a hypothetical 0.9 pre-release format
# that stored flat wavelength/psi/delta lists at the top level:
#
# @register_migration("0.9")
# def _migrate_0_9_to_1_0(raw: dict) -> dict:
#     for key in ("wavelengths_nm", "psi_deg", "delta_deg"):
#         if key in raw:
#             values = raw.pop(key)
#             raw.setdefault("spectrum", {})[key] = {
#                 "encoding": "text",
#                 "dtype": "float64",
#                 "shape": [len(values)],
#                 "data": "[" + ", ".join(f"{v:.17g}" for v in values) + "]",
#             }
#     raw["schema_version"] = "1.0"
#     return raw
