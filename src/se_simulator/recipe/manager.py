"""RecipeManager: load, save, validate, and decompose recipe YAML files."""
from __future__ import annotations

import contextlib
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from ruamel.yaml import YAML

from se_simulator.config.recipe import (
    FitResults,
    MeasurementRecipe,
    SimulationRecipe,
    StackRef,
    _inline_dict_to_stack,
)
from se_simulator.config.schemas import (
    DataCollectionConfig,
    SampleConfig,
    SimConditions,
    Stack,
    SystemConfig,
    WavelengthSpec,
)

_RECENT_FILE = Path.home() / ".config" / "se_rcwa_simulator" / "recent_recipes.json"
_MAX_RECENT = 20

# Legacy prefixes that may appear in FloatingParameter.target_field
_LEGACY_PREFIXES = [
    "forward_model.sample.inline.",
    "sample.inline.",
    "forward_model.",
]


class RecipeValidationError(ValueError):
    """Raised when a recipe file fails validation."""


def _make_yaml() -> YAML:
    """Return a ruamel.yaml instance configured for round-trip preservation."""
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.preserve_quotes = True
    return yaml


def _load_raw(path: Path) -> dict[str, Any]:
    """Load a YAML file and return a plain Python dict."""
    yaml = _make_yaml()
    with open(path) as fh:
        data = yaml.load(fh)
    if data is None:
        return {}
    # ruamel returns CommentedMap; convert to plain dict for Pydantic
    return _to_plain(data)


def _to_plain(obj: Any) -> Any:
    """Recursively convert ruamel CommentedMap/CommentedSeq to plain dict/list."""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    return obj


def _save_yaml(data: dict[str, Any], path: Path) -> None:
    """Write *data* to *path* using ruamel.yaml."""
    yaml = _make_yaml()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.dump(data, fh)


def _update_recent(path: Path, recipe_type: str) -> None:
    """Prepend (path, type) to the recent-recipes list, capping at _MAX_RECENT."""
    _RECENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing: list[list[str]] = json.loads(_RECENT_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        existing = []

    entry = [str(path.resolve()), recipe_type]
    # Remove any prior entry for the same path
    existing = [e for e in existing if e[0] != entry[0]]
    existing.insert(0, entry)
    _RECENT_FILE.write_text(json.dumps(existing[:_MAX_RECENT], indent=2))


def _migrate_target_field(path: str) -> str:
    """Strip legacy recipe-envelope prefixes from a FloatingParameter target_field.

    Old paths like ``forward_model.sample.inline.layers[0].thickness_nm`` are
    trimmed to the short Stack-rooted form ``layers[0].thickness_nm``.  A
    ``DeprecationWarning`` is emitted for any path that required stripping.
    """
    for prefix in _LEGACY_PREFIXES:
        if path.startswith(prefix):
            short = path[len(prefix):]
            warnings.warn(
                f"FloatingParameter target_field '{path}' uses a legacy prefix. "
                f"Use '{short}' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            return short
    return path


def _inline_to_sample_config(inline: dict[str, Any]) -> SampleConfig:
    """Convert an inline sample dict from a recipe into a SampleConfig.

    Deprecated: prefer Stack.to_sample_config() via _inline_dict_to_stack().
    Kept for backward compatibility with external callers.
    """
    return _inline_dict_to_stack(inline).to_sample_config()


def _dc_and_embed_to_sim_conditions(
    dc: "DataCollectionConfig",
    embed: "SimulationConditionsEmbed",
) -> SimConditions:
    """Build a SimConditions from DataCollectionConfig + SimulationConditionsEmbed.

    Optical/wavelength settings come from *dc*; algorithm knobs from *embed*.
    """
    return SimConditions(
        aoi_deg=dc.aoi_deg,
        azimuth_deg=dc.azimuth_deg,
        wavelengths=WavelengthSpec(
            range=(dc.wavelength_start_nm, dc.wavelength_end_nm, dc.wavelength_step_nm)
        ),
        n_harmonics_x=embed.n_harmonics_x,
        n_harmonics_y=embed.n_harmonics_y,
        li_factorization=embed.li_factorization,
        parallel_wavelengths=embed.parallel_wavelengths,
        output_jones=embed.output_jones,
        output_orders=embed.output_orders,
        engine_override=embed.engine_override,
    )


def _resolve_stack_ref(stack_ref: StackRef, recipe_path: Path | None) -> Stack:
    """Resolve a StackRef to a Stack object.

    If ``stack_ref.inline`` is set, return it directly.  If ``stack_ref.ref``
    is set, load the referenced YAML file (relative to *recipe_path*) and
    validate as a ``Stack``.  As a fallback, if the file validates as a
    legacy ``SampleConfig`` (not a ``Stack``), it is bridged through
    ``_sampleconfig_to_stack()``.
    """
    if stack_ref.inline is not None:
        return stack_ref.inline

    # ref path
    ref_path = Path(stack_ref.ref)  # type: ignore[arg-type]
    if not ref_path.is_absolute() and recipe_path is not None:
        ref_path = recipe_path.parent / ref_path

    raw = _load_raw(ref_path)

    # Try Stack first; fall back to SampleConfig for legacy YAML files
    try:
        return Stack.model_validate(raw)
    except Exception:  # noqa: BLE001
        pass

    try:
        from se_simulator.config.schemas import SampleConfig

        sample_cfg = SampleConfig.model_validate(raw)
        return _sampleconfig_to_stack(sample_cfg)
    except Exception:  # noqa: BLE001
        pass

    # Final attempt: treat as raw inline dict (old SampleRef.inline format)
    return _inline_dict_to_stack(raw)


def _sampleconfig_to_stack(sample_config: SampleConfig) -> Stack:
    """Bridge a legacy SampleConfig into a Stack object.

    This is a best-effort conversion: it reconstructs MaterialSpecs from the
    ``materials`` dict and maps ``GratingLayer.background_material`` back to
    the correct ``StackLayer.material``.
    """
    from se_simulator.config.schemas import MaterialSpec, StackLayer

    def _get_or_make_spec(name: str) -> MaterialSpec:
        if name in sample_config.materials:
            return sample_config.materials[name]
        return MaterialSpec(name=name, source="library", library_name=name)

    sup_spec = _get_or_make_spec(sample_config.superstrate_material)
    sub_spec = _get_or_make_spec(sample_config.substrate_material)

    layers: list[StackLayer] = []
    for gl in sample_config.layers:
        mat_spec = _get_or_make_spec(gl.background_material)
        sl = StackLayer(
            name=gl.name,
            type=gl.type,
            thickness_nm=gl.thickness_nm,
            Lx_nm=gl.Lx_nm,
            Ly_nm=gl.Ly_nm,
            material=mat_spec,
            shapes=gl.shapes,
            incoherent=gl.incoherent,
        )
        layers.append(sl)

    return Stack(
        schema_version=sample_config.schema_version,
        sample_id=sample_config.sample_id,
        superstrate=sup_spec,
        substrate=sub_spec,
        layers=layers,
        metadata=sample_config.metadata,
    )


class RecipeManager:
    """Load, save, validate, and decompose recipe YAML files."""

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_simulation_recipe(self, path: Path) -> SimulationRecipe:
        """Load and validate a SimulationRecipe from *path*."""
        raw = _load_raw(path)
        try:
            recipe = SimulationRecipe.model_validate(raw)
        except ValidationError as exc:
            raise RecipeValidationError(str(exc)) from exc
        with contextlib.suppress(Exception):
            _update_recent(path, "simulation")
        return recipe

    def load_measurement_recipe(self, path: Path) -> MeasurementRecipe:
        """Load and validate a MeasurementRecipe from *path*."""
        raw = _load_raw(path)
        try:
            recipe = MeasurementRecipe.model_validate(raw)
        except ValidationError as exc:
            raise RecipeValidationError(str(exc)) from exc
        # Migrate legacy target_field prefixes
        for fp in recipe.floating_parameters:
            fp.target_field = _migrate_target_field(fp.target_field)
        with contextlib.suppress(Exception):
            _update_recent(path, "measurement")
        return recipe

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_simulation_recipe(self, recipe: SimulationRecipe, path: Path) -> None:
        """Serialise *recipe* to YAML at *path*, auto-setting the created timestamp."""
        data = recipe.model_dump(exclude_none=True)
        if not data["metadata"].get("created"):
            data["metadata"]["created"] = datetime.now(tz=UTC).isoformat()
        _save_yaml(data, path)

    def save_measurement_recipe(self, recipe: MeasurementRecipe, path: Path) -> None:
        """Serialise *recipe* to YAML at *path*, including the results block if present."""
        data = recipe.model_dump(exclude_none=False)
        # Drop None results block unless it has content
        if data.get("results") is None:
            data.pop("results", None)
        _save_yaml(data, path)

    # ------------------------------------------------------------------
    # Decompose
    # ------------------------------------------------------------------

    def decompose_simulation(
        self, recipe: SimulationRecipe, recipe_path: Path | None = None
    ) -> tuple[Stack, SimConditions]:
        """Convert a SimulationRecipe into (Stack, SimConditions).

        Uses ``recipe.stack`` (the new StackRef).  If the recipe was loaded
        from a legacy YAML with a ``sample:`` key, the model_validator in
        ``SimulationRecipe`` will have already migrated it to ``recipe.stack``.

        Returns a :class:`Stack` object.  Pass it directly to
        ``RCWAEngine.run()`` which accepts ``Stack`` natively.
        """
        stack = _resolve_stack_ref(recipe.stack, recipe_path)  # type: ignore[arg-type]
        sim_conditions = _dc_and_embed_to_sim_conditions(
            recipe.data_collection, recipe.simulation_conditions
        )
        return stack, sim_conditions

    def decompose_measurement(
        self, recipe: MeasurementRecipe, recipe_path: Path | None = None
    ) -> tuple[Stack, SimConditions, DataCollectionConfig, SystemConfig, Any, Any]:
        """Convert a MeasurementRecipe into its six sub-objects.

        Returns
        -------
        (Stack, SimConditions, DataCollectionConfig, SystemConfig,
         list[FloatingParameter], FittingConfiguration)

        * ``Stack`` — pass directly to ``RCWAEngine.run()`` or ``TmmDirectFitter``
        * ``DataCollectionConfig`` — per-measurement optical settings
          (AOI, polarizer/analyzer/compensator angles, wavelength range)
        * ``SystemConfig`` — instrument hardware config (calibration, geometry)
        """
        fm = recipe.forward_model

        # --- sample (via stack) ---
        stack = _resolve_stack_ref(fm.stack, recipe_path)  # type: ignore[arg-type]

        # --- data collection (optical + wavelength settings) ---
        dc = fm.data_collection

        # --- simulation conditions (algorithm knobs + promoted optical settings) ---
        sim_conditions = _dc_and_embed_to_sim_conditions(dc, fm.simulation_conditions)

        # --- system config ---
        system_config: SystemConfig | None = None
        if fm.system_config_ref:
            sys_path = Path(fm.system_config_ref)
            if not sys_path.is_absolute() and recipe_path is not None:
                sys_path = recipe_path.parent / sys_path
            if sys_path.exists():
                from se_simulator.config.manager import ConfigManager

                system_config = ConfigManager().load_system(sys_path)

        if system_config is None:
            system_config = SystemConfig.default()

        return (
            stack,
            sim_conditions,
            dc,
            system_config,
            recipe.floating_parameters,
            recipe.fitting_configuration,
        )

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def validate(self, path: Path) -> list[str]:
        """Run validation checks on *path* and return a list of error strings.

        Returns an empty list if the file is valid.  Up to 6 checks:
        1. File exists
        2. Parses as YAML
        3. Contains ``metadata.recipe_type``
        4. Validates as the correct Pydantic model
        5. For measurement: floating parameter ``target_field`` paths are accessible
        6. For measurement: ``min <= initial <= max`` for each floating parameter
        """
        errors: list[str] = []

        # Check 1: file exists
        if not path.exists():
            errors.append(f"File not found: {path}")
            return errors

        # Check 2: parse as YAML
        try:
            raw = _load_raw(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"YAML parse error: {exc}")
            return errors

        # Check 3: metadata.recipe_type present
        recipe_type = raw.get("metadata", {}).get("recipe_type")
        if recipe_type not in ("simulation", "measurement"):
            errors.append(
                f"metadata.recipe_type must be 'simulation' or 'measurement'; got {recipe_type!r}"
            )
            return errors

        # Check 4: Pydantic model validation
        recipe = None
        if recipe_type == "simulation":
            try:
                recipe = SimulationRecipe.model_validate(raw)
            except ValidationError as exc:
                for e in exc.errors():
                    errors.append(f"Validation error at {'.'.join(str(x) for x in e['loc'])}: {e['msg']}")
                return errors
        else:
            try:
                recipe = MeasurementRecipe.model_validate(raw)
            except ValidationError as exc:
                for e in exc.errors():
                    errors.append(f"Validation error at {'.'.join(str(x) for x in e['loc'])}: {e['msg']}")
                return errors

        # Checks 5 & 6: measurement-specific floating parameter checks
        if recipe_type == "measurement" and isinstance(recipe, MeasurementRecipe):
            recipe_dict = raw
            for fp in recipe.floating_parameters:
                # Check 5: target_field resolves
                from se_simulator.recipe.dotpath import resolve_get

                # Migrate the path for validation (don't emit warning — just strip prefix)
                migrated_path = fp.target_field
                for prefix in _LEGACY_PREFIXES:
                    if migrated_path.startswith(prefix):
                        migrated_path = migrated_path[len(prefix):]
                        break

                try:
                    resolve_get(recipe_dict, fp.target_field)
                except (KeyError, IndexError, TypeError) as exc:
                    errors.append(
                        f"FloatingParameter '{fp.name}': target_field '{fp.target_field}' "
                        f"does not resolve: {exc}"
                    )

                # Check 6: min <= initial <= max
                if not (fp.min <= fp.initial <= fp.max):
                    errors.append(
                        f"FloatingParameter '{fp.name}': initial={fp.initial} is not in "
                        f"[{fp.min}, {fp.max}]"
                    )

        return errors

    # ------------------------------------------------------------------
    # Export / results
    # ------------------------------------------------------------------

    def export_as_simulation(self, recipe: MeasurementRecipe) -> SimulationRecipe:
        """Strip fitting config from a MeasurementRecipe, pinning floated params to initial.

        Returns a SimulationRecipe with each floating parameter's ``initial``
        value substituted into the forward model stack.
        """
        from se_simulator.config.recipe import (
            RecipeMetadata,
            SimulationRecipe,
            SimulationRecipeOutputOptions,
        )
        from se_simulator.recipe.dotpath import resolve_set

        fm = recipe.forward_model

        # Prefer the new stack-based path; fall back to legacy sample-based path
        if fm.stack is not None and fm.stack.inline is not None:
            # Mutate a deep-copy of the Stack dict and apply initial values
            stack_dict = fm.stack.inline.model_dump()
            for fp in recipe.floating_parameters:
                local_path = _migrate_target_field(fp.target_field)
                with contextlib.suppress(KeyError, IndexError, TypeError):
                    resolve_set(stack_dict, local_path, fp.initial)
            stack_pinned = Stack.model_validate(stack_dict)
            stack_ref_pinned = StackRef(inline=stack_pinned)

            sim_meta = RecipeMetadata(
                recipe_type="simulation",
                version=recipe.metadata.version,
                author=recipe.metadata.author,
                description=recipe.metadata.description,
                material_db_version=recipe.metadata.material_db_version,
            )
            return SimulationRecipe(
                metadata=sim_meta,
                stack=stack_ref_pinned,
                data_collection=fm.data_collection,
                simulation_conditions=fm.simulation_conditions,
                output_options=SimulationRecipeOutputOptions(),
            )

        # Legacy fallback: sample-based forward model
        from se_simulator.config.recipe import SampleRef, SimulationConditionsEmbed

        fm_dict = fm.model_dump()
        for fp in recipe.floating_parameters:
            local_path = fp.target_field
            prefix = "forward_model."
            if local_path.startswith(prefix):
                local_path = local_path[len(prefix):]
            with contextlib.suppress(KeyError, IndexError, TypeError):
                resolve_set(fm_dict, local_path, fp.initial)

        sim_meta = RecipeMetadata(
            recipe_type="simulation",
            version=recipe.metadata.version,
            author=recipe.metadata.author,
            description=recipe.metadata.description,
            material_db_version=recipe.metadata.material_db_version,
        )
        return SimulationRecipe(
            metadata=sim_meta,
            sample=SampleRef.model_validate(fm_dict["sample"]),
            data_collection=fm.data_collection,
            simulation_conditions=SimulationConditionsEmbed.model_validate(
                fm_dict["simulation_conditions"]
            ),
            output_options=SimulationRecipeOutputOptions(),
        )

    def append_results(self, results: FitResults, path: Path) -> None:
        """Append a FitResults block to an existing measurement recipe YAML.

        Uses ruamel.yaml so that existing comments are preserved.
        """
        yaml = _make_yaml()
        with open(path) as fh:
            data = yaml.load(fh)

        data["results"] = results.model_dump()
        with open(path, "w") as fh:
            yaml.dump(data, fh)

    # ------------------------------------------------------------------
    # Recent list
    # ------------------------------------------------------------------

    def get_recent(self, n: int = 10) -> list[tuple[Path, str]]:
        """Return up to *n* most recent (path, recipe_type) tuples."""
        try:
            entries: list[list[str]] = json.loads(_RECENT_FILE.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        return [(Path(e[0]), e[1]) for e in entries[:n]]


# ---------------------------------------------------------------------------
# Helper: build a minimal SystemConfig from polarizer/analyzer angles in a recipe
# ---------------------------------------------------------------------------

def _default_system_config(recipe: MeasurementRecipe) -> SystemConfig:
    """Build a minimal SystemConfig from the forward-model simulation conditions."""
    sc = recipe.forward_model.simulation_conditions
    return SystemConfig(
        instrument_name="SE Simulator",
        polarizer_angle_deg=sc.polarizer_degrees,
        analyzer_angle_deg=sc.analyzer_degrees,
    )
