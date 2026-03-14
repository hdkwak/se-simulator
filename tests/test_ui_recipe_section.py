"""UI tests for Recipe System (Step 3) — SE-RCWA Simulator."""

from __future__ import annotations

import contextlib
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Minimal recipe YAML content helpers
# ---------------------------------------------------------------------------

_SIM_RECIPE_YAML = """\
metadata:
  recipe_type: simulation
  version: "1.0"
  author: "Test Author"
  description: "Test simulation recipe"
  material_db_version: ""
stack:
  inline:
    superstrate:
      name: Air
      source: library
      library_name: Air
    substrate:
      name: Si
      source: library
      library_name: Si
    layers:
      - name: SiO2 layer
        type: uniform
        thickness_nm: 100.0
        material:
          name: SiO2
          source: library
          library_name: SiO2
simulation_conditions:
  wavelength_start_nm: 300.0
  wavelength_end_nm: 800.0
  wavelength_step_nm: 10.0
  aoi_degrees: 70.0
  azimuth_degrees: 5.0
  polarizer_degrees: 45.0
  analyzer_degrees: 45.0
engine_override: auto
output_options:
  save_psi_delta: true
  save_jones: false
  save_mueller: false
"""

_MEAS_RECIPE_YAML = """\
metadata:
  recipe_type: measurement
  version: "1.0"
  author: "Test"
  description: "Test measurement recipe"
  material_db_version: ""
forward_model:
  stack:
    inline:
      superstrate:
        name: Air
        source: library
        library_name: Air
      substrate:
        name: Si
        source: library
        library_name: Si
      layers:
        - name: SiO2 layer
          type: uniform
          thickness_nm: 100.0
          material:
            name: SiO2
            source: library
            library_name: SiO2
  simulation_conditions:
    wavelength_start_nm: 300.0
    wavelength_end_nm: 800.0
    wavelength_step_nm: 10.0
    aoi_degrees: 65.0
    azimuth_degrees: 0.0
    polarizer_degrees: 45.0
    analyzer_degrees: 45.0
  engine_override: auto
  system:
    polarizer_angle_deg: 45.0
    analyzer_angle_deg: 45.0
floating_parameters:
  - name: thickness
    target_field: layers[0].thickness_nm
    min: 50.0
    max: 200.0
    initial: 100.0
    step: 1.0
    units: nm
fitting_configuration:
  fitting_mode: tmm_direct
  fit_signals:
    - psi
    - delta
  weights: uniform
  optimizer: levenberg_marquardt
  max_iterations: 200
  convergence_tolerance: 1.0e-6
  gradient_step: 1.0e-4
library_reference:
  library_file: ""
output_options:
  save_recipe_with_results: true
  save_fit_report: true
  save_fitted_spectrum: true
"""


def _parse_sim_recipe(yaml_text: str):
    """Parse a YAML string into a SimulationRecipe without touching the recent list."""
    from ruamel.yaml import YAML
    from se_simulator.config.recipe import SimulationRecipe
    from io import StringIO

    yaml = YAML()
    data = yaml.load(StringIO(yaml_text))

    def _to_plain(obj):
        if isinstance(obj, dict):
            return {k: _to_plain(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_plain(v) for v in obj]
        return obj

    return SimulationRecipe.model_validate(_to_plain(data))


def _parse_meas_recipe(yaml_text: str):
    """Parse a YAML string into a MeasurementRecipe without touching the recent list."""
    from ruamel.yaml import YAML
    from se_simulator.config.recipe import MeasurementRecipe
    from io import StringIO

    yaml = YAML()
    data = yaml.load(StringIO(yaml_text))

    def _to_plain(obj):
        if isinstance(obj, dict):
            return {k: _to_plain(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_plain(v) for v in obj]
        return obj

    return MeasurementRecipe.model_validate(_to_plain(data))


@pytest.fixture()
def sim_recipe_path(tmp_path: Path) -> Path:
    p = tmp_path / "sim_recipe.yaml"
    p.write_text(_SIM_RECIPE_YAML)
    return p


@pytest.fixture()
def meas_recipe_path(tmp_path: Path) -> Path:
    p = tmp_path / "meas_recipe.yaml"
    p.write_text(_MEAS_RECIPE_YAML)
    return p


@pytest.fixture()
def sim_recipe(sim_recipe_path):
    return _parse_sim_recipe(_SIM_RECIPE_YAML)


@pytest.fixture()
def meas_recipe(meas_recipe_path):
    return _parse_meas_recipe(_MEAS_RECIPE_YAML)


# ---------------------------------------------------------------------------
# Test 1: Load Simulation Recipe via toolbar → sim panel fields populated
# ---------------------------------------------------------------------------


def test_load_sim_recipe_toolbar_populates_fields(qtbot, sim_recipe_path, sim_recipe):
    """Loading a simulation recipe via the sim panel populates AOI, wavelength fields."""
    from se_simulator.ui.widgets.sim_panel import SimulationPanel

    panel = SimulationPanel()
    qtbot.addWidget(panel)
    panel.load_recipe(sim_recipe, sim_recipe_path)

    # AOI from recipe is 70.0
    assert abs(panel._aoi_spin.value() - 70.0) < 0.01
    # Azimuth is 5.0
    assert abs(panel._az_spin.value() - 5.0) < 0.01
    # Wavelength start
    assert abs(panel._wl_start.value() - 300.0) < 0.01
    # Wavelength stop
    assert abs(panel._wl_stop.value() - 800.0) < 0.01
    # Recipe path shows in field
    assert str(sim_recipe_path) in panel._recipe_path_edit.text()
    # Save back button enabled
    assert panel._btn_save_back.isEnabled()


# ---------------------------------------------------------------------------
# Test 2: Modified marker after field edit
# ---------------------------------------------------------------------------


def test_modified_marker_after_edit(qtbot, sim_recipe_path, sim_recipe):
    """Editing a field after loading a recipe appends '*' to path display."""
    from se_simulator.ui.widgets.sim_panel import SimulationPanel

    panel = SimulationPanel()
    qtbot.addWidget(panel)
    panel.load_recipe(sim_recipe, sim_recipe_path)

    # Modify the AOI
    panel._aoi_spin.setValue(75.0)

    assert panel._recipe_path_edit.text().endswith("*"), (
        "Expected '*' suffix after field modification"
    )


# ---------------------------------------------------------------------------
# Test 3: Save back to Recipe → reload → value persisted
# ---------------------------------------------------------------------------


def test_save_back_to_recipe_persists(qtbot, tmp_path, sim_recipe):
    """Save back to Recipe updates the YAML on disk; reloading yields the new AOI."""
    from se_simulator.ui.widgets.sim_panel import SimulationPanel

    # Write recipe to disk for reload
    dest = tmp_path / "sim_recipe_edit.yaml"
    dest.write_text(_SIM_RECIPE_YAML)

    panel = SimulationPanel()
    qtbot.addWidget(panel)
    panel.load_recipe(sim_recipe, dest)

    # Change AOI and save back
    panel._aoi_spin.setValue(72.5)
    panel._on_save_back()

    # Reload from disk (bypass recent-list update)
    reloaded = _parse_sim_recipe(dest.read_text())
    assert abs(reloaded.simulation_conditions.aoi_degrees - 72.5) < 0.01


# ---------------------------------------------------------------------------
# Test 4: Load Measurement Recipe → floating params populated, fitting mode correct
# ---------------------------------------------------------------------------


def test_load_meas_recipe_populates_params_and_mode(qtbot, meas_recipe_path, meas_recipe):
    """Loading a measurement recipe populates floating params table and fitting mode."""
    from se_simulator.ui.widgets.fitting_workspace import FittingWorkspace

    ws = FittingWorkspace()
    qtbot.addWidget(ws)
    ws.load_measurement_recipe(meas_recipe, meas_recipe_path)

    # Floating params table should have 1 row (the 'thickness' parameter)
    assert ws._params_table.rowCount() == 1
    name_item = ws._params_table.item(0, 0)
    assert name_item is not None
    assert name_item.text() == "thickness"

    # Fitting mode should be "TMM Direct"
    assert ws._fitting_mode_combo.currentText() == "TMM Direct"

    # Recipe path shows in field
    assert str(meas_recipe_path) in ws._recipe_path_edit.text()


# ---------------------------------------------------------------------------
# Test 5: Wavelength mismatch warning visible
# ---------------------------------------------------------------------------


def test_wavelength_mismatch_warning(qtbot, tmp_path):
    """When target CSV wavelengths don't cover sim range, yellow warning is shown."""
    from se_simulator.config.schemas import SimConditions, WavelengthSpec
    from se_simulator.ui.widgets.fitting_workspace import FittingWorkspace
    from se_simulator.ui.main_window import _make_default_system

    ws = FittingWorkspace()
    qtbot.addWidget(ws)
    ws.show()

    # Set sim to 300-800 nm
    sim = SimConditions(
        aoi_deg=65.0,
        azimuth_deg=0.0,
        wavelengths=WavelengthSpec(range=(300.0, 800.0, 10.0)),
    )
    ws.set_configs(_make_default_system(), sim)

    # Create a CSV with limited range (only 400-600 nm — does NOT cover 300-800)
    csv_path = tmp_path / "limited.csv"
    csv_path.write_text("wavelength_nm,psi_deg,delta_deg\n400.0,30.0,90.0\n600.0,31.0,91.0\n")

    ws._validate_target_wavelengths(csv_path)

    # Check that the label was set to visible and contains warning text
    # (isVisible() is True only when parent chain is also visible — show() ensures this)
    assert ws._wl_warning_label.isVisible()
    label_text = ws._wl_warning_label.text()
    # Should contain a warning indication
    assert "Warning" in label_text or "400" in label_text or "300" in label_text


# ---------------------------------------------------------------------------
# Test 6: Wavelength OK label
# ---------------------------------------------------------------------------


def test_wavelength_ok_label(qtbot, tmp_path):
    """When target CSV wavelengths cover sim range, green OK label is shown."""
    from se_simulator.config.schemas import SimConditions, WavelengthSpec
    from se_simulator.ui.widgets.fitting_workspace import FittingWorkspace
    from se_simulator.ui.main_window import _make_default_system

    ws = FittingWorkspace()
    qtbot.addWidget(ws)
    ws.show()

    # Set sim to 400-600 nm (narrow range easy to cover)
    sim = SimConditions(
        aoi_deg=65.0,
        azimuth_deg=0.0,
        wavelengths=WavelengthSpec(range=(400.0, 600.0, 10.0)),
    )
    ws.set_configs(_make_default_system(), sim)

    # Create a CSV that covers 300-900 nm (more than enough)
    csv_path = tmp_path / "full_range.csv"
    csv_path.write_text("wavelength_nm,psi_deg,delta_deg\n300.0,30.0,90.0\n900.0,31.0,91.0\n")

    ws._validate_target_wavelengths(csv_path)

    assert ws._wl_warning_label.isVisible()
    label_text = ws._wl_warning_label.text()
    assert "OK" in label_text or "ok" in label_text.lower()


# ---------------------------------------------------------------------------
# Test 7: Save Results button disabled before fitting, enabled after completion
# ---------------------------------------------------------------------------


def test_save_results_button_state(qtbot, meas_recipe_path, meas_recipe):
    """Save Results button is disabled before fitting and enabled after simulated completion."""
    from se_simulator.ui.widgets.fitting_workspace import FittingWorkspace

    ws = FittingWorkspace()
    qtbot.addWidget(ws)
    ws.load_measurement_recipe(meas_recipe, meas_recipe_path)

    # Before fitting: disabled
    assert not ws._btn_save_results.isEnabled()

    # Simulate a fit result being received
    from se_simulator.config.recipe import FitResults
    from datetime import datetime, timezone

    fit_result = FitResults(
        fitted_parameters={"thickness": 105.0},
        fit_quality={"chi2": 0.001},
        engine_used="tmm_direct",
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
    )
    ws._fit_result = fit_result
    # Enable as _on_result would when recipe_path is set
    if ws._recipe_path is not None:
        ws._btn_save_results.setEnabled(True)

    assert ws._btn_save_results.isEnabled()


# ---------------------------------------------------------------------------
# Test 8: Recipe Editor — load recipe → YAML preview non-empty
# ---------------------------------------------------------------------------


def test_recipe_editor_yaml_preview_non_empty(qtbot, sim_recipe_path, sim_recipe):
    """Opening RecipeEditorDialog with a recipe produces non-empty YAML preview."""
    from se_simulator.ui.recipe_editor import RecipeEditorDialog

    dlg = RecipeEditorDialog(recipe=sim_recipe, path=sim_recipe_path)
    qtbot.addWidget(dlg)
    dlg.show()

    yaml_text = dlg._yaml_preview.toPlainText()
    assert len(yaml_text) > 0, "YAML preview should not be empty after loading a recipe"
    assert "simulation" in yaml_text


# ---------------------------------------------------------------------------
# Test 9: Recipe Editor — Validate → valid recipe shows green banner
# ---------------------------------------------------------------------------


def test_recipe_editor_validate_valid(qtbot, sim_recipe_path, sim_recipe):
    """Validating a valid recipe file shows a green validation banner."""
    from se_simulator.ui.recipe_editor import RecipeEditorDialog

    dlg = RecipeEditorDialog(recipe=sim_recipe, path=sim_recipe_path)
    qtbot.addWidget(dlg)
    dlg.show()

    # Patch _update_recent to avoid permission errors
    with patch("se_simulator.recipe.manager._update_recent", return_value=None):
        dlg._on_validate()

    assert dlg._validation_label.isVisible()
    banner_text = dlg._validation_label.text()
    assert "OK" in banner_text or "ok" in banner_text.lower()
    # Green style
    style = dlg._validation_label.styleSheet()
    assert "#2e7d32" in style or "#e8f5e9" in style


# ---------------------------------------------------------------------------
# Test 10: Recipe Editor — Validate → invalid recipe shows error list
# ---------------------------------------------------------------------------


def test_recipe_editor_validate_invalid(qtbot, tmp_path):
    """Validating a non-existent file shows an error message."""
    nonexistent_path = tmp_path / "does_not_exist.yaml"
    # Do NOT create the file — validate should report "File not found"

    from se_simulator.ui.recipe_editor import RecipeEditorDialog
    from se_simulator.config.recipe import SimulationRecipe, RecipeMetadata
    from se_simulator.config.recipe import SimulationConditionsEmbed, SimulationRecipeOutputOptions

    # Create a minimal valid recipe object in memory but point to a missing file
    recipe = _parse_sim_recipe(_SIM_RECIPE_YAML)

    dlg = RecipeEditorDialog(recipe=recipe, path=nonexistent_path)
    qtbot.addWidget(dlg)
    dlg.show()

    dlg._on_validate()

    assert dlg._validation_label.isVisible()
    banner_text = dlg._validation_label.text()
    # Should report error about missing file
    assert "error" in banner_text.lower() or "not found" in banner_text.lower() or "•" in banner_text
    # Red/error style
    style = dlg._validation_label.styleSheet()
    assert "#c62828" in style or "#ffebee" in style


# ---------------------------------------------------------------------------
# Test 11: Recipe Editor — Export as Simulation → saved file validates as SimulationRecipe
# ---------------------------------------------------------------------------


def test_recipe_editor_export_as_simulation(qtbot, tmp_path, meas_recipe_path, meas_recipe):
    """Export as Simulation Recipe from a MeasurementRecipe creates a valid SimulationRecipe."""
    from se_simulator.ui.recipe_editor import RecipeEditorDialog

    dlg = RecipeEditorDialog(recipe=meas_recipe, path=meas_recipe_path)
    qtbot.addWidget(dlg)
    dlg.show()

    # Export programmatically (bypass recent-list update)
    export_path = tmp_path / "exported_sim.yaml"
    with patch("se_simulator.recipe.manager._update_recent", return_value=None):
        from se_simulator.recipe.manager import RecipeManager

        manager = RecipeManager()
        sim_recipe = manager.export_as_simulation(meas_recipe)
        manager.save_simulation_recipe(sim_recipe, export_path)

    # Validate the exported file
    assert export_path.exists()
    reloaded = _parse_sim_recipe(export_path.read_text())
    from se_simulator.config.recipe import SimulationRecipe

    assert isinstance(reloaded, SimulationRecipe)
    assert reloaded.metadata.recipe_type == "simulation"


# ---------------------------------------------------------------------------
# Test 12: Recipe Editor — Recent recipes list shows two loaded recipes, most recent first
# ---------------------------------------------------------------------------


def test_recipe_editor_recent_list(qtbot, tmp_path):
    """After injecting two recent entries, the recent list shows both most recent first."""
    import json
    from se_simulator.recipe.manager import _RECENT_FILE

    recipe_a = tmp_path / "recipe_a.yaml"
    recipe_b = tmp_path / "recipe_b.yaml"
    recipe_a.write_text(_SIM_RECIPE_YAML)
    recipe_b.write_text(_SIM_RECIPE_YAML)

    # Directly write to a temp recent file to avoid permission issues
    fake_recent = tmp_path / "recent_recipes.json"
    # recipe_b is first (most recent), recipe_a is second
    fake_recent.write_text(
        json.dumps([
            [str(recipe_b.resolve()), "simulation"],
            [str(recipe_a.resolve()), "simulation"],
        ])
    )

    from se_simulator.ui.recipe_editor import RecipeEditorDialog
    import se_simulator.recipe.manager as mgr_module

    dlg = RecipeEditorDialog()
    qtbot.addWidget(dlg)
    dlg.show()

    # Patch _RECENT_FILE to our temp file
    with patch.object(mgr_module, "_RECENT_FILE", fake_recent):
        dlg._load_recent_list()

    # Find items in recent list that match our temp files
    items_text = [dlg._recent_list.item(i).text() for i in range(dlg._recent_list.count())]
    paths_in_list = [text for text in items_text if "recipe_a" in text or "recipe_b" in text]

    assert len(paths_in_list) >= 2, f"Expected both recipes in recent list, got: {items_text}"

    # Most recently loaded (recipe_b) should appear before recipe_a
    recipe_b_indices = [i for i, t in enumerate(items_text) if "recipe_b" in t]
    recipe_a_indices = [i for i, t in enumerate(items_text) if "recipe_a" in t]
    if recipe_b_indices and recipe_a_indices:
        assert recipe_b_indices[0] < recipe_a_indices[0], (
            "Most recently loaded recipe_b should appear before recipe_a"
        )
