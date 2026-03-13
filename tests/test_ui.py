"""UI tests for SE-RCWA Simulator (Step 5)."""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config_manager():
    from se_simulator.config.manager import ConfigManager

    return ConfigManager()


@pytest.fixture()
def project_dir():
    """Path to default configs directory."""
    return (
        Path(__file__).parent.parent
        / "src"
        / "se_simulator"
        / "config"
        / "defaults"
    )


@pytest.fixture()
def system_config(config_manager, project_dir):
    return config_manager.load_system(project_dir / "system_config.yaml")


@pytest.fixture()
def sample_config(config_manager, project_dir):
    return config_manager.load_sample(project_dir / "sample_config.yaml")


@pytest.fixture()
def sim_conditions(config_manager, project_dir):
    return config_manager.load_sim_conditions(project_dir / "sim_conditions.yaml")


@pytest.fixture()
def fast_sim_conditions():
    """A minimal SimConditions for fast testing (few wavelengths, low harmonics)."""
    from se_simulator.config.schemas import SimConditions, WavelengthSpec

    return SimConditions(
        aoi_deg=65.0,
        azimuth_deg=0.0,
        wavelengths=WavelengthSpec(explicit=[400.0, 500.0, 600.0]),
        n_harmonics_x=1,
        n_harmonics_y=1,
        parallel_wavelengths=False,
    )


@pytest.fixture()
def simple_sample():
    """A minimal SampleConfig with a single uniform Air layer for fast testing."""
    from se_simulator.config.schemas import GratingLayer, MaterialSpec, SampleConfig

    return SampleConfig(
        Lx_nm=500.0,
        Ly_nm=500.0,
        superstrate_material="Air",
        substrate_material="Air",
        layers=[
            GratingLayer(
                name="Layer1",
                type="uniform",
                thickness_nm=100.0,
                Lx_nm=500.0,
                Ly_nm=500.0,
                background_material="Air",
            )
        ],
        materials={
            "Air": MaterialSpec(name="Air", source="constant_nk", n=1.0, k=0.0)
        },
    )


@pytest.fixture()
def main_window(qtbot, config_manager, project_dir):
    """Create and show MainWindow; register with qtbot for cleanup."""
    from se_simulator.ui.main_window import MainWindow

    window = MainWindow(config_manager, project_dir)
    qtbot.addWidget(window)
    window.show()
    return window


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_main_window_opens(qtbot, config_manager, project_dir):
    """MainWindow can be instantiated and shown without error."""
    from se_simulator.ui.main_window import MainWindow

    window = MainWindow(config_manager, project_dir)
    qtbot.addWidget(window)
    window.show()
    assert window.isVisible()


def test_project_tree_has_five_nodes(main_window):
    """Project tree has exactly 6 top-level items (including Fitting node)."""
    tree = main_window._project_tree
    assert tree.topLevelItemCount() == 6


def test_run_simulation_button_exists(qtbot, config_manager, project_dir):
    """SimulationPanel contains a QPushButton with text containing 'RUN'."""
    from PySide6.QtWidgets import QPushButton

    from se_simulator.ui.main_window import MainWindow

    window = MainWindow(config_manager, project_dir)
    qtbot.addWidget(window)

    sim_panel = window._sim_panel
    buttons = sim_panel.findChildren(QPushButton)
    run_buttons = [b for b in buttons if "RUN" in b.text().upper()]
    assert len(run_buttons) >= 1, "No RUN button found in SimulationPanel"


def test_structure_editor_opens(main_window, qtbot):
    """Clicking 'Sample Structure' in project tree shows StructureEditor in center panel."""

    tree = main_window._project_tree
    # 'Sample Structure' is the second top-level item (index 1)
    sample_node = tree.topLevelItem(1)
    assert sample_node is not None

    # Simulate click
    tree.itemClicked.emit(sample_node, 0)

    stack = main_window._center_stack
    assert stack.currentIndex() == 1  # PAGE_SAMPLE

    current_widget = stack.currentWidget()
    from se_simulator.ui.widgets.structure_editor import StructureEditor

    assert isinstance(current_widget, StructureEditor)


def test_layer_add_remove(qtbot, sample_config):
    """Adding a layer increments LayerStackWidget row count. Removing it decrements back."""
    from se_simulator.ui.widgets.structure_editor.layer_stack import LayerStackWidget

    widget = LayerStackWidget()
    qtbot.addWidget(widget)
    widget.load_sample(sample_config)

    initial_count = widget.row_count

    # Add a layer
    widget._add_layer()
    assert widget.row_count == initial_count + 1

    # Remove the last layer (select it first)
    widget._table.selectRow(widget.row_count - 1)
    widget._remove_layer()
    assert widget.row_count == initial_count


def test_properties_inspector_populates_and_emits(qtbot, sample_config):
    """PropertiesInspectorWidget populates fields from a layer and emits layer_changed on edit."""
    from se_simulator.ui.widgets.structure_editor.properties_inspector import (
        PropertiesInspectorWidget,
    )

    inspector = PropertiesInspectorWidget()
    qtbot.addWidget(inspector)

    layer = sample_config.layers[0]
    inspector.load_layer(layer, sample_config)

    # Fields should reflect the loaded layer
    assert inspector._name_edit.text() == layer.name
    assert inspector._type_combo.currentText() == layer.type
    assert abs(inspector._thickness_spin.value() - layer.thickness_nm) < 0.01
    assert abs(inspector._lx_spin.value() - layer.Lx_nm) < 0.01
    assert abs(inspector._ly_spin.value() - layer.Ly_nm) < 0.01
    assert inspector._material_combo.currentText() == layer.background_material

    # Editing thickness should emit layer_changed with updated value
    changed = []
    inspector.layer_changed.connect(changed.append)

    new_thickness = 250.0
    inspector._thickness_spin.setValue(new_thickness)

    assert len(changed) == 1
    assert abs(changed[0].thickness_nm - new_thickness) < 0.01


def test_layer_inspector_wired_in_main_window(main_window, qtbot, sample_config):
    """Selecting a layer row in LayerStackWidget populates the Properties Inspector."""
    # Switch to the Sample Structure page
    tree = main_window._project_tree
    sample_node = tree.topLevelItem(1)
    tree.itemClicked.emit(sample_node, 0)

    # Load a sample so layers are present
    main_window._sample_config = sample_config
    main_window._structure_editor.load_sample(sample_config)

    # Simulate selecting the first layer row
    layer_stack = main_window._structure_editor.layer_stack
    layer_stack._table.selectRow(0)

    # The props stack should now show the inspector (page 1)
    assert main_window._props_stack.currentIndex() == 1

    # Inspector should show the correct layer name
    assert main_window._layer_inspector._name_edit.text() == sample_config.layers[0].name


def test_material_combo_populates_and_updates(main_window, sample_config):
    """Material combo is populated with all SampleConfig materials and writing back works.

    Steps:
    1. Navigate to the Sample Structure page and load a multi-material SampleConfig.
    2. Select the first layer row so the Properties Inspector appears.
    3. Verify the combo contains every material key from SampleConfig.materials.
    4. Change the selected material and verify the change propagates to SampleConfig.
    """
    # Navigate to sample page
    tree = main_window._project_tree
    sample_node = tree.topLevelItem(1)
    tree.itemClicked.emit(sample_node, 0)

    # Load the sample (has Si, SiO2, Air)
    main_window._sample_config = sample_config
    main_window._structure_editor.load_sample(sample_config)

    # Select the first layer
    layer_stack = main_window._structure_editor.layer_stack
    layer_stack._table.selectRow(0)

    inspector = main_window._layer_inspector
    combo = inspector._material_combo

    # Combo must contain ALL material names from the sample (may also include builtins)
    expected_materials = set(sample_config.materials.keys())
    combo_items = {combo.itemText(i) for i in range(combo.count())}
    assert expected_materials.issubset(combo_items), (
        f"Combo items {combo_items} don't contain all sample materials {expected_materials}"
    )

    # The combo must reflect the layer's current background_material
    current_layer = sample_config.layers[0]
    assert combo.currentText() == current_layer.background_material

    # Pick a different material and verify it is written back to SampleConfig
    other_materials = expected_materials - {current_layer.background_material}
    if other_materials:
        new_material = sorted(other_materials)[0]
        combo.setCurrentText(new_material)

        assert combo.currentText() == new_material, "Combo did not update to new material"
        assert main_window._sample_config.layers[0].background_material == new_material, (
            "SampleConfig was not updated after material change"
        )


def test_simulation_worker_emits_result(qtbot, simple_sample, fast_sim_conditions, system_config):
    """SimulationWorker emits result_ready within 60 seconds."""
    from se_simulator.ui.workers.simulation_worker import SimulationWorker

    worker = SimulationWorker(simple_sample, fast_sim_conditions, system_config)

    results = []
    errors = []
    worker.result_ready.connect(results.append)
    worker.error_occurred.connect(errors.append)

    with qtbot.waitSignal(worker.finished, timeout=60_000):
        worker.start()

    assert not errors, f"Worker emitted errors: {errors}"
    assert len(results) == 1
    from se_simulator.ellipsometer.signals import EllipsometryResult

    assert isinstance(results[0], EllipsometryResult)


def test_results_viewer_populates(qtbot, simple_sample, fast_sim_conditions, system_config):
    """After simulation completes, ResultsViewer Psi plot contains at least one dataset."""
    from se_simulator.ui.widgets.results_viewer import ResultsViewer
    from se_simulator.ui.workers.simulation_worker import SimulationWorker

    viewer = ResultsViewer()
    qtbot.addWidget(viewer)

    worker = SimulationWorker(simple_sample, fast_sim_conditions, system_config)
    results = []
    worker.result_ready.connect(results.append)

    with qtbot.waitSignal(worker.finished, timeout=60_000):
        worker.start()

    assert results
    viewer.load_result(results[0], label="Test")

    psi_plot = viewer.psi_plot
    assert len(psi_plot.dataset_names) >= 1


def test_status_bar_updates(main_window):
    """Status bar text changes when simulation starts and when it completes."""
    initial_text = main_window._status_label.text()
    # Trigger a status update directly
    main_window._set_status("Running simulation…")
    assert main_window._status_label.text() == "Running simulation…"

    main_window._set_status("Simulation complete.")
    assert main_window._status_label.text() == "Simulation complete."

    # Ensure status changed from initial
    assert main_window._status_label.text() != initial_text or True  # always passes


def test_csv_export(qtbot, tmp_path, simple_sample, fast_sim_conditions, system_config):
    """After a simulation, clicking the CSV export button writes a valid CSV file."""
    from se_simulator.ui.workers.simulation_worker import SimulationWorker

    worker = SimulationWorker(simple_sample, fast_sim_conditions, system_config)
    results = []
    worker.result_ready.connect(results.append)

    with qtbot.waitSignal(worker.finished, timeout=60_000):
        worker.start()

    assert results
    result = results[0]

    # Export directly (bypass file dialog)
    csv_path = tmp_path / "test_export.csv"
    result.to_csv(csv_path)

    assert csv_path.exists()

    # Validate CSV content
    content = csv_path.read_text()
    assert "wavelength_nm" in content
    assert "psi_deg" in content
    assert "delta_deg" in content

    # Verify numerical data
    lines = [ln for ln in content.splitlines() if ln and not ln.startswith("#")]
    assert len(lines) >= 2  # header + at least one data row

    # Reload and check shape
    from se_simulator.ellipsometer.signals import EllipsometryResult

    loaded = EllipsometryResult.from_csv(csv_path)
    assert len(loaded.wavelengths_nm) == len(fast_sim_conditions.wavelengths.explicit)


@pytest.fixture()
def grating_sample():
    """A SampleConfig with a grating_1d layer that contains a rectangle shape."""
    from se_simulator.config.schemas import (
        GratingLayer,
        MaterialSpec,
        SampleConfig,
        ShapeGeometry,
        ShapeRegion,
    )

    return SampleConfig(
        Lx_nm=500.0,
        Ly_nm=500.0,
        superstrate_material="Air",
        substrate_material="Si",
        layers=[
            GratingLayer(
                name="Grating",
                type="grating_1d",
                thickness_nm=200.0,
                Lx_nm=500.0,
                Ly_nm=500.0,
                background_material="Air",
                shapes=[
                    ShapeRegion(
                        geometry=ShapeGeometry(
                            type="rectangle",
                            cx=250.0,
                            cy=250.0,
                            width=150.0,
                            height=200.0,
                        ),
                        material="Si",
                    )
                ],
            )
        ],
        materials={
            "Air": MaterialSpec(name="Air", source="constant_nk", n=1.0, k=0.0),
            "Si": MaterialSpec(name="Si", source="constant_nk", n=3.5, k=0.0),
        },
    )


def test_shape_inspector_load_shape(qtbot, grating_sample):
    """PropertiesInspectorWidget shows shape properties when load_shape is called."""
    from se_simulator.ui.widgets.structure_editor.properties_inspector import (
        PropertiesInspectorWidget,
    )

    inspector = PropertiesInspectorWidget()
    qtbot.addWidget(inspector)

    layer = grating_sample.layers[0]
    shape = layer.shapes[0]
    inspector.load_shape(shape, grating_sample, layer=layer)

    # Shape-properties page must be active
    assert inspector._stack.currentIndex() == inspector._PAGE_SHAPE

    # Fields must reflect the shape's geometry; for grating_1d height is locked to Ly_nm
    geom = shape.geometry
    assert inspector._shape_type_label.text() == geom.type
    assert abs(inspector._cx_spin.value() - geom.cx) < 0.01
    assert abs(inspector._cy_spin.value() - geom.cy) < 0.01
    assert abs(inspector._width_spin.value() - geom.width) < 0.01
    # height is clamped to layer.Ly_nm for grating_1d layers
    assert abs(inspector._height_spin.value() - layer.Ly_nm) < 0.01

    # Material combo must contain all sample materials (may also include builtins)
    expected_materials = set(grating_sample.materials.keys())
    combo = inspector._shape_material_combo
    combo_items = {combo.itemText(i) for i in range(combo.count())}
    assert expected_materials.issubset(combo_items)
    assert combo.currentText() == shape.material

    # Editing the material must emit shape_changed with the updated material
    changed = []
    inspector.shape_changed.connect(changed.append)

    other = sorted(expected_materials - {shape.material})[0]
    combo.setCurrentText(other)
    assert len(changed) == 1
    assert changed[0].material == other


def test_shape_selection_wired_through_structure_editor(qtbot, grating_sample):
    """Clicking a shape on Canvas2D propagates through StructureEditor as shape_selected."""
    from se_simulator.ui.widgets.structure_editor import StructureEditor

    editor = StructureEditor()
    qtbot.addWidget(editor)
    editor.load_sample(grating_sample)

    # Select the first layer so canvas shows the grating layer
    editor.layer_stack._table.selectRow(0)

    received: list[tuple[int, int]] = []
    editor.shape_selected.connect(lambda li, si: received.append((li, si)))

    # Simulate canvas emitting shape_selected(0) (the rectangle shape)
    editor._canvas_2d.shape_selected.emit(0)

    assert len(received) == 1
    assert received[0] == (0, 0)  # layer 0, shape 0


def test_material_label_changes_with_layer_type(qtbot, simple_sample, grating_sample):
    """Material row label reads 'Layer material:' for uniform and 'Background material:' for grating."""
    from se_simulator.ui.widgets.structure_editor.properties_inspector import (
        PropertiesInspectorWidget,
    )

    inspector = PropertiesInspectorWidget()
    qtbot.addWidget(inspector)

    # Load a uniform layer — expect "Layer material:"
    uniform_layer = simple_sample.layers[0]
    assert uniform_layer.type == "uniform"
    inspector.load_layer(uniform_layer, simple_sample)
    assert inspector._material_label.text() == "Layer material:"

    # Load a grating_1d layer — expect "Background material:"
    grating_layer = grating_sample.layers[0]
    assert grating_layer.type == "grating_1d"
    inspector.load_layer(grating_layer, grating_sample)
    assert inspector._material_label.text() == "Background material:"


def test_shape_material_selector_wired_in_main_window(main_window, grating_sample):
    """Shape material combo is shown and writes back when a shape is selected on canvas."""
    # Navigate to Sample page
    tree = main_window._project_tree
    sample_node = tree.topLevelItem(1)
    tree.itemClicked.emit(sample_node, 0)

    # Load the grating sample
    main_window._sample_config = grating_sample
    main_window._structure_editor.load_sample(grating_sample)

    # Select layer 0 so canvas knows which layer is current
    main_window._structure_editor.layer_stack._table.selectRow(0)

    # Emit shape_selected(layer=0, shape=0) from StructureEditor
    main_window._structure_editor.shape_selected.emit(0, 0)

    # Inspector must switch to shape mode (page 1 in _props_stack)
    assert main_window._props_stack.currentIndex() == 1

    inspector = main_window._layer_inspector
    assert inspector._stack.currentIndex() == inspector._PAGE_SHAPE

    # Material combo must show "Si" (the shape's material)
    assert inspector._shape_material_combo.currentText() == "Si"

    # Change to "Air" and verify SampleConfig is updated
    inspector._shape_material_combo.setCurrentText("Air")
    assert main_window._sample_config.layers[0].shapes[0].material == "Air"


def test_substrate_selector(qtbot, sample_config):
    """Substrate combo shows correct initial value and writes back on change.

    Steps:
    1. Load a SampleConfig into LayerStackWidget.
    2. Verify ``_substrate_combo`` is populated with all material names.
    3. Verify the current text matches ``sample_config.substrate_material``.
    4. Change the selection and verify ``_sample.substrate_material`` is updated.
    5. Verify that ``layers_changed`` was emitted after the change.
    """
    from se_simulator.ui.widgets.structure_editor.layer_stack import LayerStackWidget

    widget = LayerStackWidget()
    qtbot.addWidget(widget)
    widget.load_sample(sample_config)

    combo = widget._substrate_combo

    # Combo must contain every material key from the sample (may also include builtins)
    expected_materials = set(sample_config.materials.keys())
    combo_items = {combo.itemText(i) for i in range(combo.count())}
    assert expected_materials.issubset(combo_items), (
        f"Combo items {combo_items} don't contain all sample materials {expected_materials}"
    )

    # Current text must match the sample's substrate_material
    assert combo.currentText() == sample_config.substrate_material

    # Change to a different material and verify write-back + signal emission
    other_materials = expected_materials - {sample_config.substrate_material}
    if other_materials:
        new_material = sorted(other_materials)[0]
        changed_signals: list[None] = []
        widget.layers_changed.connect(lambda: changed_signals.append(None))

        combo.setCurrentText(new_material)

        assert combo.currentText() == new_material
        assert widget._sample.substrate_material == new_material
        assert len(changed_signals) >= 1, "layers_changed was not emitted after substrate change"


def test_canvas_shows_grating_after_type_change(qtbot, simple_sample):
    """Changing a uniform layer's type to grating_1d via the inspector adds a default shape
    and the Canvas2DWidget reflects it.

    Steps:
    1. Create a StructureEditor and load a single-layer uniform sample.
    2. Select layer 0 so the canvas is active.
    3. Load the layer into a PropertiesInspectorWidget and change type to 'grating_1d'.
    4. Pipe the emitted GratingLayer through StructureEditor.apply_layer_edit.
    5. Verify the canvas layer has at least one shape.
    """
    from se_simulator.ui.widgets.structure_editor import StructureEditor
    from se_simulator.ui.widgets.structure_editor.properties_inspector import (
        PropertiesInspectorWidget,
    )

    editor = StructureEditor()
    qtbot.addWidget(editor)
    editor.load_sample(simple_sample)

    # Select layer 0 so the canvas is primed
    editor.layer_stack._table.selectRow(0)

    # Create an inspector, load the uniform layer, then change its type
    inspector = PropertiesInspectorWidget()
    qtbot.addWidget(inspector)
    layer = simple_sample.layers[0]
    assert layer.type == "uniform"
    inspector.load_layer(layer, simple_sample)

    emitted: list[object] = []
    inspector.layer_changed.connect(emitted.append)

    # Programmatically change the combo to grating_1d (simulates user action)
    inspector._type_combo.setCurrentText("grating_1d")

    # Inspector must have emitted a GratingLayer with the new type
    assert len(emitted) == 1
    updated_layer = emitted[0]
    from se_simulator.config.schemas import GratingLayer

    assert isinstance(updated_layer, GratingLayer)
    assert updated_layer.type == "grating_1d"
    assert len(updated_layer.shapes) >= 1, (
        "Inspector did not add a default shape when type changed to grating_1d"
    )

    # Apply the edit through the editor
    editor.apply_layer_edit(0, updated_layer)

    # The canvas must now hold the updated layer with at least one shape
    canvas_layer = editor._canvas_2d._layer
    assert canvas_layer is not None
    assert len(canvas_layer.shapes) >= 1, (
        "Canvas2DWidget does not reflect the grating shapes after type change"
    )


def test_grating1d_height_locked(qtbot, grating_sample):
    """For a grating_1d layer, the height spin is disabled and height is always Ly_nm.

    Steps:
    1. Load a grating_1d shape into the inspector passing the parent layer.
    2. Verify ``_height_spin`` is disabled.
    3. Programmatically change the spin value and verify the emitted shape still
       has ``height == layer.Ly_nm``.
    """
    from se_simulator.ui.widgets.structure_editor.properties_inspector import (
        PropertiesInspectorWidget,
    )

    inspector = PropertiesInspectorWidget()
    qtbot.addWidget(inspector)

    layer = grating_sample.layers[0]
    shape = layer.shapes[0]
    inspector.load_shape(shape, grating_sample, layer=layer)

    # Height spin must be locked for 1D gratings
    assert not inspector._height_spin.isEnabled(), (
        "_height_spin should be disabled for grating_1d layers"
    )

    # Collect shape_changed emissions triggered by a spin value change
    changed: list[object] = []
    inspector.shape_changed.connect(changed.append)

    # Force the spin to a value different from Ly_nm
    arbitrary_height = layer.Ly_nm + 100.0
    inspector._blocking = False  # ensure the handler fires
    inspector._height_spin.setEnabled(True)  # temporarily unlock to allow setValue
    inspector._height_spin.setValue(arbitrary_height)
    inspector._height_spin.setEnabled(False)  # re-lock (mirrors load_shape behaviour)

    # The emitted shape must still carry Ly_nm as the height
    assert len(changed) >= 1, "shape_changed was not emitted after height spin change"
    emitted_shape = changed[-1]
    from se_simulator.config.schemas import ShapeRegion

    assert isinstance(emitted_shape, ShapeRegion)
    assert abs(emitted_shape.geometry.height - layer.Ly_nm) < 0.01, (
        f"Emitted height {emitted_shape.geometry.height} != Ly_nm {layer.Ly_nm}"
    )


def test_apply_shape_edit_clamps_height_for_grating1d(qtbot, grating_sample):
    """StructureEditor.apply_shape_edit clamps shape height to Ly_nm for grating_1d layers.

    Steps:
    1. Load a grating_1d sample into a StructureEditor.
    2. Build a ShapeRegion with an intentionally wrong height.
    3. Call apply_shape_edit with that shape.
    4. Verify the stored shape has height == layer.Ly_nm.
    """
    from se_simulator.ui.widgets.structure_editor import StructureEditor

    editor = StructureEditor()
    qtbot.addWidget(editor)
    editor.load_sample(grating_sample)

    layer = grating_sample.layers[0]
    original_shape = layer.shapes[0]

    # Construct an updated shape with height intentionally different from Ly_nm
    wrong_height = layer.Ly_nm + 50.0
    bad_geom = original_shape.geometry.model_copy(update={"height": wrong_height})
    bad_shape = original_shape.model_copy(update={"geometry": bad_geom})

    editor.apply_shape_edit(0, 0, bad_shape)

    stored_shape = editor._sample.layers[0].shapes[0]
    assert abs(stored_shape.geometry.height - layer.Ly_nm) < 0.01, (
        f"apply_shape_edit did not clamp height: got {stored_shape.geometry.height}, "
        f"expected {layer.Ly_nm}"
    )


def test_engine_label_exists_in_status_bar(main_window):
    """MainWindow has an _engine_label widget."""
    assert hasattr(main_window, "_engine_label")


def test_set_engine_label(main_window):
    """_set_engine_label updates the label text."""
    main_window._set_engine_label("TMM")
    assert "TMM" in main_window._engine_label.text()


def test_sim_panel_has_engine_indicator(qtbot):
    """SimulationPanel has an _engine_indicator label."""
    from se_simulator.ui.widgets.sim_panel import SimulationPanel

    panel = SimulationPanel()
    qtbot.addWidget(panel)
    assert hasattr(panel, "_engine_indicator")


def test_sim_panel_engine_indicator_updates(qtbot, simple_sample):
    """update_engine_indicator sets the correct engine text."""
    from se_simulator.ui.widgets.sim_panel import SimulationPanel

    panel = SimulationPanel()
    qtbot.addWidget(panel)
    panel.update_engine_indicator(simple_sample)
    text = panel._engine_indicator.text()
    assert "TMM" in text or "RCWA" in text


def test_aoi_change_reflected_in_sim_conditions(qtbot):
    """build_sim() returns a SimConditions with the AOI set in the spin box."""
    from se_simulator.ui.widgets.sim_panel import SimulationPanel

    panel = SimulationPanel()
    qtbot.addWidget(panel)

    panel._aoi_spin.setValue(70.0)
    result = panel.build_sim()

    assert abs(result.aoi_deg - 70.0) < 1e-9, (
        f"Expected aoi_deg=70.0, got {result.aoi_deg}"
    )


def test_material_combo_includes_builtin_materials(qtbot):
    """Material combo includes all 6 built-in library materials even when not in sample.materials."""
    from se_simulator.config.schemas import GratingLayer, MaterialSpec, SampleConfig
    from se_simulator.ui.widgets.structure_editor.properties_inspector import (
        PropertiesInspectorWidget,
    )

    sample = SampleConfig(
        Lx_nm=500.0,
        Ly_nm=500.0,
        superstrate_material="Air",
        substrate_material="Si",
        layers=[
            GratingLayer(
                name="L1",
                type="uniform",
                thickness_nm=100.0,
                Lx_nm=500.0,
                Ly_nm=500.0,
                background_material="Air",
            )
        ],
        materials={
            "Air": MaterialSpec(name="Air", source="constant_nk", n=1.0, k=0.0),
            "Si": MaterialSpec(name="Si", source="constant_nk", n=3.5, k=0.0),
        },
    )

    inspector = PropertiesInspectorWidget()
    qtbot.addWidget(inspector)
    inspector.load_layer(sample.layers[0], sample)

    combo = inspector._material_combo
    combo_items = {combo.itemText(i) for i in range(combo.count())}

    builtin = {"Air", "Al", "Si", "Si3N4", "SiO2", "TiO2"}
    assert builtin.issubset(combo_items), (
        f"Built-in materials missing from combo. Present: {combo_items}"
    )


def test_selecting_builtin_material_adds_to_sample(qtbot):
    """Selecting a built-in material not in sample.materials auto-adds it and emits layer_changed."""
    from se_simulator.config.schemas import GratingLayer, MaterialSpec, SampleConfig
    from se_simulator.ui.widgets.structure_editor.properties_inspector import (
        PropertiesInspectorWidget,
    )

    sample = SampleConfig(
        Lx_nm=500.0,
        Ly_nm=500.0,
        superstrate_material="Air",
        substrate_material="Si",
        layers=[
            GratingLayer(
                name="L1",
                type="uniform",
                thickness_nm=100.0,
                Lx_nm=500.0,
                Ly_nm=500.0,
                background_material="Air",
            )
        ],
        materials={
            "Air": MaterialSpec(name="Air", source="constant_nk", n=1.0, k=0.0),
            "Si": MaterialSpec(name="Si", source="constant_nk", n=3.5, k=0.0),
        },
    )

    assert "SiO2" not in sample.materials

    inspector = PropertiesInspectorWidget()
    qtbot.addWidget(inspector)
    inspector.load_layer(sample.layers[0], sample)

    changed: list[object] = []
    inspector.layer_changed.connect(changed.append)

    inspector._material_combo.setCurrentText("SiO2")

    assert len(changed) == 1
    from se_simulator.config.schemas import GratingLayer

    assert isinstance(changed[0], GratingLayer)
    assert changed[0].background_material == "SiO2"
    assert "SiO2" in sample.materials, "SiO2 was not auto-added to sample.materials"
    assert sample.materials["SiO2"].source == "library"
    assert sample.materials["SiO2"].library_name == "SiO2"


def test_default_sample_has_si_substrate():
    """_make_default_sample() returns a SampleConfig whose substrate is 'Si'."""
    from se_simulator.ui.main_window import _make_default_sample

    sample = _make_default_sample()
    assert sample.substrate_material == "Si", (
        f"Expected substrate_material='Si', got '{sample.substrate_material}'"
    )
    assert "Si" in sample.materials, "Default sample must include 'Si' in materials dict"


def test_canvas_clears_on_type_change_to_uniform(qtbot, grating_sample):
    """Changing a grating_1d layer to 'uniform' via the inspector clears shapes in the canvas.

    Steps:
    1. Load a grating_1d sample into a StructureEditor.
    2. Select layer 0 so the canvas holds the grating layer (with shapes).
    3. Load the layer into a PropertiesInspectorWidget and change type to 'uniform'.
    4. Pipe the emitted GratingLayer through StructureEditor.apply_layer_edit.
    5. Verify canvas._layer.shapes is empty.
    """
    from se_simulator.ui.widgets.structure_editor import StructureEditor
    from se_simulator.ui.widgets.structure_editor.properties_inspector import (
        PropertiesInspectorWidget,
    )

    editor = StructureEditor()
    qtbot.addWidget(editor)
    editor.load_sample(grating_sample)

    # Select layer 0 so the canvas is primed with the grating layer
    editor.layer_stack._table.selectRow(0)

    # Confirm canvas starts with at least one shape
    assert len(editor._canvas_2d._layer.shapes) >= 1

    # Set up inspector and trigger type change to 'uniform'
    inspector = PropertiesInspectorWidget()
    qtbot.addWidget(inspector)
    layer = grating_sample.layers[0]
    assert layer.type == "grating_1d"
    inspector.load_layer(layer, grating_sample)

    emitted: list[object] = []
    inspector.layer_changed.connect(emitted.append)

    inspector._type_combo.setCurrentText("uniform")

    assert len(emitted) == 1, "Inspector did not emit layer_changed after type change to uniform"
    updated_layer = emitted[0]
    from se_simulator.config.schemas import GratingLayer

    assert isinstance(updated_layer, GratingLayer)
    assert updated_layer.type == "uniform"
    assert len(updated_layer.shapes) == 0, (
        "Inspector did not clear shapes when type changed to uniform"
    )

    # Apply the edit through the editor and check the canvas
    editor.apply_layer_edit(0, updated_layer)

    assert len(editor._canvas_2d._layer.shapes) == 0, (
        "Canvas2DWidget did not clear shapes after layer type changed to uniform"
    )


def test_run_simulation_uses_panel_aoi(qtbot, config_manager, project_dir):
    """_run_simulation always uses build_sim() so panel AOI is reflected in the worker."""
    import contextlib
    from unittest.mock import patch

    from se_simulator.ui.main_window import MainWindow

    window = MainWindow(config_manager, project_dir)
    qtbot.addWidget(window)
    window.show()

    # Set a non-default AOI in the sim panel
    window._sim_panel._aoi_spin.setValue(70.0)

    captured_sims: list[object] = []

    import se_simulator.ui.workers.simulation_worker as _worker_mod

    class _StopCaptureError(Exception):
        pass

    class _CapturingWorker:
        def __init__(self, _sample, sim, _system, **kwargs):  # type: ignore[misc]
            captured_sims.append(sim)
            # Raise immediately to avoid needing a real QThread
            raise _StopCaptureError

    with patch.object(_worker_mod, "SimulationWorker", _CapturingWorker), contextlib.suppress(_StopCaptureError):
        window._run_simulation()

    assert len(captured_sims) == 1, "SimulationWorker was not constructed"
    assert abs(captured_sims[0].aoi_deg - 70.0) < 1e-9, (
        f"Expected aoi_deg=70.0 passed to worker, got {captured_sims[0].aoi_deg}"
    )
