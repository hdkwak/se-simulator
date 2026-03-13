"""Structure editor sub-package."""

from se_simulator.ui.widgets.structure_editor.layer_stack import LayerStackWidget
from se_simulator.ui.widgets.structure_editor.properties_inspector import (
    PropertiesInspectorWidget,
)
from se_simulator.ui.widgets.structure_editor.structure_editor import StructureEditor

__all__ = ["LayerStackWidget", "PropertiesInspectorWidget", "StructureEditor"]
