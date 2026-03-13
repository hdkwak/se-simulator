"""AboutDialog: application information dialog."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

import se_simulator


class AboutDialog(QDialog):
    """Simple About dialog showing application metadata."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About SE-RCWA Simulator")
        self.setModal(True)
        self.setMinimumWidth(350)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        title_label = QLabel("<h2>SE-RCWA Simulator</h2>")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        version = getattr(se_simulator, "__version__", "1.0.0")
        info_label = QLabel(
            f"<p>Version: <b>{version}</b></p>"
            "<p>Spectroscopic Ellipsometry simulation using<br>"
            "Rigorous Coupled-Wave Analysis (RCWA).</p>"
            "<p>Built with PySide6, pyqtgraph, NumPy, SciPy, and Pydantic.</p>"
        )
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
