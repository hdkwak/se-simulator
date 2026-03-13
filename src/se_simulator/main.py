"""Application entry point."""
from __future__ import annotations

from pathlib import Path


def _apply_color_scheme(app) -> None:  # type: ignore[no-untyped-def]
    """Apply a neutral dark-ish palette."""
    from PySide6.QtGui import QColor, QPalette

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(45, 45, 48))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(45, 45, 48))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Button, QColor(60, 60, 64))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 120, 215))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)


def _copy_defaults_if_missing(config_dir: Path, defaults_dir: Path) -> None:
    """Copy default YAML configs to config_dir if they don't exist."""
    config_dir.mkdir(parents=True, exist_ok=True)
    for yaml_file in defaults_dir.glob("*.yaml"):
        dest = config_dir / yaml_file.name
        if not dest.exists():
            dest.write_text(yaml_file.read_text())


def main() -> None:
    """Launch the SE Simulator GUI."""
    import sys

    from PySide6.QtWidgets import QApplication

    from se_simulator.config.manager import ConfigManager

    app = QApplication(sys.argv)
    app.setApplicationName("SE Simulator")
    app.setOrganizationName("SE-RCWA")
    _apply_color_scheme(app)

    defaults_dir = Path(__file__).parent / "config" / "defaults"
    config_dir = Path.cwd() / "configs"
    _copy_defaults_if_missing(config_dir, defaults_dir)

    manager = ConfigManager()

    from se_simulator.ui.main_window import MainWindow

    window = MainWindow(manager, config_dir)
    window.show()

    sys.exit(app.exec())
