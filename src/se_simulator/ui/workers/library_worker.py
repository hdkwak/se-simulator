"""LibraryWorker: QThread for non-blocking spectral library generation."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from se_simulator.fitting.library import LibrarySpec


class LibraryWorker(QThread):
    """Generate a spectral library in a background thread.

    Signals
    -------
    progress(int, int):
        Emits (completed_entries, total_entries) as generation proceeds.
    finished_ok():
        Emitted when generation completes successfully.
    error_occurred(str):
        Emitted if any exception is raised.
    finished():
        Always emitted when the thread exits.
    """

    progress = Signal(int, int)
    finished_ok = Signal()
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(
        self,
        spec: LibrarySpec,
        output_path: Path,
        n_workers: int | None = None,
        resume: bool = True,
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._spec = spec
        self._output_path = Path(output_path)
        self._n_workers = n_workers
        self._resume = resume

    def run(self) -> None:
        """Generate the library. Called automatically by QThread.start()."""
        try:
            from se_simulator.fitting.library import LibraryGenerator, LibraryStore

            store = LibraryStore(self._output_path)
            if not self._output_path.exists() or not self._resume:
                store.create(self._spec, overwrite=True)

            generator = LibraryGenerator(self._spec, store, n_workers=self._n_workers)

            def _cb(done: int, total: int) -> None:
                if self.isInterruptionRequested():
                    return
                self.progress.emit(done, total)

            generator.generate(progress_callback=_cb, resume=self._resume)
            self.finished_ok.emit()

        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(str(exc))
        finally:
            self.finished.emit()
