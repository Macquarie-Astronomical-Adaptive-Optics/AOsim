from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSplitter, QVBoxLayout, QWidget

import numpy as np

from scripts.widgets.pgcanvas import PGCanvas


class Reconstructor_tab(QWidget):
    """Tab showing phase, reconstruction, DM surface, residuals, and recon matrices."""

    req_emit = Signal()

    def __init__(self, params: dict, scheduler, parent=None):
        super().__init__(parent)
        self.params = params
        self.scheduler = scheduler

        layout = QHBoxLayout(self)

        vsplitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(vsplitter)

        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        vsplitter.addWidget(top_splitter)
        vsplitter.addWidget(bottom_splitter)

        # ---- top row ----
        self.raw_screen = self._add_canvas_panel(top_splitter, "Phase Screen")
        self.reconstruct_screen = self._add_canvas_panel(top_splitter, "Reconstructed Wavefront")
        self.dm_surface = self._add_canvas_panel(top_splitter, "DM Surface")
        self.residual_screen = self._add_canvas_panel(top_splitter, "DM Corrected Residuals")

        # ---- bottom row ----
        self.recon_mat = self._add_canvas_panel(bottom_splitter, "Reconstruction Matrix")
        self.recon_mat_inv = self._add_canvas_panel(bottom_splitter, "Reconstruction Matrix Inverse")

        # Avoid repeatedly reconfiguring colorbars on every frame; do once.
        self.raw_screen.set_colorbar(label="phase", units="rad")
        self.reconstruct_screen.set_colorbar(label="phase", units="rad")
        self.residual_screen.set_colorbar(label="phase", units="rad")
        self.dm_surface.set_colorbar(label="surface deformation", units="m")

        self.scheduler._reconstructor_enabled = False
        self.req_emit.connect(self.scheduler._emit_once)

    @staticmethod
    def _styled_frame() -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.NoFrame)

        vlayout = QVBoxLayout(frame)
        vlayout.setContentsMargins(0, 0, 0, 5)
        vlayout.setSpacing(0)

        bg = frame.palette().color(frame.backgroundRole())
        frame.setStyleSheet(f"QFrame {{ background-color: {bg.name()}; }}")
        return frame

    def _add_canvas_panel(self, parent_splitter: QSplitter, title: str) -> PGCanvas:
        frame = self._styled_frame()
        vlayout: QVBoxLayout = frame.layout()  # type: ignore[assignment]

        label = QLabel(title)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        canvas = PGCanvas()
        vlayout.addWidget(canvas)
        vlayout.addWidget(label)

        parent_splitter.addWidget(frame)
        return canvas

    @staticmethod
    def _absmax(a: np.ndarray) -> float:
        return float(np.max(np.abs(a))) if a.size else 0.0

    def update_mat_view(self, matr: np.ndarray, inv_matr: np.ndarray) -> None:
        absol1 = self._absmax(np.asarray(matr))
        absol2 = self._absmax(np.asarray(inv_matr))

        self.recon_mat.queue_image(matr, cmap="Greys", levels=(-absol1, absol1), auto_levels=False)
        self.recon_mat_inv.queue_image(inv_matr, cmap="Greys", levels=(-absol2, absol2), auto_levels=False)

    def update_view(
        self,
        raw: np.ndarray,
        reconstructed: np.ndarray,
        dm_surf: np.ndarray,
        corrected: np.ndarray,
    ) -> None:
        raw = np.asarray(raw)
        reconstructed = np.asarray(reconstructed)
        corrected = np.asarray(corrected)
        dm_surf = np.asarray(dm_surf)

        abmax = max(self._absmax(raw), self._absmax(reconstructed))

        self.raw_screen.queue_image(raw, cmap="seismic", levels=(-abmax, abmax), auto_levels=False)
        self.reconstruct_screen.queue_image(reconstructed, cmap="seismic", levels=(-abmax, abmax), auto_levels=False)

        # Scale without mutating the caller's array.
        poke_amp = float(self.params.get("poke_amplitude", 1.0))
        dm_scaled = dm_surf * poke_amp
        dm_abs = self._absmax(dm_scaled)
        self.dm_surface.queue_image(dm_scaled, cmap="seismic", levels=(-dm_abs, dm_abs), auto_levels=False)

        self.residual_screen.queue_image(corrected, cmap="seismic", levels=(-abmax, abmax), auto_levels=False)

    def showEvent(self, event):
        self.scheduler._reconstructor_enabled = True
        self.req_emit.emit()
        return super().showEvent(event)

    def hideEvent(self, event):
        self.scheduler._reconstructor_enabled = False
        return super().hideEvent(event)
