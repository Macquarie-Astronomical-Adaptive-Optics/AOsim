from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSplitter, QVBoxLayout, QWidget, QCheckBox

import numpy as np

from scripts.widgets.config_table import Config_table
from scripts.widgets.pgcanvas import PGCanvas, TwoLineGraph

ARCSEC2RAD = np.pi / (180.0 * 3600.0)
RAD2ARCSEC = (180.0 * 3600.0) / np.pi


class Loop_tab(QWidget):
    def __init__(self, params: dict, scheduler, parent=None):
        super().__init__(parent)
        self.params = params
        self.scheduler = scheduler

        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        left = QSplitter(Qt.Orientation.Vertical)
        mid1 = QSplitter(Qt.Orientation.Vertical)
        mid2 = QSplitter(Qt.Orientation.Vertical)
        right = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(left)
        splitter.addWidget(mid1)
        splitter.addWidget(mid2)
        splitter.addWidget(right)

        # ---- Left column ----
        self.psf_canvas = self._add_canvas_panel(left, "PSF", show_colorbar=False)
        self.residual_canvas = self._add_canvas_panel(left, "Corrected Wavefront Residuals")
        self.residual_canvas.set_colorbar("wavefront error", units="m")

        # ---- Middle column ----
        self.fwhm_graph_corrected = self._add_graph_panel(
            mid1,
            "Corrected FWHM vs Time",
            TwoLineGraph(
                xlabel="time",
                xunit="s",
                ylabel="FWHM",
                y1label="instanteneous",
                y2label="averaged",
                yunit="arcsec",
            ),
        )
        self.r0_graph_corrected = self._add_graph_panel(
            mid1,
            "Residual Effective r0 (500 nm)",
            TwoLineGraph(
                xlabel="time",
                xunit="s",
                ylabel="r0",
                y1label="instantaneous",
                y2label="averaged",
                yunit="m",
            ),
        )
        self.fwhm_graph_uncorrected = self._add_graph_panel(
            mid2,
            "Uncorrected FWHM vs Time",
            TwoLineGraph(
                xlabel="time",
                xunit="s",
                ylabel="FWHM",
                y1label="instanteneous",
                y2label="averaged",
                yunit="arcsec",
            ),
        )
        self.r0_graph_uncorrected = self._add_graph_panel(
            mid2,
            "Uncorrected Effective r0 (500 nm)",
            TwoLineGraph(
                xlabel="time",
                xunit="s",
                ylabel="r0",
                y1label="instantaneous",
                y2label="averaged",
                yunit="m",
            ),
        )

        # ---- Right column ----
        checkbox_layout = QVBoxLayout(right)
        psf_toggle = QCheckBox("Display Uncorrected PSF")
        fwhm_toggle = QCheckBox("Calculate r0 only")

        checkbox_layout.addWidget(psf_toggle)
        checkbox_layout.addWidget(fwhm_toggle)

        fwhm_toggle.checkStateChanged.connect(lambda x: self._loop_r0_only(fwhm_toggle.isChecked()))

        
        self.config_table = Config_table(["loop_gain", "loop_leak"], self.params)
        self.config_table.params_changed.connect(self.update_loop_params)
        right.addWidget(self.config_table)

        self.peak_max = 0.0
        self.scheduler.loop_ready.connect(self.update_graphs)

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

    def _add_canvas_panel(self, parent_splitter: QSplitter, title: str, show_colorbar: bool = True) -> PGCanvas:
        frame = self._styled_frame()
        vlayout: QVBoxLayout = frame.layout()  # type: ignore[assignment]

        label = QLabel(title)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        canvas = PGCanvas(show_colorbar=show_colorbar)
        vlayout.addWidget(canvas)
        vlayout.addWidget(label)

        parent_splitter.addWidget(frame)
        return canvas

    def _add_graph_panel(self, parent_splitter: QSplitter, title: str, graph_widget: QWidget) -> QWidget:
        frame = self._styled_frame()
        vlayout: QVBoxLayout = frame.layout()  # type: ignore[assignment]

        label = QLabel(title)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        vlayout.addWidget(graph_widget)
        vlayout.addWidget(label)
        parent_splitter.addWidget(frame)
        return graph_widget

    @Slot(dict)
    def update_loop_params(self, params: dict) -> None:
        self.params = params
        self.scheduler._loop_gain = params.get("loop_gain")
        self.scheduler._loop_leak = params.get("loop_leak")

        print(
            "Updated loop parameters: "
            "\n  gain:", self.scheduler._loop_gain,
            "\n  leak:", self.scheduler._loop_leak,
        )

    @Slot(object, object, float, float, float, float, object)
    def update_graphs(self, psf, residual, r0_eff_corrected, r0_eff_uncorrected, fwhm1, fwhm2, time, coord) -> None:

        if time == 0:
            self.r0_graph_corrected.reset_data()
            self.r0_graph_uncorrected.reset_data()
            self.fwhm_graph_corrected.reset_data()
            self.fwhm_graph_uncorrected.reset_data()
            return

        self.r0_graph_corrected.append_data(time, r0_eff_corrected,         float(np.mean(self.r0_graph_corrected.y1data[-1000:])))
        self.r0_graph_uncorrected.append_data(time, r0_eff_uncorrected,     float(np.mean(self.r0_graph_uncorrected.y1data[-1000:])))

        rad_to_opd = float(self.params.get("science_lambda")) / (2 * np.pi)
        residual_opd = residual * rad_to_opd  # don't mutate the caller's residual buffer
        absol = float(np.max(np.abs(residual_opd))) if residual_opd.size else 0.0
        self.residual_canvas.queue_image(residual_opd, cmap="seismic", levels=(-absol, absol), auto_levels=False)

        if psf is None:
            return
        
        self.fwhm_graph_corrected.append_data(time, fwhm1 * RAD2ARCSEC,     float(np.mean(self.fwhm_graph_corrected.y1data[-1000:])))
        self.fwhm_graph_uncorrected.append_data(time, fwhm2 * RAD2ARCSEC,   float(np.mean(self.fwhm_graph_uncorrected.y1data[-1000:])))
    
        psf = np.asarray(psf)
        residual = np.asarray(residual)

        self.peak_max = float(np.max(psf)) #max(self.peak_max, float(np.max(psf)))
        psf_levels = (float(np.min(psf)), self.peak_max)
        self.psf_canvas.queue_image(psf, levels=psf_levels, auto_levels=False)
        self.psf_canvas.set_circle_overlay("fwhm", coord[0], coord[1], coord[2])

        

       


    def _loop_r0_only(self, r0_bool):
        self.scheduler._loop_r0_only = r0_bool
        print("sim calculating r0 only:", self.scheduler._loop_r0_only)


    def showEvent(self, event):
        self.scheduler._overview_enabled = False
        self.scheduler._reconstructor_enabled = False
        self.scheduler._detailed_enabled = False
        self.scheduler._loop_enabled = True
        return super().showEvent(event)

    def hideEvent(self, event):
        self.scheduler._loop_enabled = False
        return super().hideEvent(event)
