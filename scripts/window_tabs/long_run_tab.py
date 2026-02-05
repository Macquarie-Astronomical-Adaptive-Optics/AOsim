import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from scripts.widgets.pgcanvas import PGCanvas

RAD2ARCSEC = (180.0 * 3600.0) / np.pi


class LongRun_tab(QWidget):
    """GUI tab to run a long closed-loop batch and compute PSF/statistics at the end."""

    req_long_run = Signal(dict)
    req_cancel_long_run = Signal()

    def __init__(self, params: dict, scheduler, parent=None):
        super().__init__(parent)
        self.params = params
        self.scheduler = scheduler

        # ---- UI ----
        root = QHBoxLayout(self)

        left = QFrame()
        left.setFrameShape(QFrame.Box)
        left.setLineWidth(1)
        left_layout = QVBoxLayout(left)

        title = QLabel("Long run batch")
        title.setAlignment(Qt.AlignLeft)
        title.setStyleSheet("font-weight: 600; font-size: 14px;")
        left_layout.addWidget(title)

        grid = QGridLayout()
        row = 0

        grid.addWidget(QLabel("Duration (s)"), row, 0)
        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(0.1, 600.0)
        self.spin_duration.setSingleStep(1.0)
        self.spin_duration.setDecimals(1)
        self.spin_duration.setValue(1.0)
        self.spin_duration.valueChanged.connect(self._update_frames_label)
        grid.addWidget(self.spin_duration, row, 1)

        grid.addWidget(QLabel("FPS (sim)"), row, 2)
        self.lbl_fps = QLabel("1000")
        grid.addWidget(self.lbl_fps, row, 3)
        row += 1

        grid.addWidget(QLabel("Frames"), row, 0)
        self.lbl_frames = QLabel("0")
        grid.addWidget(self.lbl_frames, row, 1)

        grid.addWidget(QLabel("Chunk frames"), row, 2)
        self.spin_chunk = QSpinBox()
        self.spin_chunk.setRange(8, 8192)
        self.spin_chunk.setSingleStep(64)
        self.spin_chunk.setValue(64)
        grid.addWidget(self.spin_chunk, row, 3)
        row += 1

        grid.addWidget(QLabel("Discard first (s)"), row, 0)
        self.spin_discard = QDoubleSpinBox()
        self.spin_discard.setRange(0.0, 600.0)
        self.spin_discard.setSingleStep(0.5)
        self.spin_discard.setDecimals(1)
        self.spin_discard.setValue(0.0)
        self.spin_discard.setToolTip("Warm-up time excluded from final PSF/FWHM/r0 fits; all frames are still recorded if enabled.")
        self.spin_discard.valueChanged.connect(self._update_frames_label)
        grid.addWidget(self.spin_discard, row, 1)

        hint = QLabel("(metrics only)")
        hint.setAlignment(Qt.AlignLeft)
        hint.setStyleSheet("color: #666;")
        grid.addWidget(hint, row, 2, 1, 2)
        row += 1

        grid.addWidget(QLabel("PSF ROI (px)"), row, 0)
        self.spin_roi = QSpinBox()
        self.spin_roi.setRange(16, 512)
        self.spin_roi.setSingleStep(16)
        self.spin_roi.setValue(128)
        grid.addWidget(self.spin_roi, row, 1)

        grid.addWidget(QLabel("TT removal"), row, 2)
        # Simple cycle: none / fit / hybrid
        self.btn_tt_mode = QPushButton("fit")
        self.btn_tt_mode.setCheckable(True)
        self.btn_tt_mode.setChecked(False)
        self.btn_tt_mode.clicked.connect(self._cycle_tt_mode)
        self._tt_mode = "fit"
        grid.addWidget(self.btn_tt_mode, row, 3)
        row += 1

        self.chk_record_psfs = QCheckBox("Record per-frame PSFs")
        self.chk_record_psfs.setChecked(True)
        grid.addWidget(self.chk_record_psfs, row, 0, 1, 2)

        row += 1

        self.chk_save_tt = QCheckBox("Save TT time series")
        self.chk_save_tt.setChecked(True)
        grid.addWidget(self.chk_save_tt, row, 0, 1, 2)

        self.chk_reset_before = QCheckBox("Reset sim before run")
        self.chk_reset_before.setChecked(True)
        grid.addWidget(self.chk_reset_before, row, 2, 1, 2)
        row += 1

        grid.addWidget(QLabel("Output dir"), row, 0)
        self.edit_out = QLineEdit()
        base = Path("output") / "batch_runs"
        self.edit_out.setText(str(base))
        grid.addWidget(self.edit_out, row, 1, 1, 3)
        row += 1

        left_layout.addLayout(grid)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Start long run")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_cancel)
        left_layout.addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        left_layout.addWidget(self.progress)

        self.lbl_progress = QLabel("Idle")
        left_layout.addWidget(self.lbl_progress)

        # Results
        self.lbl_results = QLabel("")
        self.lbl_results.setWordWrap(True)
        self.lbl_results.setTextFormat(Qt.PlainText)
        self.lbl_results.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_results.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        left_layout.addWidget(self.lbl_results)
        
        left_layout.addWidget(self.lbl_results)

        # Open folder button
        self.btn_open_dir = QPushButton("Open output folder")
        self.btn_open_dir.setEnabled(False)
        self.btn_open_dir.clicked.connect(self._open_out_dir)
        left_layout.addWidget(self.btn_open_dir)

        left_layout.addStretch(1)

        # Right: PSF canvases
        right = QFrame()
        right.setFrameShape(QFrame.Box)
        right.setLineWidth(1)
        right_layout = QVBoxLayout(right)

        self.canvas_corr = PGCanvas(show_colorbar=False)
        self.canvas_unc = PGCanvas(show_colorbar=False)

        right_layout.addWidget(QLabel("Long exposure PSF (corrected)"))
        right_layout.addWidget(self.canvas_corr)
        right_layout.addWidget(QLabel("Long exposure PSF (uncorrected)"))
        right_layout.addWidget(self.canvas_unc)


        root.addWidget(left, 0)
        root.addWidget(right, 1)

        # ---- Wiring ----
        self.btn_start.clicked.connect(self._start_clicked)
        self.btn_cancel.clicked.connect(self._cancel_clicked)

        self.req_long_run.connect(self.scheduler.start_long_run)
        self.req_cancel_long_run.connect(self.scheduler.cancel_long_run)

        self.scheduler.long_run_started.connect(self._on_long_run_started)
        self.scheduler.long_run_progress.connect(self._on_long_run_progress)
        self.scheduler.long_run_finished.connect(self._on_long_run_finished)
        self.scheduler.long_run_failed.connect(self._on_long_run_failed)

        self._last_out_dir: Optional[str] = None

        # Initialize labels
        self._update_frames_label()

    def showEvent(self, event):
        # Avoid per-tick GUI work in other tabs while this tab is visible.
        self.scheduler._overview_enabled = False
        self.scheduler._reconstructor_enabled = False
        self.scheduler._detailed_enabled = False
        self.scheduler._loop_enabled = False
        return super().showEvent(event)

    # ---- UI helpers ----
    def _cycle_tt_mode(self):
        modes = ["none", "fit"]
        i = modes.index(self._tt_mode)
        self._tt_mode = modes[(i + 1) % len(modes)]
        self.btn_tt_mode.setText(self._tt_mode)

    def _update_frames_label(self):
        dt = float(getattr(self.scheduler, "dt_s", 0.001))
        fps = 1.0 / max(dt, 1e-12)
        self.lbl_fps.setText(f"{fps:.0f}")
        dur = float(self.spin_duration.value())
        # keep discard <= duration for convenience
        if hasattr(self, "spin_discard"):
            if float(self.spin_discard.value()) > dur:
                self.spin_discard.blockSignals(True)
                self.spin_discard.setValue(dur)
                self.spin_discard.blockSignals(False)

        n = int(round(dur / max(dt, 1e-12)))
        self.lbl_frames.setText(str(n))

    def _open_out_dir(self):
        if self._last_out_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._last_out_dir))

    # ---- Actions ----
    def _start_clicked(self):
        dt = float(getattr(self.scheduler, "dt_s", 0.001))
        duration_s = float(self.spin_duration.value())
        n_frames = int(round(duration_s / max(dt, 1e-12)))

        out_dir = self.edit_out.text().strip()
        if not out_dir:
            out_dir = str(Path.cwd() / "output" / "batch_runs")

        cfg: Dict[str, Any] = {
            "duration_s": duration_s,
            "n_frames": n_frames,
            "discard_first_s": float(self.spin_discard.value()) if hasattr(self, "spin_discard") else 0.0,
            "chunk_frames": int(self.spin_chunk.value()),
            "psf_roi": int(self.spin_roi.value()),
            "record_psfs": bool(self.chk_record_psfs.isChecked()),
            "save_tt_series": bool(self.chk_save_tt.isChecked()),
            "reset_before": bool(self.chk_reset_before.isChecked()),
            "out_dir": out_dir,
            "psf_tt_mode": str(self._tt_mode),
            "progress_every": int(self.spin_chunk.value()),
        }

        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_open_dir.setEnabled(False)
        self.progress.setValue(0)
        self.lbl_progress.setText("Starting…")
        self.lbl_results.setText("")
        self._last_out_dir = None

        self.req_long_run.emit(cfg)

    def _cancel_clicked(self):
        self.req_cancel_long_run.emit()
        self.lbl_progress.setText("Cancelling…")

    # ---- Scheduler callbacks ----
    @Slot(object)
    def _on_long_run_started(self, meta: object):
        try:
            m = dict(meta)
            self._last_out_dir = m.get("out_dir")
        except Exception:
            self._last_out_dir = None
        self.lbl_progress.setText("Running…")

    @Slot(int, int, float, str)
    def _on_long_run_progress(self, done: int, total: int, fps: float, msg: str):
        total = max(1, int(total))
        done = max(0, int(done))
        p = int(round(100.0 * done / total))
        self.progress.setValue(max(0, min(100, p)))
        self.lbl_progress.setText(f"{msg}  {done}/{total} frames  ({fps:.0f} fps)")

    @Slot(object)
    def _on_long_run_finished(self, result: object):
        res = dict(result or {})
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)

        self.lbl_results.setStyleSheet("")

        out_dir = res.get("out_dir")
        if out_dir:
            self._last_out_dir = str(out_dir)
            self.btn_open_dir.setEnabled(True)

        cancelled = bool(res.get("cancelled", False))
        done = int(res.get("n_frames_done", 0))
        total = int(res.get("n_frames_requested", done))
        self.progress.setValue(100 if (not cancelled) else int(round(100.0 * done / max(1, total))))
        self.lbl_progress.setText("Done" + (" (cancelled)" if cancelled else ""))

        # Show summary
        fwhm_c = float(res.get("fwhm_arcsec_corrected", 0.0))
        fwhm_u = float(res.get("fwhm_arcsec_uncorrected", 0.0))
        fwhm_cmof = float(res.get("fwhm_arcsec_corrected_moffat", 0.0))
        fwhm_umof = float(res.get("fwhm_arcsec_uncorrected_moffat", 0.0))
        gauss_corr = res.get("gauss_corrected")
        gauss_unc = res.get("gauss_uncorrected")
        moff_corr = res.get("moffat_corrected")
        moff_unc = res.get("moffat_uncorrected")
        def _fmt_fit(label: str, fwhm_arcsec: float, d: Dict[str, Any]) -> str:
            try:
                x0 = float(d.get("x0_px", float("nan")))
                y0 = float(d.get("y0_px", float("nan")))
                r = float(d.get("r_px", float("nan")))
            except Exception:
                x0 = y0 = r = float("nan")
            return f"{label}: {fwhm_arcsec:.3f} arcsec (x0={x0:.2f}, y0={y0:.2f}, r={r:.2f} px)"
        used = int(res.get("n_frames_used", done))
        discard_frames = int(res.get("discard_first_frames", 0))
        discard_s = float(res.get("discard_first_s", 0.0))
        r0_est = float(res.get("r0_uncorrected_m", 0.0))
        lines = [
            f"Frames: {done}/{total}",
            f"Used for fit: {used} (discarded {discard_frames} frames = {discard_s:.1f} s)",
            _fmt_fit("Gaussian FWHM corrected", fwhm_c, gauss_corr),
            _fmt_fit("Gaussian FWHM uncorrected", fwhm_u, gauss_unc),
            _fmt_fit("Moffat FWHM corrected", fwhm_cmof, moff_corr),
            _fmt_fit("Moffat FWHM uncorrected", fwhm_umof, moff_unc),
            f"Uncorrected r0: {r0_est:.3f} m",
        ]
        if "r0_effective_m" in res:
            lines.append(f"Effective r0: {float(res['r0_effective_m']):.4f} m")
        elif "r0_est_m" in res:
            lines.append(f"Estimated r0: {float(res['r0_est_m']):.4f} m")

        self.lbl_results.setText("\n".join(lines))

        # Display images (log scale for visibility)
        def _show(canvas: PGCanvas, img: Optional[np.ndarray]):
            if img is None:
                return
            a = np.asarray(img)
            if a.ndim != 2:
                return
            # avoid log(0)
            a = np.maximum(a, 0)
            a = np.log10(a + 1e-12)
            canvas.queue_image(a)

        img_corr = res.get("psf_long_corrected")
        img_unc = res.get("psf_long_uncorrected")

        def _apply_overlays(canvas: PGCanvas, img: np.ndarray, g: Dict[str, Any], m: Dict[str, Any]):
            if img is None:
                return
            # Defaults if fit failed
            xg = float(g.get("x0_px", img.shape[1] * 0.5))
            yg = float(g.get("y0_px", img.shape[0] * 0.5))
            rg = float(g.get("r_px", 0.5 * float(g.get("fwhm_px", 0.0))))
            xm = float(m.get("x0_px", img.shape[1] * 0.5))
            ym = float(m.get("y0_px", img.shape[0] * 0.5))
            rm = float(m.get("r_px", 0.5 * float(m.get("fwhm_px", 0.0))))

            pen_g = pg.mkPen((255, 0, 0), width=2)
            pen_m = pg.mkPen((255, 0, 222), width=2)

            canvas.set_circle_overlay("fwhm_gauss", xg, yg, rg, pen=pen_g)
            canvas.set_circle_overlay("fwhm_moffat", xm, ym, rm, pen=pen_m)

            legend_html = (
                '<div style="background-color:rgba(0,0,0,180); padding:4px; border-radius:4px;">'
                '<span style="color:rgb(255,0,0);">●</span> Gaussian FWHM<br>'
                '<span style="color:rgb(0,255,0);">●</span> Moffat FWHM'
                '</div>'
            )
            canvas.set_text_overlay("legend", legend_html, anchor=(0.0, 0.0), offset=(8.0, 8.0))

        _show(self.canvas_corr, img_corr)
        _apply_overlays(self.canvas_corr, np.asarray(img_corr) if img_corr is not None else None, gauss_corr, moff_corr)

        _show(self.canvas_unc, img_unc)
        _apply_overlays(self.canvas_unc, np.asarray(img_unc) if img_unc is not None else None, gauss_unc, moff_unc)

    @Slot(str)
    def _on_long_run_failed(self, err: str):
        self.lbl_progress.setText("Failed")
        self.lbl_results.setStyleSheet("font-family: Consolas, monospace;")
        self.lbl_results.setText(str(err))

        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setValue(0)

        self.btn_open_dir.setEnabled(bool(self._last_out_dir))