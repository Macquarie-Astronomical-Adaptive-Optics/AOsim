from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
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

        self.chk_record_psfs = QCheckBox("Record per-frame PSFs (memmap)")
        self.chk_record_psfs.setChecked(True)
        grid.addWidget(self.chk_record_psfs, row, 0, 1, 2)

        self.chk_record_ttremoved = QCheckBox("Also record TT-removed corrected PSFs")
        self.chk_record_ttremoved.setChecked(False)
        grid.addWidget(self.chk_record_ttremoved, row, 2, 1, 2)
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
        base = Path(self.params.get("data_path", "output")) / "batch_runs"
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
        self.canvas_tt = PGCanvas(show_colorbar=False)

        right_layout.addWidget(QLabel("Long exposure PSF (corrected)"))
        right_layout.addWidget(self.canvas_corr)
        right_layout.addWidget(QLabel("Long exposure PSF (uncorrected)"))
        right_layout.addWidget(self.canvas_unc)
        right_layout.addWidget(QLabel("Long exposure PSF (corrected, TT-removed)"))
        right_layout.addWidget(self.canvas_tt)

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
        modes = ["none", "fit", "hybrid"]
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
            "record_psfs_ttremoved": bool(self.chk_record_ttremoved.isChecked()),
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
        used = int(res.get("n_frames_used", done))
        discard_frames = int(res.get("discard_first_frames", 0))
        discard_s = float(res.get("discard_first_s", 0.0))
        lines = [
            f"Frames: {done}/{total}",
            f"Used for fit: {used} (discarded {discard_frames} frames = {discard_s:.1f} s)",
            f"FWHM corrected: {fwhm_c:.3f} arcsec",
            f"FWHM uncorrected: {fwhm_u:.3f} arcsec",
        ]
        if "fwhm_arcsec_corrected_ttremoved" in res:
            lines.append(f"FWHM corrected (TT-removed): {float(res['fwhm_arcsec_corrected_ttremoved']):.3f} arcsec")
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

        _show(self.canvas_corr, res.get("psf_long_corrected"))
        _show(self.canvas_unc, res.get("psf_long_uncorrected"))
        _show(self.canvas_tt, res.get("psf_long_corrected_ttremoved"))

    @Slot(str)
    def _on_long_run_failed(self, err: str):
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setValue(0)
        self.lbl_progress.setText("Failed")
        self.lbl_results.setText(str(err))
        self.btn_open_dir.setEnabled(bool(self._last_out_dir))
