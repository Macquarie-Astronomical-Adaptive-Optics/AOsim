import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QSplitter,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
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
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(True)
        root.addWidget(self.splitter)

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
        self.lbl_frames = QLabel("1000")
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

        # Off-axis evaluation points (arcsec): semicolon-separated "dx,dy" pairs.
        # Off-axis evaluation points
        off_hdr = QFrame()
        off_hdr_l = QHBoxLayout(off_hdr)
        off_hdr_l.setContentsMargins(0, 0, 0, 0)
        off_hdr_l.addWidget(QLabel("Off-axis points"))
        self.cmb_offaxis_coords = QComboBox()
        self.cmb_offaxis_coords.addItems(["dx,dy (arcsec)", "θ,r (deg, arcsec)"])
        self.cmb_offaxis_coords.setCurrentIndex(0)
        self.cmb_offaxis_coords.currentIndexChanged.connect(self._on_offaxis_coord_mode_changed)
        off_hdr_l.addWidget(self.cmb_offaxis_coords)
        off_hdr_l.addStretch(1)
        grid.addWidget(off_hdr, row, 0)
        self.edit_offaxis = QLineEdit()
        self.edit_offaxis.setPlaceholderText("dx,dy; dx,dy; ...  (leave blank for on-axis only)")
        grid.addWidget(self.edit_offaxis, row, 1, 1, 3)
        row += 1

        self.chk_eval_unc_offaxis = QCheckBox("Compute off-axis uncorrected PSFs (slower)")
        self.chk_eval_unc_offaxis.setChecked(True)
        grid.addWidget(self.chk_eval_unc_offaxis, row, 0, 1, 2)

        self.chk_write_offaxis_fits = QCheckBox("Write off-axis long-exposure FITS")
        self.chk_write_offaxis_fits.setChecked(True)
        grid.addWidget(self.chk_write_offaxis_fits, row, 2, 1, 2)
        row += 1

        self.chk_write_unc_offaxis_fits = QCheckBox("Write uncorrected off-axis FITS")
        self.chk_write_unc_offaxis_fits.setChecked(False)
        grid.addWidget(self.chk_write_unc_offaxis_fits, row, 0, 1, 2)

        hint2 = QLabel("(needs off-axis uncorrected enabled)")
        hint2.setAlignment(Qt.AlignLeft)
        hint2.setStyleSheet("color: #666;")
        grid.addWidget(hint2, row, 2, 1, 2)
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

        # Off-axis results table (shown only if off-axis evaluation was run)
        self.grp_offaxis = QGroupBox("Long-exposure PSFs FWHM")
        self.grp_offaxis.setVisible(False)
        off_lay = QVBoxLayout(self.grp_offaxis)
        self.lbl_offaxis_stats = QLabel("")
        self.lbl_offaxis_stats.setWordWrap(True)
        self.lbl_offaxis_stats.setTextFormat(Qt.PlainText)
        off_lay.addWidget(self.lbl_offaxis_stats)

        self.tbl_offaxis = QTableWidget()
        self.tbl_offaxis.setColumnCount(6)
        self.tbl_offaxis.setHorizontalHeaderLabels(
            ["dx\n(arcsec)", "dy\n(arcsec)", "Gauss corr\n(as)", "Moffat corr\n(as)", "Gauss unc\n(as)", "Moffat unc\n(as)"]
        )
        # PySide6 enums live on QAbstractItemView (not on QTableWidget)
        self.tbl_offaxis.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_offaxis.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_offaxis.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        off_lay.addWidget(self.tbl_offaxis)

        left_layout.addWidget(self.grp_offaxis)


        # Open folder button
        self.btn_open_dir = QPushButton("Open output folder")
        self.btn_open_dir.setEnabled(False)
        self.btn_open_dir.clicked.connect(self._open_out_dir)
        left_layout.addWidget(self.btn_open_dir)

        left_layout.addStretch(1)

        # Right: PSF canvases + optional controls (always visible)
        right = QFrame()
        right.setFrameShape(QFrame.NoFrame)
        right_outer = QVBoxLayout(right)

        topbar = QHBoxLayout()
        topbar.addStretch(1)

        topbar.addWidget(QLabel("Display"))
        self.cmb_display_scale = QComboBox()
        self.cmb_display_scale.addItems(["Log", "Linear"])
        self.cmb_display_scale.setCurrentText("Log")
        self.cmb_display_scale.currentIndexChanged.connect(self._on_display_scale_changed)
        topbar.addWidget(self.cmb_display_scale)

        topbar.addSpacing(12)
        topbar.addWidget(QLabel("FWHM fit"))
        self.cmb_fit_model = QComboBox()
        # Display-only selection; computation is done by the worker and returned in the result dict.
        self.cmb_fit_model.addItems(["Moffat (wings)", "Moffat", "Gaussian", "Both"])
        self.cmb_fit_model.setCurrentText("Moffat (wings)")
        self.cmb_fit_model.currentIndexChanged.connect(self._on_fit_model_changed)
        topbar.addWidget(self.cmb_fit_model)


        right_outer.addLayout(topbar)

        right_layout = QHBoxLayout()
        self._psf_corr = self._make_psf_panel("Long exposure PSF (corrected)")
        self._psf_unc = self._make_psf_panel("Long exposure PSF (uncorrected)")

        right_layout.addWidget(self._psf_corr["widget"])
        right_layout.addSpacing(8)
        right_layout.addWidget(self._psf_unc["widget"])

        right_outer.addLayout(right_layout)


        self.splitter.addWidget(left)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        try:
            self.splitter.setSizes([360, 1000])
        except Exception:
            pass
        self._splitter_saved_sizes = self.splitter.sizes()
        self.splitter.splitterMoved.connect(self._on_splitter_moved)

        self._psf_display_mode = "log"
        self._last_long_run_result: Optional[dict] = None

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

        # PSF display mode for the canvases ("log" or "linear")
        self._psf_display_mode = "log"
        self._last_long_run_result: Optional[dict] = None

    # ---- PSF panel helpers ----
    def _make_psf_panel(self, title: str) -> Dict[str, Any]:
        """PSF panel.

        Instead of separate marginal PlotWidgets, we draw the X/Y profiles as
        curve overlays *in the same ViewBox as the PSF image*, but shifted into
        a reserved margin region to the bottom and right of the image.
        """
        box = QGroupBox(title)
        outer = QVBoxLayout(box)
        # Leave space for the groupbox title.
        outer.setContentsMargins(8, 14, 8, 8)
        canvas = PGCanvas(show_colorbar=False)

        canvas.setMinimumHeight(360)
        outer.addWidget(canvas)
        return {"widget": box, "canvas": canvas}

    def _update_profiles(
        self,
        panel: Dict[str, Any],
        img: Optional[np.ndarray],
        gfit: Dict[str, Any],
        mfit: Dict[str, Any],
        plate_rad_per_pix: float,
    ):
        """Draw X/Y cross-sections as overlays on the PSF canvas.

        The trick is to reserve a margin region to the bottom (for X profile)
        and right (for Y profile) in the **same ViewBox** as the image.
        Then we map each pixel's intensity into that margin.

        This makes each profile sample correspond 1:1 to a pixel coordinate
        in the displayed image.
        """
        if img is None:
            return

        a = np.asarray(img, dtype=float)
        if a.ndim != 2 or a.size == 0:
            return

        canvas: PGCanvas = panel["canvas"]
        H, W = a.shape

        def _finite_center(d: Dict[str, Any]) -> tuple[float, float] | None:
            try:
                x0 = float(d.get("x0_px", float("nan")))
                y0 = float(d.get("y0_px", float("nan")))
            except Exception:
                return None
            if not np.isfinite(x0) or not np.isfinite(y0):
                return None
            return (float(np.clip(x0, 0.0, W - 1.0)), float(np.clip(y0, 0.0, H - 1.0)))

        c_g = _finite_center(gfit)
        c_m = _finite_center(mfit)
        if c_g is None and c_m is None:
            yy, xx = np.unravel_index(int(np.argmax(a)), a.shape)
            c_g = (float(xx), float(yy))

        # One data slice only (like classic astronomy tools): take it through a
        # single reference center, then overlay both models evaluated along that
        # same cut.
        c_ref = c_g if c_g is not None else c_m

        # --- reserve margins for the profiles (in image pixel units) ---
        mx = int(max(40, min(0.35 * W, 0.6 * W)))
        my = int(max(40, min(0.35 * H, 0.6 * H)))
        pad = 4.0
        x_im_max = float(W - 0.5)
        y_im_max = float(H - 0.5)
        x_base = x_im_max + pad
        y_base = y_im_max + pad

        # Expand the ViewBox so the margin region is visible.
        try:
            canvas.vb.setRange(
                xRange=(-0.5, x_base + float(mx)),
                yRange=(-0.5, y_base + float(my)),
                padding=0.0,
            )
        except Exception:
            pass

        # Sub-pixel slice via linear interpolation (so it matches the fitted centers)
        def _row_at(y0: float) -> np.ndarray:
            y_f = int(np.floor(y0))
            y_c = min(y_f + 1, H - 1)
            ty = y0 - float(y_f)
            return (1.0 - ty) * a[y_f, :] + ty * a[y_c, :]

        def _col_at(x0: float) -> np.ndarray:
            x_f = int(np.floor(x0))
            x_c = min(x_f + 1, W - 1)
            tx = x0 - float(x_f)
            return (1.0 - tx) * a[:, x_f] + tx * a[:, x_c]

        eps = 1e-12

        # Bundle curves (log10 intensity) so we can normalize all to the same display scale.
        curves_x: Dict[str, np.ndarray] = {}
        curves_y: Dict[str, np.ndarray] = {}

        # Data slice: ONE cut only (classic astronomy tool behavior).
        # We take it through the reference center (Gaussian if available, else Moffat, else peak)
        # and then evaluate *both* models along that same cut.
        if c_ref is None:
            return
        x0r, y0r = c_ref
        curves_x["data"] = np.log10(np.maximum(_row_at(y0r), 0.0) + eps)
        curves_y["data"] = np.log10(np.maximum(_col_at(x0r), 0.0) + eps)

        # Gaussian model (evaluated along the data cut)
        try:
            A = float(gfit.get("amp", np.nan))
            s = float(gfit.get("sigma_px", np.nan))
            b0 = float(gfit.get("bkg", 0.0))
            if c_g is not None and np.isfinite(A) and np.isfinite(s) and s > 0:
                x0g, y0g = c_g
                dx = np.arange(W, dtype=float) - x0g
                dy = np.arange(H, dtype=float) - y0g
                # Cross-section through y=y0r (row) and x=x0r (col)
                dy0 = float(y0r) - y0g
                dx0 = float(x0r) - x0g
                g_row = A * np.exp(-0.5 * ((dx * dx) + (dy0 * dy0)) / (s * s)) + b0
                g_col = A * np.exp(-0.5 * ((dy * dy) + (dx0 * dx0)) / (s * s)) + b0
                curves_x["g_model"] = np.log10(np.maximum(g_row, 0.0) + eps)
                curves_y["g_model"] = np.log10(np.maximum(g_col, 0.0) + eps)
        except Exception:
            pass

        # Moffat model (evaluated along the data cut)
        try:
            A = float(mfit.get("amp", np.nan))
            gamma = float(mfit.get("gamma_px", np.nan))
            alpha = float(mfit.get("alpha", np.nan))
            b0 = float(mfit.get("bkg", 0.0))
            if c_m is not None and np.isfinite(A) and np.isfinite(gamma) and np.isfinite(alpha) and gamma > 0 and alpha > 0:
                x0m, y0m = c_m
                dx = np.arange(W, dtype=float) - x0m
                dy = np.arange(H, dtype=float) - y0m
                dy0 = float(y0r) - y0m
                dx0 = float(x0r) - x0m
                m_row = A * (1.0 + ((dx * dx) + (dy0 * dy0)) / (gamma * gamma)) ** (-alpha) + b0
                m_col = A * (1.0 + ((dy * dy) + (dx0 * dx0)) / (gamma * gamma)) ** (-alpha) + b0
                curves_x["m_model"] = np.log10(np.maximum(m_row, 0.0) + eps)
                curves_y["m_model"] = np.log10(np.maximum(m_col, 0.0) + eps)
        except Exception:
            pass

        def _finite_minmax(vals: list[np.ndarray]) -> tuple[float, float] | None:
            mins = []
            maxs = []
            for v in vals:
                v = np.asarray(v, dtype=float)
                if v.size == 0:
                    continue
                mn = float(np.nanmin(v))
                mx_ = float(np.nanmax(v))
                if np.isfinite(mn) and np.isfinite(mx_):
                    mins.append(mn)
                    maxs.append(mx_)
            if not mins or not maxs:
                return None
            mn = min(mins)
            mx_ = max(maxs)
            if not np.isfinite(mn) or not np.isfinite(mx_) or abs(mx_ - mn) < 1e-12:
                return None
            return (mn, mx_)

        mmx = _finite_minmax(list(curves_x.values()))
        mmy = _finite_minmax(list(curves_y.values()))
        if mmx is None or mmy is None:
            return
        zx0, zx1 = mmx
        zy0, zy1 = mmy

        x_pix = np.arange(W, dtype=float)
        y_pix = np.arange(H, dtype=float)

        def _map_h(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            n = (np.asarray(z, dtype=float) - zx0) / (zx1 - zx0)
            n = np.clip(n, 0.0, 1.0)
            yy = y_base + (1.0 - n) * float(my)
            return x_pix, yy

        def _map_v(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            n = (np.asarray(z, dtype=float) - zy0) / (zy1 - zy0)
            n = np.clip(n, 0.0, 1.0)
            xx = x_base + (1.0 - n) * float(mx)
            return xx, y_pix

        # Pens
        pen_data = pg.mkPen((200, 200, 200), width=1)
        pen_g_model = pg.mkPen((255, 0, 0), width=2)
        pen_m_model = pg.mkPen((255, 0, 222), width=2)
        # FWHM markers (match circle overlays)
        pen_g_fwhm = pg.mkPen((255, 0, 0), width=1, style=Qt.DashLine)
        pen_m_fwhm = pg.mkPen((255, 0, 222), width=1, style=Qt.DashLine)

        # Remove legacy per-fit data cuts (from earlier versions)
        for legacy in ("prof_x_g_data", "prof_y_g_data", "prof_x_m_data", "prof_y_m_data"):
            canvas.remove_overlay(legacy)

        # Draw/update curves
        if "data" in curves_x:
            x, y = _map_h(curves_x["data"])
            canvas.set_curve_overlay("prof_x_data", x, y, pen=pen_data)
        else:
            canvas.remove_overlay("prof_x_data")
        if "data" in curves_y:
            x, y = _map_v(curves_y["data"])
            canvas.set_curve_overlay("prof_y_data", x, y, pen=pen_data)
        else:
            canvas.remove_overlay("prof_y_data")

        if "g_model" in curves_x:
            x, y = _map_h(curves_x["g_model"])
            canvas.set_curve_overlay("prof_x_g_model", x, y, pen=pen_g_model)
        else:
            canvas.remove_overlay("prof_x_g_model")
        if "g_model" in curves_y:
            x, y = _map_v(curves_y["g_model"])
            canvas.set_curve_overlay("prof_y_g_model", x, y, pen=pen_g_model)
        else:
            canvas.remove_overlay("prof_y_g_model")

        if "m_model" in curves_x:
            x, y = _map_h(curves_x["m_model"])
            canvas.set_curve_overlay("prof_x_m_model", x, y, pen=pen_m_model)
        else:
            canvas.remove_overlay("prof_x_m_model")
        if "m_model" in curves_y:
            x, y = _map_v(curves_y["m_model"])
            canvas.set_curve_overlay("prof_y_m_model", x, y, pen=pen_m_model)
        else:
            canvas.remove_overlay("prof_y_m_model")

        # ---- FWHM boundary markers on the profiles (match the circle overlays) ----
        def _radius_px(d: Dict[str, Any]) -> Optional[float]:
            try:
                r = float(d.get("r_px", float("nan")))
            except Exception:
                r = float("nan")
            if np.isfinite(r) and r > 0:
                return r
            try:
                f = float(d.get("fwhm_px", float("nan")))
            except Exception:
                f = float("nan")
            if np.isfinite(f) and f > 0:
                return 0.5 * f
            return None

        def _set_bounds(prefix: str, center: Optional[tuple[float, float]], r_px: Optional[float], pen):
            # bottom margin (X-profile): vertical lines at x = x0 ± r
            # right margin (Y-profile): horizontal lines at y = y0 ± r
            names = (
                f"prof_x_{prefix}_fwhm_p",
                f"prof_x_{prefix}_fwhm_m",
                f"prof_y_{prefix}_fwhm_p",
                f"prof_y_{prefix}_fwhm_m",
            )
            if center is None or r_px is None or (not np.isfinite(r_px)) or r_px <= 0:
                for n in names:
                    canvas.remove_overlay(n)
                return

            x0, y0 = center
            rp = float(r_px)

            # Clip to image extent to avoid drawing off-canvas
            x_p = float(np.clip(x0 + rp, -0.5, W - 0.5))
            x_m = float(np.clip(x0 - rp, -0.5, W - 0.5))
            y_p = float(np.clip(y0 + rp, -0.5, H - 0.5))
            y_m = float(np.clip(y0 - rp, -0.5, H - 0.5))

            # X-profile bounds: vertical lines spanning the bottom margin region
            canvas.set_curve_overlay(names[0], np.array([x_p, x_p]), np.array([y_base, y_base + float(my)]), pen=pen)
            canvas.set_curve_overlay(names[1], np.array([x_m, x_m]), np.array([y_base, y_base + float(my)]), pen=pen)

            # Y-profile bounds: horizontal lines spanning the right margin region
            canvas.set_curve_overlay(names[2], np.array([x_base, x_base + float(mx)]), np.array([y_p, y_p]), pen=pen)
            canvas.set_curve_overlay(names[3], np.array([x_base, x_base + float(mx)]), np.array([y_m, y_m]), pen=pen)

        _set_bounds("g", c_g, _radius_px(gfit), pen_g_fwhm)
        _set_bounds("m", c_m, _radius_px(mfit), pen_m_fwhm)

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


    def _on_display_scale_changed(self, *_args):
        """Switch between log/linear PSF display and redraw immediately."""
        try:
            txt = str(self.cmb_display_scale.currentText()).strip().lower()
        except Exception:
            txt = "log"
        self._psf_display_mode = "log" if txt.startswith("log") else "linear"
        if getattr(self, "_last_long_run_result", None) is not None:
            # Re-render using stored results (no re-run).
            self._on_long_run_finished(self._last_long_run_result)


    def _fit_mode(self) -> str:
        try:
            t = str(self.cmb_fit_model.currentText()).strip().lower()
        except Exception:
            t = "moffat"
        if "both" in t:
            return "both"
        if "gauss" in t:
            return "gaussian"
        if "wings" in t:
            return "moffat_wings"
        return "moffat"

    def _offaxis_coord_mode(self) -> str:
        """Return "cart" (dx,dy) or "polar" (theta,r)."""
        try:
            t = str(self.cmb_offaxis_coords.currentText()).lower()
        except Exception:
            t = ""
        return "polar" if ("θ" in t or "theta" in t or "polar" in t) else "cart"

    def _on_fit_model_changed(self, *_args):
        # Redraw immediately using the last completed result.
        if self._last_long_run_result is not None:
            self._on_long_run_finished(self._last_long_run_result)

    def _on_offaxis_coord_mode_changed(self, *_args):
        # Update input placeholder and redraw the results table/mosaic if available.
        try:
            if self._offaxis_coord_mode() == "polar":
                self.edit_offaxis.setPlaceholderText("θ_deg,r_arcsec; θ_deg,r_arcsec; ...  (leave blank for on-axis only)")
            else:
                self.edit_offaxis.setPlaceholderText("dx,dy; dx,dy; ...  (leave blank for on-axis only)")
        except Exception:
            pass
        if self._last_long_run_result is not None:
            self._on_long_run_finished(self._last_long_run_result)

    def _on_splitter_moved(self, *_args):
        """Keep the toggle label in sync with manual splitter drags."""
        try:
            sizes = list(self.splitter.sizes())
        except Exception:
            return
        if len(sizes) < 2:
            return
        if sizes[0] <= 10:
            if hasattr(self, "btn_toggle_settings"):
                self.btn_toggle_settings.setText("Show settings")
        else:
            self._splitter_saved_sizes = sizes
            if hasattr(self, "btn_toggle_settings"):
                self.btn_toggle_settings.setText("Hide settings")

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

        # ---- Off-axis evaluation config ----
        eval_pts = None
        try:
            txt = self.edit_offaxis.text().strip() if hasattr(self, "edit_offaxis") else ""
        except Exception:
            txt = ""
        if txt:
            import re as _re
            eval_pts = []
            for chunk in _re.split(r"[;\n]+", txt):
                c = chunk.strip()
                if not c:
                    continue
                c = c.strip().strip("()[]{}")
                parts = [p for p in _re.split(r"[\s,]+", c) if p]
                if len(parts) < 2:
                    continue
                try:
                    a0 = float(parts[0])
                    a1 = float(parts[1])
                except Exception:
                    continue

                if self._offaxis_coord_mode() == "polar":
                    # θ in degrees, r in arcsec → convert to (dx,dy) arcsec
                    th = np.deg2rad(a0)
                    r_as = a1
                    dx = float(r_as * np.cos(th))
                    dy = float(r_as * np.sin(th))
                else:
                    dx = float(a0)
                    dy = float(a1)

                eval_pts.append((dx, dy))
            if len(eval_pts) == 0:
                eval_pts = None

        if eval_pts is not None:
            cfg["eval_offaxis_arcsec"] = eval_pts
            cfg["write_eval_fits"] = bool(self.chk_write_offaxis_fits.isChecked()) if hasattr(self, "chk_write_offaxis_fits") else True

            eval_unc = bool(self.chk_eval_unc_offaxis.isChecked()) if hasattr(self, "chk_eval_unc_offaxis") else True
            cfg["eval_accumulate_uncorrected"] = eval_unc
            cfg["write_eval_uncorrected_fits"] = bool(self.chk_write_unc_offaxis_fits.isChecked()) and eval_unc if hasattr(self, "chk_write_unc_offaxis_fits") else eval_unc


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
        self._last_long_run_result = res
        self._last_long_run_result = res
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
        # On-axis FWHM (arcsec). The scheduler usually provides these, but we also
        # support fallbacks from radian values if needed.
        fwhm_c = float(res.get("fwhm_arcsec_corrected", 0.0))
        fwhm_u = float(res.get("fwhm_arcsec_uncorrected", 0.0))
        fwhm_cmof = float(res.get("fwhm_arcsec_corrected_moffat", 0.0))
        fwhm_umof = float(res.get("fwhm_arcsec_uncorrected_moffat", 0.0))

        fwhm_cmof_w = float(res.get("fwhm_arcsec_corrected_moffat_wings", float("nan")))
        fwhm_umof_w = float(res.get("fwhm_arcsec_uncorrected_moffat_wings", float("nan")))

        gauss_corr = res.get("gauss_corrected") or {}
        gauss_unc = res.get("gauss_uncorrected") or {}
        moff_corr = res.get("moffat_corrected") or {}
        moff_unc = res.get("moffat_uncorrected") or {}
        moffw_corr = res.get("moffat_wings_corrected") or {}
        moffw_unc = res.get("moffat_wings_uncorrected") or {}

        used = int(res.get("n_frames_used", done))
        discard_frames = int(res.get("discard_first_frames", 0))
        discard_s = float(res.get("discard_first_s", 0.0))
        r0_est = float(res.get("r0_uncorrected_m", 0.0))
        lines = [
            f"Frames: {done}/{total}",
            f"Used for fit: {used} (discarded {discard_frames} frames = {discard_s:.1f} s)",
            f"Uncorrected r0: {r0_est:.3f} m",
        ]
        if "r0_effective_m" in res:
            lines.append(f"Effective r0: {float(res['r0_effective_m']):.4f} m")
        elif "r0_est_m" in res:
            lines.append(f"Estimated r0: {float(res['r0_est_m']):.4f} m")

        # Off-axis summary (if present)
        off_metrics = res.get("offaxis_metrics") or []
        off_stats = res.get("offaxis_stats") or {}

        def _fmt_stats_line(label: str, key: str) -> Optional[str]:
            try:
                s = dict(off_stats.get(key) or {})
                n = int(s.get("n", 0))
                if n <= 0:
                    return None
                mean = float(s.get("mean", float("nan")))
                std = float(s.get("std", float("nan")))
                mn = float(s.get("min", float("nan")))
                mx = float(s.get("max", float("nan")))
                return f"{label}: mean={mean:.3f}±{std:.3f} as  (min={mn:.3f}, max={mx:.3f}, n={n})"
            except Exception:
                return None

        if len(off_metrics) > 0:
            lines.append("")
            lines.append(f"Off-axis points: {len(off_metrics)}")
            

        self.lbl_results.setText("\n".join(lines))

        # Populate results table (includes on-axis as first row)
        if hasattr(self, "grp_offaxis") and hasattr(self, "tbl_offaxis"):
            fit_mode = self._fit_mode()
            coord_mode = self._offaxis_coord_mode()

            show_g = fit_mode in ("gaussian", "both")
            show_m = fit_mode in ("moffat", "both", "moffat_wings")
            use_wings = (fit_mode == "moffat_wings")

            # Choose which Moffat values to display
            mc_on = fwhm_cmof_w if use_wings else fwhm_cmof
            mu_on = fwhm_umof_w if use_wings else fwhm_umof

            # Build rows: on-axis first, then off-axis
            rows: list[dict[str, Any]] = []
            rows.append({
                "dx": 0.0, "dy": 0.0,
                "gc": fwhm_c, "gu": fwhm_u,
                "mc": mc_on, "mu": mu_on,
                "is_onaxis": True,
            })

            for m in off_metrics:
                dx = float(m.get("dx_arcsec", 0.0))
                dy = float(m.get("dy_arcsec", 0.0))
                gc = float(m.get("fwhm_arcsec_corrected", float("nan")))
                gu = float(m.get("fwhm_arcsec_uncorrected", float("nan")))
                if use_wings:
                    mc = float(m.get("fwhm_arcsec_corrected_moffat_wings", float("nan")))
                    mu = float(m.get("fwhm_arcsec_uncorrected_moffat_wings", float("nan")))
                else:
                    mc = float(m.get("fwhm_arcsec_corrected_moffat", float("nan")))
                    mu = float(m.get("fwhm_arcsec_uncorrected_moffat", float("nan")))
                rows.append({
                    "dx": dx, "dy": dy,
                    "gc": gc, "gu": gu,
                    "mc": mc, "mu": mu,
                    "is_onaxis": False,
                })

            # Show group whenever we have at least the on-axis row
            self.grp_offaxis.setVisible(True)
            self.tbl_offaxis.setRowCount(len(rows))

            # Column headers
            if coord_mode == "polar":
                h0, h1 = "θ\n(deg)", "r\n(arcsec)"
            else:
                h0, h1 = "dx\n(arcsec)", "dy\n(arcsec)"
            mlabel = "Moffat(w)" if use_wings else "Moffat"
            self.tbl_offaxis.setHorizontalHeaderLabels([
                h0, h1,
                "Gauss corr\n(as)", f"{mlabel} corr\n(as)",
                "Gauss unc\n(as)", f"{mlabel} unc\n(as)",
            ])

            # Determine whether uncorrected columns should be shown for the selected fit(s)
            def _n_from_stats(key: str) -> int:
                try:
                    return int((off_stats.get(key) or {}).get("n", 0))
                except Exception:
                    return 0

            want_g_unc = show_g and (np.isfinite(fwhm_u) or (_n_from_stats("gauss_uncorrected_arcsec") > 0))
            m_unc_key = "moffat_wings_uncorrected_arcsec" if use_wings else "moffat_uncorrected_arcsec"
            want_m_unc = show_m and (np.isfinite(mu_on) or (_n_from_stats(m_unc_key) > 0))
            show_unc = bool(want_g_unc or want_m_unc)

            # Show/hide fit columns
            self.tbl_offaxis.setColumnHidden(2, not show_g)
            self.tbl_offaxis.setColumnHidden(4, (not show_g) or (not show_unc))
            self.tbl_offaxis.setColumnHidden(3, not show_m)
            self.tbl_offaxis.setColumnHidden(5, (not show_m) or (not show_unc))

            def _it(v: str, tooltip: str = ""):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if tooltip:
                    item.setToolTip(tooltip)
                return item

            def _coord_strings(dx_as: float, dy_as: float) -> tuple[str, str]:
                if coord_mode == "polar":
                    r_as = float(np.hypot(dx_as, dy_as))
                    th = float(np.degrees(np.arctan2(dy_as, dx_as)))
                    if not np.isfinite(th):
                        th = 0.0
                    # Wrap to [0,360) for nicer display
                    th = th % 360.0
                    return (f"{th:.1f}", f"{r_as:.1f}")
                return (f"{dx_as:.0f}", f"{dy_as:.0f}")

            # Fill rows
            for i, r in enumerate(rows):
                dx = float(r["dx"]); dy = float(r["dy"])
                gc = float(r["gc"]); gu = float(r["gu"])
                mc = float(r["mc"]); mu = float(r["mu"])
                tip = "On-axis" if r.get("is_onaxis") else ""
                c0, c1 = _coord_strings(dx, dy)
                self.tbl_offaxis.setItem(i, 0, _it(c0, tip))
                self.tbl_offaxis.setItem(i, 1, _it(c1, tip))
                self.tbl_offaxis.setItem(i, 2, _it("" if (not show_g or not np.isfinite(gc)) else f"{gc:.3f}", tip))
                self.tbl_offaxis.setItem(i, 3, _it("" if (not show_m or not np.isfinite(mc)) else f"{mc:.3f}", tip))
                self.tbl_offaxis.setItem(i, 4, _it("" if (not show_g or not show_unc or not np.isfinite(gu)) else f"{gu:.3f}", tip))
                self.tbl_offaxis.setItem(i, 5, _it("" if (not show_m or not show_unc or not np.isfinite(mu)) else f"{mu:.3f}", tip))

            # Stats label: on-axis + off-axis aggregates (for selected fits)
            stats_lines: list[str] = []
            on_parts: list[str] = []
            if show_g:
                on_parts.append(f"G={fwhm_c:.3f} as")
            if show_m:
                on_parts.append(f"{mlabel}={mc_on:.3f} as")
            if show_unc:
                if show_g and np.isfinite(fwhm_u):
                    on_parts.append(f"G_unc={fwhm_u:.3f} as")
                if show_m and np.isfinite(mu_on):
                    on_parts.append(f"{mlabel}_unc={mu_on:.3f} as")
            if on_parts:
                stats_lines.append("On-axis: " + ", ".join(on_parts))

            def _fmt_stats_line(label: str, key: str) -> Optional[str]:
                try:
                    s = dict(off_stats.get(key) or {})
                    n = int(s.get("n", 0))
                    if n <= 0:
                        return None
                    mean = float(s.get("mean", float("nan")))
                    std = float(s.get("std", float("nan")))
                    mn = float(s.get("min", float("nan")))
                    mx = float(s.get("max", float("nan")))
                    return f"{label} (off-axis): mean={mean:.3f}±{std:.3f} as  (min={mn:.3f}, max={mx:.3f}, n={n})"
                except Exception:
                    return None

            if len(off_metrics) > 0:
                if show_g:
                    line = _fmt_stats_line("Gaussian corrected", "gauss_corrected_arcsec")
                    if line:
                        stats_lines.append(line)
                if show_m:
                    m_corr_key = "moffat_wings_corrected_arcsec" if use_wings else "moffat_corrected_arcsec"
                    line = _fmt_stats_line(f"{mlabel} corrected", m_corr_key)
                    if line:
                        stats_lines.append(line)
                if show_unc:
                    if show_g:
                        line = _fmt_stats_line("Gaussian uncorrected", "gauss_uncorrected_arcsec")
                        if line:
                            stats_lines.append(line)
                    if show_m:
                        m_unc_key = "moffat_wings_uncorrected_arcsec" if use_wings else "moffat_uncorrected_arcsec"
                        line = _fmt_stats_line(f"{mlabel} uncorrected", m_unc_key)
                        if line:
                            stats_lines.append(line)

            self.lbl_offaxis_stats.setText("\n".join(stats_lines))


        # What to show (display-only; worker always computes Gaussian + Moffat + Moffat(wings) when available)
        fit_mode = self._fit_mode()
        show_g = fit_mode in ("gaussian", "both")
        show_m = fit_mode in ("moffat", "both", "moffat_wings")
        use_wings = (fit_mode == "moffat_wings")
        mlabel = "Moffat(w)" if use_wings else "Moffat"

        # Select which Moffat fit dicts to use for overlays/profiles
        moff_corr_show = moffw_corr if use_wings else moff_corr
        moff_unc_show  = moffw_unc if use_wings else moff_unc

        plate = float(res.get("plate_rad_per_pix", 0.0) or 0.0)

        # Display images (log or linear)
        use_log = (getattr(self, "_psf_display_mode", "log") == "log")
        def _show(panel: Dict[str, Any], img: Optional[np.ndarray]):
            if img is None:
                return
            a = np.asarray(img)
            if a.ndim != 2:
                return
            a = np.maximum(a, 0)
            # Render immediately so we can safely expand the ViewBox for profile margins.
            if use_log:
                panel["canvas"].set_image(np.log10(a + 1e-12))
            else:
                panel["canvas"].set_image(a.astype(np.float32, copy=False))

        def _apply_overlays(canvas: PGCanvas, img: np.ndarray, g: Dict[str, Any], m: Dict[str, Any]):
            if img is None:
                return

            # Use each fit's own center; fall back to the peak if a fit failed.
            yy, xx = np.unravel_index(int(np.argmax(img)), img.shape)
            xpk, ypk = float(xx), float(yy)

            xg = float(g.get("x0_px", xpk))
            yg = float(g.get("y0_px", ypk))
            if not np.isfinite(xg) or not np.isfinite(yg):
                xg, yg = xpk, ypk

            xm = float(m.get("x0_px", xpk))
            ym = float(m.get("y0_px", ypk))
            if not np.isfinite(xm) or not np.isfinite(ym):
                xm, ym = xpk, ypk

            # Radii from fits
            rg = float(g.get("r_px", 0.5 * float(g.get("fwhm_px", 0.0))))
            rm = float(m.get("r_px", 0.5 * float(m.get("fwhm_px", 0.0))))

            pen_g = pg.mkPen((255, 0, 0), width=2)
            pen_m = pg.mkPen((255, 0, 222), width=2)


            # Only draw what the user selected.
            if show_g and np.isfinite(rg) and rg > 0:
                canvas.set_circle_overlay("fwhm_gauss", xg, yg, rg, pen=pen_g)
            else:
                canvas.remove_overlay("fwhm_gauss")

            if show_m and np.isfinite(rm) and rm > 0:
                canvas.set_circle_overlay("fwhm_moffat", xm, ym, rm, pen=pen_m)
            else:
                canvas.remove_overlay("fwhm_moffat")

            legend_lines = []
            if show_g:
                legend_lines.append('<span style="color:rgb(255,0,0);">●</span> Gaussian FWHM')
            if show_m:
                legend_lines.append(f'<span style="color:rgb(255,0,222);">●</span> {mlabel} FWHM')

            if legend_lines:
                legend_html = (
                    '<div style="background-color:rgba(0,0,0,180); padding:4px; border-radius:4px;">'
                    + "<br>".join(legend_lines)
                    + "</div>"
                )
                canvas.set_text_overlay("legend", legend_html, anchor=(0.0, 1.0), offset=(8.0, -8.0))
            else:
                canvas.remove_overlay("legend")


        def _build_mosaic(imgs, gap: int = 24, pad: int = 12):
            """Build a mosaic that reserves per-tile margins for X/Y profiles.

            Each tile becomes a *block* of size:
            (tile_h + my + pad) x (tile_w + mx + pad)

            The PSF lives in the top-left tile_w x tile_h region. The extra bottom
            and right regions are left as NaN so we can draw the profile overlays
            there without overlapping neighboring tiles.
            """
            imgs = [np.asarray(im) for im in imgs if im is not None]
            if len(imgs) == 0:
                return None, [], 0, 0, 0, 0, 0, 0, 0, 0, 0
            h, w = imgs[0].shape
            n = len(imgs)
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / max(cols, 1)))

            # Profile margins per tile (tuned to avoid overlap/visual clutter).
            mx = int(max(56, min(0.50 * w, 0.95 * w)))
            my = int(max(56, min(0.50 * h, 0.95 * h)))

            block_w = int(w + mx + pad)
            block_h = int(h + my + pad)

            H = rows * block_h + max(0, rows - 1) * gap
            W = cols * block_w + max(0, cols - 1) * gap
            mosaic = np.full((H, W), np.nan, dtype=np.float32)
            offsets = []

            for i, im in enumerate(imgs):
                r = i // cols
                c = i % cols
                y0 = r * (block_h + gap)
                x0 = c * (block_w + gap)

                tile_lin = np.maximum(im.astype(np.float32, copy=False), 0.0)
                tile = np.log10(tile_lin + 1e-12) if use_log else tile_lin
                mosaic[y0 : y0 + h, x0 : x0 + w] = tile
                offsets.append((x0, y0))

            return mosaic, offsets, rows, cols, h, w, mx, my, pad, block_h, block_w

        def _show_mosaic(panel, imgs, gfits, mfits, labels, prefix: str):
            canvas = panel["canvas"]

            # Clear previous overlays (tiles + profiles)
            for i in range(512):
                canvas.remove_overlay(f"{prefix}_g_{i}")
                canvas.remove_overlay(f"{prefix}_m_{i}")
                canvas.remove_overlay(f"{prefix}_t_{i}")

                for name in (
                    "prof_x_data",
                    "prof_y_data",
                    "prof_x_g_model",
                    "prof_y_g_model",
                    "prof_x_m_model",
                    "prof_y_m_model",
                    "prof_x_g_fwhm_p",
                    "prof_x_g_fwhm_m",
                    "prof_y_g_fwhm_p",
                    "prof_y_g_fwhm_m",
                    "prof_x_m_fwhm_p",
                    "prof_x_m_fwhm_m",
                    "prof_y_m_fwhm_p",
                    "prof_y_m_fwhm_m",
                ):
                    canvas.remove_overlay(f"{prefix}_{name}_{i}")

            # Remove single-image overlays (in case we were showing on-axis only previously)
            for name in (
                "fwhm_gauss",
                "fwhm_moffat",
                "legend",
                "prof_x_data",
                "prof_y_data",
                "prof_x_g_model",
                "prof_y_g_model",
                "prof_x_m_model",
                "prof_y_m_model",
                "prof_x_g_fwhm_p",
                "prof_x_g_fwhm_m",
                "prof_y_g_fwhm_p",
                "prof_y_g_fwhm_m",
                "prof_x_m_fwhm_p",
                "prof_x_m_fwhm_m",
                "prof_y_m_fwhm_p",
                "prof_y_m_fwhm_m",
            ):
                canvas.remove_overlay(name)

            mosaic, offsets, rows, cols, h, w, mx, my, pad, block_h, block_w = _build_mosaic(imgs)
            if mosaic is None:
                return

            canvas.set_image(mosaic)

            # Legend (reflect current fit selection)
            legend_items = []
            if show_g:
                legend_items.append("<span style='color:yellow;'>●</span> Gaussian FWHM")
            if show_m:
                legend_items.append(f"<span style='color:magenta;'>●</span> {mlabel} FWHM")
            legend_items.append("<span style='color:rgb(200,200,200);'>—</span> Data slice")
            legend_html = "<div style='background-color:rgba(0,0,0,140); padding:6px;'>" + "&nbsp;&nbsp;".join(legend_items[:-1]) + "<br>" + legend_items[-1] + "</div>"
            canvas.set_text_overlay("legend", legend_html, anchor=(0, -10), offset=(8, 8), relative_to="image")

            pen_g = pg.mkPen((255, 0 ,0), width=2)
            pen_m = pg.mkPen((255, 0, 255), width=2)

            # Profile pens
            pen_data = pg.mkPen((200, 200, 200), width=1)
            pen_g_model = pg.mkPen((255, 0 ,0), width=2)
            pen_m_model = pg.mkPen((255, 0, 255), width=2)
            pen_g_fwhm = pg.mkPen((255, 0 ,0), width=1, style=Qt.DashLine)
            pen_m_fwhm = pg.mkPen((255, 0, 255), width=1, style=Qt.DashLine)

            eps = 1e-12

            def _finite_center(d: Dict[str, Any], W: int, H: int) -> Optional[tuple[float, float]]:
                try:
                    x0 = float(d.get("x0_px", float("nan")))
                    y0 = float(d.get("y0_px", float("nan")))
                except Exception:
                    return None
                if not np.isfinite(x0) or not np.isfinite(y0):
                    return None
                return (float(np.clip(x0, 0.0, W - 1.0)), float(np.clip(y0, 0.0, H - 1.0)))

            def _radius_px(d: Dict[str, Any]) -> Optional[float]:
                try:
                    r = float(d.get("r_px", float("nan")))
                except Exception:
                    r = float("nan")
                if np.isfinite(r) and r > 0:
                    return r
                try:
                    f = float(d.get("fwhm_px", float("nan")))
                except Exception:
                    f = float("nan")
                if np.isfinite(f) and f > 0:
                    return 0.5 * f
                return None

            def _tile_profiles(i: int, im: np.ndarray, g: Dict[str, Any], m: Dict[str, Any], x0: int, y0: int):
                a = np.asarray(im, dtype=float)
                if a.ndim != 2 or a.size == 0:
                    return

                # Honor the current fit selection (avoid drawing stale model overlays).
                if not show_g:
                    g = {}
                if not show_m:
                    m = {}

                Ht, Wt = a.shape

                c_g = _finite_center(g, Wt, Ht)
                c_m = _finite_center(m, Wt, Ht)
                if c_g is None and c_m is None:
                    yy, xx = np.unravel_index(int(np.argmax(a)), a.shape)
                    c_ref = (float(xx), float(yy))
                else:
                    c_ref = c_g if c_g is not None else c_m

                x0r, y0r = c_ref

                # Sub-pixel slice (linear interpolation)
                def _row_at(yv: float) -> np.ndarray:
                    yf = int(np.floor(yv))
                    yc = min(yf + 1, Ht - 1)
                    ty = yv - float(yf)
                    return (1.0 - ty) * a[yf, :] + ty * a[yc, :]

                def _col_at(xv: float) -> np.ndarray:
                    xf = int(np.floor(xv))
                    xc = min(xf + 1, Wt - 1)
                    tx = xv - float(xf)
                    return (1.0 - tx) * a[:, xf] + tx * a[:, xc]

                curves_x: Dict[str, np.ndarray] = {}
                curves_y: Dict[str, np.ndarray] = {}

                curves_x["data"] = np.log10(np.maximum(_row_at(y0r), 0.0) + eps)
                curves_y["data"] = np.log10(np.maximum(_col_at(x0r), 0.0) + eps)

                # Gaussian model along the same cut
                try:
                    A = float(g.get("amp", np.nan))
                    s = float(g.get("sigma_px", np.nan))
                    b0 = float(g.get("bkg", 0.0))
                    if c_g is not None and np.isfinite(A) and np.isfinite(s) and s > 0:
                        x0g, y0g = c_g
                        dx = np.arange(Wt, dtype=float) - x0g
                        dy = np.arange(Ht, dtype=float) - y0g
                        dy0 = float(y0r) - y0g
                        dx0 = float(x0r) - x0g
                        g_row = A * np.exp(-0.5 * ((dx * dx) + (dy0 * dy0)) / (s * s)) + b0
                        g_col = A * np.exp(-0.5 * ((dy * dy) + (dx0 * dx0)) / (s * s)) + b0
                        curves_x["g_model"] = np.log10(np.maximum(g_row, 0.0) + eps)
                        curves_y["g_model"] = np.log10(np.maximum(g_col, 0.0) + eps)
                except Exception:
                    pass

                # Moffat model along the same cut
                try:
                    A = float(m.get("amp", np.nan))
                    gamma = float(m.get("gamma_px", np.nan))
                    alpha = float(m.get("alpha", np.nan))
                    b0 = float(m.get("bkg", 0.0))
                    if c_m is not None and np.isfinite(A) and np.isfinite(gamma) and np.isfinite(alpha) and gamma > 0 and alpha > 0:
                        x0m, y0m = c_m
                        dx = np.arange(Wt, dtype=float) - x0m
                        dy = np.arange(Ht, dtype=float) - y0m
                        dy0 = float(y0r) - y0m
                        dx0 = float(x0r) - x0m
                        m_row = A * (1.0 + ((dx * dx) + (dy0 * dy0)) / (gamma * gamma)) ** (-alpha) + b0
                        m_col = A * (1.0 + ((dy * dy) + (dx0 * dx0)) / (gamma * gamma)) ** (-alpha) + b0
                        curves_x["m_model"] = np.log10(np.maximum(m_row, 0.0) + eps)
                        curves_y["m_model"] = np.log10(np.maximum(m_col, 0.0) + eps)
                except Exception:
                    pass

                def _finite_minmax(vals: list[np.ndarray]) -> Optional[tuple[float, float]]:
                    mins = []
                    maxs = []
                    for v in vals:
                        v = np.asarray(v, dtype=float)
                        if v.size == 0:
                            continue
                        mn = float(np.nanmin(v))
                        mxv = float(np.nanmax(v))
                        if np.isfinite(mn) and np.isfinite(mxv):
                            mins.append(mn)
                            maxs.append(mxv)
                    if not mins or not maxs:
                        return None
                    mn = min(mins)
                    mxv = max(maxs)
                    if not np.isfinite(mn) or not np.isfinite(mxv) or abs(mxv - mn) < 1e-12:
                        return None
                    return (mn, mxv)

                mmx = _finite_minmax(list(curves_x.values()))
                mmy = _finite_minmax(list(curves_y.values()))
                if mmx is None or mmy is None:
                    return
                zx0, zx1 = mmx
                zy0, zy1 = mmy

                x_pix = (x0 + np.arange(Wt, dtype=float))
                y_pix = (y0 + np.arange(Ht, dtype=float))

                x_im_max = float(x0 + (Wt - 0.5))
                y_im_max = float(y0 + (Ht - 0.5))
                x_base = x_im_max + float(pad)
                y_base = y_im_max + float(pad)

                def _map_h(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                    nrm = (np.asarray(z, dtype=float) - zx0) / (zx1 - zx0)
                    nrm = np.clip(nrm, 0.0, 1.0)
                    yy = y_base + (1.0 - nrm) * float(my)
                    return x_pix, yy

                def _map_v(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                    nrm = (np.asarray(z, dtype=float) - zy0) / (zy1 - zy0)
                    nrm = np.clip(nrm, 0.0, 1.0)
                    xx = x_base + (1.0 - nrm) * float(mx)
                    return xx, y_pix

                # Curves
                x, y = _map_h(curves_x["data"])
                canvas.set_curve_overlay(f"{prefix}_prof_x_data_{i}", x, y, pen=pen_data)
                x, y = _map_v(curves_y["data"])
                canvas.set_curve_overlay(f"{prefix}_prof_y_data_{i}", x, y, pen=pen_data)

                if "g_model" in curves_x:
                    x, y = _map_h(curves_x["g_model"])
                    canvas.set_curve_overlay(f"{prefix}_prof_x_g_model_{i}", x, y, pen=pen_g_model)
                if "g_model" in curves_y:
                    x, y = _map_v(curves_y["g_model"])
                    canvas.set_curve_overlay(f"{prefix}_prof_y_g_model_{i}", x, y, pen=pen_g_model)

                if "m_model" in curves_x:
                    x, y = _map_h(curves_x["m_model"])
                    canvas.set_curve_overlay(f"{prefix}_prof_x_m_model_{i}", x, y, pen=pen_m_model)
                if "m_model" in curves_y:
                    x, y = _map_v(curves_y["m_model"])
                    canvas.set_curve_overlay(f"{prefix}_prof_y_m_model_{i}", x, y, pen=pen_m_model)

                # FWHM boundary markers
                def _bounds(prefix2: str, center: Optional[tuple[float, float]], r_px: Optional[float], pen):
                    names = (
                        f"{prefix}_prof_x_{prefix2}_fwhm_p_{i}",
                        f"{prefix}_prof_x_{prefix2}_fwhm_m_{i}",
                        f"{prefix}_prof_y_{prefix2}_fwhm_p_{i}",
                        f"{prefix}_prof_y_{prefix2}_fwhm_m_{i}",
                    )
                    if center is None or r_px is None or (not np.isfinite(r_px)) or r_px <= 0:
                        for n in names:
                            canvas.remove_overlay(n)
                        return

                    cx, cy = center
                    rp = float(r_px)

                    x_p = float(np.clip(cx + rp, 0.0, Wt - 1.0)) + float(x0)
                    x_m = float(np.clip(cx - rp, 0.0, Wt - 1.0)) + float(x0)
                    y_p = float(np.clip(cy + rp, 0.0, Ht - 1.0)) + float(y0)
                    y_m = float(np.clip(cy - rp, 0.0, Ht - 1.0)) + float(y0)

                    # X-profile bounds: vertical lines spanning the bottom margin region
                    canvas.set_curve_overlay(names[0], np.array([x_p, x_p]), np.array([y_base, y_base + float(my)]), pen=pen)
                    canvas.set_curve_overlay(names[1], np.array([x_m, x_m]), np.array([y_base, y_base + float(my)]), pen=pen)

                    # Y-profile bounds: horizontal lines spanning the right margin region
                    canvas.set_curve_overlay(names[2], np.array([x_base, x_base + float(mx)]), np.array([y_p, y_p]), pen=pen)
                    canvas.set_curve_overlay(names[3], np.array([x_base, x_base + float(mx)]), np.array([y_m, y_m]), pen=pen)

                _bounds("g", c_g, _radius_px(g), pen_g_fwhm)
                _bounds("m", c_m, _radius_px(m), pen_m_fwhm)

            for i, (x0, y0) in enumerate(offsets):
                im = np.asarray(imgs[i])
                yy, xx = np.unravel_index(int(np.nanargmax(im)), im.shape)
                xpk, ypk = float(xx), float(yy)

                g = gfits[i] if i < len(gfits) else None
                m = mfits[i] if i < len(mfits) else None
                if not isinstance(g, dict):
                    g = {}
                if not isinstance(m, dict):
                    m = {}

                gx = float(g.get("x0_px", xpk))
                gy = float(g.get("y0_px", ypk))
                gr = float(g.get("r_px", 0.5 * float(g.get("fwhm_px", 0.0) or 0.0)))
                if not np.isfinite(gr) or gr <= 0:
                    gr = 0.0

                mx0 = float(m.get("x0_px", xpk))
                my0 = float(m.get("y0_px", ypk))
                mr = float(m.get("r_px", 0.5 * float(m.get("fwhm_px", 0.0) or 0.0)))
                if not np.isfinite(mr) or mr <= 0:
                    mr = 0.0

                if show_g and gr > 0:
                    canvas.set_circle_overlay(f"{prefix}_g_{i}", x0 + gx, y0 + gy, gr, pen=pen_g)
                else:
                    canvas.remove_overlay(f"{prefix}_g_{i}")

                if show_m and mr > 0:
                    canvas.set_circle_overlay(f"{prefix}_m_{i}", x0 + mx0, y0 + my0, mr, pen=pen_m)
                else:
                    canvas.remove_overlay(f"{prefix}_m_{i}")
                if labels and i < len(labels):
                    lbl = labels[i]
                    html = f"<div style='background-color:rgba(0,0,0,140); padding:3px;'><span style='color:white;'>{lbl}</span></div>"
                    if hasattr(canvas, "set_text_overlay_at"):
                        canvas.set_text_overlay_at(f"{prefix}_t_{i}", html, x0 + 6, y0 + 6, anchor=(0, 0))

                # Cross-section profiles for each tile (data + both fit models)
                _tile_profiles(i, im, g, m, x0, y0)

        img_corr = res.get("psf_long_corrected")
        img_unc = res.get("psf_long_uncorrected")
        img_corr_a = np.asarray(img_corr) if img_corr is not None else None
        img_unc_a = np.asarray(img_unc) if img_unc is not None else None

        # Off-axis arrays (optional)
        off_corr = res.get("psf_long_corrected_offaxis")
        off_unc = res.get("psf_long_uncorrected_offaxis")
        off_corr_a = np.asarray(off_corr) if off_corr is not None else None
        off_unc_a = np.asarray(off_unc) if off_unc is not None else None

        coord_mode_disp = self._offaxis_coord_mode()

        def _coord_lbl(dx_as: float, dy_as: float) -> str:
            if coord_mode_disp == "polar":
                r_as = float(np.hypot(dx_as, dy_as))
                th = float(np.degrees(np.arctan2(dy_as, dx_as)))
                if not np.isfinite(th):
                    th = 0.0
                th = th % 360.0
                return f"{th:.1f}° , {r_as:.1f} as"
            return f"{dx_as:+.2f},{dy_as:+.2f} as"

        def _fit_lbl(m: dict, corrected: bool) -> str:
            parts = []
            if show_g:
                k = "fwhm_arcsec_corrected" if corrected else "fwhm_arcsec_uncorrected"
                try:
                    v = float(m.get(k, float("nan")))
                except Exception:
                    v = float("nan")
                if np.isfinite(v):
                    parts.append(f"G {v:.3f}")
            if show_m:
                k = ("fwhm_arcsec_corrected_moffat_wings" if corrected else "fwhm_arcsec_uncorrected_moffat_wings") if use_wings else ("fwhm_arcsec_corrected_moffat" if corrected else "fwhm_arcsec_uncorrected_moffat")
                try:
                    v = float(m.get(k, float("nan")))
                except Exception:
                    v = float("nan")
                if np.isfinite(v):
                    mp = "Mw" if use_wings else "M"
                    parts.append(f"{mp} {v:.3f}")
            return "  ".join(parts) if parts else ""


        if img_corr_a is not None and img_corr_a.ndim == 2:
            if off_corr_a is not None and off_corr_a.ndim == 3 and (len(off_metrics) > 0):
                # Mosaic: on-axis + off-axis
                imgs = [img_corr_a] + [off_corr_a[i] for i in range(off_corr_a.shape[0])]
                gfits = [gauss_corr if show_g else {}] + [((m.get("gauss_corrected") or {}) if show_g else {}) for m in off_metrics]
                m_key = "moffat_wings_corrected" if use_wings else "moffat_corrected"
                mfits = [moff_corr_show if show_m else {}] + [((m.get(m_key) or {}) if show_m else {}) for m in off_metrics]

                on_m = {
                    "fwhm_arcsec_corrected": fwhm_c,
                    "fwhm_arcsec_corrected_moffat": fwhm_cmof,
                    "fwhm_arcsec_corrected_moffat_wings": fwhm_cmof_w,
                }
                on_lbl = "on-axis"
                extra = _fit_lbl(on_m, corrected=True)
                if extra:
                    on_lbl = on_lbl + "\n" + extra

                labels = [on_lbl] + [f"{_coord_lbl(float(m.get('dx_arcsec',0.0)), float(m.get('dy_arcsec',0.0)))}\n{_fit_lbl(m, corrected=True)}" for m in off_metrics]
                _show_mosaic(self._psf_corr, imgs, gfits, mfits, labels, prefix="corr")
            else:
                _show(self._psf_corr, img_corr_a)
                _apply_overlays(self._psf_corr["canvas"], img_corr_a, (gauss_corr if show_g else {}), (moff_corr_show if show_m else {}))
                self._update_profiles(self._psf_corr, img_corr_a, (gauss_corr if show_g else {}), (moff_corr_show if show_m else {}), plate)

        if img_unc_a is not None and img_unc_a.ndim == 2:
            if off_unc_a is not None and off_unc_a.ndim == 3 and (len(off_metrics) > 0):
                imgs = [img_unc_a] + [off_unc_a[i] for i in range(off_unc_a.shape[0])]
                gfits = [gauss_unc if show_g else {}] + [((m.get("gauss_uncorrected") or {}) if show_g else {}) for m in off_metrics]
                m_key = "moffat_wings_uncorrected" if use_wings else "moffat_uncorrected"
                mfits = [moff_unc_show if show_m else {}] + [((m.get(m_key) or {}) if show_m else {}) for m in off_metrics]

                on_m = {
                    "fwhm_arcsec_uncorrected": fwhm_u,
                    "fwhm_arcsec_uncorrected_moffat": fwhm_umof,
                    "fwhm_arcsec_uncorrected_moffat_wings": fwhm_umof_w,
                }
                on_lbl = "on-axis"
                extra = _fit_lbl(on_m, corrected=False)
                if extra:
                    on_lbl = on_lbl + "\n" + extra

                labels = [on_lbl] + [f"{_coord_lbl(float(m.get('dx_arcsec',0.0)), float(m.get('dy_arcsec',0.0)))}\n{_fit_lbl(m, corrected=False)}" for m in off_metrics]
                _show_mosaic(self._psf_unc, imgs, gfits, mfits, labels, prefix="unc")
            else:
                _show(self._psf_unc, img_unc_a)
                _apply_overlays(self._psf_unc["canvas"], img_unc_a, (gauss_unc if show_g else {}), (moff_unc_show if show_m else {}))
                self._update_profiles(self._psf_unc, img_unc_a, (gauss_unc if show_g else {}), (moff_unc_show if show_m else {}), plate)

    @Slot(str)
    def _on_long_run_failed(self, err: str):
        self.lbl_progress.setText("Failed")
        self.lbl_results.setStyleSheet("font-family: Consolas, monospace;")
        self.lbl_results.setText(str(err))

        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setValue(0)

        self.btn_open_dir.setEnabled(bool(self._last_out_dir))
