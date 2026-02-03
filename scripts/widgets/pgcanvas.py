from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import cupy as cp
import pyqtgraph as pg
from PySide6.QtCore import Signal, Slot, QTimer, QThread, Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget, QSizePolicy
from PySide6 import QtGui

import matplotlib.cm as mpl_cm

pg.setConfigOptions(imageAxisOrder="row-major")


def _to_numpy(a):
    """CuPy -> NumPy, otherwise np.asarray."""
    if isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def _mpl_lut(cmap: str, n: int = 256) -> np.ndarray:
    """
    Return (n,3) uint8 LUT suitable for pyqtgraph ImageItem.setLookupTable.
    Matplotlib-like colormap appearance.
    """
    cm = mpl_cm.get_cmap(cmap, n)
    rgba = (cm(np.linspace(0, 1, n)) * 255).astype(np.ubyte)  # (n,4)
    return rgba[:, :3].copy()  # (n,3) RGB


def _as_display_array(img):
    img = _to_numpy(img)
    # Keep uint8 as-is (perfect for LUT 0..255)
    if img.dtype == np.uint8:
        return img
    return img.astype(np.float32, copy=False)


def _pg_colormap_from_lut(lut_rgb_u8: np.ndarray):
    """Create a pyqtgraph ColorMap from an (N,3) uint8 LUT."""
    try:
        ColorMap = pg.ColorMap
    except AttributeError:
        # older pyqtgraph
        from pyqtgraph.colormap import ColorMap  # type: ignore

    pos = np.linspace(0.0, 1.0, lut_rgb_u8.shape[0], dtype=np.float32)
    return ColorMap(pos, lut_rgb_u8.astype(np.ubyte, copy=False))


# Prefer ColorBarItem if available; otherwise fall back to HistogramLUTItem
try:
    from pyqtgraph.graphicsItems.ColorBarItem import ColorBarItem  # type: ignore
except Exception:
    ColorBarItem = None  # type: ignore

try:
    from pyqtgraph.graphicsItems.HistogramLUTItem import HistogramLUTItem  # type: ignore
except Exception:
    HistogramLUTItem = None  # type: ignore


@dataclass
class OverlaySpec:
    item: Any
    kind: str
    z: int

class LineGraph(QWidget):
    def __init__(self, parent=None, title="", xlabel="x", ylabel="y", xunit="px", yunit="a.u."):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        self.plot = pg.PlotWidget(title=title)
        lay.addWidget(self.plot)

        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("bottom", xlabel, units=xunit)   # change as needed
        self.plot.setLabel("left", ylabel, units=yunit)   # change as needed

        self.curve = self.plot.plot()  # returns PlotDataItem
        self.xdata = []
        self.ydata = []

    def set_line(self, x, y):
        self.curve.setData(x, y)

    def reset_data(self):
        self.xdata = []
        self.ydata = []

    def append_data(self, x, y):
        self.xdata.append(x)
        self.ydata.append(y)

        self.curve.setData(self.xdata, self.ydata)

class TwoLineGraph(QWidget):
    def __init__(self, parent=None, title="", xlabel="x", ylabel="y", y1label="y1", y2label="y2", xunit="px", yunit="a.u."):
        super().__init__(parent)
        lay = QVBoxLayout(self)

        self.plot = pg.PlotWidget(title=title)
        lay.addWidget(self.plot)

        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.addLegend()
        self.plot.setLabel("bottom", xlabel, units=xunit) 
        self.plot.setLabel("left", ylabel, units=yunit)

        # Two curves
        self.curve1 = self.plot.plot(name=y1label)
        self.curve2 = self.plot.plot(name=y2label)

        # Optional: thicker lines
        self.curve1.setPen(pg.mkPen((200, 200, 200), width=2))  # grey
        self.curve2.setPen(pg.mkPen((0, 255, 0), width=2))

        self.xdata = []
        self.y1data = []
        self.y2data = []

    def set_labels(self, x_label="x", x_units="", y_label="y", y_units=""):
        self.plot.setLabel("bottom", x_label, units=x_units)
        self.plot.setLabel("left", y_label, units=y_units)

    def set_data(self, x, y1, y2):
        x = np.asarray(x)
        self.curve1.setData(x, np.asarray(y1))
        self.curve2.setData(x, np.asarray(y2))

    def reset_data(self):
        self.xdata = []
        self.y1data = []
        self.y2data = []

    def append_data(self, x, y1, y2):
        self.xdata.append(x)
        self.y1data.append(y1)
        self.y2data.append(y2)

        self.curve1.setData(self.xdata, self.y1data)
        self.curve2.setData(self.xdata, self.y2data)

class PGCanvas(QWidget):
    """
    PyQtGraph canvas with:
      - base scalar image mapped through a matplotlib colormap (LUT)
      - optional pupil RGBA overlay
      - named overlay registry (image/scatter/rgba/quiver)
      - OPTIONAL colorbar (with label + units) synced to base image levels
    """
    grab = Signal(object)  # emits QPixmap from view.grab()
    _sig_set_image = Signal(object)
    _sig_set_image_robust = Signal(object)
    _sig_queue_image = Signal(object)
    _sig_queue_image_robust = Signal(object)


    def __init__(
        self,
        parent=None,
        emit_grab_on_update: bool = False,
        max_fps: int = 120,
        show_colorbar: bool = True,
        colorbar_label: str = "Value",
        colorbar_units: str = "",
        colorbar_width: int = 22,
    ):
        super().__init__(parent)
        self._emit_grab = bool(emit_grab_on_update)

        self.layout = QVBoxLayout(self)
        self.view = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.view)

        # Put image ViewBox in (0,0). Colorbar will live in (0,1) if enabled.
        self.vb = self.view.addViewBox(row=0, col=0)
        self.vb.setAspectLocked(True)
        self.vb.invertY(True)

        sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)

        # Base image (scalar)
        self.image_item = pg.ImageItem(border="w")
        self.vb.addItem(self.image_item)

        # Pupil overlay (RGBA)
        self.pupil_item = pg.ImageItem()
        self.pupil_item.setZValue(10)
        self.vb.addItem(self.pupil_item)

        # Cache: cmap -> LUT
        self._lut_cache: Dict[str, np.ndarray] = {}
        self._current_cmap = "viridis"

        # ---- thread-safe UI update signals (workers -> GUI thread) ----
        
        self._sig_set_image.connect(self._on_set_image, Qt.QueuedConnection)
        self._sig_set_image_robust.connect(self._on_set_image_robust, Qt.QueuedConnection)
        self._sig_queue_image.connect(self._on_queue_image, Qt.QueuedConnection)
        self._sig_queue_image_robust.connect(self._on_queue_image_robust, Qt.QueuedConnection)

        # ---- autorange on first frame / shape change / show ----
        self._last_image_shape = None
        self._autorange_on_show = False


        # --- Colorbar state ---
        self._show_colorbar = bool(show_colorbar)
        self._cbar_label = str(colorbar_label)
        self._cbar_units = str(colorbar_units)
        self._cbar_width = int(colorbar_width)
        self._cbar_item = None  # ColorBarItem or HistogramLUTItem
        self._cbar_label_item = None  # optional pg.LabelItem

        # colorbar sizing policy (fraction of widget width, clamped)
        self._cbar_frac = 0.04   # 8% of canvas width
        self._cbar_min_px = 18
        self._cbar_max_px = 55

        self._last_shape = None
        self._autorange_pending = False

        # Init colormap + (optionally) colorbar
        self._set_cmap(self._current_cmap)
        if self._show_colorbar:
            self._ensure_colorbar()
            self._sync_colorbar_cmap()
            self._sync_colorbar_label()
            

        # Named overlays
        self._overlays: Dict[str, OverlaySpec] = {}
        self._quiver_items: Dict[str, List[pg.ArrowItem]] = {}

        # --- queued rendering state ---
        self._pending = None
        self._pending_kwargs = None   # (mode, cmap, levels, auto_levels, p_lo, p_hi)
        self._max_fps = max(1, int(max_fps))

        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._render_latest)
        self._render_timer.start(int(1000 / self._max_fps))

    def _request_post_layout_fix(self, shape):
        """Call after setting an image. Ensures range + colorbar sizing once layout is real."""
        shape = tuple(shape) if shape is not None else None
        if shape is not None and shape != self._last_shape:
            self._last_shape = shape
            self._autorange_pending = True  # new data size → re-fit once

        # If not visible / not sized yet, defer to showEvent
        if (not self.isVisible()) or self.width() < 10 or self.height() < 10:
            self._autorange_pending = True
            return

        # Run after Qt finishes layout this event-loop cycle
        QTimer.singleShot(0, self._apply_post_layout_fix)

    def _apply_post_layout_fix(self):
        # 1) Fit view to image bounds once we have a real size
        if self._autorange_pending:
            br = self.image_item.boundingRect()
            if br.width() > 0 and br.height() > 0:
                self.vb.setRange(rect=br, padding=0.0)
            self._autorange_pending = False

        # 2) Re-apply colorbar geometry (your dynamic width code)
        if hasattr(self, "_update_colorbar_geometry"):
            self._update_colorbar_geometry()

    def showEvent(self, ev):
        super().showEvent(ev)
        # If we received frames while hidden, apply fix once shown
        if self._autorange_pending:
            QTimer.singleShot(0, self._apply_post_layout_fix)

    # -------------------------
    # Colorbar helpers
    # -------------------------
    def set_colorbar(self, label: str = "Value", units: str = "", visible: bool = True):
        """Set colorbar label/units and show/hide it."""
        self._cbar_label = str(label)
        self._cbar_units = str(units)
        self._show_colorbar = bool(visible)

        if self._show_colorbar:
            self._ensure_colorbar()
            self._sync_colorbar_cmap()
            self._sync_colorbar_label()
            self._sync_colorbar_levels_from_image()
        else:
            self._remove_colorbar()

    def _remove_colorbar(self):
        if self._cbar_item is not None:
            try:
                self.view.removeItem(self._cbar_item)
            except Exception:
                pass
            self._cbar_item = None

        if self._cbar_label_item is not None:
            try:
                self.view.removeItem(self._cbar_label_item)
            except Exception:
                pass
            self._cbar_label_item = None

    def _ensure_colorbar(self):
        if self._cbar_item is not None:
            return

        # Option A: ColorBarItem (preferred)
        if ColorBarItem is not None:
            try:
                cm = _pg_colormap_from_lut(self._lut_cache[self._current_cmap])
                self._cbar_item = ColorBarItem(
                    colorMap=cm,
                    values=(0.0, 1.0),
                    width=self._cbar_width,
                    interactive=False,
                )
                # attach to base image
                if hasattr(self._cbar_item, "setImageItem"):
                    self._cbar_item.setImageItem(self.image_item)
                self.view.addItem(self._cbar_item, row=0, col=1)
                return
            except Exception:
                self._cbar_item = None  # fall through

        # Option B: HistogramLUTItem fallback
        if HistogramLUTItem is not None:
            self._cbar_item = HistogramLUTItem()
            if hasattr(self._cbar_item, "setImageItem"):
                self._cbar_item.setImageItem(self.image_item)
            self.view.addItem(self._cbar_item, row=0, col=1)
            return

        # If neither exists, silently do nothing
        self._cbar_item = None
        self._update_colorbar_geometry()
        QTimer.singleShot(0, self._apply_post_layout_fix)


    def _sync_colorbar_cmap(self):
        if self._cbar_item is None:
            return
        lut = self._lut_cache[self._current_cmap]
        cm = _pg_colormap_from_lut(lut)

        # ColorBarItem typically has setColorMap; HistogramLUTItem uses .gradient.setColorMap / restoreState
        if hasattr(self._cbar_item, "setColorMap"):
            try:
                self._cbar_item.setColorMap(cm)
                return
            except Exception:
                pass
        if hasattr(self._cbar_item, "gradient") and hasattr(self._cbar_item.gradient, "setColorMap"):
            try:
                self._cbar_item.gradient.setColorMap(cm)
                return
            except Exception:
                pass

    def _sync_colorbar_label(self):
        if self._cbar_item is None:
            return
        text = self._cbar_label
        units = self._cbar_units

        # Try common APIs
        if hasattr(self._cbar_item, "setLabel"):
            try:
                # many versions accept (text=..., units=...)
                self._cbar_item.setLabel(text=text, units=units)
                return
            except Exception:
                pass
        if hasattr(self._cbar_item, "axis") and hasattr(self._cbar_item.axis, "setLabel"):
            try:
                self._cbar_item.axis.setLabel(text=text, units=units)
                return
            except Exception:
                pass

        # Last resort: add a LabelItem below the bar (works for both)
        if self._cbar_label_item is None:
            self._cbar_label_item = pg.LabelItem(justify="left")
            self.view.addItem(self._cbar_label_item, row=1, col=1)
        label_str = f"{text} [{units}]" if units else text
        self._cbar_label_item.setText(label_str)

    def _sync_colorbar_levels(self, vmin: float, vmax: float):
        if self._cbar_item is None:
            return
        vmin = float(vmin)
        vmax = float(vmax)
        if vmax <= vmin:
            vmax = vmin + 1e-12

        # ColorBarItem often uses setLevels or setValues
        if hasattr(self._cbar_item, "setLevels"):
            try:
                self._cbar_item.setLevels((vmin, vmax))
                return
            except Exception:
                pass
        if hasattr(self._cbar_item, "setValues"):
            try:
                self._cbar_item.setValues((vmin, vmax))
                return
            except Exception:
                pass

        # HistogramLUTItem usually tracks the image item levels itself; nothing needed.

    def _sync_colorbar_levels_from_image(self):
        """Best-effort pull current levels from image_item and push into colorbar."""
        if self._cbar_item is None:
            return

        lv = None
        if hasattr(self.image_item, "getLevels"):
            try:
                lv = self.image_item.getLevels()
            except Exception:
                lv = None
        if lv is None:
            lv = getattr(self.image_item, "levels", None)

        if lv is None:
            return

        try:
            vmin, vmax = float(lv[0]), float(lv[1])
        except Exception:
            return

        self._sync_colorbar_levels(vmin, vmax)

    def set_colorbar_sizing(self, frac: float = 0.08, min_px: int = 18, max_px: int = 55):
        self._cbar_frac = float(frac)
        self._cbar_min_px = int(min_px)
        self._cbar_max_px = int(max_px)
        self._update_colorbar_geometry()

    def _update_colorbar_geometry(self):
        """Keep colorbar from stealing space on small canvases."""
        if not self._show_colorbar or self._cbar_item is None:
            return

        total_w = max(1, self.view.width())
        bar_w = int(np.clip(total_w * self._cbar_frac, self._cbar_min_px, self._cbar_max_px))

        # Column 0 = image, Column 1 = colorbar
        try:
            gl = self.view.ci.layout  # QGraphicsGridLayout
            gl.setColumnFixedWidth(1, bar_w)
            gl.setColumnStretchFactor(0, 10)  # image gets all remaining space
            gl.setColumnStretchFactor(1, 0)
        except Exception:
            pass


    # -------------------------
    # Base image (scalar + LUT)
    # -------------------------
    def _set_image_impl(self, img, cmap="viridis", levels=None, auto_levels=True):
        self._set_cmap(cmap)

        if auto_levels:
            self.image_item.setImage(img, autoLevels=True)
            if self._show_colorbar:
                self._sync_colorbar_levels_from_image()
        else:
            if levels is None:
                finite = np.isfinite(img)
                if np.any(finite):
                    vmin = float(np.min(img[finite]))
                    vmax = float(np.max(img[finite]))
                else:
                    vmin, vmax = 0.0, 1.0
                if vmax <= vmin:
                    vmax = vmin + 1.0
                levels = (vmin, vmax)

            self.image_item.setImage(img, autoLevels=False, levels=levels)
            self.image_item.setLevels(levels)

            if self._show_colorbar:
                self._sync_colorbar_levels(float(levels[0]), float(levels[1]))

        self._maybe_autorange(img.shape)

        if self._emit_grab:
            self.grab.emit(self.view.grab())

    def _set_image_robust_impl(self, img, cmap="viridis", p_lo=0.5, p_hi=99.5):
        self._set_cmap(cmap)

        finite = np.isfinite(img)
        if np.any(finite):
            vmin, vmax = np.percentile(img[finite], [p_lo, p_hi])
            if vmax <= vmin:
                vmin, vmax = float(np.min(img[finite])), float(np.max(img[finite]))
        else:
            vmin, vmax = 0.0, 1.0

        self.image_item.setImage(img, autoLevels=False)
        self.image_item.setLevels((float(vmin), float(vmax)))

        if self._show_colorbar:
            self._sync_colorbar_levels(float(vmin), float(vmax))

        self._maybe_autorange(img.shape)

        if self._emit_grab:
            self.grab.emit(self.view.grab())

    def _set_cmap(self, cmap: str):
        if cmap not in self._lut_cache:
            self._lut_cache[cmap] = _mpl_lut(cmap, 256)
        self.image_item.setLookupTable(self._lut_cache[cmap])
        self._current_cmap = cmap
        if self._show_colorbar:
            self._ensure_colorbar()
            self._sync_colorbar_cmap()

    def set_image(self, img, cmap="viridis", levels=None, auto_levels=True):
        img = _as_display_array(img)

        if QThread.currentThread() != self.thread():
            self._sig_set_image.emit((img, cmap, levels, bool(auto_levels)))
            return

        self._set_image_impl(img, cmap=cmap, levels=levels, auto_levels=bool(auto_levels))
        self._request_post_layout_fix(img.shape)


    def set_image_robust(self, img, cmap="viridis", p_lo=0.5, p_hi=99.5):
        img = _as_display_array(img)

        if QThread.currentThread() != self.thread():
            self._sig_set_image_robust.emit((img, cmap, float(p_lo), float(p_hi)))
            return

        self._set_image_robust_impl(img, cmap=cmap, p_lo=float(p_lo), p_hi=float(p_hi))
        self._request_post_layout_fix(img.shape)



    # -------------------------
    # Pupil overlay
    # -------------------------
    def set_pupil_mask(self, pupil, color=(255, 255, 255), alpha_inside=255):
        """
        pupil: 2D binary array (1 inside pupil).
        Draws RGBA overlay: color inside pupil, transparent outside.
        """
        pupil = _to_numpy(pupil)
        pupil = (pupil > 0).astype(bool)

        H, W = pupil.shape
        rgba = np.zeros((H, W, 4), dtype=np.ubyte)
        rgba[pupil, 0] = color[0]
        rgba[pupil, 1] = color[1]
        rgba[pupil, 2] = color[2]
        rgba[pupil, 3] = np.ubyte(alpha_inside)

        self.pupil_item.setImage(rgba, autoLevels=False)

        if self._emit_grab:
            self.grab.emit(self.view.grab())

    def clear_pupil_mask(self):
        self.pupil_item.clear()
        if self._emit_grab:
            self.grab.emit(self.view.grab())

    def set_image_masked(
        self,
        img,
        mask=None,
        cmap: str = "viridis",
        levels: Optional[Tuple[float, float]] = None,
        auto_levels: bool = True,
        masked_value: float = np.nan,
    ):
        """
        Backwards-compatible masked image display.
        """
        img = _as_display_array(img)

        if mask is not None:
            m = _to_numpy(mask).astype(bool)
            if m.shape != img.shape:
                raise ValueError(f"mask shape {m.shape} must match img shape {img.shape}")
            out = img.copy()
            out[~m] = masked_value
            img = out

        if levels is None and not auto_levels:
            finite = np.isfinite(img)
            if np.any(finite):
                vmin = float(np.min(img[finite]))
                vmax = float(np.max(img[finite]))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                levels = (vmin, vmax)
            else:
                levels = (0.0, 1.0)

        self.set_image(img, cmap=cmap, levels=levels, auto_levels=auto_levels)

    # -------------------------
    # Overlay registry helpers
    # -------------------------
    def remove_overlay(self, name: str):
        spec = self._overlays.pop(name, None)
        if spec is None:
            return
        self.vb.removeItem(spec.item)
        if name in self._quiver_items:
            for a in self._quiver_items.pop(name):
                self.vb.removeItem(a)

        if self._emit_grab:
            self.grab.emit(self.view.grab())

    def clear_overlays(self):
        for name in list(self._overlays.keys()):
            self.remove_overlay(name)

    # -------------------------
    # Scalar image overlay (with cmap + opacity)
    # -------------------------
    def set_image_overlay(
        self,
        name: str,
        img,
        cmap: str = "magma",
        levels: Optional[Tuple[float, float]] = None,
        opacity: float = 0.6,
        z: int = 15,
        auto_levels: bool = True,
    ):
        img = _as_display_array(img)

        if name not in self._overlays:
            item = pg.ImageItem()
            item.setZValue(z)
            item.setOpacity(float(opacity))
            self.vb.addItem(item)
            self._overlays[name] = OverlaySpec(item=item, kind="image", z=z)
        else:
            item = self._overlays[name].item
            item.setZValue(z)
            item.setOpacity(float(opacity))

        if cmap not in self._lut_cache:
            self._lut_cache[cmap] = _mpl_lut(cmap, 256)
        item.setLookupTable(self._lut_cache[cmap])

        if levels is not None:
            item.setImage(img, autoLevels=False)
            item.setLevels((float(levels[0]), float(levels[1])))
        else:
            item.setImage(img, autoLevels=bool(auto_levels))

        if self._emit_grab:
            self.grab.emit(self.view.grab())

    # -------------------------
    # RGBA overlay (mask-like)
    # -------------------------
    def set_rgba_overlay(self, name: str, rgba, opacity: float = 1.0, z: int = 20):
        rgba = _to_numpy(rgba)
        if rgba.dtype != np.ubyte:
            rgba = rgba.astype(np.ubyte, copy=False)

        if name not in self._overlays:
            item = pg.ImageItem()
            item.setZValue(z)
            item.setOpacity(float(opacity))
            self.vb.addItem(item)
            self._overlays[name] = OverlaySpec(item=item, kind="rgba", z=z)
        else:
            item = self._overlays[name].item
            item.setZValue(z)
            item.setOpacity(float(opacity))

        item.setImage(rgba, autoLevels=False)

        if self._emit_grab:
            self.grab.emit(self.view.grab())

    # -------------------------
    # Scatter overlay
    # -------------------------
    def set_scatter_overlay(
        self,
        name: str,
        x,
        y,
        size: float = 6,
        pen=None,
        brush=None,
        z: int = 30,
    ):
        x = _to_numpy(x).ravel()
        y = _to_numpy(y).ravel()

        if name not in self._overlays:
            item = pg.ScatterPlotItem(size=size, pen=pen, brush=brush)
            item.setZValue(z)
            self.vb.addItem(item)
            self._overlays[name] = OverlaySpec(item=item, kind="scatter", z=z)
        else:
            item = self._overlays[name].item
            item.setZValue(z)

        item.setData(x=x, y=y)

        if self._emit_grab:
            self.grab.emit(self.view.grab())

    # -------------------------
    # Quiver overlay (arrows)
    # -------------------------
    def set_quiver_overlay(
        self,
        name: str,
        centroids,
        moved_centroids,
        threshold: float = 1e-3,
        color=(0, 255, 0),
        min_len: float = 1.0,
        z: int = 40,
    ):
        c = _to_numpy(centroids)
        m = _to_numpy(moved_centroids)
        c = np.asarray(c, dtype=np.float32)
        m = np.asarray(m, dtype=np.float32)

        dy = m[:, 0] - c[:, 0]
        dx = m[:, 1] - c[:, 1]

        moved_mask = (np.hypot(dx, dy) > threshold)
        stationary_mask = ~moved_mask

        self.set_scatter_overlay(
            name=name + "_stationary",
            x=c[stationary_mask, 1],
            y=c[stationary_mask, 0],
            size=6,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(255, 0, 0),
            z=z,
        )

        for a in self._quiver_items.get(name, []):
            self.vb.removeItem(a)
        self._quiver_items[name] = []

        pen = pg.mkPen(color)
        brush = pg.mkBrush(*color)

        for x0, y0, ddx, ddy in zip(
            c[moved_mask, 1], c[moved_mask, 0],
            dx[moved_mask], dy[moved_mask]
        ):
            angle = np.degrees(np.arctan2(ddy, ddx))
            length = max(float(np.hypot(ddx, ddy)), float(min_len))
            arrow = pg.ArrowItem(
                pos=(float(x0), float(y0)),
                angle=float(angle),
                tipAngle=30,
                headLen=6,
                tailLen=0,
                brush=brush,
                pen=pen,
            )
            arrow.setScale(length)
            arrow.setZValue(z + 1)
            self.vb.addItem(arrow)
            self._quiver_items[name].append(arrow)

        if self._emit_grab:
            self.grab.emit(self.view.grab())

    def set_points_and_quivers(
        self,
        centroids,
        moved_centroids,
        threshold: float = 1e-3,
        color=(0, 255, 0),
        min_len: float = 1.0,
    ):
        self.set_quiver_overlay(
            name="centroids",
            centroids=centroids,
            moved_centroids=moved_centroids,
            threshold=threshold,
            color=color,
            min_len=min_len,
            z=40,
        )

    # -------------------------
    # Grab control
    # -------------------------
    def emit_grab_now(self):
        self.grab.emit(self.view.grab())

    # -------------------------
    # queued rendering
    # -------------------------
    def _render_latest(self):
        if self._pending is None or self._pending_kwargs is None:
            return

        img = self._pending
        mode, cmap, levels, auto_levels, p_lo, p_hi = self._pending_kwargs

        self._pending = None
        self._pending_kwargs = None

        if mode == "robust":
            self.set_image_robust(img, cmap=cmap, p_lo=p_lo, p_hi=p_hi)
        else:
            self.set_image(img, cmap=cmap, levels=levels, auto_levels=auto_levels)


    def queue_image(self, img, cmap="viridis", levels=None, auto_levels=True):
        img = _as_display_array(img)
        payload = (img, cmap, levels, bool(auto_levels))

        if QThread.currentThread() != self.thread():
            self._sig_queue_image.emit(payload)
            return

        self._pending = img
        self._pending_kwargs = ("normal", cmap, levels, bool(auto_levels), None, None)

    def queue_image_robust(self, img, cmap="viridis", p_lo=0.5, p_hi=99.5):
        img = _as_display_array(img)
        payload = (img, cmap, float(p_lo), float(p_hi))

        if QThread.currentThread() != self.thread():
            self._sig_queue_image_robust.emit(payload)
            return

        self._pending = img
        self._pending_kwargs = ("robust", cmap, None, False, float(p_lo), float(p_hi))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._update_colorbar_geometry()

    def _maybe_autorange(self, shape):
        if shape is None:
            return
        if self._last_image_shape != tuple(shape):
            self._last_image_shape = tuple(shape)
            if self.isVisible():
                self.vb.autoRange(padding=0.0)
            else:
                self._autorange_on_show = True

    @Slot(object)
    def _on_set_image(self, payload):
        img, cmap, levels, auto_levels = payload
        self._set_image_impl(img, cmap=cmap, levels=levels, auto_levels=auto_levels)

    @Slot(object)
    def _on_set_image_robust(self, payload):
        img, cmap, p_lo, p_hi = payload
        self._set_image_robust_impl(img, cmap=cmap, p_lo=p_lo, p_hi=p_hi)

    @Slot(object)
    def _on_queue_image(self, payload):
        img, cmap, levels, auto_levels = payload
        self._pending = img
        self._pending_kwargs = ("normal", cmap, levels, bool(auto_levels), None, None)

    @Slot(object)
    def _on_queue_image_robust(self, payload):
        img, cmap, p_lo, p_hi = payload
        self._pending = img
        self._pending_kwargs = ("robust", cmap, None, False, float(p_lo), float(p_hi))

    def set_circle_overlay(
        self,
        name: str,
        x: float,
        y: float,
        r: float,
        pen=None,
        brush=None,
        z: int = 100,
    ):
        """
        Draw/update a circle centered at (x,y) with radius r in image/data coords.
        Coordinates match your ImageItem: x = column, y = row (since you invertY).
        """
        from pyqtgraph.Qt import QtWidgets

        if pen is None:
            pen = pg.mkPen(255, 255, 0, width=2)
        if brush is None:
            brush = pg.mkBrush(0, 0, 0, 0)

        if name not in self._overlays:
            item = QtWidgets.QGraphicsEllipseItem()
            item.setPen(pen)
            item.setBrush(brush)
            item.setZValue(z)
            self.vb.addItem(item)
            self._overlays[name] = OverlaySpec(item=item, kind="circle", z=z)
        else:
            item = self._overlays[name].item
            item.setZValue(z)
            item.setPen(pen)
            item.setBrush(brush)

        item.setRect(float(x - r), float(y - r), float(2*r), float(2*r))

        if self._emit_grab:
            self.grab.emit(self.view.grab())

    def set_ellipse_overlay(
        self,
        name: str,
        x: float,
        y: float,
        rx: float,
        ry: float,
        angle_deg: float = 0.0,
        pen=None,
        brush=None,
        z: int = 100,
    ):
        """
        Draw/update an ellipse centered at (x,y).
        rx, ry are radii in x/y (image/data coords). angle_deg rotates around center.
        x = column, y = row (works with invertY(True)).
        """
        from pyqtgraph.Qt import QtWidgets, QtCore
        import pyqtgraph as pg

        if pen is None:
            pen = pg.mkPen(255, 255, 0, width=2)
        if brush is None:
            brush = pg.mkBrush(0, 0, 0, 0)

        if name not in self._overlays:
            item = QtWidgets.QGraphicsEllipseItem()
            item.setZValue(z)
            self.vb.addItem(item)
            self._overlays[name] = OverlaySpec(item=item, kind="ellipse", z=z)
        else:
            item = self._overlays[name].item
            item.setZValue(z)

        item.setPen(pen)
        item.setBrush(brush)

        # base ellipse rect (unrotated)
        item.setRect(float(x - rx), float(y - ry), float(2 * rx), float(2 * ry))

        # rotate about the center (x,y)
        tr = QtGui.QTransform()  # <-- need QtGui import
        tr.translate(float(x), float(y))
        tr.rotate(float(angle_deg))
        tr.translate(float(-x), float(-y))
        item.setTransform(tr)

        if self._emit_grab:
            self.grab.emit(self.view.grab())



    def showEvent(self, ev):
        super().showEvent(ev)
        if self._autorange_on_show:
            self._autorange_on_show = False
            self.vb.autoRange(padding=0.0)

