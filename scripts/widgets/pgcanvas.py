from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import cupy as cp
import pyqtgraph as pg
from PySide6.QtCore import Signal, QTimer, QSize
from PySide6.QtWidgets import QVBoxLayout, QWidget, QSizePolicy

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

@dataclass
class OverlaySpec:
    item: Any
    kind: str
    z: int


class PGCanvas(QWidget):
    """
    PyQtGraph canvas with:
      - base scalar image mapped through a matplotlib colormap (LUT)
      - optional pupil RGBA overlay
      - named overlay registry (image/scatter/rgba/quiver)
    """
    grab = Signal(object)  # emits QPixmap from view.grab()

    def __init__(self, parent=None, emit_grab_on_update: bool = False, max_fps: int = 120):
        super().__init__(parent)
        self._emit_grab = bool(emit_grab_on_update)

        self.layout = QVBoxLayout(self)
        self.view = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.view)

        self.vb = self.view.addViewBox()
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
        self._set_cmap(self._current_cmap)

        # Named overlays
        self._overlays: Dict[str, OverlaySpec] = {}
        self._quiver_items: Dict[str, List[pg.ArrowItem]] = {}

        # --- queued rendering state ---
        self._pending = None          # latest frame (numpy)
        self._pending_kwargs = None   # (cmap, levels, auto_levels, robust, p_lo, p_hi)
        self._max_fps = max(1, int(max_fps))

        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._render_latest)
        self._render_timer.start(int(1000 / self._max_fps))

    # -------------------------
    # Base image (scalar + LUT)
    # -------------------------
    def _set_cmap(self, cmap: str):
        if cmap not in self._lut_cache:
            self._lut_cache[cmap] = _mpl_lut(cmap, 256)
        self.image_item.setLookupTable(self._lut_cache[cmap])
        self._current_cmap = cmap

    def set_image(
        self,
        img,
        cmap: str = "viridis",
        levels: Optional[Tuple[float, float]] = None,
        auto_levels: bool = True,
    ):
        """
        img: 2D scalar array (numpy or cupy).
        cmap: matplotlib colormap name.
        levels: (vmin, vmax). If None and auto_levels=True, use min/max.
        """
        img = _as_display_array(img)
        self._set_cmap(cmap)

        if auto_levels:
            # pyqtgraph will compute levels
            self.image_item.setImage(img, autoLevels=True)
        else:
            # Need explicit levels for float images in pyqtgraph 0.14+
            if levels is None:
                # try to reuse existing levels if present
                try:
                    levels = self.image_item.levels
                except Exception:
                    levels = None

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

        if self._emit_grab:
            self.grab.emit(self.view.grab())

    def set_image_robust(
        self,
        img,
        cmap: str = "viridis",
        p_lo: float = 0.5,
        p_hi: float = 99.5,
    ):
        """
        Matplotlib-ish 'robust' scaling (percentiles). Nice for turbulence screens.
        """
        img = _as_display_array(img)
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

        if self._emit_grab:
            self.grab.emit(self.view.grab())

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

        img: (H,W) scalar
        mask: (H,W) bool/0-1; True = keep, False = masked out

        Implementation: sets masked pixels to NaN (or masked_value) and displays with LUT.
        """
        img = _as_display_array(img)

        if mask is not None:
            m = _to_numpy(mask).astype(bool)
            if m.shape != img.shape:
                raise ValueError(f"mask shape {m.shape} must match img shape {img.shape}")
            # make a writable copy only if needed
            out = img.copy()
            out[~m] = masked_value
            img = out

        # If user didn't supply levels, compute them from unmasked pixels
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

        # Use normal set_image pipeline
        self.set_image(img, cmap=cmap, levels=levels, auto_levels=auto_levels)


    # -------------------------
    # Overlay registry helpers
    # -------------------------
    def remove_overlay(self, name: str):
        spec = self._overlays.pop(name, None)
        if spec is None:
            return
        self.vb.removeItem(spec.item)
        # if it was a quiver overlay, also remove arrows
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
        """
        Add/update a scalar image overlay. Great for showing e.g. another layer, residual, etc.
        """
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

        # colormap LUT
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
        """
        Add/update an RGBA overlay (H,W,4 uint8).
        Useful for pupil masks, flagged pixels, segmentation, etc.
        """
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
        centroids,         # (N,2) as (row=y, col=x) or (y,x)
        moved_centroids,   # (N,2)
        threshold: float = 1e-3,
        color=(0, 255, 0),
        min_len: float = 1.0,
        z: int = 40,
    ):
        """
        Draw stationary points as scatter + moved points as arrows.
        Uses ArrowItem per moved centroid (fine for ~1600 points; avoid huge counts).
        """
        c = _to_numpy(centroids)
        m = _to_numpy(moved_centroids)
        c = np.asarray(c, dtype=np.float32)
        m = np.asarray(m, dtype=np.float32)

        # Interpret inputs as (y,x) pairs (your original code did that)
        dy = m[:, 0] - c[:, 0]
        dx = m[:, 1] - c[:, 1]

        moved_mask = (np.hypot(dx, dy) > threshold)
        stationary_mask = ~moved_mask

        # stationary scatter
        self.set_scatter_overlay(
            name=name + "_stationary",
            x=c[stationary_mask, 1],
            y=c[stationary_mask, 0],
            size=6,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(255, 0, 0),
            z=z,
        )

        # remove old arrows
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
        """
        Backwards-compatible wrapper.

        Draw stationary centroids as scatter and moved ones as arrows.
        Keeps the same signature you used previously.
        """
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
        """Manually emit a grab without doing it on every update."""
        self.grab.emit(self.view.grab())


    # -------------------------
    # queued rendering
    # -------------------------
    def _render_latest(self):
        if self._pending is None or self._pending_kwargs is None:
            return

        img = self._pending
        mode, cmap, levels, auto_levels, p_lo, p_hi = self._pending_kwargs

        # clear pending *first* so if a new frame arrives during render we don't lose it
        self._pending = None
        self._pending_kwargs = None

        if mode == "robust":
            # robust scaling is expensive — only do this if you explicitly queued it
            self.set_image_robust(img, cmap=cmap, p_lo=p_lo, p_hi=p_hi)
        else:
            self.set_image(img, cmap=cmap, levels=levels, auto_levels=auto_levels)


    def queue_image(
        self,
        img,
        cmap: str = "viridis",
        levels: Optional[Tuple[float, float]] = None,
        auto_levels: bool = False,
    ):
        """
        Queue an image for rendering. Drops intermediate frames automatically.
        Use this for high-rate updates from threads/signals.
        """
        img = _as_display_array(img)
        self._pending = img
        self._pending_kwargs = ("normal", cmap, levels, bool(auto_levels), None, None)

    def queue_image_robust(
        self,
        img,
        cmap: str = "viridis",
        p_lo: float = 0.5,
        p_hi: float = 99.5,
    ):
        """
        Queue an image that will be rendered with robust scaling.
        Warning: robust scaling is CPU-heavy; don't call at high rate.
        """
        img = _as_display_array(img)
        self._pending = img
        self._pending_kwargs = ("robust", cmap, None, False, float(p_lo), float(p_hi))

