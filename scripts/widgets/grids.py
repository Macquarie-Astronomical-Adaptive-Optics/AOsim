from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QGridLayout, QLabel, QVBoxLayout, QFrame
import numpy as np
import cupy as cp

from scripts.widgets.pgcanvas import PGCanvas
from scripts.utilities import Pupil_tools

pupil_alpha=55
_pens_rgba = [
            (255, 0, 0, pupil_alpha),
            (0, 255, 0, pupil_alpha),
            (0, 128, 255, pupil_alpha),
            (255, 165, 0, pupil_alpha),
            (255, 0, 255, pupil_alpha),
            (0, 255, 255, pupil_alpha),
            (255, 255, 0, pupil_alpha),
        ]

def rgba_to_css(color):
    r, g, b, a = color
    return f"rgba({r}, {g}, {b}, {a/255:.3f})"

class SensorPSFGrid(QWidget):
    """
    Displays PSFs for all sensors in a grid arranged by their field angles.

    thetas_xy_rad: list/array shape (S,2) with theta_x, theta_y in radians.
    """
    def __init__(self, sensors, title="Sensor PSFs", parent=None):
        super().__init__(parent)
        self.sensors = sensors
        self.thetas = np.asarray([s.field_angle for s in self.sensors.values()])# (S,2)
        self.S = self.thetas.shape[0]

        outer = QVBoxLayout(self)
        outer.addWidget(QLabel(title))

        self.grid = QGridLayout()
        outer.addLayout(self.grid)

        # compute grid coordinates from thetas
        self.pos = self._theta_to_grid(self.thetas)  # list of (row,col)

        # create a canvas for each sensor
        self.canvases = []
        
        self._pens_rgba = [(r, g, b, 255) for (r, g, b, _a) in _pens_rgba]

        for s in range(self.S):

            frame = QFrame()
            frame.setFrameShape(QFrame.NoFrame)
            
            vlayout = QVBoxLayout(frame)
            vlayout.setContentsMargins(0, 0, 0, 5)
            vlayout.setSpacing(0)

            bg = frame.palette().color(frame.backgroundRole())
            frame.setStyleSheet(
                f"QFrame {{ background-color: {bg.name()}; }}"
            )
            layername = QLabel(list(self.sensors.keys())[s])
            layername.setAlignment(Qt.AlignmentFlag.AlignCenter)
            css_color = rgba_to_css(self._pens_rgba[s])
            layername.setStyleSheet(f"color: {css_color};")
            
            c = PGCanvas()
            self.canvases.append(c)
            r, col = self.pos[s]

            vlayout.addWidget(c)
            vlayout.addWidget(layername)

            
            self.grid.addWidget(frame, r, col)

    @staticmethod
    def _theta_to_grid(thetas):
        """
        Map theta values to integer grid indices.
        Uses the smallest nonzero spacing across x/y to define a grid step.
        """
        th_arcsec = thetas * (180.0 * 3600.0 / np.pi)

        # pick a step size that matches your constellation
        vals = np.unique(np.round(np.abs(th_arcsec).flatten(), 6))
        vals = vals[vals > 1e-6]
        step = float(np.min(vals)) if vals.size else 1.0

        ix = np.rint(th_arcsec[:, 0] / step).astype(int)
        iy = np.rint(th_arcsec[:, 1] / step).astype(int)

        # shift to positive grid coords and flip y so +theta_y appears "up"
        ix0, iy0 = ix - ix.min(), iy.max() - iy
        return list(zip(iy0.tolist(), ix0.tolist()))

    def update_psfs(self, psf_stack_or_dict, cmap="viridis"):
        """
        Accepts:
          - dict {sensor_idx: image}
          - stack shape (S,H,W)
        """
        if isinstance(psf_stack_or_dict, dict):
            items = psf_stack_or_dict.items()
        else:
            arr = psf_stack_or_dict
            if isinstance(arr, cp.ndarray):
                arr = arr.get()
            arr = np.asarray(arr)
            items = enumerate(arr)

        for s, img in items:
            if 0 <= s < self.S:
                self.canvases[s].queue_image(img, cmap=cmap, auto_levels=True)



import numpy as np
import cupy as cp
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QGridLayout, QLabel, QVBoxLayout
import matplotlib.cm as mpl_cm

def _mpl_lut(cmap: str, n: int = 256) -> np.ndarray:
    cm = mpl_cm.get_cmap(cmap, n)
    rgba = (cm(np.linspace(0, 1, n)) * 255).astype(np.ubyte)
    return rgba[:, :3].copy()

def _to_numpy(a):
    if isinstance(a, cp.ndarray):
        return a.get()
    return np.asarray(a)

class LayerFootprintGrid(QWidget):
    def __init__(
        self,
        sensors,
        layers_cfg,
        N,
        dx,
        patch_center,
        patch_size_px,
        parent=None,
        title="Layers + Sensor Footprints",
        overlay_opacity=0.55,
        overlay_cmap="viridis",
        overlay_z=50,
        pupil_center_obscuration=None,
    ):
        super().__init__(parent)

        self.layers_cfg = list(layers_cfg)  # may include active/unloaded flags
        self.sensors = sensors
        self.thetas = np.asarray([s.field_angle for s in self.sensors.values()])  # (S,2)
        self.N = int(N)
        self.dx = float(dx)

        self.cx0, self.cy0 = float(patch_center[0]), float(patch_center[1])
        self.patch_size0 = float(patch_size_px)

        self.overlay_opacity = float(overlay_opacity)
        self.overlay_cmap = overlay_cmap
        self.overlay_z = int(overlay_z)
        self.pupil_alpha = int(pupil_alpha)

        # --- pupil param state / dirty flag ---
        self._pupil_center_obscuration = pupil_center_obscuration  # None -> utilities default
        self._pupil_dirty = True
        self._pupil_key = None  # (M, obsc, alpha, S)

        self._lut_cache = {}

        outer = QVBoxLayout(self)
        outer.addWidget(QLabel(title))
        self.grid = QGridLayout()
        outer.addLayout(self.grid)

        # on-axis sensor = closest theta to (0,0)
        self.center_idx = int(np.argmin(np.sum(self.thetas**2, axis=1)))

        # --- dynamic layout state ---
        self._active_layer_idxs = []
        self._layer_to_canvas = {}
        self.layer_canvases = []

        # overlays keyed by layer_idx
        self._overlay_items = {}     # layer_idx -> [ImageItem for each sensor]
        self._pupil_items = {}       # layer_idx -> [ImageItem for each sensor]

        self._pupil_rgba_by_sensor = None
        self._pupil_rgba_M = None

        # NEW: remember last patch size we drew, so "show" can refresh stamps without update_layers()
        self._last_patch_M = None

        self._rebuild_grid()

    def set_pupil_params(self, *, center_obscuration=None, pupil_alpha=None):

        if center_obscuration is not None:
            self._pupil_center_obscuration = float(center_obscuration)
        if pupil_alpha is not None:
            self.pupil_alpha = int(pupil_alpha)
            # also update palette alphas
            _pens_rgba = [(r, g, b, self.pupil_alpha) for (r, g, b, _a) in _pens_rgba]

        self._invalidate_pupil_stamps()
        self._refresh_pupil_overlays()

    def _invalidate_pupil_stamps(self):
        self._pupil_rgba_by_sensor = None
        self._pupil_rgba_M = None
        self._pupil_key = None
        self._pupil_dirty = True

    # -------------------------
    # refresh when shown 
    # -------------------------
    def _layer_is_active(self, layer: dict) -> bool:
        # treat unloaded as inactive for display
        if layer.get("unloaded", False):
            return False
        return bool(layer.get("active", True))

    def set_layers_cfg(self, layers_cfg):
        """Call this whenever active/unloaded flags change."""
        self.layers_cfg = list(layers_cfg)
        self._rebuild_grid()

    def _rebuild_grid(self):
        """Rebuild the grid layout to include only active layers."""
        # compute active layer indices
        self._active_layer_idxs = [i for i, L in enumerate(self.layers_cfg) if self._layer_is_active(L)]

        # clear layout widgets
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

        self.layer_canvases = []
        self._layer_to_canvas = {}
        self._overlay_items = {}
        self._pupil_items = {}

        L_active = len(self._active_layer_idxs)
        if L_active == 0:
            return

        ncol = int(np.ceil(np.sqrt(L_active)))
        lut = self._get_lut(self.overlay_cmap)
        S = self.thetas.shape[0]

        for ci, layer_idx in enumerate(self._active_layer_idxs):
            canvas = PGCanvas()
            self.layer_canvases.append(canvas)
            self._layer_to_canvas[layer_idx] = ci

            r, c = divmod(ci, ncol)

            frame = QFrame()
            frame.setFrameShape(QFrame.NoFrame)
            
            vlayout = QVBoxLayout(frame)
            vlayout.setContentsMargins(0, 0, 0, 5)
            vlayout.setSpacing(0)

            bg = frame.palette().color(frame.backgroundRole())
            frame.setStyleSheet(
                f"QFrame {{ background-color: {bg.name()}; }}"
            )

            layername = QLabel(self.layers_cfg[layer_idx]["name"])
            layername.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            vlayout.addWidget(canvas)
            vlayout.addWidget(layername)

            self.grid.addWidget(frame, r, c)

            # create overlay items for this layer
            patch_items = []
            pupil_items = []

            for s in range(S):
                it = pg.ImageItem()
                it.setLookupTable(lut)

                if s == self.center_idx:
                    it.setZValue(self.overlay_z + 999)
                    it.setOpacity(1.0)
                else:
                    it.setZValue(self.overlay_z + s)
                    it.setOpacity(self.overlay_opacity)

                canvas.vb.addItem(it)
                patch_items.append(it)

                pit = pg.ImageItem()
                pit.setZValue(4500 + s)
                pit.setOpacity(1.0)  # alpha is in RGBA
                canvas.vb.addItem(pit)
                pupil_items.append(pit)

            self._overlay_items[layer_idx] = patch_items
            self._pupil_items[layer_idx] = pupil_items

    # -------------------------
    # LUT + pupil stamps
    # -------------------------
    def _get_lut(self, cmap: str):
        if cmap not in self._lut_cache:
            self._lut_cache[cmap] = _mpl_lut(cmap, 256)
        return self._lut_cache[cmap]

    def _generate_pupil_mask(self, M: int) -> np.ndarray:
        """
        Create the pupil mask at resolution M using current pupil params.
        Uses Pupil_tools.generate_pupil(...) with kwargs when available.
        """
        try:
            if self._pupil_center_obscuration is None:
                pup = Pupil_tools.generate_pupil(grid_size=int(M))
            else:
                pup = Pupil_tools.generate_pupil(
                    grid_size=int(M),
                    telescope_center_obscuration=float(self._pupil_center_obscuration),
                )
        except TypeError:
            # fallback for older signature: generate_pupil(M) positional
            pup = Pupil_tools.generate_pupil(int(M))

        return _to_numpy(pup).astype(bool)
    
    import numpy as np

    def outline(self, rgba: np.ndarray, thickness: int = 1, inplace: bool = True) -> np.ndarray:
        if not inplace:
            rgba = rgba.copy()

        a = rgba[..., 3]
        mask = a > 0 

        if thickness <= 0 or not mask.any():
            return rgba

        def erode4(m: np.ndarray) -> np.ndarray:
            # 4-neighbour erosion
            out = np.zeros_like(m, dtype=bool)
            out[1:-1, 1:-1] = (
                m[1:-1, 1:-1] &
                m[:-2, 1:-1] &
                m[2:, 1:-1] &
                m[1:-1, :-2] &
                m[1:-1, 2:]
            )
            return out

        core = mask
        for _ in range(thickness):
            core = erode4(core)
            if not core.any():
                break

        edge_band = mask & ~core 
        a[edge_band] = np.min([self.pupil_alpha*2, 255])
        return rgba

    
    def _ensure_pupil_stamps(self, M: int, force: bool = False):
        M = int(M)
        S = int(self.thetas.shape[0])
        obsc = None if self._pupil_center_obscuration is None else float(self._pupil_center_obscuration)
        key = (M, obsc, int(self.pupil_alpha), S)

        if not force and not self._pupil_dirty and self._pupil_key == key and self._pupil_rgba_M == M:
            return

        pupil = self._generate_pupil_mask(M)  # (M,M) bool

        stamps = []
        for s in range(S):
            r, g, b, a = _pens_rgba[s % len(_pens_rgba)]
            rgba = np.zeros((M, M, 4), dtype=np.uint8)
            rgba[pupil, 0] = r
            rgba[pupil, 1] = g
            rgba[pupil, 2] = b
            rgba[pupil, 3] = a

            stamps.append(self.outline(rgba))

        self._pupil_rgba_by_sensor = stamps
        self._pupil_rgba_M = M
        self._pupil_key = key
        self._pupil_dirty = False

    def _refresh_pupil_overlays(self, force: bool = False):
        """
        Rebuild stamps if needed and re-apply them to *all* existing pupil ImageItems
        without requiring update_layers() to run again.
        """
        if self._last_patch_M is None:
            return  # nothing drawn yet

        self._ensure_pupil_stamps(self._last_patch_M, force=force)

        if self._pupil_rgba_by_sensor is None:
            return

        S = self.thetas.shape[0]
        for layer_idx, pupil_items in self._pupil_items.items():
            # pupil_items is list[ImageItem] length S
            for s in range(min(S, len(pupil_items))):
                pupil_items[s].setImage(self._pupil_rgba_by_sensor[s], autoLevels=False)

    # -------------------------
    # Overlay update
    # -------------------------
    def _update_one(self, layer_idx: int, sensor_idx: int, patch_img: np.ndarray, base_levels=None):
        """Update one overlay patch + pupil stamp for (layer_idx, sensor_idx)."""
        if layer_idx not in self._layer_to_canvas:
            return

        patch = np.asarray(patch_img, dtype=np.float32)
        M = int(patch.shape[0])

        self._last_patch_M = M

        self._ensure_pupil_stamps(M)

        # display scale: ORIGINAL N coords -> displayed image coords
        # Here we assume the base image on canvas has same shape as patch (common in your overview).
        H, W = patch.shape
        scale_x = W / self.N
        scale_y = H / self.N

        cx = self.cx0 * scale_x
        cy = self.cy0 * scale_y

        layer_cfg = self.layers_cfg[layer_idx]
        h_m = float(layer_cfg.get("altitude_m", layer_cfg.get("h", 0.0)))

        thx, thy = float(self.thetas[sensor_idx, 0]), float(self.thetas[sensor_idx, 1])
        sx = ((h_m * thx) / self.dx) * scale_x
        sy = ((h_m * thy) / self.dx) * scale_y

        # put patch centered at (cx+sx, cy+sy)
        cxs = cx + sx
        cys = cy + sy
        x0 = float(cxs - M / 2.0)
        y0 = float(cys - M / 2.0)

        # levels
        if base_levels is None:
            finite = np.isfinite(patch)
            if np.any(finite):
                vmin = float(np.min(patch[finite]))
                vmax = float(np.max(patch[finite]))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                levels = (vmin, vmax)
            else:
                levels = (0.0, 1.0)
        else:
            levels = base_levels

        # patch overlay
        it = self._overlay_items[layer_idx][sensor_idx]
        it.setImage(patch, autoLevels=False, levels=levels)
        it.setPos(x0, y0)

        # pupil RGBA overlay
        pit = self._pupil_items[layer_idx][sensor_idx]
        pit.setImage(self._pupil_rgba_by_sensor[sensor_idx], autoLevels=False)

        # cone shrink factor
        gs_range = list(self.sensors.values())[sensor_idx].gs_range_m
        try:
            gs_range = cp.asnumpy(gs_range)
            alpha = 1.0 - (h_m / gs_range)
        except:
             alpha = 1.0 - (h_m / gs_range)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        pit.resetTransform()
        pit.setScale(alpha)

        # keep it centered inside the patch
        dxy = (1.0 - alpha) * M * 0.5
        pit.setPos(x0 + dxy, y0 + dxy)

    # -------------------------
    # Main entry: update
    # -------------------------
    def update_layers(self, layer_stack_or_dict, cmap="viridis", layer_first=True):
        """
        Accepts:
          - dict {layer_idx: (S,M,M)} or {layer_name: (S,M,M)}
          - array (L_active,S,M,M)  -> mapped onto active layers in order
          - array (L_total,S,M,M)   -> skips inactive layers
        """
        arr = layer_stack_or_dict

        # dictionary mode
        if isinstance(arr, dict):
            for k, layer_patches in arr.items():
                if isinstance(k, str):
                    # find layer by name
                    layer_idx = next((i for i, L in enumerate(self.layers_cfg) if L.get("name") == k), None)
                    if layer_idx is None:
                        continue
                else:
                    layer_idx = int(k)

                layer_patches = _to_numpy(layer_patches)
                layer_patches = np.asarray(layer_patches)

                if layer_patches.ndim != 3:
                    continue  # expect (S,M,M)

                # base image = center sensor patch
                base = np.asarray(layer_patches[self.center_idx])
                if layer_idx in self._layer_to_canvas:
                    self.layer_canvases[self._layer_to_canvas[layer_idx]].queue_image(base, cmap=cmap, levels=(0, 255))
                    try:
                        base_levels = self.layer_canvases[self._layer_to_canvas[layer_idx]].image_item.levels
                    except Exception:
                        base_levels = None
                else:
                    base_levels = None

                for s in range(layer_patches.shape[0]):
                    self._update_one(layer_idx, s, layer_patches[s], base_levels=base_levels)
            return

        # array mode
        if isinstance(arr, cp.ndarray):
            arr = arr.get()
        arr = np.asarray(arr)

        if arr.ndim != 4:
            return  # expect (L,S,M,M)

        L_in, S, M, _ = arr.shape
        L_total = len(self.layers_cfg)
        L_active = len(self._active_layer_idxs)

        # Determine mapping from arr layer axis -> true layer indices
        if L_in == L_active:
            layer_indices = self._active_layer_idxs
        elif L_in == L_total:
            layer_indices = list(range(L_total))
        else:
            # unknown layout; assume first L_in layers
            layer_indices = list(range(min(L_in, L_total)))

        for j, layer_idx in enumerate(layer_indices):
            if j >= L_in:
                break
            if not (0 <= layer_idx < L_total):
                continue
            if not self._layer_is_active(self.layers_cfg[layer_idx]):
                continue  # skip inactive

            layer_patches = arr[j]  # (S,M,M)

            # base image = center sensor patch
            base = np.asarray(layer_patches[self.center_idx])
            ci = self._layer_to_canvas.get(layer_idx, None)
            if ci is None:
                continue

            self.layer_canvases[ci].queue_image(base, cmap=cmap, levels=(0, 255))
            try:
                base_levels = self.layer_canvases[ci].image_item.levels
            except Exception:
                base_levels = None

            for s in range(S):
                self._update_one(layer_idx, s, layer_patches[s], base_levels=base_levels)
