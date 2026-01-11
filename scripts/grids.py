from PySide6.QtWidgets import QWidget, QGridLayout, QLabel, QVBoxLayout, QSizePolicy
import numpy as np
import cupy as cp

from scripts.pgcanvas import PGCanvas
from scripts.utilities import Pupil_tools

class SensorPSFGrid(QWidget):
    """
    Displays PSFs for all sensors in a grid arranged by their field angles.

    thetas_xy_rad: list/array shape (S,2) with theta_x, theta_y in radians.
    """
    def __init__(self, thetas_xy_rad, parent=None, title="Sensor PSFs"):
        super().__init__(parent)
        self.thetas = np.asarray(thetas_xy_rad, dtype=np.float32)  # (S,2)
        self.S = self.thetas.shape[0]

        outer = QVBoxLayout(self)
        outer.addWidget(QLabel(title))

        self.grid = QGridLayout()
        outer.addLayout(self.grid)

        # compute grid coordinates from thetas
        self.pos = self._theta_to_grid(self.thetas)  # list of (row,col)

        # create a canvas for each sensor
        self.canvases = []
        
        
        for s in range(self.S):
            c = PGCanvas()
            self.canvases.append(c)
            r, col = self.pos[s]
            self.grid.addWidget(c, r, col)

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
                self.canvases[s].set_image_robust(img, cmap=cmap)


import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QGridLayout, QLabel, QVBoxLayout
import numpy as np
import cupy as cp
import pyqtgraph as pg
import matplotlib.cm as mpl_cm


def _mpl_lut(cmap: str, n: int = 256) -> np.ndarray:
    cm = mpl_cm.get_cmap(cmap, n)
    rgba = (cm(np.linspace(0, 1, n)) * 255).astype(np.ubyte)
    return rgba[:, :3].copy()


class LayerFootprintGrid(QWidget):
    """
    Grid of layer phase screens with per-sensor overlay patches showing the turbulence
    inside each sensor's footprint on that layer.
    """
    def __init__(
        self,
        layers_cfg,
        thetas_xy_rad,
        N,
        dx,
        patch_center,
        patch_size_px,
        parent=None,
        title="Layers + Sensor Footprints",
        overlay_opacity=0.55,
        overlay_cmap="viridis",
        overlay_z=50,
    ):
        super().__init__(parent)

        self.layers_cfg = list(layers_cfg)
        self.thetas = np.asarray(thetas_xy_rad, dtype=np.float32)  # (S,2)
        self.N = int(N)
        self.dx = float(dx)

        self.cx0, self.cy0 = float(patch_center[0]), float(patch_center[1])  # in ORIGINAL N-pixel coords
        self.patch_size0 = float(patch_size_px)  # in ORIGINAL N-pixel coords

        self.overlay_opacity = float(overlay_opacity)
        self.overlay_cmap = overlay_cmap
        self.overlay_z = int(overlay_z)

        self._lut_cache = {}

        outer = QVBoxLayout(self)
        outer.addWidget(QLabel(title))

        self.grid = QGridLayout()
        outer.addLayout(self.grid)

        self.layer_canvases = []
        # overlay_items[layer_idx][sensor_idx] = pg.ImageItem
        self._overlay_items = []

        L = len(self.layers_cfg)
        ncol = int(np.ceil(np.sqrt(L)))

        # sensor color pens
        self._pens = [
            (255, 0, 0, 55),
            (0, 255, 0, 55),
            (0, 128, 255, 55),
            (255, 165, 0, 55),
            (255, 0, 255, 55),
            (0, 255, 255, 55),
            (255, 255, 0, 55)
        ]

        # middle sensor index = closest theta to (0,0)
        self.center_idx = int(np.argmin(np.sum(self.thetas**2, axis=1)))

        S = self.thetas.shape[0]
        L = len(self.layers_cfg)
        self._pupil_items = [[None for _ in range(S)] for _ in range(L)]



        for li in range(L):
            c = PGCanvas()
            self.layer_canvases.append(c)
            r, col = divmod(li, ncol)
            self.grid.addWidget(c, r, col)

        self._create_overlay_items()

    def _get_lut(self, cmap: str):
        if cmap not in self._lut_cache:
            self._lut_cache[cmap] = _mpl_lut(cmap, 256)
        return self._lut_cache[cmap]

    def _create_overlay_items(self):
        """Create overlay ImageItems once (one per layer per sensor)."""
        self._overlay_items = []
        lut = self._get_lut(self.overlay_cmap)

        for li in range(len(self.layers_cfg)):
            items = []
            canvas = self.layer_canvases[li]

            for s in range(self.thetas.shape[0]):
                it = pg.ImageItem()

                if s == 0:
                    it.setZValue(self.overlay_z+999)
                    it.setOpacity(1.00)
                else:
                    it.setZValue(self.overlay_z + s)
                    it.setOpacity(self.overlay_opacity)
                    
                it.setLookupTable(lut)

                # Add to this layer canvas
                canvas.vb.addItem(it)
                items.append(it)

            self._overlay_items.append(items)

    def rebuild_overlays(self, layers_cfg=None, thetas_xy_rad=None, dx=None, patch_center=None, patch_size_px=None):
        """
        Call this if geometry changes (thetas, dx, altitudes, patch size).
        """
        if layers_cfg is not None:
            self.layers_cfg = list(layers_cfg)
        if thetas_xy_rad is not None:
            self.thetas = np.asarray(thetas_xy_rad, dtype=np.float32)
        if dx is not None:
            self.dx = float(dx)
        if patch_center is not None:
            self.cx0, self.cy0 = float(patch_center[0]), float(patch_center[1])
        if patch_size_px is not None:
            self.patch_size0 = float(patch_size_px)

        # Remove old overlay items from viewboxes
        for li, items in enumerate(self._overlay_items):
            canvas = self.layer_canvases[li]
            for it in items:
                try:
                    canvas.vb.removeItem(it)
                except Exception:
                    pass

        self._create_overlay_items()

    def make_pupil_rgba(pupil_mask: np.ndarray, color=(255,255,255), alpha=80):
        """
        pupil_mask: bool array (M,M) True inside pupil
        returns RGBA uint8 (M,M,4)
        """
        m = np.asarray(pupil_mask).astype(bool)
        H, W = m.shape
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[m, 0] = color[0]
        rgba[m, 1] = color[1]
        rgba[m, 2] = color[2]
        rgba[m, 3] = alpha
        return rgba


    def _update_layer_overlays(self, li: int, ii: int, layer_img: np.ndarray):
        """
        Update overlay patch images/positions for one layer given the displayed layer image.
        layer_img is the displayed layer image (might be downsampled).
        """
        H, W = layer_img.shape
        # scale from ORIGINAL coords (N) to DISPLAY coords (W)
        scale_x = W / self.N
        scale_y = H / self.N

        # center + footprint size in DISPLAY coords
        cx = self.cx0 * scale_x
        cy = self.cy0 * scale_y

        # base levels from the main layer image, if available
        base_levels = None
        try:
            base_levels = self.layer_canvases[li].image_item.levels
        except Exception:
            base_levels = None

        s = ii
        thx, thy = float(self.thetas[s, 0]), float(self.thetas[s, 1])
        h_m = float(self.layers_cfg[li].get("altitude_m", self.layers_cfg[li].get("h", 0.0)))

        # shift in ORIGINAL pixels then scale to display pixels
        sx0 = (h_m * thx) / self.dx
        sy0 = (h_m * thy) / self.dx
        sx = sx0 * scale_x
        sy = sy0 * scale_y

        # desired footprint rectangle (in display coords)

        it = self._overlay_items[li][s]

        # patch = layer_img[y0c:y1c, x0c:x1c].astype(np.float32, copy=False)
        patch = layer_img.astype(np.float32, copy=False)

        # IMPORTANT for pyqtgraph float rendering: ensure levels exist
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

        # --- CIRCLE overlay ---
        pen = self._pens[s % len(self._pens)]
        _pupil_alpha = Pupil_tools.circle(self.patch_size0/2, self.patch_size0)  # (M,M)
        
        m = cp.asarray(_pupil_alpha).astype(bool)      # (M,M)
        H, W = m.shape

        rgba = cp.zeros((H, W, 4), dtype=cp.uint8)
        rgba[m, 0] = pen[0]   # R
        rgba[m, 1] = pen[1]   # G
        rgba[m, 2] = pen[2]   # B
        rgba[m, 3] = pen[3]
        rgba = cp.asnumpy(rgba)
        # radius: show the patch footprint radius (circle around the patch)
        cxs = cx + sx
        cys = cy + sy

        pit = self._pupil_items[li][s]
        if pit is None:
            pit = pg.ImageItem()
            pit.setZValue(4500 + s)
            pit.setOpacity(1.0)  # alpha is in the RGBA itself
            self.layer_canvases[li].vb.addItem(pit)
            self._pupil_items[li][s] = pit

        pit.setImage(rgba, autoLevels=False)

        # place it centered at (cxs, cys)
        Mpatch = rgba.shape[0]
        pit.setPos(float(cxs - Mpatch/2.0), float(cys - Mpatch/2.0))

        # Paint turbulence patch in the footprint region
        it.setImage(patch, autoLevels=False, levels=levels)
        it.setPos(sx, sy)

    def update_layers(self, layer_stack_or_dict, cmap="viridis", layer_first=True):
        """
        Update base layer images + per-sensor overlay patches.

        Accepts:
        - (L,S,H,W) per-layer per-sensor patches (layer_first=True)

        """

        arr = layer_stack_or_dict
        if isinstance(arr, cp.ndarray):
            arr = arr.get()
        arr = np.asarray(arr)

        for li, layer in enumerate(arr):
            for ii, img in enumerate(layer): 
                if 0 <= li < len(self.layer_canvases):
                    img = np.asarray(img, dtype=np.float32)
                    self.layer_canvases[li].set_image_robust(img, cmap=cmap)
                    self._update_layer_overlays(li, ii, img)
