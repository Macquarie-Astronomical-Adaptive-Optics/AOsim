from PySide6.QtCore import Signal, Slot, QObject
from PySide6.QtWidgets import (
    QVBoxLayout, QWidget,
)
import pyqtgraph as pg
import numpy as np
import cupy as cp

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scripts.utilities import Analysis
from scripts.pg_colormaps import apply_mpl_cmap

pg.setConfigOptions(imageAxisOrder='row-major')

class PGCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.view = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.view)
        self.vb = self.view.addViewBox()
        self.vb.setAspectLocked(True)
        self.vb.invertY(True)

        # Base image
        self.image_item = pg.ImageItem(border='w')
        self.vb.addItem(self.image_item)

        # Pupil overlay
        self.pupil_item = pg.ImageItem()
        self.pupil_item.setZValue(10)
        self.vb.addItem(self.pupil_item)

        # scatter for stationary centroids
        self.scatter = pg.ScatterPlotItem(size=6, pen=pg.mkPen(None), brush=pg.mkBrush(255,0,0))
        self.scatter.setZValue(20)
        self.vb.addItem(self.scatter)

        # quiver arrows for moved centroids
        self.quivers = []

    def set_points_and_quivers(self, centroids, moved_centroids, threshold=1e-3, color=(0,255,0), min_len=1.0):
        """Draw stationary centroids as scatter, moved as arrows"""
        if isinstance(centroids, cp.ndarray): centroids = centroids.get()
        if isinstance(moved_centroids, cp.ndarray): moved_centroids = moved_centroids.get()
        centroids = np.asarray(centroids)
        moved_centroids = np.asarray(moved_centroids)

        dx = moved_centroids[:,1] - centroids[:,1]
        dy = moved_centroids[:,0] - centroids[:,0]

        # mask moved vs stationary
        moved_mask = (np.hypot(dx, dy) > threshold)
        stationary_mask = ~moved_mask

        # update scatter for stationary
        self.scatter.setData(x=centroids[stationary_mask,1], y=centroids[stationary_mask,0])

        # remove old quivers
        for q in self.quivers:
            self.vb.removeItem(q)
        self.quivers = []

        # add arrows for moved centroids
        for px, py, mx, my in zip(centroids[moved_mask,1], centroids[moved_mask,0],
                                  dx[moved_mask], dy[moved_mask]):
            arrow = pg.ArrowItem(pos=(px, py), angle=np.degrees(np.arctan2(my, mx)),
                                 tipAngle=30, headLen=6, tailLen=0,
                                 brush=pg.mkBrush(color), pen=pg.mkPen(color))
            arrow.setScale(max(np.hypot(mx, my), min_len))
            self.vb.addItem(arrow)
            self.quivers.append(arrow)

    def set_image(self, img, cmap="viridis"):
        """img: numpy array (H,W)"""
        # Ensure numpy
        if isinstance(img, cp.ndarray):
            img = img.get()
        img = np.asarray(img)
        apply_mpl_cmap(self.image_item, img, cmap=cmap)

    def set_pupil_mask(self, pupil):
        """pupil: binary array (1 inside pupil) -> draws white where pupil==1, alpha elsewhere 0"""
        if isinstance(pupil, cp.ndarray):
            pupil = pupil.get()
        pupil = np.asarray(pupil)
        # Create RGBA white where pupil==1, transparent elsewhere
        H, W = pupil.shape
        rgba = np.zeros((H, W, 4), dtype=np.ubyte)
        mask = pupil == 1
        rgba[..., 0][mask] = 255
        rgba[..., 1][mask] = 255
        rgba[..., 2][mask] = 255
        rgba[..., 3][mask] = 255  # opaque inside pupil
        # outside pupil alpha stays 0 (transparent)
        self.pupil_item.setImage(rgba, autoLevels=False)

    def set_points(self, x, y, **kwargs):
        """x,y numpy arrays"""
        if isinstance(x, cp.ndarray):
            x = x.get()
        if isinstance(y, cp.ndarray):
            y = y.get()
        x = np.asarray(x); y = np.asarray(y)
        pts = [{'pos': (float(px), float(py))} for px, py in zip(x, y)]
        self.scatter.setData(pts, **kwargs)

    def set_image_masked(self, img, mask=None, cmap='viridis', autoLevels=True):
        
        # convert CuPy -> numpy if needed
        if isinstance(img, cp.ndarray):
            img = img.get()
        img = np.asarray(img, dtype=np.float32)

        # normalize 0-1
        img_min, img_max = np.nanmin(img), np.nanmax(img)
        if img_max > img_min:
            img_norm = (img - img_min) / (img_max - img_min)
        else:
            img_norm = np.zeros_like(img)

        # apply colormap
        colormap = cm.get_cmap(cmap)
        rgba = colormap(img_norm)  # H,W,4 float 0-1
        rgba = (rgba * 255).astype(np.uint8)

        # apply pupil mask to alpha
        if mask is not None:
            if isinstance(mask, cp.ndarray):
                mask = mask.get()
            mask = np.asarray(mask).astype(bool)
            # where mask==0, set color to masked_color and alpha=255
            rgba[~mask, 0:3] = (255,255,255)
            rgba[~mask, 3] = 255


        apply_mpl_cmap(self.image_item, rgba, cmap=cmap)



from PySide6.QtGui import QColor, QBrush, QPen
from PySide6.QtWidgets import QGraphicsRectItem

class PGCanvasWithBoxes(PGCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.wfs_boxes = []

    def add_wfs_box(self, x_start, x_end, y_start, y_end, color=(0,255,0,100)):
        """
        Draw a semi-transparent rectangle for a WFS FOV
        x_start, x_end, y_start, y_end : indices on the phase screen
        color : RGBA tuple (0-255)
        """
        # remove old box if needed
        rect = QGraphicsRectItem(x_start, y_start, x_end-x_start, y_end-y_start)
        pen = QPen(QColor(*color))
        pen.setWidth(2)
        rect.setPen(pen)
        brush = QBrush(QColor(*color))
        rect.setBrush(brush)
        self.vb.addItem(rect)
        self.wfs_boxes.append(rect)

    def remove_rect(self, rect):
        if rect in self.wfs_boxes:
            self.vb.removeItem(rect)
            self.wfs_boxes.remove(rect)
            
    def clear_wfs_boxes(self):
        for rect in self.wfs_boxes:
            self.vb.removeItem(rect)
        self.wfs_boxes = []
