from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QWidget,
    QSplitter
)


from scripts.widgets.grids import LayerFootprintGrid, SensorPSFGrid

class SensorView_tab(QWidget):

    def __init__(self, params, layers_cfg, thetas, N, dx, patch_center, patch_size_px, scheduler, parent=None):
        super().__init__(parent)
        self.params = params
        self.scheduler = scheduler

        layout = QHBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        self.psf_grid = SensorPSFGrid(thetas, title="Sensor PSFs")
        self.layer_grid = LayerFootprintGrid(
            layers_cfg=layers_cfg,
            thetas_xy_rad=thetas,
            N=N, dx=dx,
            patch_center=patch_center,
            patch_size_px=patch_size_px,
            title="Layers + Sensor Footprints",
        )

        splitter.addWidget(self.psf_grid)
        splitter.addWidget(self.layer_grid)
        splitter.setSizes([400, 800])  # tweak

    def showEvent(self, event):
        self.scheduler._overview_enabled = True
        return super().showEvent(event)
    
    def hideEvent(self, event):
        self.scheduler._overview_enabled = False
        return super().hideEvent(event)
