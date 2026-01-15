from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QHBoxLayout, QWidget,
    QSplitter
)


from scripts.widgets.grids import LayerFootprintGrid, SensorPSFGrid

class SensorView_tab(QWidget):

    def __init__(self, sensors, params, layers_cfg, N, dx, patch_center, patch_size_px, scheduler, parent=None):
        super().__init__(parent)
        self.params = params
        self.sensors = sensors
        self.scheduler = scheduler

        layout = QHBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        self.psf_grid = SensorPSFGrid(sensors, title="WFSensor Phase")
        self.layer_grid = LayerFootprintGrid(
            sensors=sensors,
            layers_cfg=layers_cfg,
            N=N, dx=dx,
            patch_center=patch_center,
            patch_size_px=patch_size_px,
            title="Layers + Sensor Footprints",
        )

        splitter.addWidget(self.psf_grid)
        splitter.addWidget(self.layer_grid)
        splitter.setSizes([400, 800])  # tweak

    @Slot(float)
    def updateLayerGrid(self, new_obsc):
        self.layer_grid.set_pupil_params(center_obscuration=new_obsc) 

    def showEvent(self, event):
        self.scheduler._overview_enabled = True
        return super().showEvent(event)
    
    def hideEvent(self, event):
        self.scheduler._overview_enabled = False
        return super().hideEvent(event)
