from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QHBoxLayout, QSplitter, QWidget

from scripts.widgets.grids import LayerFootprintGrid, SensorPSFGrid


class SensorView_tab(QWidget):
    def __init__(
        self,
        sensors: dict,
        params: dict,
        layers_cfg,
        N: int,
        dx: float,
        patch_center,
        patch_size_px: int,
        scheduler,
        parent=None,
    ):
        super().__init__(parent)
        self.params = params
        self.sensors = sensors
        self.scheduler = scheduler

        layout = QHBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        self.psf_grid = SensorPSFGrid(sensors, title="WFSensor Phase")
        self.layer_grid = LayerFootprintGrid(
            sensors=sensors,
            layers_cfg=layers_cfg,
            N=N,
            dx=dx,
            patch_center=patch_center,
            patch_size_px=patch_size_px,
            title="Layers + Sensor Footprints",
        )

        splitter.addWidget(self.psf_grid)
        splitter.addWidget(self.layer_grid)
        splitter.setSizes([400, 800])  # tweak

        self.scheduler._overview_enabled = False

    # Keep old name for external signal-slot connections.
    @Slot(float)
    def updateLayerGrid(self, new_obsc: float) -> None:
        self.update_layer_grid(new_obsc)

    def update_layer_grid(self, new_obsc: float) -> None:
        self.layer_grid.set_pupil_params(center_obscuration=float(new_obsc))

    def showEvent(self, event):
        self.scheduler._overview_enabled = True
        return super().showEvent(event)

    def hideEvent(self, event):
        self.scheduler._overview_enabled = False
        return super().hideEvent(event)
