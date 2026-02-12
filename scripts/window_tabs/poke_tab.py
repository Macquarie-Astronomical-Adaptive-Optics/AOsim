import math
from typing import Dict
import logging

logger = logging.getLogger(__name__)
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QWidget

import scripts.utilities as ut
from scripts.widgets.config_table import Config_table
from scripts.widgets.wrap_tab import DetachableTabWidget
from scripts.window_tabs.wfsensor_tab import SensorTabWidget

ARCSEC2RAD = math.pi / (180.0 * 3600.0)
RAD2ARCSEC = (180.0 * 3600.0) / math.pi

class Poke_tab(QWidget):
    update_request = Signal(int, int)
    sensors_changed = Signal(dict)
    pupil_changed = Signal(float)

    def __init__(self, config_dict: dict):
        super().__init__()
        self.params = config_dict
        ut.set_params(self.params)

        self.wfsensors: Dict[str, object] = {}
        self._build_sensors()

        # layout for entire tab
        main_layout = QHBoxLayout(self)

        # ---- Left: parameter configuration ----
        left_layout = QVBoxLayout()

        ftable = QFrame()
        ftable.setMaximumWidth(250)
        ftable.setMinimumWidth(218)

        ftable_layout = QVBoxLayout(ftable)
        self.config_table = Config_table(
            ["telescope_diameter", "telescope_center_obscuration", "actuators", "grid_size", "poke_amplitude"],
            self.params,
        )
        ftable_layout.addWidget(self.config_table)
        left_layout.addWidget(ftable)
        main_layout.addLayout(left_layout)

        # ---- Middle: sensor editor tabs ----
        main_middle_layout = QVBoxLayout()
        sensor_selector_h = QHBoxLayout()

        sensor_tabs = DetachableTabWidget()
        sensor_tabs.setMovable(True)

        logger.info("Starting sensor tab editor")

        self.tab_pages = []
        for key, sensor in self.wfsensors.items():
            # The "science" channel is a dedicated imager (not a WFS) and does not
            # support the WFS sensor editor UI.
            if key.lower() == "science" or bool(getattr(sensor, "is_science_sensor", False)):
                continue
            tab = SensorTabWidget(dict(self.params), key, sensor)
            tab.sensor_changed.connect(self.update_sensor)

            self.tab_pages.append(tab)
            sensor_tabs.addTab(tab, key)

        sensor_selector_h.addWidget(sensor_tabs)

        self.config_table.params_changed.connect(ut.set_params)
        self.config_table.params_changed.connect(self.update_tabs)

        main_middle_layout.addLayout(sensor_selector_h)
        main_layout.addLayout(main_middle_layout)

        # Some code expects an emission after UI construction as well.
        self.sensors_changed.emit(self.wfsensors)

    def _build_sensors(self) -> None:
        logger.info("Creating sensors")
        # Note: keep off_x_px=0.0 to preserve existing behaviour (placeholder for future scaling).
        off_x_px = 0.0  # (0.5 * telescope_diameter) / (telescope_diameter / grid_size)

        def add(key: str, sensor) -> None:
            self.wfsensors[key] = sensor

        # Science channel
        add(
            "science",
            ut.WFSensor_tools.ScienceImager(
                wavelength=float(self.params.get("science_lambda", 2.150e-06)),
                grid_size=int(self.params.get("grid_size", 512)),
            ),
        )

        # LGS constellation (units consistent with original code)
        add(
            "LGS_NE",
            ut.WFSensor_tools.ShackHartmann(
                90_000,
                n_sub=40,
                dx=2.828 * 60,
                dy=2.828 * 60,
                lgs_thickness_m=10_000.0,  # ~10 km sodium thickness
                lgs_launch_offset_px=(off_x_px, 0.0),
                lgs_remove_tt=True,
                wavelength=589e-9,
            ),
        )
        add(
            "LGS_NW",
            ut.WFSensor_tools.ShackHartmann(
                90_000,
                n_sub=40,
                dx=-2.828 * 60,
                dy=2.828 * 60,
                lgs_thickness_m=10_000.0,
                lgs_launch_offset_px=(-off_x_px, 0.0),
                lgs_remove_tt=True,
                wavelength=589e-9,
            ),
        )
        add(
            "LGS_SE",
            ut.WFSensor_tools.ShackHartmann(
                90_000,
                n_sub=40,
                dx=2.828 * 60,
                dy=-2.828 * 60,
                lgs_thickness_m=10_000.0,
                lgs_launch_offset_px=(0.0, off_x_px),
                lgs_remove_tt=True,
                wavelength=589e-9,
            ),
        )
        add(
            "LGS_SW",
            ut.WFSensor_tools.ShackHartmann(
                90_000,
                n_sub=40,
                dx=-2.828 * 60,
                dy=-2.828 * 60,
                lgs_thickness_m=10_000.0,
                lgs_launch_offset_px=(0.0, -off_x_px),
                lgs_remove_tt=True,
                wavelength=589e-9,
            ),
        )

        # NGS quad
        # for name, dx, dy in [
        #     ("NGS_NE", 4.95 * 60, 4.95 * 60),
        #     ("NGS_NW", -4.95 * 60, 4.95 * 60),
        #     ("NGS_SE", 4.95 * 60, -4.95 * 60),
        #     ("NGS_SW", -4.95 * 60, -4.95 * 60),
        # ]:
        #     add(name, ut.WFSensor_tools.ShackHartmann(n_sub=2, dx=dx, dy=dy, wavelength=1650e-9))
        add("NGS", ut.WFSensor_tools.ShackHartmann(n_sub=2, dx=0, dy=0, wavelength=1650e-9))

        self.sensors_changed.emit(self.wfsensors)

        for key, sensor in self.wfsensors.items():
            dx_arcsec = getattr(sensor, "dx", 0.0) * RAD2ARCSEC
            dy_arcsec = getattr(sensor, "dy", 0.0) * RAD2ARCSEC
            logger.info(" - %s: (%.0f, %.0f) arcsec", key, dx_arcsec, dy_arcsec)

    def update_tabs(self, params: dict) -> None:
        # also update sensor view tab while we're at it
        self.pupil_changed.emit(params.get("telescope_center_obscuration"))
        for tab in self.tab_pages:
            tab.main_params_changed(params)

    def update_sensor(self, sensor) -> None:
        # SensorTabWidget emits this after its worker finishes a full (re)compute.
        # Sensors are mutated in-place; just re-emit the dictionary.
        self.sensors_changed.emit(self.wfsensors)