import math
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QWidget

import scripts.utilities as ut
from scripts.core.config_store import ConfigStore
from scripts.core.config_proxy import OverlayConfig
from scripts.widgets.config_table import Config_table
from scripts.widgets.wrap_tab import DetachableTabWidget
from scripts.window_tabs.wfsensor_tab import SensorTabWidget, SENSOR_PARAM_KEYS

ARCSEC2RAD = math.pi / (180.0 * 3600.0)
RAD2ARCSEC = (180.0 * 3600.0) / math.pi

class Poke_tab(QWidget):
    update_request = Signal(int, int)
    sensors_changed = Signal(dict)
    pupil_changed = Signal(float)
    # Global geometry changed (requires turbulence + reconstructor rebuild).
    hard_geometry_changed = Signal(dict)

    def __init__(self, config_dict: dict | ConfigStore):
        super().__init__()

        self.config_store = config_dict if isinstance(config_dict, ConfigStore) else ConfigStore(dict(config_dict))
        self.params = self.config_store.data

        # Backwards compatible migration: ensure sensors live in the central config.
        self._ensure_default_sensors()

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
            config_store=self.config_store,
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
            # Per-sensor overrides live under params["sensors"][name].
            sens_cfg: Dict[str, Any] = self.params.setdefault("sensors", {}).setdefault(key, {})
            merged_params = OverlayConfig(self.params, sens_cfg, overlay_keys=SENSOR_PARAM_KEYS)
            tab = SensorTabWidget(merged_params, key, sensor, config_store=self.config_store)
            tab.sensor_changed.connect(self.update_sensor)

            self.tab_pages.append(tab)
            sensor_tabs.addTab(tab, key)

        sensor_selector_h.addWidget(sensor_tabs)

        self.config_table.params_changed.connect(self.update_tabs)

        # Track signature of geometry knobs that require a pipeline rebuild.
        self._hard_sig = self._compute_hard_sig(dict(self.params))

        # Refresh tables if the central config is loaded from disk.
        self.config_store.changed.connect(self._on_store_changed)

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

        # Build from params["sensors"].
        sensors_cfg: Dict[str, Dict[str, Any]] = self.params.setdefault("sensors", {})
        for name, cfg in sensors_cfg.items():
            stype = str(cfg.get("type", "ShackHartmann"))

            if stype == "ScienceImager":
                add(
                    name,
                    ut.WFSensor_tools.ScienceImager(
                        wavelength=float(cfg.get("wavelength", self.params.get("science_lambda", 2.150e-6))),
                        grid_size=int(cfg.get("grid_size", self.params.get("grid_size", 512))),
                    ),
                )
                continue

            if stype == "ShackHartmann":
                # dx/dy in config are stored in arcsec (ut.ShackHartmann converts internally)
                add(
                    name,
                    ut.WFSensor_tools.ShackHartmann(
                        cfg.get("gs_range_m", float("inf")),
                        n_sub=cfg.get("sub_apertures", cfg.get("n_sub", self.params.get("sub_apertures", 40))),
                        dx=cfg.get("dx", 0.0),
                        dy=cfg.get("dy", 0.0),
                        wavelength=cfg.get("wfs_lambda", cfg.get("wavelength", self.params.get("wfs_lambda", 500e-9))),
                        lgs_thickness_m=cfg.get("lgs_thickness_m", 0.0),
                        lgs_launch_offset_px=tuple(cfg.get("lgs_launch_offset_px", (0.0, 0.0))),
                        lgs_remove_tt=bool(cfg.get("lgs_remove_tt", True)),
                    ),
                )
                continue

            raise ValueError(f"Unknown sensor type {stype!r} for sensor {name!r}")

        self.sensors_changed.emit(self.wfsensors)

        for key, sensor in self.wfsensors.items():
            dx_arcsec = getattr(sensor, "dx", 0.0) * RAD2ARCSEC
            dy_arcsec = getattr(sensor, "dy", 0.0) * RAD2ARCSEC
            logger.info(" - %s: (%.0f, %.0f) arcsec", key, dx_arcsec, dy_arcsec)

    def update_tabs(self, params: dict) -> None:
        # also update sensor view tab while we're at it
        self.pupil_changed.emit(params.get("telescope_center_obscuration"))

        # Only request a heavy rebuild when core geometry changed.
        try:
            sig = self._compute_hard_sig(params)
            if sig != getattr(self, "_hard_sig", None):
                self._hard_sig = sig
                self.hard_geometry_changed.emit({
                    "telescope_diameter": params.get("telescope_diameter"),
                    "telescope_center_obscuration": params.get("telescope_center_obscuration"),
                    "actuators": params.get("actuators"),
                    "grid_size": params.get("grid_size"),
                })
        except Exception:
            pass

        for tab in self.tab_pages:
            tab.main_params_changed(params)

    @staticmethod
    def _compute_hard_sig(params: dict) -> tuple:
        """Signature of geometry params that require a pipeline rebuild."""
        def _as_int(x, default=0):
            try:
                return int(x)
            except Exception:
                return int(default)

        def _as_float(x, default=0.0):
            try:
                return float(x)
            except Exception:
                return float(default)

        return (
            _as_float(params.get("telescope_diameter"), 0.0),
            _as_float(params.get("telescope_center_obscuration"), 0.0),
            _as_int(params.get("actuators"), 0),
            _as_int(params.get("grid_size"), 0),
        )

    def _on_store_changed(self) -> None:
        # Update global-dependent views.
        self.update_tabs(dict(self.params))
        # Keep sensor-tab tables visually in sync on load.
        for tab in self.tab_pages:
            if hasattr(tab, "reload_from_store"):
                try:
                    tab.reload_from_store()
                except Exception:
                    pass

    def _ensure_default_sensors(self) -> None:
        """Ensure params has a default sensor constellation definition.

        This migrates older configs that didn't include a "sensors" section.
        """
        if "sensors" in self.params and isinstance(self.params["sensors"], dict) and self.params["sensors"]:
            return

        # Match the previous hard-coded defaults.
        off_x_px = 0.0
        self.params["sensors"] = {
            "science": {
                "type": "ScienceImager",
                "wavelength": float(self.params.get("science_lambda", 2.150e-6)),
                "grid_size": int(self.params.get("grid_size", 512)),
            },
            "LGS_NE": {
                "type": "ShackHartmann",
                "gs_range_m": 90_000,
                "sub_apertures": 40,
                "dx": 2.828 * 60,
                "dy": 2.828 * 60,
                "wfs_lambda": 589e-9,
                "lgs_thickness_m": 10_000.0,
                "lgs_launch_offset_px": (off_x_px, 0.0),
                "lgs_remove_tt": True,
            },
            "LGS_NW": {
                "type": "ShackHartmann",
                "gs_range_m": 90_000,
                "sub_apertures": 40,
                "dx": -2.828 * 60,
                "dy": 2.828 * 60,
                "wfs_lambda": 589e-9,
                "lgs_thickness_m": 10_000.0,
                "lgs_launch_offset_px": (-off_x_px, 0.0),
                "lgs_remove_tt": True,
            },
            "LGS_SE": {
                "type": "ShackHartmann",
                "gs_range_m": 90_000,
                "sub_apertures": 40,
                "dx": 2.828 * 60,
                "dy": -2.828 * 60,
                "wfs_lambda": 589e-9,
                "lgs_thickness_m": 10_000.0,
                "lgs_launch_offset_px": (0.0, off_x_px),
                "lgs_remove_tt": True,
            },
            "LGS_SW": {
                "type": "ShackHartmann",
                "gs_range_m": 90_000,
                "sub_apertures": 40,
                "dx": -2.828 * 60,
                "dy": -2.828 * 60,
                "wfs_lambda": 589e-9,
                "lgs_thickness_m": 10_000.0,
                "lgs_launch_offset_px": (0.0, -off_x_px),
                "lgs_remove_tt": True,
            },
            "NGS": {
                "type": "ShackHartmann",
                "gs_range_m": float("inf"),
                "sub_apertures": 2,
                "dx": 0.0,
                "dy": 0.0,
                "wfs_lambda": 1650e-9,
            },
        }

        # Notify listeners that the store now contains the migrated structure.
        try:
            self.config_store.changed.emit()
        except Exception:
            pass

    def update_sensor(self, sensor) -> None:
        # SensorTabWidget emits this after its worker finishes a full (re)compute.
        # Sensors are mutated in-place; just re-emit the dictionary.
        self.sensors_changed.emit(self.wfsensors)