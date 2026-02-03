# sensor_tab_widget.py

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFrame, QLabel,
    QSpinBox, QSlider, QTableWidget, QTableWidgetItem
)
import numpy as np

import scripts.utilities as ut
from scripts.widgets.pgcanvas import PGCanvas
from scripts.worker import CalculateWorker  # singleton shared worker
from scripts.widgets.config_table import Config_table
from data.CONFIG_DTYPES import enforce_config_types

ARCSEC2RAD = 3.141592653589793 / (180.0 * 3600.0)
RAD2ARCSEC = (180.0 * 3600.0) / 3.141592653589793

# Keys that are sensor-specific and must NOT be overwritten by main/global config updates
SENSOR_PARAM_KEYS = {
    "sub_apertures",
    "wfs_lambda",
    "dx",
    "dy",
    "gs_range_m",
    "lgs_launch_offset_px",
    "lgs_thickness_m",
    "lenslet_f_m",
    "pixel_pitch_m",
}


class SensorTabWidget(QWidget):
    """
    WFS sensor tab using the shared calculate worker.
    Heavy computations run in a single shared thread.
    """

    actuator_changed = Signal(int, int)  # (actuator index, job_id)
    sensor_changed = Signal(object)

    def __init__(self, params: dict, sensor_name: str = "main_sensor", sensor=None, parent=None):
        super().__init__(parent)
        self.params = params
        self.sensor_name = sensor_name

        if sensor is None:
            sensor = ut.WFSensor_tools.ShackHartmann(
                n_sub=self.params.get("sub_apertures"),
                wavelength=self.params.get("wfs_lambda"),
                pupil=None,
                grid_size=self.params.get("grid_size")
            )

        self.sensor = sensor
        self._seed_params_from_sensor()
        enforce_config_types(self.params)

        self.job_id = 0
        self.pending_index = 0

        # Cached CPU-side plotting data (avoid repeated GPU->CPU transfers / recomputation)
        self._pupil_mask = None
        self._sub_aps = None
        self._ref_centroids_single = None
        self._ref_points = None
        self._subap_mask_cache = {}  # size -> mask

        # build UI
        self._build_ui()

        # slider/spinbox connections
        self.act_select_slider.valueChanged.connect(self._on_spin_changed)
        self.act_select_spinbox.valueChanged.connect(self._on_spin_changed)
        self.act_select_spinbox.valueChanged.connect(self.act_select_slider.setValue)
        self.act_select_slider.valueChanged.connect(self.act_select_spinbox.setValue)

        # config table triggers
        self.config_table.params_changed.connect(self.recalculate)
        self.config_table.params_changed.connect(ut.set_params)

        # connect actuator updates to shared worker
        self.actuator_changed.connect(
            lambda k, jid: CalculateWorker.instance().process_subap(self.sensor, k, jid, self.params)
        )


        # connect shared worker signals
        worker = CalculateWorker.instance()
        worker.initialized_result.connect(self.on_calculation_finished)
        worker.subap_finished.connect(self.on_worker_finished)

        # request initial compute
        self._schedule_update(0)
        CalculateWorker.instance().request_initialization(self.sensor, self.params)

        self.initialized = False

    
    def _seed_params_from_sensor(self) -> None:
        """Initialize sensor-specific params from the provided sensor object."""
        self.params["sub_apertures"] = int(getattr(self.sensor, "n_sub", self.params.get("sub_apertures") or 0))
        self.params["wfs_lambda"] = float(getattr(self.sensor, "wavelength", self.params.get("wfs_lambda") or 500e-9))
        # Sensor stores angles in radians; UI/config uses arcsec.
        self.params["dx"] = float(getattr(self.sensor, "dx", 0.0)) * RAD2ARCSEC
        self.params["dy"] = float(getattr(self.sensor, "dy", 0.0)) * RAD2ARCSEC

        if hasattr(self.sensor, "gs_range_m"):
            self.params["gs_range_m"] = float(getattr(self.sensor, "gs_range_m"))
        if hasattr(self.sensor, "lgs_launch_offset_px"):
            self.params["lgs_launch_offset_px"] = tuple(getattr(self.sensor, "lgs_launch_offset_px"))
        if hasattr(self.sensor, "lgs_thickness_m"):
            self.params["lgs_thickness_m"] = float(getattr(self.sensor, "lgs_thickness_m"))
        if hasattr(self.sensor, "lenslet_f_m"):
            self.params["lenslet_f_m"] = float(getattr(self.sensor, "lenslet_f_m"))
        if hasattr(self.sensor, "pixel_pitch_m"):
            self.params["pixel_pitch_m"] = float(getattr(self.sensor, "pixel_pitch_m"))

    def _apply_sensor_params_to_sensor(self) -> None:
        """Apply current sensor-specific params to the underlying sensor object."""
        kw = {}
        if self.params.get("sub_apertures") is not None:
            kw["n_sub"] = int(self.params["sub_apertures"])
        if self.params.get("wfs_lambda") is not None:
            kw["wavelength"] = float(self.params["wfs_lambda"])

        for k in ("dx", "dy", "gs_range_m", "lgs_launch_offset_px", "lgs_thickness_m", "lenslet_f_m", "pixel_pitch_m"):
            if k in self.params and self.params[k] is not None:
                kw[k] = self.params[k]

        if kw:
            self.sensor.recompute(**kw)

    def _build_ui(self):
        main_layout = QHBoxLayout(self)

        # Left: config + busy + table
        left_v = QVBoxLayout()
        config_keys = ["sub_apertures","wfs_lambda","dx","dy","gs_range_m","lgs_launch_offset_px","lgs_thickness_m","lenslet_f_m","pixel_pitch_m"]
        self.config_table = Config_table(config_keys, self.params)
        left_v.addWidget(self.config_table)

        self.busy_label = QLabel("Waiting")
        self.busy_label.setAlignment(Qt.AlignCenter)
        self.busy_label.setStyleSheet("font-weight: bold;")
        left_v.addWidget(self.busy_label)

        self.calc_values_table = QTableWidget(5, 2)
        self.calc_values_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.calc_values_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.calc_values_table.setItem(0, 0, QTableWidgetItem("Strehl"))
        self.calc_values_table.setItem(1, 0, QTableWidgetItem("Poke FWHM (mas)"))
        self.calc_values_table.setItem(2, 0, QTableWidgetItem("Unperturbed FWHM (mas)"))
        self.calc_values_table.setItem(3, 0, QTableWidgetItem("Diffraction Limit (mas)"))

        # Pre-create value cells to avoid allocating new QTableWidgetItems on every update.
        self._value_items = {}
        for row in (0, 1, 2, 3):
            it = QTableWidgetItem("—")
            self.calc_values_table.setItem(row, 1, it)
            self._value_items[row] = it
        self.calc_values_table.setWordWrap(True)
        self.calc_values_table.resizeRowsToContents()
        self.calc_values_table.setMinimumWidth(200)
        left_v.addWidget(self.calc_values_table)

        main_layout.addLayout(left_v)

        # Middle: overview + subap + science
        mid_v = QVBoxLayout()
        header_h = QHBoxLayout()
        header_h.addWidget(QLabel(f"Sensor: {self.sensor_name}"))
        mid_v.addLayout(header_h)

        top_h = QHBoxLayout()
        # overview canvas
        f_over = QFrame()
        f_over.setFrameShape(QFrame.Box)
        f_over.setLineWidth(1)
        over_v = QVBoxLayout(f_over)
        over_v.addWidget(QLabel("Sub Aperture Centroids"))
        self.canvas_overview = PGCanvas()
        self.canvas_overview.set_colorbar(label="surface deformation", units="m")

        over_v.addWidget(self.canvas_overview)
        top_h.addWidget(f_over)

        # subap canvas
        f_sub = QFrame()
        f_sub.setFrameShape(QFrame.Box)
        f_sub.setLineWidth(1)
        sub_v = QVBoxLayout(f_sub)
        sub_v.addWidget(QLabel("Sub Aperture Images"))
        selector_h = QHBoxLayout()
        selector_h.addWidget(QLabel("Poke Actuator Idx"))
        self.act_select_spinbox = QSpinBox()
        self.act_select_slider = QSlider(Qt.Horizontal)
        selector_h.addWidget(self.act_select_spinbox)
        selector_h.addWidget(self.act_select_slider)
        sub_v.addLayout(selector_h)
        self.canvas_subap = PGCanvas(show_colorbar = False)

        sub_v.addWidget(self.canvas_subap)
        top_h.addWidget(f_sub)

        mid_v.addLayout(top_h)

        # bottom: science canvases
        bottom_h = QHBoxLayout()
        f_science = QFrame()
        f_science.setFrameShape(QFrame.Box)
        f_science.setLineWidth(1)
        sc_v = QVBoxLayout(f_science)
        sc_v.addWidget(QLabel("Poke Science Image"))
        self.canvas_science = PGCanvas(show_colorbar = False)
        sc_v.addWidget(self.canvas_science)

        bottom_h.addWidget(f_science)

        f_unpert = QFrame()
        f_unpert.setFrameShape(QFrame.Box)
        f_unpert.setLineWidth(1)
        up_v = QVBoxLayout(f_unpert)
        up_v.addWidget(QLabel("Unperturbed Science Image"))
        self.canvas_science_pupil = PGCanvas(show_colorbar = False)

        up_v.addWidget(self.canvas_science_pupil)
        bottom_h.addWidget(f_unpert)

        mid_v.addLayout(bottom_h)
        main_layout.addLayout(mid_v)

        # debounce timer
        self.debounce_timer = QTimer(self)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.setInterval(40)
        self.debounce_timer.timeout.connect(self._emit_worker_request)

    def _on_spin_changed(self, v):
        self._schedule_update(v)

    def _schedule_update(self, v):
        self.pending_index = int(v)
        self.debounce_timer.start()
        # Avoid spamming external listeners for every slider tick.

    def _emit_worker_request(self):
        self.job_id += 1
        self.actuator_changed.emit(int(self.pending_index), self.job_id)
        self.busy_label.setText("Computing…")

    @Slot(object, object)
    def on_calculation_finished(self, sensor, result):
        if sensor is not self.sensor:
            return
    
        self.busy_label.setText("Ready")

        self.initialized = True

        # Cache CPU-side plotting arrays
        self._pupil_mask = result["pupil_mask"]
        self._sub_aps = result["sub_aps"].astype(np.float32, copy=False)
        self._ref_centroids_single = result["ref_centroids_single"].astype(np.float32, copy=False)
        self._ref_points = self._ref_centroids_single
       
        # update the unperturbed science canvas
        self.canvas_science_pupil.queue_image(result["ref_science_plot"])

        # update table values
        self._value_items[2].setText(f"{RAD2ARCSEC * 1000 * result['ref_fwhm']:.3f}")
        self._value_items[3].setText(f"{RAD2ARCSEC * 1000 * result['diff_lim']:.3f}")

        n_act = result["n_act"]
        self.act_select_spinbox.setRange(0, n_act - 1)
        self.act_select_slider.setRange(0, n_act - 1)

        self._schedule_update(self.act_select_spinbox.value())

        # Emit a single sensor-changed notification after (re)initialization.
        self.sensor_changed.emit(self.sensor)

    @Slot(object, object, int)
    def on_worker_finished(self, sensor, result, job_id):
        if job_id != self.job_id:
            return
        if sensor is not self.sensor:
            return

        if not isinstance(result, dict) or result.get("error"):
            self.busy_label.setText("Error")
            return

        self.busy_label.setText("Plotting")
        k = int(self.pending_index)

        centroids_single = result["centroids_single"]
        influence_map = result["influence_map"]
        science_plot = result["science_plot"]
        strehl = float(result["strehl"])
        img_fwhm = float(result["img_fwhm"])

        self.canvas_science.queue_image(science_plot)

        if self._pupil_mask is not None:
            self.canvas_overview.set_image_masked(influence_map, mask=self._pupil_mask)

        if self._ref_points is not None and self._sub_aps is not None:
            self.canvas_overview.set_points_and_quivers(
                self._ref_points + self.params["field_padding"],
                centroids_single + self.params["field_padding"],
            )

        self._value_items[0].setText(f"{strehl:.3f}")
        self._value_items[1].setText(f"{RAD2ARCSEC * 1000 * img_fwhm:.3f}")

        img = np.asarray(result["subap_image"])
        obsc = self.params.get("telescope_center_obscuration")
        key = (tuple(img.shape), obsc)

        mask = self._subap_mask_cache.get(key)
        if mask is None:
            if img.ndim == 2 and img.shape[0] == img.shape[1]:
                m = ut.Pupil_tools.generate_pupil(grid_size=int(img.shape[0]))
                mask = (np.asarray(m.get()) > 0)
            else:
                # Fallback: no masking for non-square stitched sensor images
                mask = None
            self._subap_mask_cache[key] = mask

        if mask is None:
            self.canvas_subap.queue_image(img)
        else:
            self.canvas_subap.set_image_masked(img, mask=mask)
        self.busy_label.setText("Waiting")
        
    def recalculate(self, params):
        """Queue a full recompute on the shared worker whenever this tab's config changes params."""
        self.job_id += 1
        enforce_config_types(params)

        # Apply only the sensor-specific changes from this table.
        self.params.update(params)
        self._apply_sensor_params_to_sensor()

        self.busy_label.setText("Recomputing…")

        # Full recompute uses merged (global + per-sensor) params.
        CalculateWorker.instance().request_recompute(self.sensor, dict(self.params))
        self._schedule_update(self.act_select_spinbox.value())

        self.sensor_changed.emit(self.sensor)


    def main_params_changed(self, params):
        """Apply *global* config changes without clobbering per-sensor values (e.g. wavelength)."""
        
        self.job_id += 1
        enforce_config_types(params)

        # Update only non-sensor-specific keys from the main/global config.
        for k, v in params.items():
            if k in SENSOR_PARAM_KEYS:
                continue
            self.params[k] = v

        # Keep the local sensor object consistent for any immediate UI reads.
        if self.params.get("grid_size") is not None:
            try:
                self.sensor.recompute(grid_size=int(self.params["grid_size"]))
            except Exception:
                self.sensor.grid_size = int(self.params["grid_size"])
        self.busy_label.setText("Recomputing…")


        # Recompute with merged params (ensures per-sensor wfs_lambda/sub_apertures are preserved).
        CalculateWorker.instance().request_recompute(self.sensor, dict(self.params))
        self._schedule_update(self.act_select_spinbox.value())
        self.sensor_changed.emit(self.sensor)

