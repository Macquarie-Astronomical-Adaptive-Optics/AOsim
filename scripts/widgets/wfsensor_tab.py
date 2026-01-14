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
        for key, val in self.sensor.__dict__.items():
            self.params[key] = val

        self.params["dx"] = self.sensor.dx * RAD2ARCSEC
        self.params["dy"] = self.sensor.dy * RAD2ARCSEC
        print(" -", self.sensor_name)

        params["sub_apertures"] = self.sensor.n_sub
        params["wfs_lambda"] = self.sensor.wavelength 
        print("    ", self.sensor.n_sub, "sub apertures")
        print("     at wavelength/lambda:", self.sensor.wavelength, "m")

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
        self.calc_values_table.setItem(1, 0, QTableWidgetItem("Poke FWHM (mrad)"))
        self.calc_values_table.setItem(2, 0, QTableWidgetItem("Unperturbed FWHM (mrad)"))
        self.calc_values_table.setItem(3, 0, QTableWidgetItem("Diffraction Limit (mrad)"))

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
        self.canvas_subap = PGCanvas()
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
        self.canvas_science = PGCanvas()
        sc_v.addWidget(self.canvas_science)
        bottom_h.addWidget(f_science)

        f_unpert = QFrame()
        f_unpert.setFrameShape(QFrame.Box)
        f_unpert.setLineWidth(1)
        up_v = QVBoxLayout(f_unpert)
        up_v.addWidget(QLabel("Unperturbed Science Image"))
        self.canvas_science_pupil = PGCanvas()
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

        # Cache CPU-side plotting arrays
        self._pupil_mask = result["pupil_mask"]
        self._sub_aps = result["sub_aps"].astype(np.float32, copy=False)
        self._ref_centroids_single = result["ref_centroids_single"].astype(np.float32, copy=False)
        self._ref_points = self._ref_centroids_single + self._sub_aps
       
        # update the unperturbed science canvas
        self.canvas_science_pupil.set_image(result["ref_science_plot"])

        # update table values
        self._value_items[2].setText(f"{result['ref_fwhm']*1e3:.3e}")
        self._value_items[3].setText(f"{result['diff_lim']*1e3:.3e}")

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

        self.canvas_science.set_image(science_plot)

        if self._pupil_mask is not None:
            self.canvas_overview.set_image_masked(influence_map, mask=self._pupil_mask)

        if self._ref_points is not None and self._sub_aps is not None:
            self.canvas_overview.set_points_and_quivers(
                self._ref_points,
                centroids_single + self._sub_aps,
            )

        self._value_items[0].setText(f"{strehl:.3f}")
        self._value_items[1].setText(f"{1e3 * img_fwhm:.3e}")

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
            self.canvas_subap.set_image(img)
        else:
            self.canvas_subap.set_image_masked(img, mask=mask)
        self.busy_label.setText("Waiting")

    def recalculate(self, params):
        """Queue a full recompute on the shared worker whenever this tab's config changes params."""
        self.job_id += 1
        enforce_config_types(params)
        self.sensor.recompute(
            **params
            )

        self.busy_label.setText("Recomputing…")
        
        CalculateWorker.instance().request_recompute(self.sensor, params)
        self._schedule_update(self.act_select_spinbox.value())

        self.sensor_changed.emit(self.sensor)
        

    @Slot(object)
    def main_params_changed(self, params):
        """Queue a full recompute on the shared worker whenever main window config table changes params"""
        self.job_id += 1
        enforce_config_types(params)
        self.params.update(params)
        self.sensor.recompute(grid_size=int(params.get("grid_size")))

        self.busy_label.setText("Recomputing…")
        
        CalculateWorker.instance().request_recompute(self.sensor, params)
        self._schedule_update(self.act_select_spinbox.value())

        self.sensor_changed.emit(self.sensor)

