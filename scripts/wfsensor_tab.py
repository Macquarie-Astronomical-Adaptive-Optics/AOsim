# sensor_tab_widget.py

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFrame, QLabel,
    QSpinBox, QSlider, QTableWidget, QTableWidgetItem
)
import cupy as cp
import numpy as np

import scripts.utilities as ut
from scripts.pgcanvas import PGCanvas
from scripts.worker import CalculateWorker  # singleton shared worker
from scripts.config_table import Config_table
from data.CONFIG_DTYPES import enforce_config_types


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

        params["sub_apertures"] = self.sensor.n_sub
        params["wfs_lambda"] = self.sensor.wavelength 


        self.job_id = 0
        self.pending_index = 0

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
        worker.instance().request_initialization(self.sensor, self.params)

    def _build_ui(self):
        main_layout = QHBoxLayout(self)

        # Left: config + busy + table
        left_v = QVBoxLayout()
        config_keys = ["sub_apertures","wfs_lambda"]
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
        self.sensor_changed.emit(self.sensor)

    def _emit_worker_request(self):
        self.job_id += 1
        self.actuator_changed.emit(int(self.pending_index), self.job_id)
        self.busy_label.setText("Waiting")

    @Slot(object)
    def on_calculation_finished(self, sensor, result):
        if sensor is not self.sensor:
            return
    
        self.busy_label.setText("Calculation Finished")
        self.pupil = result["pupil"]
        self.act_centers = result["act_centers"]
        self.influence_maps = result["influence_maps"]
        self.active_sub_aps = result["active_sub_aps"]
        self.sub_aps = result["sub_aps"]
        self.sub_aps_idx = result["sub_aps_idx"]
        self.sub_ap_width = result["sub_ap_width"]
        self.sub_slice = result["sub_slice"]
        self.sub_pupils = result["sub_pupils"]
        self.ref_centroids = result["ref_centroids"]
        self.ref_images = result["ref_images"]
        self.ref_science_image = result["ref_science_image"]
        self.ref_strehl = result["ref_strehl"]
        self.normalized_image = result["normalized_image"]
        self.science_image_plot = result["science_image_plot"]
       
        # update the unperturbed science canvas
        self.canvas_science_pupil.set_image(result["science_image_plot"])

        # update table values
        self.calc_values_table.setItem(2, 1, QTableWidgetItem(f"{result['ref_fwhm']*1e3:.3e}"))
        self.calc_values_table.setItem(3, 1, QTableWidgetItem(f"{result['diff_lim']*1e3:.3e}"))

        n_act = result["n_act"]
        self.act_select_spinbox.setRange(0, n_act - 1)
        self.act_select_slider.setRange(0, n_act - 1)

        self._schedule_update(self.act_select_spinbox.value())

    @Slot(object, int)
    def on_worker_finished(self, sensor, result, job_id):
        if job_id != self.job_id:
            return
        if sensor is not self.sensor:
            return

        enforce_config_types(self.params)


        self.busy_label.setText("Plotting")
        k = int(self.pending_index)
        centroids, subap_images = result
        centroids = centroids - self.params.get("field_padding")
        centroids_single = centroids.squeeze(0)
        
        ref_centroids_single = (self.ref_centroids - self.params.get("field_padding")).squeeze(0)

        influence_phase_scale = 4 * cp.pi / float(self.params.get("science_lambda"))
        influence_map_phase = self.influence_maps[k] * influence_phase_scale

        science_img, strehl = ut.Analysis.generate_science_image(self.pupil, influence_map_phase)
        self.science_image = science_img
        self.strehl = strehl
        self.normalized_image = self.science_image / self.science_image.sum()
        self.science_image_plot = cp.log10(self.normalized_image + 1e-12)
        self.canvas_science.set_image(self.science_image_plot)

        self.canvas_overview.set_image_masked(self.influence_maps[k].get(), mask=self.pupil.get())
        self.canvas_overview.set_points_and_quivers(
            ref_centroids_single + self.sub_aps,
            centroids_single + self.sub_aps 
        )

        plate_rad = (self.params.get("science_lambda") * self.params.get("grid_size")
                     / (self.params.get("telescope_diameter") * self.science_image.shape[0]))
        img_fwhm = plate_rad * ut.Analysis.fwhm_radial(self.science_image)
        self.calc_values_table.setItem(0, 1, QTableWidgetItem(f"{strehl:.3f}"))
        self.calc_values_table.setItem(1, 1, QTableWidgetItem(f"{1e3 * img_fwhm:.3e}"))

        if isinstance(subap_images, cp.ndarray):
            img = subap_images.get()
        else: 
            img = np.asarray(subap_images)
        self.canvas_subap.set_image_masked(img, mask=ut.Pupil_tools.generate_pupil(img.shape[0]))
        self.busy_label.setText("Waiting")

    def recalculate(self, params):
        """Queue a full recompute on the shared worker whenever this tab's config changes params."""
        self.job_id += 1
        self.sensor.recompute(n_sub=int(params.get("sub_apertures")), wavelength=float(params.get("wfs_lambda")))

        self.busy_label.setText("Recomputing…")
        
        CalculateWorker.instance().request_recompute(self.sensor, params)
        self._schedule_update(self.act_select_spinbox.value())
        

    @Slot(object)
    def main_params_changed(self, params):
        """Queue a full recompute on the shared worker whenever main window config table changes params"""
        self.job_id += 1
        self.sensor.recompute(grid_size=int(params.get("grid_size")))

        self.busy_label.setText("Recomputing…")
        
        CalculateWorker.instance().request_recompute(self.sensor, params)
        self._schedule_update(self.act_select_spinbox.value())

