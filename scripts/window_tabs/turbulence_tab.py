from PySide6.QtCore import Signal, Slot, QThread
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QWidget,
    QLabel, QFrame,
    QPushButton, QSizePolicy,
    QSpinBox,
)

import json
from pathlib import Path

from scripts.utilities import Pupil_tools
from scripts.widgets.config_table import Config_table
from scripts.widgets.wrap_tab import DetachableTabWidget
from scripts.widgets.pgcanvas import PGCanvas
from scripts.widgets.loading_label import DotLoadingLabel
from scripts.phase_screen.infinite_vonkarman import LayeredInfinitePhaseScreen
from scripts.schedulerGPU import SimWorker
from scripts.widgets.dual_list_selector import DualListSelector
from scripts.window_tabs.sensor_view_tab import SensorView_tab

import cupy as cp

ARCSEC2RAD = cp.pi / (180.0 * 3600.0)
RAD2ARCSEC = (180.0 * 3600.0) / cp.pi


class Turbulence_tab(QWidget):
    active_changed = Signal(object)
    req_add_layer = Signal(object)
    req_step = Signal(bool)
    req_step_n = Signal(int, int)
    req_overview_enabled = Signal(bool)
    req_emit_overview = Signal()
    req_visible_sensor = Signal(int)

    def __init__(self, config_dict, sensors):
        super().__init__()

        self.params = config_dict
        self.sensors = sensors

        self.available_funcs = {"InfVonKarman": LayeredInfinitePhaseScreen}

        screen_directory = Path(__file__).parent.parent.parent / "turbulence"

        # load turbulence configs
        self.turbulence_screens = []
        for file_path in screen_directory.iterdir():
            if file_path.is_file():
                with file_path.open("r") as f:
                    content = json.load(f)
                content_name = file_path.stem
                content_data = {k: v for k, v in content.items() if k != "function"}
                self.turbulence_screens.append({
                    "name": content_name,
                    "data": content_data,
                    "function": self.available_funcs[content["function"]],
                })

        # pick first config for now
        screen = self.turbulence_screens[0]
        data = screen["data"]

        # split base kwargs vs layer list (IMPORTANT: avoids double-adding layers)
        layers = list(data.get("layers", []))
        base_kwargs = {k: v for k, v in data.items() if k != "layers"}
        N = int(base_kwargs["N"])
        base_kwargs["N"] = N*3

        self.turbWidgets = {}
        self.layer_last = {}  # cache numpy layer frames

        # build widgets from layer configs (GUI side only)
        for i, layer in enumerate(layers):
            name = f"Layer {i+1}"
            turbWidget = TurbulenceWidget(name, screen["function"], layer)
            self.turbWidgets[i] = turbWidget

        # sensors -> angles (radians)
        thetas = [s.field_angle for s in self.sensors.values()]  # list of (theta_x, theta_y) rad
        ranges_m = [getattr(s, 'gs_range_m', float('inf')) for s in self.sensors.values()]  # meters
        print("Loaded sensors at angles")
        for t in thetas:
            print(f" - ({t[0]*RAD2ARCSEC:.0f}, {t[1]*RAD2ARCSEC:.0f}) arcsec")

        # ---------- UI ----------
        main_layout = QHBoxLayout(self)

        # left: combined view
        left_frame = QFrame()
        left_frame.setFrameShape(QFrame.Box)
        left_frame.setLineWidth(1)
        left_layout = QVBoxLayout(left_frame)

        self.science_canvas = PGCanvas()
        left_layout.addWidget(self.science_canvas)

        self.psf_r0_label = QLabel(f"Synthetic R0 total = {base_kwargs.get('r0_total', base_kwargs.get('r0', '...'))}")
        left_layout.addWidget(self.psf_r0_label)

        self.layer_selector = DualListSelector(active=self.turbWidgets.values(), text_key="title")
        self.layer_selector.itemsChanged.connect(self.active_change)

        left_layout.addWidget(self.layer_selector)

        add_layer_btn = QPushButton("Add Layer")
        add_layer_btn.clicked.connect(self.add_layer)

        left_layout.addWidget(add_layer_btn)

        print("Active layers: ")
        [print(" -", i.title) for i in self.layer_selector.active_items()]

        main_layout.addWidget(left_frame)

        # middle: controls + total view
        fmiddle = QFrame()
        fmiddle.setFrameShape(QFrame.Box)
        fmiddle.setLineWidth(1)
        middle_layout = QVBoxLayout(fmiddle)

        self.active_canvas = PGCanvas()
        middle_layout.addWidget(self.active_canvas)

        self.scheduler_status = DotLoadingLabel("Creating turbulence screens")
        self.scheduler_status.start()
        middle_layout.addWidget(self.scheduler_status)

        phase_b_layout = QHBoxLayout()
        middle_layout.addLayout(phase_b_layout)

        self.next_button = QPushButton("Next Step")
        self.run_x_button = QPushButton("Run X Frames")
        self.spin = QSpinBox()
        self.spin.setRange(1, 100000)
        self.spin.setValue(10)
        self.run_inf_button = QPushButton("Run Indefinitely")
        self.stop_button = QPushButton("Pause")
        self.reset_button = QPushButton("Reset")

        phase_b_layout.addWidget(self.next_button)
        phase_b_layout.addWidget(self.run_x_button)
        phase_b_layout.addWidget(self.spin)
        phase_b_layout.addWidget(self.run_inf_button)
        phase_b_layout.addWidget(self.stop_button)
        phase_b_layout.addWidget(self.reset_button)

        main_layout.addWidget(fmiddle)

        # right: per-layer tabs (your plan)
        right_layout = QVBoxLayout()
        fright_top = QFrame()
        fright_top.setFrameShape(QFrame.Box)
        fright_top.setLineWidth(1)
        right_top_layout = QVBoxLayout(fright_top)

        self.tab_widget = DetachableTabWidget()
        self.tab_widget.setMovable(True)

        for i in range(len(layers)):
            page = self.turbWidgets[i]
            self.tab_widget.addTab(page, page.title)

        right_top_layout.addWidget(self.tab_widget)
        right_layout.addWidget(fright_top)
        main_layout.addLayout(right_layout)
        
        # ---------- Worker thread ----------
        print("Starting GPU Scheduler")
        self.scheduler_thread = QThread(self)

        print("Building Turbulence Screen Generator")
        self.scheduler = SimWorker(
            sim_factory=screen["function"],
            sim_kwargs=base_kwargs,
            layers=layers,
            pupil_mask=Pupil_tools.generate_pupil(N, self.params.get("telescope_center_obscuration")),
            thetas_xy_rad=thetas,
            ranges_m=ranges_m,
            patch_size_px= N,
            patch_M= N,
            dt_s=0.001,
            emit_psf=True,
        )
        self.scheduler.moveToThread(self.scheduler_thread)
        print("Scheduler moved to thread!")


        # build once thread starts, then emit a first frame
        self.req_overview_enabled.connect(self.scheduler.set_overview_enabled)
        self.req_emit_overview.connect(self.scheduler.emit_overview_once)
        self.req_visible_sensor.connect(self.scheduler.set_visible_sensor)
        self.req_add_layer.connect(self.scheduler.add_layer)
        self.req_step_n.connect(self.scheduler.step_n)
        self.req_step.connect(self.scheduler.step_once)


        self.scheduler_thread.started.connect(self.scheduler.build_sim)
        # self.scheduler_thread.started.connect(self.scheduler.step_once)

        self.scheduler.finished.connect(self.scheduler_thread.quit)

        # signals -> UI
        self.scheduler.status_log.connect(print)
        self.scheduler.status.connect(self.scheduler_status.setBaseText)
        self.scheduler.status_running.connect(self.scheduler_status.run)

        self.scheduler.fpsReady.connect(lambda fps: self.setWindowTitle(f"Turbulence — {fps:.1f} FPS"))
        self.scheduler.combined_ready.connect(self.on_combined_frame)
        self.scheduler.layer_ready.connect(self.on_layer_frame)
        self.scheduler.sensor_psf_ready.connect(self.on_sensor_psf)

        self.req_visible_sensor.emit(0)


        # controls -> worker (queued automatically across threads)
        self.next_button.clicked.connect(lambda  a: self.req_step.emit(True))
        self.run_inf_button.clicked.connect(self.scheduler.start)
        self.stop_button.clicked.connect(self.scheduler.pause)
        self.reset_button.clicked.connect(self.scheduler.reset)

        self.run_x_button.clicked.connect(self._run_x_clicked)

        # tab changes => only update visible layer
        self.tab_widget.currentChanged.connect(self._tab_changed)

        # layer editor changes => apply safely in worker thread
        for i, w in self.turbWidgets.items():
            w.screen_params_changed.connect(lambda p, idx=i: self.scheduler.apply_layer_params(idx, p))

        self.scheduler_thread.start()

        # worker emits stacks
        self.overview_tab = SensorView_tab(sensors, config_dict, layers, N, self.params.get("telescope_diameter")/N, (N/2,N/2), N, self.scheduler)
        
        self.scheduler.all_sensor_psf_ready.connect(self.overview_tab.psf_grid.update_psfs)
        self.scheduler.all_layers_ready.connect(self.overview_tab.layer_grid.update_layers)
        self.scheduler.sim_signal.connect(self.overview_tab.layer_grid.set_layers_cfg)
        
        self.scheduler.update_r0.connect(lambda r: self.psf_r0_label.setText(f"Synthetic R0 total: {r:.3f}"))

        # enable and render once (queued)
        self.req_overview_enabled.emit(True)
        self.req_emit_overview.emit()

    def add_layer(self):
        layer_params = self.turbulence_screens[0]["data"]["layers"][0].copy()


        i = len(self.turbWidgets)
        name = f"Layer {i+1}"
        layer_params['name'] = name
        layer_params['seed_offset'] = i+1

        self.req_add_layer.emit(layer_params)

        turbWidget = TurbulenceWidget(name, self.turbulence_screens[0]["function"], layer_params)
        self.turbWidgets[i] = turbWidget
        self.layer_selector._add_item(self.layer_selector.available_list, turbWidget)
        self.tab_widget.addTab(turbWidget, name)
        turbWidget.screen_params_changed.connect(lambda p, idx=i: self.scheduler.apply_layer_params(idx, p))


    def active_change(self, items, active):
        self.scheduler.set_active([turbWidg.title for turbWidg in items], active)   

    # ----------- UI slots -----------
    @Slot()
    def _run_x_clicked(self):
        n = int(self.spin.value())
        self.req_step_n.emit(n,2)
        

    @Slot(int)
    def _tab_changed(self, idx: int):
        # request updates for just the visible layer tab
        self.scheduler.set_visible_layer(idx)

        # show cached layer immediately if we have it
        if idx in self.layer_last:
            self.turbWidgets[idx].update_screen(self.layer_last[idx])

    @Slot(object)
    def on_combined_frame(self, img_np):
        # show combined in both canvases
        self.active_canvas.queue_image(img_np, cmap="viridis", auto_levels=True)
    
    @Slot(int, object)
    def on_sensor_psf(self, idx, img_np):
        # show on whatever canvas you want for that sensor
        self.science_canvas.queue_image(img_np, cmap="viridis", auto_levels=True)


    @Slot(int, object)
    def on_layer_frame(self, idx: int, img_np):
        self.layer_last[int(idx)] = img_np
        # only update visible tab widget (saves UI work)
        if int(idx) == int(self.tab_widget.currentIndex()):
            self.turbWidgets[int(idx)].update_screen(img_np)

    def closeEvent(self, event):
        # stop scheduler and wait
        if self.scheduler is not None:
            self.scheduler.stop()
        if self.scheduler_thread is not None:
            self.scheduler_thread.quit()
            self.scheduler_thread.wait(3000)
        super().closeEvent(event)


class TurbulenceWidget(QWidget):
    screen_params_changed = Signal(dict)

    def __init__(self, title, function, params, parent=None):
        super().__init__(parent)
        self.title = title
        self.function = function
        self.params = params
        self.current_screen = None

        tab_frame = QFrame(self)
        self.tab_layout = QVBoxLayout(tab_frame)

        self.turb_canvas = PGCanvas()
        self.tab_layout.addWidget(self.turb_canvas)

        table_config_key = list(self.params.keys())
        self.config_table = Config_table(table_config_key, self.params)
        self.tab_layout.addWidget(self.config_table)

        self.config_table.params_changed.connect(self._on_params_changed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(tab_frame)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    @Slot(dict)
    def _on_params_changed(self, pc: dict):
        self.params = pc
        self.screen_params_changed.emit(pc)

    @Slot(object)
    def update_screen(self, new_screen):
        self.current_screen = new_screen
        self.turb_canvas.queue_image(new_screen, cmap="viridis", levels=(0, 255), auto_levels=True)

    def showEvent(self, event):
        super().showEvent(event)
        if self.current_screen is not None:
            self.turb_canvas.queue_image(self.current_screen, cmap="viridis", levels=(0, 255), auto_levels=True)
