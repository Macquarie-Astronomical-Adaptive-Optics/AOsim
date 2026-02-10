import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from scripts.phase_screen.infinite_vonkarman import LayeredInfinitePhaseScreen
from scripts.schedulerGPU import SimWorker
from scripts.utilities import Pupil_tools
from scripts.widgets.config_table import Config_table
from scripts.widgets.dual_list_selector import DualListSelector
from scripts.widgets.loading_label import DotLoadingLabel
from scripts.widgets.pgcanvas import PGCanvas
from scripts.widgets.popout_window import PopoutWindow
from scripts.widgets.wrap_tab import DetachableTabWidget
from scripts.window_tabs.reconstructor_tab import Reconstructor_tab
from scripts.window_tabs.sensor_view_tab import SensorView_tab

ARCSEC2RAD = np.pi / (180.0 * 3600.0)
RAD2ARCSEC = (180.0 * 3600.0) / np.pi


class Turbulence_tab(QWidget):
    active_changed = Signal(object)
    req_add_layer = Signal(object)
    req_step = Signal(bool)
    req_step_n = Signal(int, int)
    req_sensor_update = Signal(dict)
    req_overview_enabled = Signal(bool)
    req_emit_overview = Signal()
    req_visible_sensor = Signal(int)

    def __init__(self, config_dict: dict, sensors: dict):
        super().__init__()

        self.params = config_dict
        self.sensors = sensors
        self.initDone = False

        self.available_funcs = {"InfVonKarman": LayeredInfinitePhaseScreen}
        self.turbulence_screens = self._load_turbulence_configs()

        # pick first config for now
        screen = self.turbulence_screens[0]
        data = screen["data"]

        # split base kwargs vs layer list (IMPORTANT: avoids double-adding layers)
        layers = list(data.get("layers", []))
        base_kwargs = {k: v for k, v in data.items() if k != "layers"}
        N = int(base_kwargs["N"])

        self.turbWidgets: Dict[int, TurbulenceWidget] = {}
        self.layer_last: Dict[int, np.ndarray] = {}  # cache numpy layer frames

        # build widgets from layer configs (GUI side only)
        for i, layer in enumerate(layers):
            name = f"Layer {i+1}"
            self.turbWidgets[i] = TurbulenceWidget(name, screen["function"], layer)

        # sensors -> angles (radians) (used for debug output)
        thetas = [s.field_angle for s in self.sensors.values()]  # list of (theta_x, theta_y) rad
        ranges_m = [getattr(s, "gs_range_m", float("inf")) for s in self.sensors.values()]  # meters
        print("Loaded sensors at angles")
        for t in thetas:
            print(f" - ({t[0] * RAD2ARCSEC:.0f}, {t[1] * RAD2ARCSEC:.0f}) arcsec")

        # ---------- UI ----------
        main_layout = QHBoxLayout(self)

        # left: combined view
        left_frame = QFrame()
        left_frame.setFrameShape(QFrame.Box)
        left_frame.setLineWidth(1)
        left_layout = QVBoxLayout(left_frame)

        self.science_canvas = PGCanvas(show_colorbar=False)
        left_layout.addWidget(self.science_canvas)

        self.psf_r0_label = QLabel(
            f"Synthetic R0 total = {base_kwargs.get('r0_total', base_kwargs.get('r0', '...'))}"
        )
        left_layout.addWidget(self.psf_r0_label)

        self.layer_selector = DualListSelector(active=self.turbWidgets.values(), text_key="title")
        self.layer_selector.itemsChanged.connect(self.active_change)
        left_layout.addWidget(self.layer_selector)

        add_layer_btn = QPushButton("Add Layer")
        add_layer_btn.clicked.connect(self.add_layer)
        left_layout.addWidget(add_layer_btn)

        print("Active layers:")
        for w in self.layer_selector.active_items():
            print(" -", w.title)

        main_layout.addWidget(left_frame)

        # middle: controls + total view
        fmiddle = QFrame()
        fmiddle.setFrameShape(QFrame.Box)
        fmiddle.setLineWidth(1)
        middle_layout = QVBoxLayout(fmiddle)

        self.active_canvas = PGCanvas()
        self.active_canvas.set_colorbar(label="phase", units="rad")
        middle_layout.addWidget(self.active_canvas)

        control_popout = QFrame()
        control_popout.setFrameShape(QFrame.NoFrame)
        self.control_popout_layout = QVBoxLayout(control_popout)
        self.control_popout_window: Optional[PopoutWindow] = None

        self.control_frame = QFrame()
        self.control_frame.setFrameShape(QFrame.NoFrame)
        control_layout = QVBoxLayout(self.control_frame)

        self.scheduler_status = DotLoadingLabel("Creating turbulence screens")
        self.scheduler_status.start()
        control_layout.addWidget(self.scheduler_status)

        phase_b_layout = QHBoxLayout()
        control_layout.addLayout(phase_b_layout)

        self.next_button = QPushButton("Next Step")
        self.run_x_button = QPushButton("Run X Frames")
        self.spin = QSpinBox()
        self.spin.setRange(1, 100000)
        self.spin.setValue(10)
        self.run_inf_button = QPushButton("Run Indefinitely")
        self.stop_button = QPushButton("Pause")
        self.reset_button = QPushButton("Reset")

        self.control_popout_layout.addWidget(self.control_frame)
        middle_layout.addWidget(control_popout)

        phase_b_layout.addWidget(self.next_button)
        phase_b_layout.addWidget(self.run_x_button)
        phase_b_layout.addWidget(self.spin)
        phase_b_layout.addWidget(self.run_inf_button)
        phase_b_layout.addWidget(self.stop_button)
        phase_b_layout.addWidget(self.reset_button)

        main_layout.addWidget(fmiddle)

        # right: per-layer tabs
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
            sensors=sensors,
            ranges_m=ranges_m,
            patch_size_px=N,
            patch_M=N,
            dt_s=0.002,
            emit_psf=True,
        )
        self.scheduler.moveToThread(self.scheduler_thread)
        print("Scheduler moved to thread!")

        # build once thread starts, then emit a first frame
        self.req_overview_enabled.connect(self.scheduler.set_overview_enabled)
        self.req_emit_overview.connect(self.scheduler.emit_overview_once)
        self.req_visible_sensor.connect(self.scheduler.set_visible_sensor)
        self.req_sensor_update.connect(self.scheduler.sensor_update)
        self.req_add_layer.connect(self.scheduler.add_layer)
        self.req_step_n.connect(self.scheduler.step_n)
        self.req_step.connect(self.scheduler.step_once)

        self.scheduler_thread.started.connect(self.scheduler.build_sim)
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
        self.next_button.clicked.connect(lambda _: self.req_step.emit(True))
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
        self.overview_tab = SensorView_tab(
            sensors,
            config_dict,
            layers,
            N,
            self.params.get("telescope_diameter") / N,
            (N / 2, N / 2),
            N,
            self.scheduler,
        )
        self.scheduler.all_sensor_phase_ready.connect(self.overview_tab.psf_grid.update_psfs)
        self.scheduler.all_layers_ready.connect(self.overview_tab.layer_grid.update_layers)
        self.scheduler.sim_signal.connect(self.overview_tab.layer_grid.set_layers_cfg)

        self.scheduler.update_r0.connect(lambda r: self.psf_r0_label.setText(f"Synthetic R0 total: {r:.3f}"))

        # enable and render once (queued)
        self.req_overview_enabled.emit(True)
        self.req_emit_overview.emit()

        self.reconstructor_tab = Reconstructor_tab(config_dict, self.scheduler)
        self.scheduler.reconstructed_ready.connect(self.reconstructor_tab.update_view)
        self.scheduler.recon_matr_ready.connect(self.reconstructor_tab.update_mat_view)

    def _load_turbulence_configs(self) -> List[Dict[str, Any]]:
        screen_directory = Path(__file__).parent.parent.parent / "turbulence"

        configs: List[Dict[str, Any]] = []
        for file_path in sorted(screen_directory.iterdir()):
            if not (file_path.is_file() and file_path.suffix.lower() == ".json"):
                continue
            with file_path.open("r", encoding="utf-8") as f:
                content = json.load(f)

            fn_name = content.get("function")
            if fn_name not in self.available_funcs:
                raise KeyError(f"Unknown turbulence screen function: {fn_name!r} in {file_path.name}")

            content_name = file_path.stem
            content_data = {k: v for k, v in content.items() if k != "function"}
            configs.append(
                {
                    "name": content_name,
                    "data": content_data,
                    "function": self.available_funcs[fn_name],
                }
            )

        if not configs:
            raise FileNotFoundError(f"No turbulence configs found in {screen_directory}")

        return configs

    def add_layer(self) -> None:
        layer_params = self.turbulence_screens[0]["data"]["layers"][0].copy()

        i = len(self.turbWidgets)
        name = f"Layer {i+1}"
        layer_params["name"] = name
        layer_params["seed_offset"] = i + 1

        self.req_add_layer.emit(layer_params)

        turbWidget = TurbulenceWidget(name, self.turbulence_screens[0]["function"], layer_params)
        self.turbWidgets[i] = turbWidget
        self.layer_selector._add_item(self.layer_selector.available_list, turbWidget)
        self.tab_widget.addTab(turbWidget, name)
        turbWidget.screen_params_changed.connect(lambda p, idx=i: self.scheduler.apply_layer_params(idx, p))

    def active_change(self, items, active) -> None:
        self.scheduler.set_active([turbWidg.title for turbWidg in items], active)

    @Slot(dict)
    def sensor_update(self, sensors: dict) -> None:
        self.sensors = sensors
        self.req_sensor_update.emit(sensors)

    # ----------- UI slots -----------
    @Slot()
    def _run_x_clicked(self) -> None:
        n = int(self.spin.value())
        self.req_step_n.emit(n, 2)

    @Slot(int)
    def _tab_changed(self, idx: int) -> None:
        # request updates for just the visible layer tab
        self.scheduler.set_visible_layer(idx)

        # show cached layer immediately if we have it
        if idx in self.layer_last:
            self.turbWidgets[idx].update_screen(self.layer_last[idx])

    @Slot(object)
    def on_combined_frame(self, img_np: np.ndarray) -> None:
        # show combined in both canvases
        img_np = np.asarray(img_np)
        absol = float(np.max(np.abs(img_np))) if img_np.size else 0.0
        self.active_canvas.queue_image(img_np, cmap="seismic", levels=(-absol, absol), auto_levels=False)

    @Slot(int, object)
    def on_sensor_psf(self, idx: int, img_np: np.ndarray) -> None:
        # show on whatever canvas you want for that sensor
        self.science_canvas.queue_image(np.asarray(img_np), cmap="viridis", auto_levels=True)
        if not self.initDone:
            self.initDone = True

    @Slot(int, object)
    def on_layer_frame(self, idx: int, img_np: np.ndarray) -> None:
        idx = int(idx)
        self.layer_last[idx] = np.asarray(img_np)
        # only update visible tab widget (saves UI work)
        if idx == int(self.tab_widget.currentIndex()):
            self.turbWidgets[idx].update_screen(img_np)

    def showEvent(self, event):
        self.scheduler._detailed_enabled = True
        return super().showEvent(event)

    def hideEvent(self, event):
        if self.initDone:
            self.scheduler._detailed_enabled = False

        timer = getattr(self.scheduler, "timer", None)
        if timer is not None and timer.isActive():
            print("exiting page while sim is running")

            def on_close_callback(returned_widget, returned_title):
                self.control_popout_layout.addWidget(returned_widget)
                self.control_popout_window = None
                return

            self.control_popout_window = PopoutWindow(self.control_frame, "Sim Controls", on_close_callback)
            self.control_popout_window.show()

        return super().hideEvent(event)

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

    def __init__(self, title: str, function, params: dict, parent=None):
        super().__init__(parent)
        self.title = title
        self.function = function
        self.params = params
        self.current_screen = None

        tab_frame = QFrame(self)
        tab_layout = QVBoxLayout(tab_frame)

        self.turb_canvas = PGCanvas()
        self.turb_canvas.set_colorbar(label="phase", units="rad")
        tab_layout.addWidget(self.turb_canvas)

        self.config_table = Config_table(list(self.params.keys()), self.params)
        tab_layout.addWidget(self.config_table)
        self.config_table.params_changed.connect(self._on_params_changed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(tab_frame)

    @Slot(dict)
    def _on_params_changed(self, pc: dict) -> None:
        self.params = pc
        self.screen_params_changed.emit(pc)

    @Slot(object)
    def update_screen(self, new_screen) -> None:
        self.current_screen = new_screen
        self.turb_canvas.queue_image(new_screen, cmap="seismic", levels=(0, 255), auto_levels=True)

    def showEvent(self, event):
        super().showEvent(event)
        if self.current_screen is not None:
            self.turb_canvas.queue_image(self.current_screen, cmap="seismic", levels=(0, 255), auto_levels=True)
