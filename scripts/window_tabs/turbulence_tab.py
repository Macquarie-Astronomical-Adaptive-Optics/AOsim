import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import logging
logger = logging.getLogger(__name__)
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
from scripts.core.config_store import ConfigStore
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

    def __init__(self, config_dict: dict | ConfigStore, sensors: dict):
        super().__init__()

        self.config_store = config_dict if isinstance(config_dict, ConfigStore) else ConfigStore(dict(config_dict))
        self.params = self.config_store.data
        self.sensors = sensors
        self.initDone = False

        self.available_funcs = {"InfVonKarman": LayeredInfinitePhaseScreen}

        # Active turbulence profile is stored as either:
        #   - inline: params["turbulence"]["profile"]
        #   - file ref: params["turbulence"]["file"] (relative to project root)
        self.active_profile_name, data, self._turb_source = self._load_active_turbulence_profile()
        self._active_profile_data = data

        fn_name = data.get("function")
        if fn_name not in self.available_funcs:
            raise KeyError(f"Unknown turbulence screen function: {fn_name!r} in profile {self.active_profile_name!r}")
        screen_func = self.available_funcs[fn_name]

        # split base kwargs vs layer list (IMPORTANT: avoids double-adding layers)
        # Keep the *same* list object from config so edits/save/load stay in sync.
        layers = data.setdefault("layers", [])
        base_kwargs = {k: v for k, v in data.items() if k not in ("layers", "function")}
        N = int(base_kwargs["N"])

        self.turbWidgets: Dict[int, TurbulenceWidget] = {}
        self.layer_last: Dict[int, np.ndarray] = {}  # cache numpy layer frames

        # build widgets from layer configs (GUI side only)
        for i, layer in enumerate(layers):
            name = layer.get("name")
            if isinstance(layer, dict) and "name" not in layer:
                layer["name"] = name
            self.turbWidgets[i] = TurbulenceWidget(name, screen_func, layer, config_store=self.config_store)

        # sensors -> angles (radians) (used for debug output)
        thetas = [s.field_angle for s in self.sensors.values()]  # list of (theta_x, theta_y) rad
        ranges_m = [getattr(s, "gs_range_m", float("inf")) for s in self.sensors.values()]  # meters
        logger.info("Loaded sensors at angles")
        for t in thetas:
            logger.info(" - (%.0f, %.0f) arcsec", t[0]*RAD2ARCSEC, t[1]*RAD2ARCSEC)

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

        logger.info("Active layers:")
        for w in self.layer_selector.active_items():
            logger.info(" - %s", w.title)

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
        logger.info("Starting GPU Scheduler")
        self.scheduler_thread = QThread(self)

        logger.info("Building Turbulence Screen Generator")
        self.scheduler = SimWorker(
            sim_factory=screen_func,
            sim_kwargs=base_kwargs,
            layers=layers,
            pupil_mask=Pupil_tools.generate_pupil(N, self.params.get("telescope_center_obscuration")),
            sensors=sensors,
            ranges_m=ranges_m,
            patch_size_px=N,
            patch_M=N,
            dt_s=(1/self.params.get("frame_rate")),
            emit_psf=True,
            params=self.params,
        )
        self.scheduler.moveToThread(self.scheduler_thread)
        logger.info("Scheduler moved to thread!")

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
        self.scheduler.status_log.connect(lambda s: logger.info(s))
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
            w.screen_params_changed.connect(lambda p, idx=i: self._on_layer_params_changed(idx, p))

        self.scheduler_thread.start()

        # worker emits stacks
        self.overview_tab = SensorView_tab(
            sensors,
            self.params,
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

        self.reconstructor_tab = Reconstructor_tab(self.params, self.scheduler)
        self.scheduler.reconstructed_ready.connect(self.reconstructor_tab.update_view)
        self.scheduler.recon_matr_ready.connect(self.reconstructor_tab.update_mat_view)

        # Keep layer tables in sync on load.
        self.config_store.changed.connect(self._on_store_changed)

    def _on_store_changed(self) -> None:
        """Best-effort refresh when the central config is loaded."""
        try:
            # Update label if r0 changed etc (worker will also emit new values when running).
            pass
        except Exception:
            pass


    def _project_root(self) -> Path:
        # turbulence_tab.py -> window_tabs -> scripts -> <project root>
        return Path(__file__).resolve().parent.parent.parent

    def _resolve_turbulence_file(self, file_ref: str) -> Path:
        root = self._project_root()
        p = Path(str(file_ref))
        if not p.is_absolute():
            p = (root / p).resolve()
        return p

    def _scan_turbulence_folder(self) -> List[Path]:
        folder = self._project_root() / "turbulence"
        if not folder.exists():
            return []
        return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".json"])

    def _load_active_turbulence_profile(self) -> tuple[str, Dict[str, Any], str]:
        """Load the active turbulence profile.

        Config schema (active only):
            params["turbulence"] = {
                "source": "file", "file": "turbulence/Paranal.json", "name": "Paranal"
            }
        or:
            params["turbulence"] = {
                "source": "inline", "name": "Custom", "profile": {...}
            }

        Back-compat:
            params["turbulence_profiles"] + params["turbulence_active"]
        """

        # Back-compat migration (legacy multi-profile schema).
        if not isinstance(self.params.get("turbulence"), dict):
            profiles = self.params.get("turbulence_profiles")
            if isinstance(profiles, dict) and profiles:
                active = self.params.get("turbulence_active")
                if not isinstance(active, str) or active not in profiles:
                    active = next(iter(profiles.keys()))
                prof = profiles.get(active)
                if not isinstance(prof, dict):
                    prof = {}
                self.params["turbulence"] = {"source": "inline", "name": active, "profile": prof}
                self.params.pop("turbulence_profiles", None)
                self.params.pop("turbulence_active", None)
            else:
                self.params["turbulence"] = {}

        tcfg = self.params.get("turbulence")
        if not isinstance(tcfg, dict):
            tcfg = {}
            self.params["turbulence"] = tcfg

        source = str(tcfg.get("source") or "").strip().lower()

        # Inline profile
        prof_inline = tcfg.get("profile")
        if source == "inline" and isinstance(prof_inline, dict):
            name = str(tcfg.get("name") or "Inline")
            tcfg.setdefault("name", name)
            return name, prof_inline, "inline"

        # File reference profile
        file_ref = tcfg.get("file") or tcfg.get("path")
        if isinstance(file_ref, str) and file_ref.strip():
            path = self._resolve_turbulence_file(file_ref)
            if not path.exists():
                raise FileNotFoundError(f"Turbulence profile file not found: {path}")
            with path.open("r", encoding="utf-8") as f:
                prof = json.load(f)
            if not isinstance(prof, dict):
                raise TypeError(f"Turbulence profile JSON must be an object, got {type(prof)} in {path}")
            name = str(tcfg.get("name") or path.stem)
            # keep config normalized
            tcfg["source"] = "file"
            tcfg["file"] = str(file_ref)
            tcfg.setdefault("name", name)
            return name, prof, "file"

        # Fallback: first JSON in turbulence folder
        files = self._scan_turbulence_folder()
        if not files:
            raise FileNotFoundError(
                "No active turbulence profile configured and no turbulence/*.json files found."
            )
        path = files[0]
        # store a relative reference by default
        try:
            rel = str(path.relative_to(self._project_root()))
        except Exception:
            rel = str(path)
        tcfg["source"] = "file"
        tcfg["file"] = rel
        tcfg["name"] = path.stem

        with path.open("r", encoding="utf-8") as f:
            prof = json.load(f)
        if not isinstance(prof, dict):
            prof = {}
        return path.stem, prof, "file"

    def _ensure_inline_turbulence(self) -> None:
        """If the active profile is file-backed, materialize it inline so edits persist in config.json."""
        tcfg = self.params.get("turbulence")
        if not isinstance(tcfg, dict):
            tcfg = {}
            self.params["turbulence"] = tcfg

        if str(tcfg.get("source") or "").strip().lower() == "inline":
            return

        # Store the *same* dict object currently used by the UI/worker so future edits persist.
        tcfg.clear()
        tcfg.update({"source": "inline", "name": self.active_profile_name, "profile": self._active_profile_data})
        self._turb_source = "inline"
        try:
            self.config_store.changed.emit()
        except Exception:
            pass

    def _on_layer_params_changed(self, idx: int, params: dict) -> None:
        # Any edit should persist in the main config; switch to inline if needed.
        self._ensure_inline_turbulence()
        self.scheduler.apply_layer_params(idx, params)

    def _load_turbulence_profiles_from_disk(self) -> Dict[str, Dict[str, Any]]:
        """Load legacy turbulence/*.json profiles from disk.

        These get migrated into params["turbulence_profiles"].
        """
        screen_directory = Path(__file__).parent.parent.parent / "turbulence"

        profiles: Dict[str, Dict[str, Any]] = {}
        if not screen_directory.exists():
            return profiles

        for file_path in sorted(screen_directory.iterdir()):
            if not (file_path.is_file() and file_path.suffix.lower() == ".json"):
                continue
            with file_path.open("r", encoding="utf-8") as f:
                content = json.load(f)
            if not isinstance(content, dict):
                continue
            profiles[file_path.stem] = content

        return profiles

    def _ensure_turbulence_profiles(self) -> None:
        """Ensure turbulence configs live in the central config file.

        Migrates the legacy turbulence/*.json files into params["turbulence_profiles"].
        """
        profiles = self.params.get("turbulence_profiles")
        if isinstance(profiles, dict) and profiles:
            return

        migrated = self._load_turbulence_profiles_from_disk()
        if migrated:
            self.params["turbulence_profiles"] = migrated
            if "turbulence_active" not in self.params:
                self.params["turbulence_active"] = next(iter(migrated.keys()))
            try:
                self.config_store.changed.emit()
            except Exception:
                pass

    def add_layer(self) -> None:
        self._ensure_inline_turbulence()
        data = self._active_profile_data
        layers = data.setdefault("layers", [])
        fn_name = data.get("function")
        screen_func = self.available_funcs[fn_name]

        # Template from first layer if available
        if layers:
            template = dict(layers[0])
        else:
            template = {
                "name": "Layer 1",
                "altitude_m": 0.0,
                "r0": 0.25,
                "L0": 25.0,
                "wind": [0.0, 0.0],
                "seed_offset": 1,
            }

        i = len(self.turbWidgets)
        name = f"Layer {i+1}"
        new_layer = dict(template)
        new_layer["name"] = name
        new_layer["seed_offset"] = int(template.get("seed_offset", 0)) + i + 1
        layers.append(new_layer)

        # Notify worker + UI
        self.req_add_layer.emit(new_layer)

        turbWidget = TurbulenceWidget(name, screen_func, new_layer, config_store=self.config_store)
        self.turbWidgets[i] = turbWidget
        self.layer_selector._add_item(self.layer_selector.available_list, turbWidget)
        self.tab_widget.addTab(turbWidget, name)
        turbWidget.screen_params_changed.connect(lambda p, idx=i: self.scheduler.apply_layer_params(idx, p))

        try:
            self.config_store.changed.emit()
        except Exception:
            pass

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
            logger.warning("Exiting page while sim is running")

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

    def __init__(self, title: str, function, params: dict, parent=None, *, config_store: ConfigStore | None = None):
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

        self.config_table = Config_table(list(self.params.keys()), self.params, config_store=config_store)
        tab_layout.addWidget(self.config_table)
        self.config_table.params_changed.connect(self._on_params_changed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(tab_frame)

    @Slot(dict)
    def _on_params_changed(self, pc: dict) -> None:
        # Config_table emits a snapshot dict; keep our backing mapping object.
        try:
            self.params.update(pc)
        except Exception:
            pass
        self.screen_params_changed.emit(dict(self.params))

    @Slot(object)
    def update_screen(self, new_screen) -> None:
        self.current_screen = new_screen
        self.turb_canvas.queue_image(new_screen, cmap="seismic", levels=(0, 255), auto_levels=True)

    def showEvent(self, event):
        super().showEvent(event)
        if self.current_screen is not None:
            self.turb_canvas.queue_image(self.current_screen, cmap="seismic", levels=(0, 255), auto_levels=True)