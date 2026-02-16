"""AOsim GUI entrypoint.

This file is intentionally small: it loads config, sets global defaults (used by many
modules), then starts the PySide6 GUI.

Examples
--------
python main.py
python main.py --config my_config.json
python main.py --no-gpu
python main.py --gpu-device 1
"""

from __future__ import annotations

import argparse
import json
import logging

from scripts.core.logging_utils import setup_logging, env_log_level
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import QTimer, Slot
from PySide6.QtWidgets import QApplication, QMainWindow

from scripts.utilities import set_params
from scripts.core.config_store import ConfigStore
from scripts.widgets.wrap_tab import DetachableTabWidget
from scripts.window_tabs.log_console import add_console
from scripts.window_tabs.long_run_tab import LongRun_tab
from scripts.window_tabs.loop_tab import Loop_tab
from scripts.window_tabs.poke_tab import Poke_tab
from scripts.window_tabs.turbulence_tab import Turbulence_tab

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config_ultimate.json")



def _read_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a config JSON file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    try:
        cfg = _read_json(config_path)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse config JSON: {config_path}: {e}") from e
    if not isinstance(cfg, dict):
        raise TypeError(f"Config must be a JSON object/dict, got {type(cfg)}")
    return cfg


def build_arg_parser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    """Build CLI parser.

    We only expose a small subset of parameters as explicit flags; everything else
    should be changed via the JSON config.
    """
    p = argparse.ArgumentParser(description='AOsim')

    p.add_argument(
        '--config',
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help='Path to a config JSON file (default: config_ultimate.json)',
    )

    # Common overrides
    p.add_argument('--telescope_diameter', type=float)
    p.add_argument('--telescope_center_obscuration', type=float)
    p.add_argument('--wfs_lambda', type=float)
    p.add_argument('--science_lambda', type=float)
    p.add_argument('--actuators', type=int)
    p.add_argument('--frame_rate', type=int)
    p.add_argument('--grid_size', type=int)
    p.add_argument('--field_padding', type=int)
    p.add_argument('--poke_amplitude', type=float)
    p.add_argument('--random_seed', type=int)
    p.add_argument('--data_path', type=str)

    # GPU controls
    p.add_argument('--gpu-device', type=int, default=0, help='CUDA device index')
    p.add_argument('--use_gpu', action='store_true')
    p.add_argument('--no-gpu', dest='use_gpu', action='store_false')
    p.set_defaults(use_gpu=bool(defaults.get('use_gpu', False)))

    # Developer / UI helpers
    p.add_argument('--log-console', action='store_true', default=False, help='Show Qt log console dock')

    return p


def merge_config(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in overrides.items():
        if v is not None:
            out[k] = v
    return out


def init_gpu(use_gpu: bool, device_index: int, logger: Optional[logging.Logger] = None) -> None:
    """Select GPU device if requested.

    This keeps CuPy optional at import-time: if you run with --no-gpu you won't
    crash on systems without CUDA.
    """
    if not use_gpu:
        return

    try:
        import cupy as cp  # local import

        cp.cuda.Device(int(device_index)).use()
        if logger:
            logger.info('Using CUDA device %s', device_index)
    except Exception as e:
        msg = f"Failed to initialize GPU device {device_index}: {e}"
        if logger:
            logger.error(msg)
        else:
            print(msg)


class MainWindow(QMainWindow):
    def __init__(self, params: Dict[str, Any], *, log_console: bool = False, gpu_device: int = 0):
        super().__init__()
        # Central shared config store. We keep one dict instance alive and mutate it
        # in-place on load, so all tabs/widgets stay in sync.
        self.config_store = ConfigStore(dict(params))
        self.params = self.config_store.data

        self.setWindowTitle('AOsim')

        # Make config available to modules that still read scripts.utilities.params
        set_params(self.params)

        self.logger: Optional[logging.Logger] = None
        if log_console:
            # Redirects stdout/stderr to the dock.
            self.logger = add_console(self)

        init_gpu(bool(self.params.get('use_gpu', False)), gpu_device, self.logger)

        # Keep references so we can fully rebuild the UI when a new config.json is loaded.
        self._tabs_widget: DetachableTabWidget | None = None
        self._poke_tab: Poke_tab | None = None
        self._turb_tab: Turbulence_tab | None = None
        self._rebuild_pending = False

        # When a new config is loaded from disk, rebuild everything so all derived
        # components (sensors, turbulence, reconstructor, etc.) recompute from the
        # new configuration.
        if hasattr(self.config_store, "loaded"):
            self.config_store.loaded.connect(self._on_config_loaded)

        self._build_tabs()

    def _build_tabs(self) -> None:
        """(Re)create all tabs from the current config_store."""
        tabs = DetachableTabWidget()
        tabs.setMovable(True)

        # --- Tabs ---
        poke = Poke_tab(self.config_store)
        tabs.addTab(poke, 'Poke Diagnostics')

        # If core AO geometry changes (diameter/actuators/grid), rebuild the
        # turbulence + reconstructor pipeline so dimensions stay consistent.
        try:
            poke.hard_geometry_changed.connect(self._on_hard_geometry_changed)
        except Exception:
            pass

        turb = Turbulence_tab(self.config_store, poke.wfsensors)
        # Keep turbulence in sync if sensors are replaced (e.g. after config load).
        try:
            poke.sensors_changed.connect(turb.sensor_update)
        except Exception:
            pass
        tabs.addTab(turb, 'Turbulence')

        sensview = turb.overview_tab
        tabs.addTab(sensview, 'SensorView')
        poke.pupil_changed.connect(sensview.updateLayerGrid)

        tabs.addTab(turb.reconstructor_tab, 'Reconstructor')
        tabs.addTab(Loop_tab(self.config_store, turb.scheduler), 'Loop')
        tabs.addTab(LongRun_tab(self.config_store, turb.scheduler), 'Long Run')

        # Swap into the window.
        old = self.centralWidget()
        self.setCentralWidget(tabs)
        if old is not None:
            old.deleteLater()

        # Update refs.
        self._tabs_widget = tabs
        self._poke_tab = poke
        self._turb_tab = turb

    @Slot(dict)
    def _on_hard_geometry_changed(self, info: dict) -> None:
        # Avoid re-entrancy if multiple changes happen quickly.
        if self._rebuild_pending:
            return
        self._rebuild_pending = True

        # Ensure module-level defaults update (some code still reads scripts.utilities.params).
        set_params(self.params)

        def _do():
            try:
                self._close_all_popouts()
                self._shutdown_workers()
            finally:
                self._build_tabs()
                self._rebuild_pending = False

        QTimer.singleShot(0, _do)

    def _shutdown_workers(self) -> None:
        """Best-effort shutdown for worker threads/timers before rebuilding."""
        turb = self._turb_tab
        if turb is None:
            return
        try:
            if getattr(turb, "scheduler", None) is not None:
                turb.scheduler.stop()
        except Exception:
            pass

    def _close_all_popouts(self) -> None:
        """Close any detached (popped-out) tab windows before rebuilding UI."""
        root = getattr(self, "_tabs_widget", None)
        turb = self._turb_tab
        if root is None:
            return

        try:
            # Close popouts on the root tab widget first.
            if isinstance(root, DetachableTabWidget):
                root.close_popouts()
        except Exception:
            pass

        # Also close popouts on any nested detachable tab widgets (e.g. sensor pages).
        try:
            for tw in root.findChildren(DetachableTabWidget):
                try:
                    tw.close_popouts()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            th = getattr(turb, "scheduler_thread", None)
            if th is not None:
                th.quit()
                th.wait(3000)
        except Exception:
            pass

    @Slot(str)
    def _on_config_loaded(self, path: str) -> None:
        # Avoid re-entrancy if multiple loads fire quickly.
        if self._rebuild_pending:
            return
        self._rebuild_pending = True

        # Ensure module-level defaults update (some code still reads scripts.utilities.params).
        set_params(self.params)

        # Defer the heavy UI rebuild to the next event loop tick so we don't
        # rebuild while a file dialog / slot stack is still unwinding.
        def _do():
            try:
                # Ensure detached windows don't keep old widgets/threads alive.
                self._close_all_popouts()
                self._shutdown_workers()
            finally:
                self._build_tabs()
                self._rebuild_pending = False

        QTimer.singleShot(0, _do)


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Logging (root logger). LOG_LEVEL env var can override.
    setup_logging(env_log_level())

    # Logging (respects LOG_LEVEL env var; default INFO)
    setup_logging(env_log_level())


    # Parse args (including --config first)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=str, default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre.parse_known_args(argv)
    config_path = Path(pre_args.config)

    cfg = load_config(config_path)
    parser = build_arg_parser(cfg)
    args = parser.parse_args(argv)

    overrides = vars(args).copy()
    # Keep parsed config path out of params
    overrides.pop('config', None)

    params = merge_config(cfg, overrides)

    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(params, log_console=bool(args.log_console), gpu_device=int(args.gpu_device))
    win.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())