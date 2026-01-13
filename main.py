import sys
import json
from pathlib import Path
import argparse

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget
)

from scripts.window_tabs.poke_tab import Poke_tab
from scripts.window_tabs.turbulence_tab import Turbulence_tab
from scripts.window_tabs.sensor_view_tab import SensorView_tab

import cupy as cp
import sys

# Main application window
class MainWindow(QMainWindow):

    # load default configs
    with open(Path(__file__).parent / "config_default.json", "r") as f:
        config = json.load(f)

    # load command line arguments if any
    parser = argparse.ArgumentParser()
    parser.add_argument("--telescope_diameter", type=float)
    parser.add_argument("--telescope_center_obscuration", type=float)
    parser.add_argument("--wfs_lambda", type=float)
    parser.add_argument("--science_lambda", type=float)
    parser.add_argument("--r0", type=float)
    parser.add_argument("--L0", type=float)
    parser.add_argument("--Vwind", type=float)
    parser.add_argument("--actuators", type=int)
    parser.add_argument("--sub_apertures", type=int)
    parser.add_argument("--frame_rate", type=int)
    parser.add_argument("--grid_size", type=int)
    parser.add_argument("--field_padding", type=int)
    parser.add_argument("--poke_amplitude", type=float)
    parser.add_argument("--random_seed", type=int)
    parser.add_argument("--use_gpu", action="store_true")  # default False
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    parser.set_defaults(use_gpu=config.get("use_gpu", False))
    parser.add_argument("--data_path", type=str)
    args, unknown = parser.parse_known_args()


    # override default config with command line arguments
    params = config.copy()
    for key, value in vars(args).items():
        if value is not None:
            params[key] = value

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AOsim")
        cp.cuda.Device(0).use()
            
        print("> Starting!")
        tabs = QTabWidget()
        tabs.setMovable(True)

        print("> Starting Poke diagnostic tab building")
        tabs.addTab(poke := Poke_tab(self.params), "Poke Diagnostics")
        print("> Poke diagnostic tab done!")

        print("> Starting Turbulence tab building")
        tabs.addTab(turb := Turbulence_tab(self.params, poke.wfsensors), "Turbulence")
        # poke.sensors_changed.connect(turb.update_sensor_tabs)
        print("> Turbulence tab done!")
        
        print("> Starting SensorView tab building")
        tabs.addTab(sensview := turb.overview_tab, "SensorView")
        poke.pupil_changed.connect(sensview.updateLayerGrid)
        print("> SensorView tab done!")


        self.setCentralWidget(tabs)
        print("> MAIN init done!")

# start application
if __name__ == "__main__":
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())