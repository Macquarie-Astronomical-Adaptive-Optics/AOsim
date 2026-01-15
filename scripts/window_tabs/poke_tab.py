from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QWidget,
    QFrame,
)

import scripts.utilities as ut
from scripts.widgets.wrap_tab import DetachableTabWidget
from scripts.widgets.wfsensor_tab import SensorTabWidget
from scripts.widgets.config_table import Config_table

ARCSEC2RAD = 3.141592653589793 / (180.0 * 3600.0)
RAD2ARCSEC = (180.0 * 3600.0) / 3.141592653589793

class Poke_tab(QWidget):
    update_request = Signal(int, int)
    sensors_changed = Signal(dict)
    pupil_changed = Signal(float)

    def __init__(self, config_dict):
        super().__init__()
        self.params = config_dict
        ut.set_params(self.params)
        self.wfsensors = {}

        print("Creating sensors")
        off_x_px = (0.5 * self.params.get("telescope_diameter")) / (self.params.get("telescope_diameter") / self.params.get("grid_size"))
        self.wfsensors["main_sensor"] = ut.WFSensor_tools.ShackHartmann()
        self.wfsensors["test_sensor_right"] = ut.WFSensor_tools.ShackHartmann(
            90_000, 
            n_sub=20, 
            dx=10*60, 
            dy=0,
            lgs_thickness_m=10_000.0,     # ~10 km sodium thickness
            lgs_launch_offset_px=(off_x_px, 0.0),
            lgs_remove_tt=True
        )
        self.wfsensors["test_sensor_left"] = ut.WFSensor_tools.ShackHartmann(
            90_000, 
            n_sub=20, 
            dx=-10*60, 
            dy=0,
            lgs_thickness_m=10_000.0,    
            lgs_launch_offset_px=(-off_x_px, 0.0),
            lgs_remove_tt=True
        )
        self.wfsensors["test_sensor_up"] = ut.WFSensor_tools.ShackHartmann(
            90_000, 
            n_sub=20, 
            dx=0, 
            dy=10*60,
            lgs_thickness_m=10_000.0,     
            lgs_launch_offset_px=(0.0, off_x_px),
            lgs_remove_tt=True
        )
        self.wfsensors["test_sensor_down"] = ut.WFSensor_tools.ShackHartmann(
            90_000, 
            n_sub=20, 
            dx=0, 
            dy=-10*60,
            lgs_thickness_m=10_000.0,   
            lgs_launch_offset_px=( 0.0, -off_x_px),
            lgs_remove_tt=True
        )
        self.sensors_changed.emit(self.wfsensors)
        for i, j in self.wfsensors.items():
            print(" -", i, f": ({j.dx*RAD2ARCSEC:.0f}, {j.dy*RAD2ARCSEC:.0f}) arcsec")

        # layout for entire tab
        main_layout = QHBoxLayout(self)

        

        left_layout = QVBoxLayout()
        ## top left -- parameter configuration
        ftable = QFrame()
        ftable.setMaximumWidth(250)
        ftable.setMinimumWidth(218)

        ftable_layout = QVBoxLayout(ftable)
        table_config_key = ["telescope_diameter","telescope_center_obscuration","actuators","grid_size","poke_amplitude"]
        self.config_table = Config_table(table_config_key, self.params)
        
        ftable_layout.addWidget(self.config_table)

        left_layout.addWidget(ftable)

        main_layout.addLayout(left_layout)       


        # middle section
        main_middle_layout = QVBoxLayout()


        # sensor viewer
        sensor_selector_h = QHBoxLayout()
        sensor_tabs = DetachableTabWidget()
        sensor_tabs.setMovable(True)
        
        print("Starting sensor tab editor: ")
        self.tab_pages = []
        for key, val in self.wfsensors.items():
            
            tab = SensorTabWidget(dict(self.params), key, val)
            tab.sensor_changed.connect(self.update_sensor)

            self.tab_pages.append(tab)
            sensor_tabs.addTab(tab, key)
            
        sensor_selector_h.addWidget(sensor_tabs)
        self.config_table.params_changed.connect(ut.set_params)
        self.config_table.params_changed.connect(self.update_tabs)
        
        main_middle_layout.addLayout(sensor_selector_h)
        main_layout.addLayout(main_middle_layout)

        self.sensors_changed.emit(self.wfsensors)

    def update_tabs(self, params):
        # also update sensor view tab while we're at it
        self.pupil_changed.emit(params.get("telescope_center_obscuration"))
        for tab in self.tab_pages:
            tab.main_params_changed(params)

    def update_sensor(self, sensor):
        self.sensors_changed.emit(self.wfsensors)
            

        


