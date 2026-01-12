from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot, QThread, Signal, Slot, QMetaObject, Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QWidget,
    QTableWidget, QTableWidgetItem, QSpinBox, QSlider, QLabel, QFrame,

)

import cupy as cp
import numpy as np

import scripts.utilities as ut
from scripts.wrap_tab import DetachableTabWidget
from scripts.wfsensor_tab import SensorTabWidget
from scripts.worker import CalculateWorker
from scripts.config_table import Config_table
from data.CONFIG_DTYPES import CONFIG_DTYPES, enforce_config_types

ARCSEC2RAD = cp.pi / (180.0 * 3600.0)
RAD2ARCSEC = (180.0 * 3600.0) / cp.pi

class Poke_tab(QWidget):
    update_request = Signal(int, int)
    sensors_changed = Signal(dict)

    def __init__(self, config_dict):
        super().__init__()
        self.params = config_dict
        self.wfsensors = {}

        print("Creating sensors")
        self.wfsensors["main_sensor"] = ut.WFSensor_tools.ShackHartmann()
        self.wfsensors["test_sensor_right"] = ut.WFSensor_tools.ShackHartmann(n_sub=20, dx=45, dy=0)
        self.wfsensors["test_sensor_left"] = ut.WFSensor_tools.ShackHartmann(n_sub=20, dx=-45, dy=0)
        self.wfsensors["test_sensor_up"] = ut.WFSensor_tools.ShackHartmann(n_sub=20, dx=0, dy=45)
        self.wfsensors["test_sensor_down"] = ut.WFSensor_tools.ShackHartmann(n_sub=20, dx=0, dy=-45)
        self.sensors_changed.emit(self.wfsensors)
        [print(" -", i, f": ({j.dx*RAD2ARCSEC:.0f}, {j.dy*RAD2ARCSEC:.0f}) arcsec") for i,j in self.wfsensors.items()]

        # calculation jobs bookkeeping
        self.job_id = 0
        self.pending_index = 0

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
            print(" -", key)
            tab = SensorTabWidget(dict(self.params), key, val)
            tab.sensor_changed.connect(self.update_sensor)

            self.tab_pages.append(tab)
            sensor_tabs.addTab(tab, key)
            
        sensor_selector_h.addWidget(sensor_tabs)
        self.config_table.params_changed.connect(self.update_tabs)
        
        main_middle_layout.addLayout(sensor_selector_h)
        main_layout.addLayout(main_middle_layout)

        self.sensors_changed.emit(self.wfsensors)

    def update_tabs(self, params):
        for tab in self.tab_pages:
            tab.main_params_changed(params)

    def update_sensor(self, sensor):
        self.sensors_changed.emit(self.wfsensors)
            

        


