from PySide6.QtCore import Qt, Signal, Slot, QThread, Signal, QMetaObject, Q_ARG
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QWidget,
    QLabel, QFrame,
    QListWidget, QPushButton, QSizePolicy,
    QListWidget, QSpinBox,
)

import json
from pathlib import Path

import scripts.utilities as ut
from scripts.widgets.config_table import Config_table
from scripts.widgets.pgcanvas import PGCanvas
from scripts.widgets.wrap_tab import DetachableTabWidget
from scripts.utilities import PhaseMap_tools, Pupil_tools
from scripts.schedulerGPU import GPUScheduler
from scripts.widgets.dual_list_selector import DualListSelector

from pyqtgraph import ImageItem
import cupy as cp
import queue

class Turbulence_tab(QWidget):
    request_add_layer  = Signal(str, object, dict)
    request_regenerate = Signal(str)
    request_active = Signal(str, bool)
    active_changed = Signal(object)

    def __init__(self, config_dict):
        super().__init__()
        
        self.params = config_dict
        self.active_screens = None
        self.sensors = {}

        # start GPU Scheduler
        print("Starting GPU Scheduler")
        self.scheduler_thread = QThread()
        self.scheduler = GPUScheduler(downsample_factor=1)
        self.scheduler.moveToThread(self.scheduler_thread)
        self.scheduler_thread.start()
        
        # connect scheduler signals to update tabs:
        self.scheduler.status.connect(lambda s: print(s))
        self.scheduler.frame_ready.connect(self.on_scheduler_frame)  # receives (name, np.ndarray)
        self.request_regenerate.connect(self.scheduler.regenerate_layer)  # receives (name, np.ndarray)
        self.request_add_layer.connect(self.scheduler.add_layer)  # queued
        self.request_active.connect(self.scheduler.set_active)  # queued

        print("GPU Scheduler started!")


        self.available_funcs = {
            "AOtools InfiniteKolmogorov": PhaseMap_tools.generate_phase_map
        } # add more in the future
        print("Available Turbulence Generators:")
        [print(" -",i) for i in list(self.available_funcs.keys())]

        screen_directory = Path(__file__).parent.parent/ "turbulence"

        # load turbulence screens from "/turbulence" folder
        print("Loading configs from /turbulence")
        self.turbulence_screens = []
        for file_path in screen_directory.iterdir():
            if file_path.is_file():   # skip subdirectories
                with file_path.open("r") as f:
                    content = json.load(f)       
                    content_name = file_path.stem
                    content_data = {k: v for k, v in content.items() if k != "function"}
                    print(" - Loaded:", content_name)
                    self.turbulence_screens.append({"name": content_name, "data": content_data, "function": self.available_funcs[content["function"]]})   

        # initalize tab/page for each loaded screen
        print("Starting turbulence widgets:")
        self.turbWidgets = {}
        for i, screen in enumerate(self.turbulence_screens):
            turbWidget = TurbulenceWidget(screen["function"], screen["data"], self.params, screen["name"])
            self.turbWidgets[screen["name"]] = turbWidget
            turbWidget.scheduler = self.scheduler
            turbWidget.request_regenerate.connect(self.scheduler.regenerate_layer)
            turbWidget.request_edit.connect(self.scheduler.edit_layer_param)
            turbWidget.request_active.connect(self.scheduler.set_active)  # queued
            turbWidget.screen_params_changed.connect(lambda pm: self.update_sensor_tabs(self.sensors))

            print(" -", turbWidget.title)

            self.request_add_layer.emit(screen["name"], screen["function"], screen["data"])
            # self.add_layer(screen["name"], screen["function"], screen["data"])
            

        print("Creating Turbulence tab layout")
        main_layout = QHBoxLayout(self)

        # left -- final psf viewer
        left_frame = QFrame()
        left_frame.setFrameShape(QFrame.Box)
        left_frame.setLineWidth(1)

        left_layout = QVBoxLayout(left_frame)
        self.science_canvas = PGCanvas()

        left_layout.addWidget(self.science_canvas)

        self.psf_r0_label = QLabel("Effective total R0 = ...")
        left_layout.addWidget(self.psf_r0_label)


        func_sel_layout = QHBoxLayout()

        self.func_selector = QListWidget()
        self.func_selector.addItems(list(self.available_funcs.keys()))
        self.func_selector.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        func_sel_layout.addWidget(self.func_selector)

        func_options_layout = QVBoxLayout()
        for i in range(5):
            func_options_layout.addWidget(QLabel(f"Placeholder {i}"))

        func_sel_layout.addLayout(func_options_layout)


        left_layout.addLayout(func_sel_layout)

        main_layout.addWidget(left_frame)

        # middle -- layered turbulence viewer
        fmiddle = QFrame()
        fmiddle.setFrameShape(QFrame.Box)
        fmiddle.setLineWidth(1)

        ## plot canvas
        middle_layout = QVBoxLayout(fmiddle)
        self.active_canvas = PGCanvas()

        middle_layout.addWidget(self.active_canvas)

        phase_b_layout = QHBoxLayout()
        middle_layout.addLayout(phase_b_layout)

        ## run/stop buttons
        self.next_button = QPushButton("Next Step")
        self.run_x_button = QPushButton("Run X Frames")
        self.run_inf_button = QPushButton("Run Indefinitely")
        self.spin = QSpinBox()
        self.spin.setRange(1, 1000)
        self.spin.setValue(10)
        self.stop_button = QPushButton("Stop")
        self.reset_button = QPushButton("Reset")

        phase_b_layout.addWidget(self.next_button)
        phase_b_layout.addWidget(self.run_x_button)
        phase_b_layout.addWidget(self.spin)
        phase_b_layout.addWidget(self.run_inf_button)
        phase_b_layout.addWidget(self.stop_button)
        phase_b_layout.addWidget(self.reset_button)

        self.next_button.clicked.connect(self.step_active)
        self.run_x_button.clicked.connect(self.run_x_active)
        self.run_inf_button.clicked.connect(self.run_active)
        self.stop_button.clicked.connect(self.stop_active)
        self.reset_button.clicked.connect(self.reset_active)
 
        ## list selectors for active/inactive screens
        print("Populating function selector")
        self.turb_selector = DualListSelector(text_key="title")
        self.turb_selector.activeChanged.connect(self.set_active)

        for i in list(self.turbWidgets.values())[1:]: 
            self.turb_selector._add_item(self.turb_selector.available_list, i)
        for i in list(self.turbWidgets.values())[:1]: 
            self.turb_selector._add_item(self.turb_selector.active_list, i)

        self.set_active(self.turb_selector.active_items())
        self.func_selector.itemDoubleClicked.connect(lambda it: self.turb_selector._add_item(self.turb_selector.available_list, self.load_turb(it.text())))

        middle_layout.addWidget(self.turb_selector)


        main_layout.addWidget(fmiddle)

 
        # right -- turbulence editor
        right_layout = QVBoxLayout()

        fright_top = QFrame()
        fright_top.setFrameShape(QFrame.Box)
        fright_top.setLineWidth(1)

        right_top_layout = QVBoxLayout(fright_top)
        self.tab_widget = DetachableTabWidget()
        self.tab_widget.setMovable(True)

        print("Adding subtabs")
        # Add tabs
        for i, page in enumerate(self.turbWidgets.values()):
            self.tab_widget.addTab(page, page.title)
            print(" -", page.title)
        
        right_top_layout.addWidget(self.tab_widget)

        right_layout.addWidget(fright_top)
        main_layout.addLayout(right_layout)

    @Slot(dict)
    def update_sensor_tabs(self, sensors):
        if sensors != self.sensors:
            self.sensors = sensors

        max_angle = 0.0
        for name, sensor in sensors.items():
            sen_angle = sensor.sen_angle
            if max_angle < sen_angle:
                max_angle = sen_angle

        for turb_tab in self.turbWidgets.values():
            extra = self.calc_extra_phase_pad(turb_tab, max_angle)
            turb_tab.set_extra_grid_size(int(extra.get()))

    def load_turb(self, func):
        content = None
        content_name = None
        content_data = None
        file_path = (Path(__file__).parent.parent/ "turbulence" / "defaults" / (str(func) + ".json"))
        if file_path.is_file():   # skip subdirectories
            with file_path.open("r") as f:
                content = json.load(f)       
                content_name = file_path.stem
                content_data = {k: v for k, v in content.items() if k != "function"}

        page = TurbulenceWidget(self.available_funcs[content["function"]], content_data, self.params, content_name)

        page = self.check_dup(page)

        self.turbulence_screens.append({"name": page.title, "data": page.params, "function": page.function})
        self.turbWidgets[page.title] = page

        max_angle = 0.0
        for name, sensor in self.sensors.items():
            sen_angle = sensor.sen_angle
            if max_angle < sen_angle:
                max_angle = sen_angle

        extra = self.calc_extra_phase_pad(page, max_angle)
        page.set_extra_grid_size(int(extra.get()))

        page.scheduler = self.scheduler
        page.request_regenerate.connect(self.scheduler.regenerate_layer)
        page.request_edit.connect(self.scheduler.edit_layer_param)
        page.request_active.connect(self.scheduler.set_active)  # queued
        page.screen_params_changed.connect(lambda pm: self.update_sensor_tabs(self.sensors))

        print("Adding new layer", page.title)

        self.request_add_layer.emit(page.title, page.function, page.params)
        self.tab_widget.addTab(page, page.title)

        return page
    
    def check_dup(self, new_page):
        existing_titles = [tp.title for tp in list(self.turbWidgets.values())]

        base_title = new_page.title
        counter = 1
        while new_page.title in existing_titles:
            new_page.title = f"{base_title} ({counter})"
            counter += 1

        return new_page
    
    def calc_extra_phase_pad(self, turb_tab, sensor_angle):
        return cp.ceil(cp.tan(sensor_angle) * turb_tab.params.get("altitude") / (self.params.get("telescope_diameter")/self.params.get("grid_size")))
    
    def set_active(self, a):
        for i in self.turbWidgets.values():
            QMetaObject.invokeMethod(self.scheduler, "set_science_active", Qt.QueuedConnection,
                Q_ARG(str, i.title),
                Q_ARG(bool, i in a)
                )
        QMetaObject.invokeMethod(self.scheduler, "redraw", Qt.QueuedConnection)
        self.active_screens = a
        self.active_changed.emit(self.active_screens)
        for i, sensor in enumerate(self.sensors.values()):
            for turbWidget in a:
                if not hasattr(turbWidget, "current_screen"):
                    print("WARN Turbulence_tab.on_scheduler_frame : No Turbulence widget found!")
                    return 

                if turbWidget.current_screen is None: continue

                extra = int(turbWidget.extra_grid_size)
            
                shift_x = int(self.calc_extra_phase_pad(turbWidget, -sensor.dx))
                shift_y = int(self.calc_extra_phase_pad(turbWidget, sensor.dy))
                
                right_pad = -extra+shift_x
                bottom_pad = -extra+shift_y

                if right_pad == 0: right_pad = turbWidget.current_screen.shape[0]
                if bottom_pad == 0: bottom_pad = turbWidget.current_screen.shape[1]

                turbWidget.draw_sensor_rect(Pupil_tools.generate_pupil(),shift_x,shift_y)

    # get currently active screens
    def get_active_screens(self):
        active = [
                    item.current_screen for item in self.turb_selector.active_items()
                    if hasattr(item, "current_screen")
                 ]
        return active
    
    def get_active_pages(self):
        active = [
                    item for item in self.turb_selector.active_items()
                    if hasattr(item, "current_screen")
                 ]
        return active
    
    def get_active_names(self):
        active = [
                    item.title for item in self.turb_selector.active_items()
                    if hasattr(item, "current_screen")
                 ]
        return active

    def regen_active(self):
        for name in self.get_active_names():
            self.request_regenerate.emit(name)

    def on_scheduler_frame(self, names, host_preview):

        science_screens = []
        active = self.get_active_names()

        for name in self.turbWidgets:
            turbWidget = self.turbWidgets[name]

            if not hasattr(turbWidget, "current_screen"):
                print("WARN Turbulence_tab.on_scheduler_frame : No Turbulence widget found!")
                return 
            
            if name in host_preview:
                new_screen = host_preview[name]

                turbWidget.current_screen = new_screen
                turbWidget.turb_canvas.set_image(new_screen)

            if not self.isVisible():
                continue

            if name not in active:
                continue

            new_screen = turbWidget.current_screen
            turbWidget.grid_size_display.setText(str(new_screen.shape))

            science_screens.append(self.center_crop(new_screen, crop_size=self.params.get("grid_size")))

        if not self.isVisible():
            return
        
        science_screens = cp.asarray(science_screens)

        if science_screens.shape[0] == 0:
            return
        else:
            total_active = cp.sum(science_screens, axis=0)
            
        self.active_canvas.set_image(total_active)

        if "science" in host_preview:
            self.science_canvas.set_image(host_preview["science"][1])
            r0s = cp.asarray([page.params.get("r0") for page in self.get_active_pages()])   # example r0 per layer
            r0 = (cp.sum(r0s**(-5/3)))**(-3/5)

            self.psf_r0_label.setText(f"Effective total R0 = {r0:.2f} cm")

    def center_crop(self, img, crop_size=512):
        H, W = img.shape[:2] 
        cy, cx = H // 2, W // 2  # center coordinates

        half = crop_size // 2
        # compute crop bounds
        y1, y2 = max(0, cy - half), min(H, cy + half)
        x1, x2 = max(0, cx - half), min(W, cx + half)

        return img[y1:y2, x1:x2]

        # for run/stop buttons
    def run_active(self):
        for page in self.turb_selector.active_items():
            page.request_active.emit(page.title, True)
        QMetaObject.invokeMethod(self.scheduler, "run_loop", Qt.QueuedConnection, Q_ARG(float, 0.03))

    def stop_active(self):
        for page in self.turb_selector.active_items():
            page.stop()

    def step_active(self):
        for page in self.turb_selector.active_items():
            page.request_active.emit(page.title, True)
        QMetaObject.invokeMethod(self.scheduler, "step_layer", Qt.QueuedConnection)

    def run_x_active(self, n):
        for page in self.turb_selector.active_items():
            page.request_active.emit(page.title, True)
        QMetaObject.invokeMethod(self.scheduler, "step_layer_N", Qt.QueuedConnection, 
                                 Q_ARG(int, int(self.spin.value())),
                                 Q_ARG(float, 0.03)
                                 )

    def reset_active(self):
        for page in self.turb_selector.active_items():
            page.reset()

    def closeEvent(self, event):
        # stop scheduler and wait
        self.scheduler.stop()
        self.scheduler_thread.quit()
        self.scheduler_thread.wait(3000)
        super().closeEvent(event)

    def showEvent(self, event):
        QMetaObject.invokeMethod(self.scheduler, "redraw", Qt.QueuedConnection)
        return super().showEvent(event)

   
class TurbulenceWidget(QWidget):
    new_frame = Signal(object) 
    screen_params_changed = Signal(dict)
    request_regenerate = Signal(str)
    request_edit = Signal(str, dict)
    request_active = Signal(str, bool)


    def __init__(self, function, params, defaults, title, parent=None):
        super().__init__(parent)
        self.function = function
        self.params = params
        self.title = title
        self.map_grid_size = defaults.get("grid_size")
        self.extra_grid_size = 0.0
        self.params["grid_size"] = self.map_grid_size + 2 * self.extra_grid_size
        self.params["px_scale"] = defaults.get("telescope_diameter")/self.params["grid_size"]

        self.current_screen = None
        self.pupil_overlay = None

        self.scheduler = None

        # layout
        tab_frame = QFrame(self)  
        self.tab_layout = QVBoxLayout(tab_frame)
        # screen view
        self.turb_canvas = PGCanvas()

        self.tab_layout.addWidget(self.turb_canvas)

        self.grid_size_display = QLabel(str(self.params.get("grid_size")))
        self.tab_layout.addWidget(self.grid_size_display)

        # --- Controls ---
        phase_b_layout = QHBoxLayout()
        self.tab_layout.addLayout(phase_b_layout)

        self.next_button = QPushButton("Next Step")
        self.run_x_button = QPushButton("Run X Frames")
        self.run_inf_button = QPushButton("Run Indefinitely")
        self.spin = QSpinBox()
        self.spin.setRange(1, 1000)
        self.spin.setValue(10)
        self.stop_button = QPushButton("Stop")
        self.reset_button = QPushButton("Reset")

        phase_b_layout.addWidget(self.next_button)
        phase_b_layout.addWidget(self.run_x_button)
        phase_b_layout.addWidget(self.spin)
        phase_b_layout.addWidget(self.run_inf_button)
        phase_b_layout.addWidget(self.stop_button)
        phase_b_layout.addWidget(self.reset_button)

        self.next_button.clicked.connect(self.page_next_button_clicked)
        self.run_x_button.clicked.connect(self.page_run_x_button_clicked)
        self.run_inf_button.clicked.connect(self.page_run_button_clicked)
        self.stop_button.clicked.connect(self.page_stop_button_clicked)
        self.reset_button.clicked.connect(self.page_reset_button_clicked)

        # --- Config table ---
        table_config_key = list(self.params.keys())
        self.config_table = Config_table(table_config_key, self.params)
        self.tab_layout.addWidget(self.config_table)

        self.config_table.params_changed.connect(lambda pc: setattr(self, "params", pc))
        self.config_table.params_changed.connect(lambda pc: self.request_edit.emit(self.title, pc))
        self.config_table.params_changed.connect(self.screen_params_changed.emit)
        self.config_table.params_changed.connect(self.page_reset_button_clicked)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(tab_frame)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    @Slot()
    def clear_sensor_rects(self):
        if self.pupil_overlay is not None:
            self.turb_canvas.vb.removeItem(self.pupil_overlay)
            self.pupil_overlay = None

    @Slot(object, int, int)
    def draw_sensor_rect(self, pupil, shift_x, shift_y):
        if self.current_screen is None:
            return
        
        self.clear_sensor_rects()

        if self.current_screen.shape > pupil.shape:
            extra = self.extra_grid_size

            right_pad = -extra+shift_x
            bottom_pad = -extra+shift_y

            if right_pad == 0: right_pad = self.current_screen.shape[0]
            if bottom_pad == 0: bottom_pad = self.current_screen.shape[1]

            rgba_mask = cp.zeros((self.current_screen.shape[0],self.current_screen.shape[1], 4))
            rgba_mask[(extra+shift_y):bottom_pad, (extra+shift_x):right_pad, 0][pupil == 1] = 0
            rgba_mask[(extra+shift_y):bottom_pad, (extra+shift_x):right_pad, 1][pupil == 1] = 0
            rgba_mask[(extra+shift_y):bottom_pad, (extra+shift_x):right_pad, 2][pupil == 1] = 1
            rgba_mask[(extra+shift_y):bottom_pad, (extra+shift_x):right_pad, 3][pupil == 1] = 0.5

            self.pupil_overlay = ImageItem(rgba_mask.get())
        

            self.pupil_overlay.setZValue(10)  # make sure it’s on top
            self.turb_canvas.vb.addItem(self.pupil_overlay)

        return
        


    # refresh viewer when tab/page is shown
    def showEvent(self, event):
        super().showEvent(event)
        
        if self.current_screen is not None:
            self.turb_canvas.set_image(self.current_screen)

    def page_next_button_clicked(self):
        # if hasattr(self, "_worker"):
        #     QMetaObject.invokeMethod(self._worker, "step", Qt.QueuedConnection)
        self.request_active.emit(self.title, True)
        QMetaObject.invokeMethod(self.scheduler, "step_layer", Qt.QueuedConnection)

        return # TODO

    def page_run_x_button_clicked(self):
        # if hasattr(self, "_worker"):
        #     for i in range(self.spin.value()):
        #         QMetaObject.invokeMethod(self._worker, "step", Qt.QueuedConnection)
        self.request_active.emit(self.title, True)
        QMetaObject.invokeMethod(self.scheduler, "step_layer_N", Qt.QueuedConnection, 
                                 Q_ARG(int, int(self.spin.value())),
                                 Q_ARG(float, 0.03)
                                 )

        return # TODO


    def page_run_button_clicked(self):
        # if hasattr(self, "_worker"):
        #     QMetaObject.invokeMethod(self._worker, "run", Qt.QueuedConnection)
        self.request_active.emit(self.title, True)
        QMetaObject.invokeMethod(self.scheduler, "run_loop", Qt.QueuedConnection, Q_ARG(float, 0.03))


        return # TODO


    def page_stop_button_clicked(self):
        # if hasattr(self, "_worker"):
        #     QMetaObject.invokeMethod(self._worker, "stop", Qt.QueuedConnection)
        self.request_active.emit(self.title, False)
        
        return # TODO


    def page_reset_button_clicked(self):
        # stop worker then recreate generator inside worker thread:
        # if hasattr(self, "_worker"):
        #     QMetaObject.invokeMethod(self._worker, "reset", Qt.QueuedConnection)
        self.request_regenerate.emit(self.title)


        return # TODO

    @Slot(int)
    def set_extra_grid_size(self, extra):
        if self.extra_grid_size == extra:
            return
        self.extra_grid_size = extra
        self.params["grid_size"] = self.map_grid_size +  2 * self.extra_grid_size
        self.request_edit.emit(self.title, self.params)
        # if hasattr(self, "_worker"):
        #     QMetaObject.invokeMethod(self._worker, "change_grid_size", Qt.QueuedConnection, Q_ARG(int, int(self.params["grid_size"])))
        self.grid_size_display.setText(str(self.params.get("grid_size")) + ", Generating new grid")




    # allow for main widget to activate threads
    @Slot()
    def stop(self):
        self.page_stop_button_clicked()

    @Slot()
    def step_once(self):
        self.page_next_button_clicked() 
        
    @Slot()
    def run(self):
        self.page_run_button_clicked()

    @Slot(int)
    def run_x(self, n):
        self.request_active.emit(self.title, True)
        QMetaObject.invokeMethod(self.scheduler, "step_layer_N", Qt.QueuedConnection, 
                                 Q_ARG(int, int(n)),
                                 Q_ARG(float, 0.03)
                                 )


    @Slot()
    def reset(self):
        self.page_reset_button_clicked() 

    # ---- kill page ----
    def kill(self):
        return # TODO

    def cleanup(self):
        try:
            self.new_frame.disconnect()
        except TypeError:
            pass



