import time
from PySide6.QtCore import QObject, Signal, Slot, QTimer
# NOTE: import cupy inside the scheduler thread methods only

class GPUScheduler(QObject):
    # emits (layer_name: str, host_preview: np.ndarray uint8)
    frame_ready = Signal(object, dict)
    status = Signal(str)

    def __init__(self, downsample_factor=4):
        super().__init__()
        self.layers = {}  # name -> dict with GPU arrays and params
        self.downsample_factor = downsample_factor
        self.last_draw = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.step)

        self.step_n_num = None

    @Slot(str, object, dict)
    def add_layer(self, name, layer_function, layer_params):
        # run in scheduler thread; do CuPy allocations here
        import cupy as cp
        self.status.emit(f"Adding layer {name}")

        # initialize GPU array and RNG for each layer
        generator, next_frame = layer_function(**layer_params)
        self.layers[name] = {
            "next_frame": next_frame,
            "generator": generator,
            "active": False,
            "function": layer_function,
            "layer_params": layer_params
        }
        host_preview = self._gpu_to_preview_host(next_frame(), self.downsample_factor)
        self.frame_ready.emit([name], {name : host_preview})
        self.last_draw = {name : host_preview}

    @Slot(str, dict)
    def edit_layer_param(self, name, layer_params):
        if name in self.layers:
            self.layers[name]["layer_params"] = layer_params
            self.status.emit(f"Changed params of layer {name}")

    @Slot(str)
    def remove_layer(self, name):
        if name in self.layers:
            del self.layers[name]
            self.status.emit(f"Removed layer {name}")

    @Slot(str, bool)
    def set_active(self, name, is_active):
        if name in self.layers:
            self.layers[name]["active"] = is_active

    @Slot(str)
    def regenerate_layer(self, name):
        import cupy as cp
        if name not in self.layers:
            print("[WARN] GPU Scheduler.regenerate_layer :", name, "not in scheduler layers!")
            return
        self.status.emit(f"Regenerating layer {name}")
        generator, next_frame = self.layers[name]["function"](**self.layers[name]["layer_params"])

        self.layers[name]["generator"] = generator
        self.layers[name]["next_frame"] = next_frame
        host_preview = self._gpu_to_preview_host(next_frame(), self.downsample_factor)
        self.frame_ready.emit([name], {name : host_preview})
        self.last_draw = {name : host_preview}
        self.status.emit(f"Regenerated layer {name}")

    @Slot()
    def step(self):
        import cupy as cp
        cp.cuda.Device().use()  # ensure context is created in this thread

        if self.step_n_num is not None:
            if self.step_n_num <= 0:
                self.stop()
        if cp.all(cp.asarray([not i.get("active", False) for n, i in list(self.layers.items())])):
            return
        
        frames = {}
        for name, info in list(self.layers.items()):
            if not info.get("active", False):
                continue
            host_preview = self._gpu_to_preview_host(info["next_frame"](), self.downsample_factor)
            frames[name] = host_preview
            if self.step_n_num is not None: 
                self.step_n_num -= 1
        self.last_draw = frames
        self.frame_ready.emit(list(frames.keys()), frames)

    @Slot()
    def redraw(self):
        frames = {}
        for name, info in list(self.layers.items()):
            if not info.get("active", False):
                continue
            host_preview = self._gpu_to_preview_host(info["next_frame"](), self.downsample_factor)
            frames[name] = host_preview

        self.frame_ready.emit(list(frames.keys()), frames)
        

    @Slot(float)
    def run_loop(self, interval_s=0.03):
        import cupy as cp
        cp.cuda.Device().use()  # ensure context is created in this thread

        self._timer.start(int(interval_s * 1000))
        self.status.emit("Scheduler running")

    @Slot(int, float)
    def step_layer_N(self, n_times=1, interval_s=0.03):
        import cupy as cp
        cp.cuda.Device().use()  # ensure context is created in this thread

        self._timer.start(int(interval_s * 1000))
        self.step_n_num = n_times
        self.status.emit("Scheduler running")
        
    @Slot()
    def step_layer(self):
        import cupy as cp
        cp.cuda.Device().use()  # ensure context is created in this thread
        frames = {}
        for name, info in list(self.layers.items()):
            if not info.get("active", False):
                continue
            host_preview = self._gpu_to_preview_host(info["next_frame"](), self.downsample_factor)
            frames[name] = host_preview

            info["active"] = False
        self.last_draw = frames
        self.frame_ready.emit(list(frames.keys()), frames)

    @Slot()
    def stop(self):
        if self._timer is not None: self._timer.stop()
        self.step_n_num = None

    def _gpu_to_preview_host(self, gpu_arr, downsample_factor):
        # import cupy as cp
        # if downsample_factor > 1:
        #     preview_gpu = gpu_arr[::downsample_factor, ::downsample_factor]
        # else:
        #     preview_gpu = gpu_arr
        # mn = preview_gpu.min()
        # mx = preview_gpu.max()
        # denom = mx - mn
        # if denom == 0:
        #     denom = 1.0
        # norm = (preview_gpu - mn) / denom
        # preview_u8 = (norm * 255.0).astype(cp.uint8)
        # host_preview = cp.asnumpy(preview_u8)
        return gpu_arr