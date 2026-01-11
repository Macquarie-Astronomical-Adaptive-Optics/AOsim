import time
from PySide6.QtCore import QObject, Signal, Slot, QTimer
from scripts.utilities import Analysis
# NOTE: import cupy inside the scheduler thread methods only

class GPUScheduler(QObject):
    # emits (layer_name: str, host_preview: np.ndarray uint8)
    frame_ready = Signal(object, dict)
    status = Signal(str)

    def __init__(self, downsample_factor=4):
        super().__init__()
        self.layers = {}  # name -> dict with GPU arrays and params
        self.downsample_factor = downsample_factor
        self.last_draw = {}

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.step)

        self.step_n_num = None

    @Slot(str, dict)
    def edit_layer_param(self, name, layer_params):
        if name in self.layers:
            if self.layers[name]["layer_params"] == layer_params:
                return
            self.layers[name]["layer_params"] = layer_params
            self.status.emit(f"Changed params of layer {name}")
            self.regenerate_layer(name)

    @Slot(str)
    def remove_layer(self, name):
        if name in self.layers:
            del self.layers[name]
            self.status.emit(f"Removed layer {name}")

    @Slot(str, bool)
    def set_active(self, name, is_active):
        if name in self.layers:
            self.layers[name]["active"] = is_active
            
    @Slot(str, bool)
    def set_science_active(self, name, is_active):
        if name in self.layers:
            self.layers[name]["active_science"] = is_active

    @Slot(str, object, dict)
    def add_layer(self, name, layer_function, layer_params):
        # run in scheduler thread; do CuPy allocations here
        import cupy as cp
        size = layer_params["grid_size"]
        self.status.emit(f"Adding layer {name} with shape ({size}, {size})")

        # initialize GPU array and RNG for each layer
        generator, next_frame = layer_function(**layer_params)
        self.layers[name] = {
            "next_frame": next_frame,
            "generator": generator,
            "active": False,
            "active_science" : False,
            "function": layer_function,
            "layer_params": layer_params
        }
        host_preview = next_frame()

        self.last_draw[name] = host_preview
        frames = {name : host_preview}
        
        frames = {k:self.frame_preview(v) for k, v in frames.items()}
        frames["science"] = self.science(self.last_draw)
        self.frame_ready.emit(list(frames.keys()), frames)
        self.status.emit(f"Added layer {name}")


    @Slot(str)
    def regenerate_layer(self, name):
        import cupy as cp
        if name not in self.layers:
            print("[WARN] GPU Scheduler.regenerate_layer :", name, "not in scheduler layers!")
            return
        size = self.layers[name]["layer_params"]["grid_size"]
        self.status.emit(f"Regenerating layer {name} with shape ({size},{size})")
        generator, next_frame = self.layers[name]["function"](**self.layers[name]["layer_params"])

        self.layers[name]["generator"] = generator
        self.layers[name]["next_frame"] = next_frame
        host_preview = next_frame()
        
        self.last_draw[name] = host_preview
        frames = {name : host_preview}
        
        frames = {k:self.frame_preview(v) for k, v in frames.items()}
        frames["science"] = self.science(self.last_draw)
        self.frame_ready.emit(list(frames.keys()), frames)

        self.status.emit(f"Regenerated layer {name}")

    @Slot()
    def step(self):
        import cupy as cp
        cp.cuda.Device().use()  # ensure context is created in this thread

        if self.step_n_num is not None:
            if self.step_n_num <= 0:
                self.stop()
                return
        if cp.all(cp.asarray([not i.get("active", False) for n, i in list(self.layers.items())])):
            return
        
        frames = {}
        for name, info in list(self.layers.items()):
            if not info.get("active", False):
                continue
            host_preview = info["next_frame"]()
            frames[name] = host_preview
            if self.step_n_num is not None: 
                self.step_n_num -= 1

        self.last_draw = frames.copy()

        frames = {k:self.frame_preview(v) for k, v in frames.items()}
        frames["science"] = self.science(self.last_draw)
        self.frame_ready.emit(list(frames.keys()), frames)


    @Slot()
    def redraw(self):
        frames = self.last_draw

        frames = {k:self.frame_preview(v) for k, v in frames.items()}
        frames["science"] = self.science(self.last_draw)
        self.frame_ready.emit(list(frames.keys()), frames)

    @Slot(object)
    def science(self, frames):
        import cupy as cp
        cp.cuda.Device().use()

        total_active = cp.asarray([self.center_crop(item) for key, item in frames.items() if self.layers[key]["active_science"]])
        total_active = cp.sum(total_active, axis=0)
        
        total_science, total_science_strehl = Analysis.generate_science_image(phase_map=total_active)
        normalized_image = total_science/total_science.sum()
        total_science_plot = cp.log10(normalized_image + 1e-12)

        return cp.asarray([total_science, total_science_plot])
        

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
        self.status.emit(f"Scheduler stepping {n_times} times")
        
    @Slot()
    def step_layer(self):
        import cupy as cp
        cp.cuda.Device().use()  # ensure context is created in this thread
         
        self._timer.start(int(0.03 * 1000))
        self.step_n_num = 1
        self.status.emit("Scheduler stepping")

    @Slot()
    def stop(self):
        if self._timer is not None: self._timer.stop()
        self.step_n_num = None
        self.redraw()

    def center_crop(self, img, crop_size=512):
        H, W = img.shape[:2] 
        cy, cx = H // 2, W // 2  # center coordinates

        half = crop_size // 2
        # compute crop bounds
        y1, y2 = max(0, cy - half), min(H, cy + half)
        x1, x2 = max(0, cx - half), min(W, cx + half)

        return img[y1:y2, x1:x2]

    def frame_preview(self, img):
        return self._gpu_to_preview_host(img, self.downsample_factor)

    def _gpu_to_preview_host(self, gpu_arr, downsample_factor):
        import cupy as cp
        if downsample_factor > 1:
            preview_gpu = gpu_arr[::downsample_factor, ::downsample_factor]
        else:
            preview_gpu = gpu_arr
        mn = preview_gpu.min()
        mx = preview_gpu.max()
        denom = mx - mn
        if denom == 0:
            denom = 1.0
        norm = (preview_gpu - mn) / denom
        preview_u8 = (norm * 255.0).astype(cp.uint8)
        host_preview = cp.asnumpy(preview_u8)
        return host_preview