# scripts/worker.py
from PySide6.QtCore import QObject, QRunnable, Signal, Slot, QTimer, Qt
import scripts.utilities as ut
import cupy as cp
import time


# ---- Generic Fire and Forget worker 
class GenericWorkerSignals(QObject):
    finished = Signal(object)  # emit the created generator
    error = Signal(Exception)

class GenericWorker(QRunnable):
    def __init__(self, generator_func, **params):
        super().__init__()
        self.generator_func = generator_func
        self.params = params
        self.signals = GenericWorkerSignals()

    def run(self):
        try:
            gen, next_frame_func = self.generator_func(**self.params)
            self.signals.finished.emit((gen, next_frame_func))
        except Exception as e:
            self.signals.error.emit(e)

# Worker for turbulence screen viewers
class FrameWorker(QObject):
    frame_ready = Signal(object)

    def __init__(self, gen_factory, params, n_frames=None, delay=0.01):
        super().__init__()
        self.gen_factory = gen_factory    # phase screen generator callable: () -> (phase_obj, next_frame_callable)
        self.n_frames = n_frames
        self.params = params
        self.delay = delay
        self._running = False
        self._gen, self._next_frame = self.gen_factory(**self.params)

        self.step()

    @Slot()
    def _ensure_gen(self):
        # create generator (run inside worker thread)
        if self._gen is None:
            obj, next_fn = self.gen_factory(**self.params)
            self._gen = obj
            self._next_frame = next_fn

    @Slot(dict)
    def change_params(self, new_params):
        self.params = new_params
        self.reset()
        
    @Slot(int)
    def change_grid_size(self, new_size):
        self.params["grid_size"] = new_size
        self.reset()

    @Slot()
    def step(self):
        try:
            self._ensure_gen()
            frame = self._next_frame()
            self.frame_ready.emit(frame)
        except Exception:
            import traceback
            traceback.print_exc()

    @Slot()
    def run(self):
        if self._running:
            return

        self._running = True
        self._ensure_gen()

        self._timer = QTimer()
        self._timer.setInterval( int(1000 / 30) )  # 30 fps
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    @Slot()
    def _tick(self):
        if not self._running:
            return
        frame = self._next_frame()
        self.frame_ready.emit(frame)

    @Slot()
    def stop(self):
        self._running = False

    @Slot()
    def reset(self):
        self.stop()
        self._gen = None
        self._next_frame = None
        self.step()



# worker.py

from PySide6.QtCore import QThread
import threading

class CalculateWorker(QObject):
    """Singleton worker that runs all WFS sensor computations in a single thread."""

    _instance = None
    _lock = threading.Lock()

    # Signals
    initialized_result = Signal(object, object)  # sensor, result
    subap_finished = Signal(object, object, int) # sensor, (centroids, imgs), job_id


    def __init__(self):
        super().__init__()
        self._queue = []        # queue of (sensor, job_type, job_data)
        self._queue_lock = threading.Lock()
        self._job_id = 0
        self._sensor_data = {}  # sensor -> shared GPU arrays

        # dedicated thread
        self.thread = QThread()
        self.moveToThread(self.thread)
        self.thread.started.connect(self._start_timer)
        self.thread.start()

    @classmethod
    def instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = CalculateWorker()
            return cls._instance

    def _start_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self._process_queue)
        self.timer.start(10)

    def request_initialization(self, sensor, params):
        """Queue a full computation job for a sensor."""
        with self._queue_lock:
            self._queue.append((sensor, "init", dict(params)))

    def request_recompute(self, sensor, params):
        """Queue a full recompute job for a sensor."""
        with self._queue_lock:
            self._queue.append((sensor, "recompute", params))

    @Slot(int, int, object)
    def process_subap(self, sensor, k, job_id, params):
        """Queue a per-actuator measurement for a sensor."""
        with self._queue_lock:
            self._queue.append((sensor, "subap", (k, job_id)))

    def _process_queue(self):
        """Process one job from the queue."""
        with self._queue_lock:
            if not self._queue:
                return
            job = self._queue.pop(0)

        sensor, job_type, job_data = job

        if job_type in ["init", "recompute"]:
            sensor, job_type, params = job
            self._do_full_compute(sensor, params)
        elif job_type == "subap":
            k, job_id = job_data
            self._process_single_subap(sensor, k, job_id)

    def _do_full_compute(self, sensor, params):
        """Perform full WFS computation for one sensor."""
        ut.set_params(params)
        grid_size = int(params.get("grid_size"))
        pupil = cp.asarray(ut.Pupil_tools.generate_pupil(grid_size=grid_size))
        act_centers = ut.Pupil_tools.generate_actuators(pupil)
        influence_maps = ut.Pupil_tools.generate_actuator_influence_map(act_centers, pupil)

        # geometry from sensor
        active_sub_aps = sensor.active_sub_aps
        sub_aps = sensor.sub_aps
        sub_aps_idx = sensor.sub_aps_idx
        sub_ap_width = sensor.sub_ap_width
        sub_slice = sensor.sub_slice
        sub_pupils = sensor.sub_pupils

        # reference centroids & images
        N = pupil.shape[0]
        zero_phase = cp.zeros((1, N, N), dtype=pupil.dtype)
        centroids_ref, slopes_ref, ref_images = sensor.measure(
            pupil=pupil,
            phase_map=zero_phase,
            poke_amplitude=0.0,
            pad=int(params.get("field_padding", 4))
        )

        ref_centroids = cp.asarray(centroids_ref[0])
        ref_images = cp.asarray(ref_images[0])
        ref_science_image, ref_strehl = ut.Analysis.generate_science_image(pupil, cp.zeros_like(pupil), pad=512)
        normalized_image = ref_science_image / ref_science_image.sum()
        science_image_plot = cp.log10(normalized_image[512:-512, 512:-512] + 1e-12)
        n_act = int(act_centers.shape[0])

        # emit result
        res = {
            "pupil": pupil,
            "act_centers": act_centers,
            "influence_maps": influence_maps,
            "active_sub_aps": active_sub_aps,
            "sub_aps": sub_aps,
            "sub_aps_idx": sub_aps_idx,
            "sub_ap_width": sub_ap_width,
            "sub_slice": sub_slice,
            "sub_pupils": sub_pupils,
            "ref_centroids": ref_centroids,
            "ref_images": ref_images,
            "ref_science_image": ref_science_image,
            "ref_strehl": ref_strehl,
            "normalized_image": normalized_image,
            "science_image_plot": science_image_plot,
            "n_act": n_act,
            "ref_fwhm": float(self._compute_ref_fwhm(ref_science_image, params)),
            "diff_lim": float(1.22 * params["science_lambda"] / params["telescope_diameter"])
        }
        self.initialized_result.emit(sensor, res)

        # save shared data for per-actuator jobs
        self._sensor_data[sensor] = {
            "pupil": pupil,
            "influence_maps": influence_maps,
            "ref_centroids": ref_centroids,
            "sub_aps": sub_aps,
            "sub_ap_width": sub_ap_width,
            "params": params
        }

    def _process_single_subap(self, sensor, k, job_id):
        """Perform per-actuator measurement using shared GPU arrays."""
        try:
            if sensor not in self._sensor_data:
                self.subap_finished.emit(sensor, (cp.zeros((0, 2)), cp.zeros((1, 1))), job_id)
                print("WARN sensor not in worker")
                return

            data = self._sensor_data[sensor]

            phase_map_for_k = cp.asarray(data["influence_maps"][k:k+1])
            centroids, slopes, sensor_images = sensor.measure(
                pupil=data["pupil"],
                phase_map=phase_map_for_k,
                poke_amplitude=data["params"].get("poke_amplitude"),
                pad=int(data["params"].get("field_padding", 4))
            )

            self.subap_finished.emit(sensor, (cp.asarray(centroids[0]), cp.asarray(sensor_images[0])), job_id)
        except Exception:
            import traceback
            traceback.print_exc()
            self.subap_finished.emit(sensor, (cp.zeros((0, 2)), cp.zeros((1, 1))), job_id)

    def _compute_ref_fwhm(self, ref_science_image, params):
        plate_rad = (params["science_lambda"] * params["grid_size"] /
                     (params["telescope_diameter"] * ref_science_image.shape[0]))
        return plate_rad * ut.Analysis.fwhm_radial(ref_science_image)
