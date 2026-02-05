# scripts/worker.py
from collections import deque
import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np
import cupy as cp

from PySide6.QtCore import QObject, Signal, Slot, QTimer, QThread

import scripts.utilities as ut
from scripts.utilities import Gaussian2DFitter, Analysis


def _to_numpy(x: Any) -> np.ndarray:
    """Convert CuPy arrays to NumPy; pass NumPy arrays through."""
    if isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


class CalculateWorker(QObject):
    """Singleton worker that runs all WFS sensor computations in a single thread.

    The worker coalesces bursty requests:
      - For each sensor, only the latest pending full compute (init/recompute) is kept.
      - For each sensor, only the latest pending per-actuator request is kept.

    This prevents heavy work on stale slider positions.
    """

    _instance: Optional["CalculateWorker"] = None
    _lock = threading.Lock()

    # Signals
    initialized_result = Signal(object, object)  # sensor, result_dict
    subap_finished = Signal(object, object, int)  # sensor, result_dict, job_id

    def __init__(self):
        super().__init__()

        # Pending-job structures (coalescing by sensor)
        self._pending_lock = threading.Lock()

        self._full_params: Dict[object, Dict[str, Any]] = {}
        self._full_order: deque[object] = deque()
        self._full_live: set[object] = set()

        self._subap_req: Dict[object, Tuple[int, int]] = {}  # sensor -> (k, job_id)
        self._subap_order: deque[object] = deque()
        self._subap_live: set[object] = set()

        # Shared GPU arrays per sensor
        self._sensor_data: Dict[object, Dict[str, Any]] = {}

        # Dedicated thread
        self.thread = QThread()
        self.moveToThread(self.thread)
        self.thread.started.connect(self._start_timer)
        self.thread.start()

    @classmethod
    def instance(cls) -> "CalculateWorker":
        with cls._lock:
            if cls._instance is None:
                cls._instance = CalculateWorker()
            return cls._instance

    def _start_timer(self) -> None:
        self._timer = QTimer()
        self._timer.timeout.connect(self._process_one)
        self._timer.start(10)

    # ---- enqueue helpers
    def _enqueue_full(self, sensor: object, params: Dict[str, Any]) -> None:
        with self._pending_lock:
            # Keep latest params
            self._full_params[sensor] = params
            if sensor not in self._full_live:
                self._full_live.add(sensor)
                self._full_order.append(sensor)

            # Any pending per-actuator jobs are stale relative to geometry/params.
            if sensor in self._subap_live:
                self._subap_live.discard(sensor)
                self._subap_req.pop(sensor, None)

    def _enqueue_subap(self, sensor: object, k: int, job_id: int) -> None:
        with self._pending_lock:
            self._subap_req[sensor] = (int(k), int(job_id))
            if sensor not in self._subap_live:
                self._subap_live.add(sensor)
                self._subap_order.append(sensor)

    # ---- public API
    def request_initialization(self, sensor: object, params: Dict[str, Any]) -> None:
        """Queue a full computation job for a sensor."""
        self._enqueue_full(sensor, dict(params))

    def request_recompute(self, sensor: object, params: Dict[str, Any]) -> None:
        """Queue a full recompute job for a sensor."""
        self._enqueue_full(sensor, dict(params))

    @Slot(object, int, int)
    def request_subap(self, sensor: object, k: int, job_id: int) -> None:
        """Queue a per-actuator measurement for a sensor."""
        self._enqueue_subap(sensor, k, job_id)

    # Backwards-compatible slot (older code passed a params arg; we ignore it).
    @Slot(object, int, int, object)
    def process_subap(self, sensor: object, k: int, job_id: int, _params: object = None) -> None:
        self.request_subap(sensor, k, job_id)

    # Optional cleanup hook if tabs/sensors are destroyed.
    def drop_sensor(self, sensor: object) -> None:
        with self._pending_lock:
            self._full_params.pop(sensor, None)
            self._subap_req.pop(sensor, None)
            self._full_live.discard(sensor)
            self._subap_live.discard(sensor)
        self._sensor_data.pop(sensor, None)

    # ---- main loop
    def _process_one(self) -> None:
        """Process a single pending job (full jobs are higher priority)."""
        sensor = None
        full_params = None
        subap = None

        with self._pending_lock:
            # Prefer full jobs
            while self._full_order:
                s = self._full_order.popleft()
                if s in self._full_live:
                    self._full_live.remove(s)
                    sensor = s
                    full_params = self._full_params.pop(s, None)
                    break

            if sensor is None:
                while self._subap_order:
                    s = self._subap_order.popleft()
                    if s in self._subap_live:
                        self._subap_live.remove(s)
                        sensor = s
                        subap = self._subap_req.pop(s, None)
                        break

        if sensor is None:
            return

        if full_params is not None:
            self._do_full_compute(sensor, full_params)
            return

        if subap is not None:
            k, job_id = subap
            self._process_single_subap(sensor, k, job_id)

    # ---- heavy jobs
    def _do_full_compute(self, sensor: object, params: Dict[str, Any]) -> None:
        """Perform full (re)initialization for one sensor.

        Uses *lazy* actuator influence maps to avoid allocating an (n_act, N, N) cube on GPU.
        """
        try:
            ut.set_params(params)

            grid_size = int(params.get("grid_size"))
            obsc = float(params.get("telescope_center_obscuration"))
            pad_fp = int(params.get("field_padding", 4))

            # Pupil on GPU (float32 to reduce memory/bandwidth)
            pupil = cp.asarray(
                ut.Pupil_tools.generate_pupil(
                    grid_size=grid_size,
                    telescope_center_obscuration=obsc,
                    dtype=cp.float32,
                ),
                dtype=cp.float32,
            )

            # Ensure sensor uses this pupil/grid and rebuilds cached geometry
            n_sub = int(params.get("sub_apertures"))
            wfs_lambda = float(params.get("wfs_lambda"))

            # Optional per-sensor geometry (safe to omit if absent).
            recompute_kwargs: Dict[str, Any] = {
                "n_sub": n_sub,
                "wavelength": wfs_lambda,
            }
            for k in (
                "dx",
                "dy",
                "gs_range_m",
                "lgs_launch_offset_px",
                "lgs_thickness_m",
                "lenslet_f_m",
                "pixel_pitch_m",
            ):
                if k in params and params[k] is not None:
                    recompute_kwargs[k] = params[k]

            recomputed = False
            # Try most feature-complete signature first; progressively fall back for older/alternate sensors.
            for extra in ({"pupil": pupil, "grid_size": grid_size}, {},):
                if recomputed:
                    break
                try:
                    sensor.recompute(**recompute_kwargs, **extra)
                    recomputed = True
                except TypeError:
                    # Some implementations may use different kw names; try common alternatives.
                    try:
                        alt = dict(recompute_kwargs)
                        alt["sub_apertures"] = alt.pop("n_sub")
                        alt["wfs_lambda"] = alt.pop("wavelength")
                        sensor.recompute(**alt, **extra)
                        recomputed = True
                    except Exception:
                        pass
                except Exception:
                    pass

            if not recomputed:
                # Final fallback: set essentials manually and rebuild apertures if supported.
                sensor.pupil = pupil
                sensor.grid_size = grid_size
                for attr, val in (
                    ("n_sub", n_sub),
                    ("sub_apertures", n_sub),
                    ("wavelength", wfs_lambda),
                    ("wfs_lambda", wfs_lambda),
                ):
                    if hasattr(sensor, attr):
                        try:
                            setattr(sensor, attr, val)
                        except Exception:
                            pass
                if hasattr(sensor, "regen_sub_apertures"):
                    try:
                        sensor.regen_sub_apertures()
                    except Exception:
                        pass

            # Actuator centers and lazy influence-map getter
            act_centers = ut.Pupil_tools.generate_actuators(
                pupil=pupil,
                actuators=int(params.get("actuators")),
                grid_size=grid_size,
            )

            get_map, act_centers = ut.Pupil_tools.generate_actuator_influence_map(
                act_centers=act_centers,
                pupil=pupil,
                actuators=int(params.get("actuators")),
                poke_amplitude=float(params.get("poke_amplitude")),
                grid_size=grid_size,
                return_mode="lazy",
                dtype=cp.float32,
            )

            # Geometry for plotting (CPU)
            sub_aps = _to_numpy(getattr(sensor, "sub_aps", np.zeros((0, 2), dtype=np.float32))).astype(
                np.float32, copy=False
            )

            # Reference centroids
            N = int(pupil.shape[0])
            zero_phase = cp.zeros((1, N, N), dtype=cp.float32)
            centroids_ref, _slopes_ref, _ = sensor.measure(
                pupil=pupil,
                phase_map=zero_phase,
                pad=pad_fp,
                return_image=True,
            )
            ref_centroids_single = (_to_numpy(centroids_ref[0]) - pad_fp).astype(np.float32, copy=False)

            # Unperturbed science image (display) + FWHM
            science_pad = 512
            ref_science_img, _ref_strehl = ut.Analysis.generate_science_image(
                pupil=pupil,
                phase_map=cp.zeros_like(pupil, dtype=cp.float32),
                pad=science_pad,
            )
            ref_fwhm = float(self._compute_img_fwhm(ref_science_img, params))

            # Plot (normalize in-place)
            ref_science_img = cp.log10(ref_science_img / ref_science_img.max() + cp.float32(1e-12)) * cp.float32(10.0)
            ref_plot = ref_science_img[science_pad:-science_pad, science_pad:-science_pad].astype(
                cp.float32, copy=False
            )
            ref_plot_cpu = cp.asnumpy(ref_plot)

            pupil_mask_cpu = cp.asnumpy(pupil > 0)
            n_act = int(act_centers.shape[0])

            res = {
                "n_act": n_act,
                "sub_aps": sub_aps,
                "ref_centroids_single": ref_centroids_single,
                "pupil_mask": pupil_mask_cpu,
                "ref_science_plot": ref_plot_cpu,
                "ref_fwhm": ref_fwhm,
                "diff_lim": float(1.22 * float(params["wfs_lambda"]) / float(params["telescope_diameter"])),
            }

            # Save shared data for per-actuator jobs
            self._sensor_data[sensor] = {
                "pupil": pupil,
                "act_centers": act_centers,
                "get_map": get_map,
                "params": dict(params),
                "science_pad": science_pad,
            }

            self.initialized_result.emit(sensor, res)

        except Exception:
            import traceback

            traceback.print_exc()
            self._sensor_data.pop(sensor, None)
            self.initialized_result.emit(sensor, {"error": "exception"})

    def _process_single_subap(self, sensor: object, k: int, job_id: int) -> None:
        """Perform per-actuator measurement + science image on the worker thread."""
        try:
            data = self._sensor_data.get(sensor)
            if data is None:
                self.subap_finished.emit(sensor, {"error": "sensor not initialized"}, job_id)
                return

            params = data["params"]
            pad_fp = int(params.get("field_padding", 4))
            science_pad = int(data.get("science_pad", 512))

            pupil = data["pupil"]
            get_map = data["get_map"]

            # Lazy influence map for this actuator (float32 on GPU)
            influence_map = get_map(int(k)).astype(cp.float32, copy=False)

            # WFS measurement
            centroids, _slopes, sensor_images = sensor.measure(
                pupil=pupil,
                phase_map=influence_map[None, :, :],
                pad=pad_fp,
                return_image=True,
            )
            centroids_single = (_to_numpy(centroids[0]) - float(pad_fp)).astype(np.float32, copy=False)
            subap_image = _to_numpy(sensor_images[0])

            # Science image (heavy) on GPU
            sci_lambda = float(params.get("science_lambda") or params.get("wfs_lambda"))
            influence_phase_scale = np.float32(4.0 * np.pi / sci_lambda)
            influence_map_phase = influence_map * influence_phase_scale

            science_img, strehl = ut.Analysis.generate_science_image(
                pupil=pupil,
                phase_map=influence_map_phase,
                pad=science_pad,
            )
            img_fwhm = float(self._compute_img_fwhm(science_img, params))

            # Plot (normalize in-place)
            science_img = cp.log10(science_img / science_img.max() + cp.float32(1e-12)) * cp.float32(10.0)
            science_plot = science_img[science_pad:-science_pad, science_pad:-science_pad].astype(cp.float32, copy=False)
            science_plot_cpu = cp.asnumpy(science_plot)

            influence_map_cpu = cp.asnumpy(influence_map)

            self.subap_finished.emit(
                sensor,
                {
                    "centroids_single": centroids_single,
                    "subap_image": subap_image,
                    "science_plot": science_plot_cpu,
                    "strehl": float(strehl),
                    "img_fwhm": img_fwhm,
                    "influence_map": influence_map_cpu,
                },
                int(job_id),
            )

        except Exception:
            import traceback

            traceback.print_exc()
            self.subap_finished.emit(sensor, {"error": "exception"}, int(job_id))

    @staticmethod
    def _compute_img_fwhm(science_image: cp.ndarray, params: Dict[str, Any]) -> float:
        plate_rad = (
            float(params["wfs_lambda"]) * float(params["grid_size"])
            / (float(params["telescope_diameter"]) * float(science_image.shape[0]))
        )
        gaussian_fitter = Gaussian2DFitter(roi_half=32, dtype=cp.float32)
        zoomed = Analysis._zoom2d(science_image, 4, cp.get_array_module(science_image))
        fwhm = gaussian_fitter.fit_frame(zoomed, zoom=4)[2][0]
        return plate_rad * float(fwhm)
