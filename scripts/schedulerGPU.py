from __future__ import annotations
# scripts/schedulerGPU.py
import time
from typing import Any, Dict, List, Optional

import cupy as cp
from PySide6.QtCore import QObject, Signal, Slot, QTimer, Qt

from scripts.reconstructor import TomoOnAxisIM_CuPy
from scripts.utilities import Pupil_tools, Analysis, Gaussian2DFitter, params


def _u8_linear(x, vmin, vmax):
    eps = 1e-12
    x = cp.nan_to_num(x, nan=vmin, posinf=vmax, neginf=vmin)
    y = (x - vmin) * (255.0 / (vmax - vmin + eps))
    return cp.clip(y, 0.0, 255.0).astype(cp.uint8)


def time_n(sync: bool = False) -> int:
    """GPU-aware timer. If sync=True, waits for default stream before timestamping."""
    if sync:
        cp.cuda.Stream.null.synchronize()
    return time.perf_counter_ns()


def remove_tt(phi, pupil):
    m = (pupil > 0)
    yy, xx = cp.indices(phi.shape, dtype=cp.float32)
    xx = xx - (phi.shape[1] - 1) * 0.5
    yy = yy - (phi.shape[0] - 1) * 0.5

    z = phi[m].ravel()
    X = cp.stack([xx[m].ravel(), yy[m].ravel(), cp.ones_like(z)], axis=1)  # [x,y,1]
    beta, *_ = cp.linalg.lstsq(X, z, rcond=None)  # plane fit
    plane = (beta[0] * xx + beta[1] * yy + beta[2]) * m
    return phi - plane


def strip_tt_from_slopes(sl_yx):  # sl shape (n_sub, 2) with [sy, sx]
    """Remove global tip/tilt component in slope-space by subtracting mean slopes."""
    sl = sl_yx.copy()
    sl[:, 0] -= cp.mean(sl[:, 0])  # remove mean sy
    sl[:, 1] -= cp.mean(sl[:, 1])  # remove mean sx
    return sl

class SimWorker(QObject):
    # status / lifecycle
    status_log = Signal(object)
    status = Signal(str)
    status_running = Signal(bool)

    fpsReady = Signal(float)
    finished = Signal()

    # main outputs
    combined_ready = Signal(object)       # cp/np (N,N)
    update_r0 = Signal(float)
    sim_signal = Signal(object)
    reconstructed_ready = Signal(object, object, object, object)
    loop_ready = Signal(object, object, float, float, float, float, float, object)

    recon_matr_ready = Signal(object, object)

    psf_save_to_file = Signal(object)

    # single views (for tabs)
    layer_ready = Signal(int, object)          # (layer_idx, cp/np (H,W))
    sensor_phase_ready = Signal(int, object)   # (sensor_idx, cp/np (M,M))
    sensor_psf_ready = Signal(int, object)     # (sensor_idx, cp/np (M,M))

    # overview outputs (for grid pages)
    all_sensor_psf_ready = Signal(object)      # cp/np (S, psf_M, psf_M)
    all_sensor_phase_ready = Signal(object)    # cp/np (S, psf_M, psf_M) or (S, patch_M, patch_M)
    all_layers_ready = Signal(object)          # cp/np (L, S, layer_M, layer_M)

    def __init__(
        self,
        sim_factory,
        sim_kwargs: Dict[str, Any],
        layers: List[Dict[str, Any]],
        sensors,
        ranges_m,
        patch_size_px,
        patch_M,
        dt_s: float = 0.001,

        # display sizes
        psf_M: Optional[int] = None,
        layer_M: Optional[int] = None,

        # PSF controls
        emit_psf: bool = True,
        pupil_mask=None,

        # long exposure psf
        long_exposure_frames = 50,
        sensor_method = "southwell",

        
        fps_cap: int = 2000,
        # single-tab update rates
        layer_tab_hz: int = 10,
        sensor_hz: int = 10,
        # overview update rates
        overview_psf_hz: int = 5,
        overview_layer_hz: int = 10,
        # loop display update rates
        loop_display_hz = 450,
        # combined update rate (display). None/0 -> every tick
        combined_hz: int = 10,

        parent=None,

    ):
        super().__init__(parent)

        self.sim_factory = sim_factory
        self.sim_kwargs = dict(sim_kwargs)
        self.layer_cfgs = list(layers)

        self.sensor_method = sensor_method  # southwell or fft

        self.dt_s = float(dt_s)
        self.fps_cap = int(fps_cap)

        # AO geometry
        self.sensors = sensors
        self._sensor_list = list(self.sensors.values())
        self._sensor_keys = list(self.sensors.keys())

        self.thetas_gpu = cp.asarray([s.field_angle for s in self._sensor_list], dtype=cp.float32)  # (S,2)
        S = int(self.thetas_gpu.shape[0])

        if ranges_m is None:
            self.ranges_gpu = cp.full((S,), cp.inf, dtype=cp.float32)
        else:
            r = cp.asarray(ranges_m, dtype=cp.float32)
            if r.ndim != 1:
                r = r.reshape(-1)
            if r.size == 1:
                r = cp.full((S,), float(r.item()), dtype=cp.float32)
            elif r.size != S:
                raise ValueError(f"ranges_m must be scalar or shape (S,), got {tuple(r.shape)} for S={S}")
            self.ranges_gpu = r

        self.patch_size_px = float(patch_size_px)
        self.patch_M = int(patch_M)

        self.psf_M = int(psf_M) if psf_M is not None else (self.patch_M // 2)
        self.layer_M = int(layer_M) if layer_M is not None else (self.patch_M // 2)

        # controls
        self.emit_psf = bool(emit_psf)
        self._overview_enabled = False
        self._reconstructor_enabled = False
        self._detailed_enabled = True
        self._loop_enabled = False
        self._loop_r0_only = False

        # profiling / timing
        self._profile_gpu_timings = False

        # view selection
        self._visible_layer: Optional[int] = 0
        self._visible_sensor: int = 0

        # rate limit periods (ticks)
        self._interval_ms = max(1, int(1000 / max(1, self.fps_cap)))
        self._combined_period = self._hz_to_period(int(combined_hz)) if (combined_hz is not None and combined_hz > 0) else 1
        self._layer_tab_period = self._hz_to_period(layer_tab_hz)
        self._sensor_period = self._hz_to_period(sensor_hz)
        self._overview_psf_period = self._hz_to_period(overview_psf_hz)
        self._overview_layer_period = self._hz_to_period(overview_layer_hz)
        self._loop_disp_period = self._hz_to_period(loop_display_hz)


        self._tick_count = 0

        # stepping state
        self._stepping = False
        self._steps_left = 0
        self._emit_every = 0
        self._batch = 20

        # fps stats (do not use for scheduling)
        self._fps_frames = 0
        self._fps_t0_wall = None
        self._prev_tick_wall = None

        # sim + masks
        self.sim = None
        self.R = None

        # reusable GPU buffers (avoid per-tick allocations)
        self._combined_buf = None
        self.pupil_mask = None if pupil_mask is None else cp.asarray(pupil_mask).astype(cp.float32)
        self._pupil_mask_psf = None
        self._pupil_mask_patch = None

        # ------------------ per-frame sampling cache ------------------
        self._frame_tick = -1
        self._frame_need_per_layer = False
        self._frame_phases_M = None        # (S, patch_M, patch_M)
        self._frame_layers_M = None        # (L, S, patch_M, patch_M) or None

        # lazily computed downsampled caches
        self._frame_phases_psfM = None     # (S, psf_M, psf_M)
        self._frame_layers_layerM = None   # (L, S, layer_M, layer_M)

        self.reconstructed = None

        # ------------------ display scaling cache ------------------
        self._display_stats_M = 64
        self._phase_k = 3.0
        self._psf_dyn = 6.0
        self._phase_mu = cp.asarray(0.0, dtype=cp.float32)
        self._phase_sig = cp.asarray(1.0, dtype=cp.float32)
        self._psf_vmax = cp.asarray(0.0, dtype=cp.float32)
        self._last_level_tick = -10**9
        self._level_period = max(1, self._hz_to_period(15))  # ~15 Hz levels

        # timer
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)

        # ------------------ optional profiling knobs ------------------
        self._sync_profile_every = 0
        self._prev_tick_end_wall = None

        self.initialized = False

        # DM control loop
        self._closed_loop = True
        self._loop_gain = 1.0
        self._loop_leak = 0.0
        self._loop_t = 0.0
        self._s_ref = None
        self._dm_cmd = None  # active coefficient/command vector (matches reconstructor basis)
        self._dm_sigma = None  # optional override
        self._dm_phi = None    # phase-to-subtract next tick (M,M)
        self._phases_res_buf = None  # (S,M,M) residual buffer
        self.lambda_m = params.get("science_lambda")
        self.gaussian_fitter = Gaussian2DFitter(roi_half=32, dtype=cp.float32)
        self.gaussian_fitter_uc = Gaussian2DFitter(roi_half=32, dtype=cp.float32)

        self.plate_rad = None
        self.long_exposure_frames = long_exposure_frames
        self.long_exposure_psf = None
        self.long_exposure_psf_uncorrected = None

    # ------------------ helpers ------------------
    def _hz_to_period(self, hz: int) -> int:
        hz = max(1, int(hz))
        return max(1, int(round(self.fps_cap / hz)))

    def _build_circular_pupil(self, M: int):
        y, x = cp.mgrid[0:M, 0:M]
        cy = (M - 1) / 2.0
        cx = (M - 1) / 2.0
        r = cp.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return (r <= (M / 2.0)).astype(cp.float32)

    def _ensure_pupil_mask(self, M: int):
        """Ensure pupil mask exists at shape (M,M). Returns boolean pupil mask."""
        if self.pupil_mask is None or self.pupil_mask.shape != (M, M):
            self.pupil_mask = self._build_circular_pupil(M)
        return (self.pupil_mask > 0)

    def _apply_dm_to_phases(self, phases_atm):
        if (not self._loop_enabled) or (self._dm_phi is None):
            return phases_atm
        return phases_atm.copy() - self._dm_phi

    def _invalidate_frame_cache(self):
        self._frame_tick = -1
        self._frame_need_per_layer = False
        self._frame_phases_M = None        # (S, patch_M, patch_M)
        self._frame_layers_M = None
        self._frame_phases_psfM = None
        self._frame_layers_layerM = None

    def _downsample_stack(self, arr, M_out: int):
        M_in = int(arr.shape[-1])
        M_out = int(M_out)
        if M_out == M_in:
            return arr

        f = max(1, M_in // M_out)
        if f > 1 and f * M_out == M_in:
            if arr.ndim == 3:
                S = arr.shape[0]
                return arr.reshape(S, M_out, f, M_out, f).mean(axis=(2, 4))
            if arr.ndim == 4:
                L, S = arr.shape[0], arr.shape[1]
                return arr.reshape(L, S, M_out, f, M_out, f).mean(axis=(3, 5))
        return arr[..., ::f, ::f]

    def _stats_view(self, x):
        M_in = int(x.shape[-1])
        M_out = int(self._display_stats_M)
        if M_in <= M_out:
            return x
        f = max(1, M_in // M_out)
        return x[..., ::f, ::f]

    def _update_levels(self, phase_like=None, psf_like=None):
        if (self._tick_count - self._last_level_tick) < self._level_period:
            return
        if phase_like is not None:
            v = self._stats_view(phase_like)
            self._phase_mu = cp.mean(v)
            self._phase_sig = cp.std(v) + cp.asarray(1e-6, dtype=cp.float32)
        if psf_like is not None:
            v = self._stats_view(psf_like)
            self._psf_vmax = cp.max(v)
        self._last_level_tick = self._tick_count

    def _phase_to_u8_cached(self, phi):
        return cp.asnumpy(phi.astype(cp.float16))

    def _logpsf_to_u8_cached(self, logI):
        """Convert a log-PSF image to a host uint8 array for UI display."""
        vmax = cp.max(logI)
        vmin = cp.min(logI)
        return cp.asnumpy(_u8_linear(logI, vmin, vmax))

    # ------------------ sampling ------------------
    def _sample_frame_once(self, need_per_layer: bool):
        if self.sim is None:
            return None, None

        if self._frame_tick == self._tick_count and (not need_per_layer or self._frame_need_per_layer):
            return self._frame_phases_M, self._frame_layers_M

        self._frame_phases_psfM = None
        self._frame_layers_layerM = None

        if need_per_layer:
            outL = self.sim.sample_patches_batched(
                thetas_xy_rad=self.thetas_gpu,
                ranges_m=self.ranges_gpu,
                size_pixels=self.patch_size_px,
                M=self.patch_M,
                angle_deg=0.0,
                remove_piston=False,
                return_gpu=True,
                return_per_layer=True,
                layer_first=True,
            )

            phases = cp.sum(outL, axis=0)
            phases = phases - cp.mean(phases, axis=(1, 2), keepdims=True)
            outL = outL - cp.mean(outL, axis=(2, 3), keepdims=True)

            self._frame_layers_M = outL
            self._frame_need_per_layer = True
        else:
            phases = self.sim.sample_patches_batched(
                thetas_xy_rad=self.thetas_gpu,
                ranges_m=self.ranges_gpu,
                size_pixels=self.patch_size_px,
                M=self.patch_M,
                angle_deg=0.0,
                remove_piston=True,
                return_gpu=True,
                return_per_layer=False,
            )
            self._frame_layers_M = None
            self._frame_need_per_layer = False

        self._frame_phases_M = phases
        self._frame_tick = self._tick_count
        return phases, self._frame_layers_M

    # ------------------ lifecycle ------------------
    @Slot()
    def build_sim(self):
        if self.sim is not None:
            return

        cp.cuda.Device().use()
        self.status_log.emit("Building LayeredPhaseScreen on worker thread...")
        self.status.emit("Building Sim")

        self.status_log.emit("   Creating sim...")
        self.sim = self.sim_factory(**self.sim_kwargs)
        for i, j in self.sim_kwargs.items():
            self.status_log.emit("     " + str((i, j)))
        max_theta = max([sensor.sen_angle for sensor in self._sensor_list]) if self._sensor_list else 0.0

        for k, layer in enumerate(self.layer_cfgs):
            self.status_log.emit(f"   Adding layer {k+1}...")
            for i, j in layer.items():
                self.status_log.emit("     " + str((i, j)))
            self.sim.add_layer(max_theta=max_theta, **layer)

        self.status_log.emit("Screen Sim built!")

        self._pupil_mask_psf = self._build_circular_pupil(self.psf_M)

        if self.pupil_mask is None:
            self._pupil_mask_patch = self._build_circular_pupil(self.patch_M)
        else:
            self._pupil_mask_patch = self.pupil_mask
            if self._pupil_mask_patch.shape != (self.patch_M, self.patch_M):
                self._pupil_mask_patch = self._build_circular_pupil(self.patch_M)

        self.status_log.emit(f"Built sim with {len(self.layer_cfgs)} layers.")
        self.status.emit("Building tomographic reconstructor")

        self.update_r0.emit(getattr(self.sim, "r0_total", 0.0))
        self.initialized = True
        self._emit_once()
        self._emit_visible_layer()

        layer_heights_m = [0.0]
        self.R = TomoOnAxisIM_CuPy(
            sensors=self._sensor_list,
            layer_heights_m=layer_heights_m,
            dx_world_m=float(self.sim.dx),
            M=self.patch_M,
            size_pixels=self.patch_M,
            reg_alpha=1e-2,
            reg_beta=1e-1,
            science_sensor_index=0,          # exclude science WFS from recon
            slopes_aggregation="stack",
            remove_waffle_in_solution=True,
            waffle_per_layer=True,
            reg_waffle_gamma=0.0,  
        )
        for h in self.R.layer_heights_m:
            for s in self.R.recon_sensors:
                sx = (h * float(s.dx)) / self.R.dx_world_m
                sy = (h * float(s.dy)) / self.R.dx_world_m
                self.status_log.emit(f"h, {h}, sx_px, {sx}, sy_px, {sy}")
        
        

        self.status_log.emit("Tomographic reconstructor initialized!")

        self.status.emit("Building interaction matrix")
        self.R.build_interaction_matrix(sensor_method=self.sensor_method)
        self.status_log.emit("Interaction matrix built!")

        self.R.factorize(rcond=1e-12)
        self.status_log.emit("Interaction matrix factorized!")

        self.R.prepare_runtime()
        self.status_log.emit("Runtime coeff matrix prepared!")

        

        At = self.R._At              # (n_modes, n_slopes)
        V = self.R._eig_V            # (n_modes, k)
        inv = self.R._eig_inv        # (k,)

        tmp = (V.T @ At) * inv[:, None]
        Rmat = V @ tmp

        A = self.R.A.astype(cp.float32)
        H0 = A.T @ A
        d_H0 = cp.mean(cp.diag(H0))
        d_beta = cp.mean(cp.diag(self.R.reg_beta * self.R.G)) if (self.R.G is not None) else 0
        d_alpha = (self.R.reg_alpha**2)

        self.status_log.emit("Reconstruction Matrix: ")
        self.status_log.emit(f"  mean diag(A^T A): {float(d_H0.get())}")
        self.status_log.emit(f"  mean diag(beta*G): {float(d_beta.get())}")
        self.status_log.emit(f"  alpha^2: {d_alpha}")
        self.status_log.emit(f"  beta*G / A^T A: {float((d_beta / (d_H0+1e-12)).get())}")
        self.status_log.emit(f"  alpha^2 / A^T A: {float((d_alpha / (d_H0+1e-12)).get())}")
        self.status_log.emit("")
        self.recon_matr_ready.emit(cp.asnumpy(self.R.A), cp.asnumpy(Rmat))
        self.status_log.emit(f"  Rmatrix min/max {cp.min(self.R.A)}, {cp.max(self.R.A)}")
        self.status_log.emit(f"  Rinv    min/max {cp.min(Rmat)}, {cp.max(Rmat)}")

        self.status.emit("Sim ready")


        self._emit_once()
        self.status_running.emit(False)

    @Slot()
    def start(self):
        if self.sim is None:
            self.build_sim()

        self._tick_count = 0
        self._fps_frames = 0
        self._fps_t0_wall = None
        self._prev_tick_wall = None
        self._prev_tick_end_wall = None

        self._invalidate_frame_cache()
        self.timer.start(self._interval_ms)
        self.status_log.emit("Running (continuous).")
        self.status.emit("Running Sim")
        self.status_running.emit(True)

    @Slot()
    def pause(self):
        self.timer.stop()
        self._stepping = False
        self._steps_left = 0
        self.status_log.emit("Paused.")
        self.status.emit("Sim Paused")
        self.status_running.emit(False)

    @Slot()
    def stop(self):
        self.timer.stop()
        self._stepping = False
        self._steps_left = 0
        self.finished.emit()
        self.status_log.emit("Stopped.")
        self.status.emit("Sim stopped")
        self.status_running.emit(False)

    @Slot(int)
    def set_sync_profile_every(self, n: int):
        """For diagnostics: if n>0, synchronize default stream every n ticks."""
        try:
            n = int(n)
        except Exception:
            n = 0
        self._sync_profile_every = max(0, n)

    # ------------------ external API (restored for GUI signals) ------------------
    @Slot(dict)
    def add_layer(self, layer_cfgs):
        if self.sim is None:
            self.build_sim()

        self.status_log.emit("Adding layer...")
        cfg = dict(layer_cfgs)

        max_theta = max([sensor.sen_angle for sensor in self._sensor_list]) if self._sensor_list else 0.0
        try:
            self.sim.add_layer(max_theta=max_theta, **cfg)
        except TypeError:
            self.sim.add_layer(**cfg)

        self.layer_cfgs.append(cfg)

        if hasattr(self.sim, "layers"):
            self.sim_signal.emit(self.sim.layers)

        self.update_r0.emit(getattr(self.sim, "r0_total", 0.0))

        self._invalidate_frame_cache()
        self._emit_once()

    @Slot(dict)
    def sensor_update(self, sensors):
        self.sensors = sensors
        self._sensor_list = list(self.sensors.values())
        self._sensor_keys = list(self.sensors.keys())

        self.thetas_gpu = cp.asarray([s.field_angle for s in self._sensor_list], dtype=cp.float32)
        S = int(self.thetas_gpu.shape[0])

        r = cp.asarray([float(getattr(s, "gs_range_m", cp.inf)) for s in self._sensor_list], dtype=cp.float32)
        if r.size != S:
            r = cp.full((S,), cp.inf, dtype=cp.float32)
        self.ranges_gpu = r

        self._invalidate_frame_cache()

    @Slot()
    def reset(self):
        if self.sim is None:
            return

        self.timer.stop()
        self._stepping = False
        self._steps_left = 0

        self._loop_t = 0.0
        self._dm_cmd = None
        self._dm_sigma = None
        self._dm_phi = None
        self._s_ref = None

        if self.long_exposure_psf is not None:
            self.long_exposure_psf.reset()
            self.long_exposure_psf_uncorrected.reset()

        if hasattr(self.sim, "hard_reset"):
            self.status.emit("Sim resetting")
            self.status_running.emit(True)

            self.sim.hard_reset()

            self.status_log.emit("Hard reset to initial frame.")
            self.status.emit("Sim reset to initial frames")
            self.status_running.emit(False)
        else:
            self.status_log.emit("Unable to hard reset! ")
            self.status.emit("Unable to hard reset! ")

        self._invalidate_frame_cache()
        self._emit_once()

    # ------------------ stepping ------------------
    @Slot(bool)
    def step_once(self, status=True):
        if self.sim is None:
            self.build_sim()

        self.timer.stop()
        self.status_running.emit(True)

        self._ensure_cache()
        self._advance(1)
        self._emit_once()

        self.status_running.emit(False)
        if status:
            self.status.emit("Sim stepped once")

    @Slot(int, int)
    def step_n(self, n: int, emit_every: int = 0):
        if self.sim is None:
            self.build_sim()

        self.timer.stop()
        self._stepping = True
        self._steps_left = max(0, int(n))
        self._emit_every = max(0, int(emit_every))
        self.status.emit(f"Sim stepping {n} times")
        self.status_running.emit(True)

        QTimer.singleShot(0, self._tick_step_n)

    def _tick_step_n(self):
        if not self._stepping or self._steps_left <= 0:
            self._stepping = False
            self._emit_once()
            self.status.emit("Sim finished stepping")
            self.status_running.emit(False)
            return

        chunk = min(self._batch, self._steps_left)

        if self._emit_every == 0:
            self._advance(chunk)
            self._steps_left -= chunk
        else:
            for _ in range(chunk):
                self._advance(1)
                self._steps_left -= 1
                if self._steps_left % self._emit_every == 0:
                    self._emit_once()

        QTimer.singleShot(0, self._tick_step_n)

    # ------------------ overview controls ------------------
    @Slot(bool)
    def set_overview_enabled(self, en: bool):
        self._overview_enabled = bool(en)
        if self._overview_enabled:
            self.emit_overview_once()

    @Slot()
    def emit_overview_once(self):
        if self.sim is None:
            self.build_sim()
        self._ensure_cache()
        self._invalidate_frame_cache()
        was = self._overview_enabled
        self._overview_enabled = True
        self._emit_overview_from_cache()
        self._overview_enabled = was

    @Slot(int)
    def set_visible_layer(self, idx: int):
        self._visible_layer = int(idx)
        self._emit_visible_layer()

    @Slot(int)
    def set_visible_sensor(self, idx: int):
        self._visible_sensor = int(idx)

    # ------------------ layer edits ------------------
    @Slot(list, bool)
    def set_active(self, names: list, active: bool):
        for name in names:
            self.sim.set_layer_active_name(name, active)

        self.update_r0.emit(self.sim.r0_total)
        self.sim_signal.emit(self.sim.layers)
        self._emit_once()

    @Slot(int, object)
    def apply_layer_params(self, idx: int, params: dict):
        if self.sim is None:
            self.build_sim()

        i = int(idx)
        p = dict(params)
        p.pop("name", None)

        if "altitude_m" in p:
            h = float(p["altitude_m"])
            if hasattr(self.sim, "set_layer_altitude"):
                self.sim.set_layer_altitude(i, h)

        if "wind" in p and isinstance(p["wind"], (list, tuple)) and len(p["wind"]) == 2:
            vx, vy = float(p["wind"][0]), float(p["wind"][1])
            if hasattr(self.sim, "set_layer_wind"):
                self.sim.set_layer_wind(i, vx, vy)
            else:
                self.sim.layers[i]["vx"] = vx
                self.sim.layers[i]["vy"] = vy

        if "L0" in p:
            L0 = float(p["L0"])
            if hasattr(self.sim, "set_layer_L0"):
                self.sim.set_layer_L0(i, L0)
            else:
                self.sim.layers[i]["L0"] = L0
                self.sim.layers[i]["phi"] = None
                self.sim.layers[i]["phi_t"] = None

        if "r0" in p:
            r0 = float(p["r0"])
            if hasattr(self.sim, "set_layer_r0"):
                self.sim.set_layer_r0(i, r0)
            else:
                self.sim.layers[i]["r0"] = r0
            self.update_r0.emit(self.sim.r0_total)

        if "seed_offset" in p:
            new_off = int(p["seed_offset"])
            old_off = int(self.layer_cfgs[i].get("seed_offset", 0))
            if new_off != old_off:
                self.layer_cfgs[i]["seed_offset"] = new_off
                self._regen_single_layer(i)

        if i < len(self.sim.layers):
            self.sim.layers[i]["phi"] = None
            self.sim.layers[i]["phi_t"] = None

        self.step_once(status=False)

    def _regen_single_layer(self, i: int):
        if hasattr(self.sim, "regen_layer"):
            self.sim.regen_layer(i, seed_offset=self.layer_cfgs[i].get("seed_offset", 0))
            return

        self.status_log.emit(f"Rebuilding sim to apply seed_offset for layer {i}...")
        sim_kwargs = dict(self.sim_kwargs)
        self.sim = self.sim_factory(**sim_kwargs)
        for layer in self.layer_cfgs:
            self.sim.add_layer(**layer)
        self._pupil_mask_psf = self._build_circular_pupil(self.psf_M)
        self.status_log.emit("Rebuilt sim.")

    # ------------------ internals ------------------
    def _advance(self, n_steps: int):
        if self.sim is None:
            return
        if hasattr(self.sim, "advance"):
            for _ in range(int(n_steps)):
                self.sim.advance(self.dt_s)
        else:
            if hasattr(self.sim, "time"):
                self.sim.time += self.dt_s * int(n_steps)
        self._invalidate_frame_cache()

    def _ensure_cache(self):
        if hasattr(self.sim, "_ensure_layer_cache"):
            self.sim._ensure_layer_cache()

    def _combined_gpu(self):
        N = int(self.sim_kwargs["N"])
        if self.sim is None or len(getattr(self.sim, "layers", [])) == 0:
            return cp.zeros((N, N), dtype=cp.float32)

        buf = self._combined_buf
        if buf is None or buf.shape != (N, N):
            buf = cp.empty((N, N), dtype=cp.float32)
            self._combined_buf = buf
        buf.fill(0.0)

        any_active = False
        for i in range(len(self.sim.layers)):
            if not self.sim._layers_obj[i].active:
                continue
            any_active = True
            buf += self.sim.get_layer_view(i, size_pixels=N, M=N, angle_deg=0.0, return_gpu=True)

        if not any_active:
            buf.fill(0.0)
            return buf

        buf -= cp.mean(buf)
        return buf

    def _phase_to_psf_logI(self, phase, pupil):
        E = pupil * cp.exp(1j * phase)
        F = cp.fft.fftshift(cp.fft.fft2(E))
        I = (F.real * F.real + F.imag * F.imag).astype(cp.float32)
        return cp.log10(I + 1e-12)

    # ------------------ emissions ------------------
    def _emit_visible_layer(self):
        if self.sim is None or self._visible_layer is None or not self._detailed_enabled:
            return
        i = int(self._visible_layer)
        if not (0 <= i < len(self.sim.layers)):
            return

        _, outL = self._sample_frame_once(need_per_layer=True)
        if outL is None:
            return

        outL = outL[i][self._visible_sensor]
        self._update_levels(phase_like=outL)
        self.layer_ready.emit(i, self._phase_to_u8_cached(outL))

    def _emit_visible_sensor_products_from_cache(self):
        if self.sim is None and self._detailed_enabled:
            return
        sidx = int(self._visible_sensor)
        S = int(self.thetas_gpu.shape[0])
        if not (0 <= sidx < S):
            return
        if not self._detailed_enabled and not self._reconstructor_enabled:
            return

        phases_atm, _ = self._sample_frame_once(need_per_layer=False)
        if phases_atm is None:
            return
        phases = self._apply_dm_to_phases(phases_atm)

        phase0 = phases[sidx]
        if self._detailed_enabled:
            self._update_levels(phase_like=phase0)
            self.sensor_phase_ready.emit(sidx, self._phase_to_u8_cached(phase0))

            if self.emit_psf:
                pupil = self._pupil_mask_patch
                psf = self._phase_to_psf_logI(phase0, pupil)
                self._update_levels(psf_like=psf)
                self.sensor_psf_ready.emit(sidx, self._logpsf_to_u8_cached(psf))

    def _compute_all_sensor_psfs_from_cache(self):
        phases_atm, _ = self._sample_frame_once(need_per_layer=False)
        if phases_atm is None:
            return None
        phases = self._apply_dm_to_phases(phases_atm)

        if self.psf_M != self.patch_M:
            if self._frame_phases_psfM is None:
                self._frame_phases_psfM = self._downsample_stack(phases, self.psf_M)
            phases_psf = self._frame_phases_psfM
        else:
            phases_psf = phases

        self._update_levels(phase_like=phases_psf)
        self.all_sensor_phase_ready.emit(self._phase_to_u8_cached(phases_psf))

        pupil = self._pupil_mask_psf
        E = pupil[None, :, :] * cp.exp(1j * phases_psf)
        F = cp.fft.fftshift(cp.fft.fft2(E, axes=(-2, -1)), axes=(-2, -1))
        I = (F.real * F.real + F.imag * F.imag).astype(cp.float32)
        return cp.log10(I + 1e-12)

    def _emit_overview_from_cache(self):
        if not self._overview_enabled or self.sim is None:
            return
        _, outL = self._sample_frame_once(need_per_layer=True)
        if outL is None:
            return

        if self.layer_M != self.patch_M:
            if self._frame_layers_layerM is None:
                self._frame_layers_layerM = self._downsample_stack(outL, self.layer_M)
            outLd = self._frame_layers_layerM
        else:
            outLd = outL

        self._update_levels(phase_like=outLd)
        self.all_layers_ready.emit(self._phase_to_u8_cached(outLd))

    def _emit_once(self):
        """One-shot emit used by GUI actions. Measures/display using residual (WFS after DM) if loop enabled."""
        if not self.initialized:
            return

        self._ensure_cache()
        self._invalidate_frame_cache()

        # Combined view is atmospheric-only (not DM-corrected)
        if self._detailed_enabled:
            combined = self._combined_gpu()
            self._update_levels(phase_like=combined)
            self.combined_ready.emit(self._phase_to_u8_cached(combined))

        if self.R is None:
            self._emit_visible_sensor_products_from_cache()
            return

        phases_atm, _ = self._sample_frame_once(need_per_layer=False)
        if phases_atm is None:
            return
        
        m = self._ensure_pupil_mask(phases_atm.shape[-1])
        phases_res = self._apply_dm_to_phases(phases_atm)

        # --- slopes from non-science sensors (measure on residual) ---
        slopes_list = []
        for s0, ph in zip(self._sensor_list[1:], phases_res[1:]):
            try:
                _, sl, _ = s0.measure(
                    phase_map=ph,
                    return_image=False,
                    assume_radians=True,
                    method=self.sensor_method,
                )
            except TypeError:
                _, sl, _ = s0.measure(phase_map=ph, return_image=False)
            slopes_list.append(sl[0])

        # --- reference slopes (flat) ---
        if self._s_ref is None:
            self._s_ref = []
            for s0, ph in zip(self._sensor_list[1:], phases_atm[1:]):
                ph0 = cp.zeros_like(ph)
                try:
                    _, sl0, _ = s0.measure(
                        phase_map=ph0,
                        return_image=False,
                        assume_radians=True,
                        method=self.sensor_method,
                    )
                except TypeError:
                    _, sl0, _ = s0.measure(phase_map=ph0, return_image=False)
                self._s_ref.append(sl0[0].copy())

        # display recon: subtract s_ref (TT removal is display-only)
        slopes_sub = [sl - ref for sl, ref in zip(slopes_list, self._s_ref)]
        rec_phase, xhat = self.R.reconstruct_onaxis(slopes_sub, return_coeffs=True)
        rec_phase -= cp.mean(rec_phase[m])
        rec_phase = remove_tt(rec_phase, self.pupil_mask)

        # show on-axis residual (what science sees this tick)
        phase0 = phases_res[0] - cp.mean(phases_res[0][m])
        phase0 = remove_tt(phase0, self.pupil_mask)

        # “perfect instantaneous” correction preview
        phi_dm_perf = -self.R.coeffs_to_onaxis_phase(xhat)

        phi_res_perf = (phases_atm[0] + phi_dm_perf) * self._pupil_mask_patch
        phi_res_perf -= cp.mean(phi_res_perf[m])
        phi_res_perf = remove_tt(phi_res_perf, self.pupil_mask)

        surf_m = Pupil_tools.dm_surface_from_commands(
            commands=-xhat,
            pupil=self.pupil_mask,
            grid_size=self.patch_M,
            sigma=None,
            dtype=cp.float32,
        )

        self.reconstructed_ready.emit(
            cp.asnumpy(phase0 * self._pupil_mask_patch),
            cp.asnumpy(rec_phase),
            cp.asnumpy(surf_m),
            cp.asnumpy(phi_res_perf),
        )

        self._emit_visible_sensor_products_from_cache()

        psfs = self._compute_all_sensor_psfs_from_cache()
        if psfs is not None:
            self._update_levels(psf_like=psfs)
            self.all_sensor_psf_ready.emit(self._logpsf_to_u8_cached(psfs))

        was = self._overview_enabled
        self._overview_enabled = True
        self._emit_overview_from_cache()
        self._overview_enabled = was

    # ------------------ main tick ------------------
    @Slot()
    def _tick(self):
        wall_start = time.perf_counter_ns()
        self._tick_count += 1

        tn = (lambda: time_n(sync=self._profile_gpu_timings))
        t0 = tn()

        self._ensure_cache()
        self._advance(1)

        tc = self._tick_count
        need_visible_sensor = ((tc % self._sensor_period) == 0)
        need_visible_tab = ((tc % self._layer_tab_period) == 0)
        need_overview_psf = (self._overview_enabled and ((tc % self._overview_psf_period) == 0))
        need_overview_layer = (self._overview_enabled and ((tc % self._overview_layer_period) == 0))
        need_combined = ((tc % self._combined_period) == 0)

        t1 = tn()

        # Sample atmosphere (cached) then apply DM (without mutating cache)
        phases_atm, _ = self._sample_frame_once(need_per_layer=False)   # (S,M,M)
        if phases_atm is None:
            return
        phase_atm0 = phases_atm[0].copy()
        phase_atm0 -= cp.mean(phases_atm[0], keepdims=True)
        

        m = self._ensure_pupil_mask(phases_atm.shape[-1])
        phases_res = self._apply_dm_to_phases(phases_atm)

        do_recon = bool(self._reconstructor_enabled and (phases_atm is not None) and (self.R is not None))
        need_slopes = bool(do_recon or self._loop_enabled)

        # on-axis phase for display (residual, TT removed only for metrics panels)
        phase0 = None
        if need_slopes:
            phase0 = (phases_res[0] - cp.mean(phases_res[0][m]))
            phase0 = remove_tt(phase0, self.pupil_mask)

        t2 = tn()

        slopes_list = []
        if need_slopes:
            # --- measure slopes on residual ---
            for s0, ph in zip(self._sensor_list[1:], phases_res[1:]):
                try:
                    _, sl, _ = s0.measure(
                        phase_map=ph,
                        return_image=False,
                        assume_radians=True,
                        method=self.sensor_method,
                    )
                except TypeError:
                    _, sl, _ = s0.measure(phase_map=ph, return_image=False)
                slopes_list.append(sl[0])   # (n_sub, 2)

            # --- one-time reference slope calibration (flat) ---
            if self._s_ref is None:
                self._s_ref = []
                for s0, ph in zip(self._sensor_list[1:], phases_res[1:]):
                    ph0 = cp.zeros_like(ph)
                    try:
                        _, sl0, _ = s0.measure(
                            phase_map=ph0,
                            return_image=False,
                            assume_radians=True,
                            method=self.sensor_method,
                        )
                    except TypeError:
                        _, sl0, _ = s0.measure(phase_map=ph0, return_image=False)

                    self._s_ref.append(sl0[0].copy())

        t3 = tn()

        # ---- optional recon display (subtract s_ref; TT removed for display) ----
        rec = None
        surf_m = cp.zeros_like(phases_atm[0])
        phi_res_disp = None

        if do_recon and slopes_list:
            slopes_sub = [sl - ref for sl, ref in zip(slopes_list, self._s_ref)]
            self.reconstructed, xhat_disp = self.R.reconstruct_onaxis(slopes_sub, return_coeffs=True)
            self.reconstructed -= cp.mean(self.reconstructed[m])
            self.reconstructed = remove_tt(self.reconstructed, self.pupil_mask)
            rec = self.reconstructed - cp.mean(self.reconstructed)

            # “perfect instantaneous” correction preview
            phi_dm_perf = -self.R.coeffs_to_onaxis_phase(xhat_disp)
            phi_res_disp = (phases_atm[0] + phi_dm_perf) * self._pupil_mask_patch

            surf_m = Pupil_tools.dm_surface_from_commands(
                commands=-xhat_disp,
                pupil=self.pupil_mask,
                grid_size=self.patch_M,
                sigma=None,
                dtype=cp.float32,
            )

        # ---- CLOSED LOOP ----
        if self._closed_loop and self._loop_enabled and slopes_list:
            slopes_err_list = [sl - ref for sl, ref in zip(slopes_list, self._s_ref)]
            slopes_err_list = [strip_tt_from_slopes(sl) if sens.gs_range_m != cp.inf else cp.zeros_like(sl) for sl, sens in zip(slopes_err_list, self._sensor_list[1:]) ]
            xhat = self.R.solve_coeffs(slopes_err_list)

            if self._dm_cmd is None:
                self._dm_cmd = cp.zeros_like(xhat)

            self._dm_cmd = (1.0 - self._loop_leak) * self._dm_cmd - self._loop_gain * xhat

            dm_phi = -self.R.coeffs_to_onaxis_phase(self._dm_cmd)
            dm_phi -= cp.mean(dm_phi[m])
            dm_phi = remove_tt(dm_phi, self.pupil_mask)
            self._dm_phi = dm_phi

            # ---- UI at limited rate ----
            if (self._fps_frames % self._loop_disp_period) == 0 or self._loop_t == 0:
                phi_sci = phases_res[0]
                m_f = self.pupil_mask.astype(cp.float32)
                m_sum = cp.sum(m_f)

                phi_sci = phi_sci - cp.sum(phi_sci * m_f) / (m_sum + 1e-12)
                phi_sci = remove_tt(phi_sci, self.pupil_mask)

                # r0 calculation
                V = cp.stack([phi_sci, phase_atm0], axis=0)   # (2,H,W)
                Ds = []
                rhos = []
                for lag in (2,4,8):
                    mx_f = (m[:, lag:] & m[:, :-lag]).astype(cp.float32)
                    my_f = (m[lag:, :] & m[:-lag, :]).astype(cp.float32)

                    dpx = V[:, :, lag:] - V[:, :, :-lag]      # (2,H,W-lag)
                    dpy = V[:, lag:, :] - V[:, :-lag, :]      # (2,H-lag,W)

                    Dx = (dpx*dpx*mx_f[None, :, :]).sum(axis=(1,2)) / (mx_f.sum() + 1e-12)
                    Dy = (dpy*dpy*my_f[None, :, :]).sum(axis=(1,2)) / (my_f.sum() + 1e-12)
                    D  = 0.5*(Dx + Dy) # Phase Structure Function estimation
                    Ds.append(D)

                    rho = self.sim.dx * lag
                    rhos.append(rho)


                Ds = cp.stack(Ds, axis=0)        # (nlag, 2)
                rhos = cp.asarray(rhos, dtype=cp.float32)[:, None]  # (nlag,1)

                slope = 5.0/3.0
                y = cp.log(Ds) - cp.log(cp.float32(6.88)) - slope*cp.log(rhos)
                log_r0 = -cp.mean(y, axis=0) / slope
                r0_eff_corrected, r0_eff_uncorrected = cp.exp(log_r0)
                            
                # Kolmogorov: Dphi(rho) = 6.88 * (rho/r0)^(5/3)
                
                if not self._loop_r0_only:
                        
                    fft_pad = self.patch_M // 2
                    I, I_uncorrected = self._sensor_list[0].measure(pupil = None, phase_map = cp.stack((phi_sci, phase_atm0), axis = 0), pad = fft_pad, assume_radians = True)[2]

                    if self.plate_rad is None:
                        self.plate_rad = (
                            float(self._sensor_list[0].wavelength) * float(self.patch_M) /
                            (float(params["telescope_diameter"]) * float(I.shape[0] + 2 * fft_pad))
                        )

                    if self.long_exposure_psf is None:
                        self.long_exposure_psf = Analysis.PSFRingMean(shape=I.shape, K=self.long_exposure_frames, save_label = "Corrected")
                        self.long_exposure_psf_uncorrected = Analysis.PSFRingMean(shape=I_uncorrected.shape, K=self.long_exposure_frames, save_label = "Uncorrected")

                    self.long_exposure_psf.add(I)
                    self.long_exposure_psf_uncorrected.add(I_uncorrected)

                    sci_long = self.long_exposure_psf.mean()
                    raw_long = self.long_exposure_psf_uncorrected.mean()

                    zoom = 2
                    zimg = Analysis._zoom2d(sci_long, zoom, cp)     # (H*zoom, W*zoom)
                    p, (xc, yc), fwhm, roi, (x0, y0), fwhm_z = self.gaussian_fitter.fit_frame(
                        zimg, search_r=24, loss="soft_l1", max_iter=10, zoom=zoom
                    )
                    zimg_uc = Analysis._zoom2d(raw_long, zoom, cp)     # (H*zoom, W*zoom)
                    _, _, fwhm_uncorrected, _, _, _  = self.gaussian_fitter_uc.fit_frame(
                        zimg_uc, search_r=24, loss="soft_l1", max_iter=10, zoom=zoom
                    )

                    self.loop_ready.emit(
                        (roi.astype(cp.float16)).get(),
                        (phi_sci * self.pupil_mask).get(), 
                        (r0_eff_corrected),
                        (r0_eff_uncorrected),
                        (self.plate_rad * fwhm[0]),
                        (self.plate_rad * fwhm_uncorrected[0]),
                        self._loop_t,
                        (p[1], p[2], fwhm_z[1]/2) # gaussian x, y, radius
                    )
                else:
                    self.loop_ready.emit(
                        None,
                        (phi_sci * self.pupil_mask).get(), 
                        (r0_eff_corrected),
                        (r0_eff_uncorrected),
                        0,
                        0,
                        self._loop_t,
                        (0, 0, 1, 1, 0) # gaussian x, y, x width, y width, rotation
                    )

            self._loop_t += self.dt_s
            


        t4 = tn()

        # ---- emissions ----
        t_c_0 = t_c_1 = 0
        if need_combined:
            t_c_0 = tn()
            if self._detailed_enabled:
                combined = self._combined_gpu()
                self._update_levels(phase_like=combined)
                self.combined_ready.emit(self._phase_to_u8_cached(combined))

            if rec is not None and do_recon and phase0 is not None and phi_res_disp is not None:
                self.reconstructed_ready.emit(
                    cp.asnumpy(phase0 * self._pupil_mask_patch),
                    cp.asnumpy(rec),
                    cp.asnumpy(surf_m),
                    cp.asnumpy(phi_res_disp),
                )

            t_c_1 = tn()

        t_s_0 = t_s_1 = 0
        if need_visible_sensor:
            t_s_0 = tn()
            self._emit_visible_sensor_products_from_cache()
            t_s_1 = tn()

        if need_visible_tab:
            self._emit_visible_layer()

        t_os_0 = t_os_1 = 0
        if need_overview_psf:
            t_os_0 = tn()
            psfs = self._compute_all_sensor_psfs_from_cache()
            if psfs is not None:
                self._update_levels(psf_like=psfs)
                self.all_sensor_psf_ready.emit(self._logpsf_to_u8_cached(psfs))
            t_os_1 = tn()

        t_ol_0 = t_ol_1 = 0
        if need_overview_layer:
            t_ol_0 = tn()
            self._emit_overview_from_cache()
            t_ol_1 = tn()

        # ---- FPS / timing stats (wall clock) ----
        wall_end = time.perf_counter_ns()

        # Optional: force a sync every N ticks so timing reflects queued GPU work.
        if self._sync_profile_every and (self._tick_count % int(self._sync_profile_every) == 0):
            try:
                cp.cuda.Stream.null.synchronize()
            except Exception:
                pass
            wall_end = time.perf_counter_ns()

        if self._fps_t0_wall is None:
            self._fps_t0_wall = wall_end
            self._fps_frames = 0
        self._fps_frames += 1

        avg_period_ms = (wall_end - self._fps_t0_wall) / 1e6 / max(1, self._fps_frames)
        tick_wall_ms = (wall_end - wall_start) / 1e6

        period_ms = None
        if self._prev_tick_wall is not None:
            period_ms = (wall_start - self._prev_tick_wall) / 1e6
        self._prev_tick_wall = wall_start

        idle_ms = None
        if self._prev_tick_end_wall is not None:
            idle_ms = (wall_start - self._prev_tick_end_wall) / 1e6
        self._prev_tick_end_wall = wall_end

        need_status = (wall_end - self._fps_t0_wall) > 0.5e9 or need_visible_tab or need_visible_sensor or need_overview_psf or need_overview_layer
        if need_status:
            fps = 1000.0 / avg_period_ms if avg_period_ms > 0 else 0.0
            self.fpsReady.emit(fps)
            extra = ""
            if period_ms is not None:
                extra += f" | period {period_ms:.1f} ms"
            if idle_ms is not None:
                extra += f" | idle {idle_ms:.1f} ms"
            self.status.emit(
                f"Running Sim {fps:.0f} fps | avg {avg_period_ms:.1f} ms/t | tick {tick_wall_ms:.1f} ms{extra}"
                f"\nAdvanceSim {(t1-t0)/1e6:.2f}ms | ScrnSample {(t2 - t1)/1e6:.1f}ms | SensorMeasure {(t3-t2)/1e6:.1f}ms | Reconstruct {(t4-t3)/1e6:.1f}ms"
                f"\nVisibleSensor {(t_s_1-t_s_0)/1e6:.1f}ms | Combined {(t_c_1-t_c_0)/1e6:.1f}ms | OverviewPSF {(t_os_1-t_os_0)/1e6:.1f}ms | OverviewLayers {(t_ol_1-t_ol_0)/1e6:.1f}ms"
            )
            self._fps_t0_wall = wall_end
            self._fps_frames = 0
