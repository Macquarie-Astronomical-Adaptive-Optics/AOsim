# scripts/schedulerGPU.py
import time
import datetime
from pathlib import Path
import json
import os
import inspect
import traceback
import math
from collections import deque
from typing import Any, Dict, List, Optional

import cupy as cp
import logging

logger = logging.getLogger(__name__)
from PySide6.QtCore import QObject, Signal, Slot, QTimer, Qt

from scripts.reconstructor import TomoOnAxisIM_CuPy, TipTiltReconstructor_CuPy, GLAOConjugateIM_CuPy
from scripts.utilities import Pupil_tools, Analysis, Gaussian2DFitter, params as default_params
from scripts.batch_runner import AOBatchRunner


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

    # long-run batch outputs
    long_run_started = Signal(object)
    long_run_progress = Signal(int, int, float, str)   # done, total, fps, message
    long_run_finished = Signal(object)
    long_run_failed = Signal(object)

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

        # parameter source (prefer explicit injection over module globals)
        params: Optional[Dict[str, Any]] = None,

        parent=None,

    ):
        super().__init__(parent)

        # Params
        self.params = params if params is not None else default_params



        # ---------------- DM servo lag (frame-delayed actuation) ----------------
        # dm_delay_frames = number of simulation frames between computing a DM command
        # and it becoming active. Default=1 matches the previous behaviour
        # ("new DM command becomes active next tick").
        try:
            self._dm_delay_frames = int(self.params.get("dm_delay_frames", self.params.get("dm_servo_lag_frames", 1)))
        except Exception:
            self._dm_delay_frames = 1
        self._dm_delay_frames = max(1, int(self._dm_delay_frames))
        self._dm_phi_pipeline = deque([None] * self._dm_delay_frames)


        self.sim_factory = sim_factory
        self.sim_kwargs = dict(sim_kwargs)
        self.layer_cfgs = list(layers)

        self.sensor_method = sensor_method  # southwell or fft


        # Prefer fast Southwell implementation (same outputs, less overhead)
        if str(self.sensor_method).lower().strip() == "southwell":
            self.sensor_method = "southwell_fast"
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
        self.loop_display_hz = loop_display_hz

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



        # long-run batch state
        self._long_run_active = False
        self._long_run_cancel = False
        self._long_run_prev_flags = None
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

        
        # Fast patch sampler (fixed geometry) + WFS measure signature cache
        self._patch_sampler = None
        self._wfs_measure_accepts = None
        self._zero_phase_patch = None

        # ---- batched Southwell WFS path (loop optimization) ----
        self._wfs_batch_enabled = False
        self._wfs_batch_template = None
        self._wfs_phase_buf = None           # (n_wfs, M, M)
        self._slopes_ref_stack = None        # (n_wfs, n_active, 2)
        self._slopes_err_stack = None        # (n_wfs, n_active, 2)
        self._slopes_sub_stack = None        # (n_wfs, n_active, 2)  (for recon display)

        # heterogeneous-WFS fallback (vectorized slope packing)
        self._slopes_meas_vec = None         # (n_slopes_total,) raw measured slopes packed
        self._slopes_ref_vec = None          # (n_slopes_total,) reference (flat) slopes packed
        self._slopes_err_vec = None          # (n_slopes_total,) slopes error (meas - ref), loop path
        self._slopes_sub_vec = None          # (n_slopes_total,) slopes error (meas - ref), recon display path
        self._wfs_slices = None              # list[(start, stop)] for each WFS in sensors[1:]
        self._wfs_n_active = None            # list[int] active subaps per WFS
        self._slopes_total_len = 0           # total packed slope length

        self._wfs_is_lgs = None              # (n_wfs,) bool
        self._wfs_wavelengths = None         # (n_wfs,) float
        self._m_f = None                     # float pupil mask cache
        self._m_sum = None
        self._sf_cache = None                # structure-function mask cache
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
        self.lambda_m = self.params.get("science_lambda")
        self.gaussian_fitter = Gaussian2DFitter(roi_half=32, dtype=cp.float32)
        self.gaussian_fitter_uc = Gaussian2DFitter(roi_half=32, dtype=cp.float32)

        # cache last loop-metrics outputs so we can update UI faster than we recompute PSFs
        self._loop_last_roi = None
        self._loop_last_fwhm = 0.0
        self._loop_last_fwhm_uncorr = 0.0
        self._loop_last_gauss = (0.0, 0.0, 1.0)

        self.plate_rad = None
        self.long_exposure_frames = long_exposure_frames
        self.long_exposure_psf = None
        self.long_exposure_psf_uncorrected = None

        # ------------------------------------------------------------
        # Tip-tilt (TT) controller (separate from DM loop)
        #
        # Physically, this represents a fast-steering mirror (FSM) in the common path.
        # In a pupil-plane phase screen model, an FSM is equivalent to adding a global
        # phase ramp: phi_tt = (sy * y + sx * x) * pupil.
        #
        # Controller runs at `tt_rate_hz` (may be different from DM loop) and can have
        # an additional discrete delay `tt_delay_frames` (in *simulation ticks*).
        # TT measurements are formed from the mean slopes of the NGS WFS (infinite range).
        # ------------------------------------------------------------
        self._tt_enabled = bool(self.params.get("tt_enabled", False))
        self._tt_gain = float(self.params.get("tt_gain", 0.25))
        self._tt_leak = float(self.params.get("tt_leak", 0.0))
        self._tt_rate_hz = float(self.params.get("tt_rate_hz", 0.0))
        self._tt_delay_frames = int(self.params.get("tt_delay_frames", 1))
        # If TT runs slower than the main simulation tick, average the TT measurements over
        # the integration period (closer to a real SH exposure). Default: enabled.
        self._tt_measure_average = bool(self.params.get("tt_measure_average", True))
        self._tt_period_ticks = 1
        self._tt_tick_phase = 0
        self._tt_cmd_yx = cp.zeros((2,), dtype=cp.float32)  # [sy, sx] command (500nm-ref units)
        # Centered coord grids for phase ramp
        yy0, xx0 = cp.indices((self.patch_M, self.patch_M), dtype=cp.float32)
        self._tt_xx = xx0 - (self.patch_M - 1) * 0.5
        self._tt_yy = yy0 - (self.patch_M - 1) * 0.5
        self._tt_phi_cmd = cp.zeros((self.patch_M, self.patch_M), dtype=cp.float32)
        self._tt_phi = cp.zeros((self.patch_M, self.patch_M), dtype=cp.float32)  # active TT phase (after delay)
        self._tt_delay_line = deque()
        self._rebuild_tt_delay_line()

    # ------------------ helpers ------------------
    def _hz_to_period(self, hz: int) -> int:
        hz = max(1, int(hz))
        return max(1, int(round(self.fps_cap / hz)))

    def _update_tt_period_ticks(self):
        """Compute TT update period in simulation ticks from tt_rate_hz."""
        if self._tt_rate_hz is None or float(self._tt_rate_hz) <= 0.0:
            self._tt_period_ticks = 1
            return
        dt = float(max(getattr(self, "dt_s", 0.0), 1e-12))
        period_s = 1.0 / float(self._tt_rate_hz)
        self._tt_period_ticks = max(1, int(round(period_s / dt)))

    def _rebuild_tt_delay_line(self):
        """(Re)create the TT delay line.

        Convention: tt_delay_frames = 1 means *one* simulation tick of delay
        (command computed at tick t becomes active at tick t+1).
        """
        try:
            d = int(self._tt_delay_frames)
        except Exception:
            d = 1
        # See note above: queue length is (d-1) because we update at end of tick.
        qlen = max(0, d - 1)
        self._tt_delay_line = deque([cp.zeros_like(self._tt_phi_cmd) for _ in range(qlen)])
        # active phase (what is applied in the next tick) starts at zero
        self._tt_phi.fill(0.0)
        self._update_tt_period_ticks()

    def _tt_cmd_to_phase(self, cmd_yx: cp.ndarray) -> cp.ndarray:
        """Convert [sy,sx] command (500nm-ref slope units) to a pupil-masked phase ramp."""
        cmd_yx = cp.asarray(cmd_yx, dtype=cp.float32).reshape(2,)
        phi = (cmd_yx[0] * self._tt_yy + cmd_yx[1] * self._tt_xx)
        # Apply pupil mask + remove piston over the pupil
        m, m_f, m_sum = self._ensure_pupil_float_cache(self.patch_M)
        phi = phi * m_f
        mu = cp.sum(phi) / cp.maximum(m_sum, cp.float32(1e-20))
        return (phi - mu) * m_f

    # ------------------ public TT controls (called by Loop tab) ------------------
    def set_tt_controller(self, enabled: bool, gain: float, leak: float, rate_hz: float):
        """Enable/configure the separate NGS tip-tilt controller."""
        self._tt_enabled = bool(enabled)
        try:
            self._tt_gain = float(gain)
        except Exception:
            pass
        try:
            self._tt_leak = float(leak)
        except Exception:
            pass
        try:
            self._tt_rate_hz = float(rate_hz)
        except Exception:
            pass

        # Recompute period in ticks (doesn't disturb controller state)
        self._update_tt_period_ticks()

        if not self._tt_enabled:
            # Clear any residual TT correction
            try:
                self._tt_cmd_yx.fill(0.0)
                self._tt_phi_cmd.fill(0.0)
                self._tt_phi.fill(0.0)
            except Exception:
                pass

    def set_tt_delay_frames(self, delay_frames: int):
        """Set TT servo-lag (in simulation ticks)."""
        try:
            self._tt_delay_frames = int(delay_frames)
        except Exception:
            self._tt_delay_frames = 1
        # Rebuild delay line using current command so we don't introduce a large step
        try:
            d = int(self._tt_delay_frames)
        except Exception:
            d = 1
        qlen = max(0, d - 1)
        self._tt_delay_line = deque([self._tt_phi_cmd.copy() for _ in range(qlen)])

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

    def _ensure_pupil_float_cache(self, M: int):
        """Return (m_bool, m_float, m_sum) cached for shape (M,M)."""
        m = self._ensure_pupil_mask(M)
        if self._m_f is None or self._m_f.shape != (M, M):
            self._m_f = m.astype(cp.float32)
            self._m_sum = cp.sum(self._m_f)
        return m, self._m_f, self._m_sum

    def _remove_piston_tt_fast(self, phi2d):
        """Fast piston+TT removal using the reconstructor's precomputed projector when available."""
        phi2d = cp.asarray(phi2d, dtype=cp.float32)
        if self.R is None or not hasattr(self.R, "_sel"):
            # fallback: keep behaviour for safety
            out = phi2d - cp.mean(phi2d)
            return remove_tt(out, self.pupil_mask)

        sel = self.R._sel
        flat = phi2d.ravel()
        z = flat[sel]
        z = z - cp.mean(z)

        if getattr(self.R, "remove_tt_in_output", True):
            beta = self.R._XtX_inv @ (self.R._Xt @ z)
            z = z - (beta[0] * self.R._xx_sel + beta[1] * self.R._yy_sel + beta[2])

        out = phi2d.copy()
        out_flat = out.ravel()
        out_flat[sel] = z
        return out

    def _apply_dm_to_phases(self, phases_atm):
        """Apply active closed-loop corrections (DM + optional TT) to a sampled phase stack."""
        if not self._loop_enabled:
            return phases_atm

        out = phases_atm
        if self._dm_phi is not None:
            out = out.copy() - self._dm_phi
        if getattr(self, "_tt_enabled", False):
            out = out.copy() - self._tt_phi
        return out

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

        use_fast = (self._patch_sampler is not None) and hasattr(self.sim, "sample_patches_from_sampler")

        if need_per_layer:
            if use_fast:
                outL = self.sim.sample_patches_from_sampler(
                    self._patch_sampler,
                    remove_piston=False,
                    return_gpu=True,
                    return_per_layer=True,
                    layer_first=True,
                    autopan=True,
                )
            else:
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
            if use_fast:
                phases = self.sim.sample_patches_from_sampler(
                    self._patch_sampler,
                    remove_piston=True,
                    return_gpu=True,
                    return_per_layer=False,
                    autopan=True,
                )
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

    def _measure_wfs_slopes(self, phases_wfs):
        """Measure slopes for sensors[1:] on provided phase maps (S-1,M,M)."""
        if phases_wfs is None:
            return []

        out = []
        acc_list = self._wfs_measure_accepts or [{} for _ in range(len(self._sensor_list) - 1)]

        for s0, ph, acc in zip(self._sensor_list[1:], phases_wfs, acc_list):
            kwargs = {"phase_map": ph, "return_image": False}
            if acc.get("assume_radians", False):
                kwargs["assume_radians"] = True
            if acc.get("method", False):
                kwargs["method"] = self.sensor_method
            _, sl, _ = s0.measure(**kwargs)
            out.append(sl[0])  # (n_sub_active, 2) [sy, sx]

        return out


    def _ensure_wfs_vec_structures(self):
        """Ensure slices + geometry groups exist for packed-slope measurement.

        This enables batching for sensors that share the same geometry, even when the
        full WFS set is heterogeneous.
        """
        wfs_list = list(self._sensor_list[1:])
        if not wfs_list:
            self._wfs_groups_vec = None
            self._wfs_slices = None
            self._wfs_n_active = None
            self._slopes_total_len = 0
            return

        # --- slices into the packed slope vector ---
        self._wfs_n_active = [int(len(getattr(s0, "sub_aps_idx"))) for s0 in wfs_list]
        lens = [2 * n for n in self._wfs_n_active]
        offs = [0]
        for ln in lens:
            offs.append(offs[-1] + int(ln))
        self._wfs_slices = [(offs[i], offs[i + 1]) for i in range(len(wfs_list))]
        self._slopes_total_len = int(offs[-1])

        # --- group sensors by geometry for batched measure() ---
        groups = {}
        for i, s0 in enumerate(wfs_list):
            if hasattr(s0, "batch_geometry_key"):
                key = s0.batch_geometry_key()
            else:
                key = (
                    type(s0).__name__,
                    int(getattr(s0, "grid_size", -1)),
                    int(getattr(s0, "n_sub", -1)),
                    int(len(getattr(s0, "sub_aps_idx"))),
                )
            groups.setdefault(key, []).append(i)

        self._wfs_groups_vec = []
        for key, idxs in groups.items():
            tmpl = wfs_list[idxs[0]]
            phase_buf = cp.empty((len(idxs), self.patch_M, self.patch_M), dtype=cp.float32)
            self._wfs_groups_vec.append({"key": key, "idxs": idxs, "template": tmpl, "phase_buf": phase_buf})


    def _measure_wfs_slopes_vec_into(self, phases_wfs_stack, out_vec):
        """Measure slopes per WFS and pack into a 1D vector (handles heterogeneous WFS shapes).

        phases_wfs_stack: (n_wfs, M, M) or list/tuple of (M,M)
        out_vec: preallocated (n_slopes_total,) cupy array
        """
        if phases_wfs_stack is None or out_vec is None:
            return None

        # Ensure slices + groups exist
        if (getattr(self, "_wfs_slices", None) is None) or (getattr(self, "_wfs_groups_vec", None) is None):
            self._ensure_wfs_vec_structures()

        acc_list = self._wfs_measure_accepts or [{} for _ in range(len(self._sensor_list) - 1)]

        groups = getattr(self, "_wfs_groups_vec", None) or []
        if not groups:
            return out_vec

        # Group-wise batching
        for grp in groups:
            idxs = grp["idxs"]
            tmpl = grp["template"]
            phase_buf = grp["phase_buf"]

            # gather phases for this geometry into a contiguous buffer
            for j, i in enumerate(idxs):
                phase_buf[j] = phases_wfs_stack[i]

            # use template's signature support flags
            acc = acc_list[idxs[0]] if (idxs and idxs[0] < len(acc_list)) else {}
            kwargs = {"phase_map": phase_buf, "return_image": False}
            if acc.get("assume_radians", False):
                kwargs["assume_radians"] = True
            if acc.get("method", False):
                kwargs["method"] = self.sensor_method

            _, sl, _ = tmpl.measure(**kwargs)
            sl = sl.astype(cp.float32, copy=False)
            if sl.ndim == 2:
                sl = sl[None, :, :]
            na = int(sl.shape[1])

            # pack per-sensor in [sx..., sy...] ordering
            for j, i in enumerate(idxs):
                a, _ = self._wfs_slices[i]
                out_vec[a : a + na] = sl[j, :, 1]
                out_vec[a + na : a + 2 * na] = sl[j, :, 0]

        return out_vec

    def _measure_wfs_slopes_stack(self, phases_wfs_stack):
        """Measure slopes for all WFS maps in one call when Southwell batching is available.

        Returns:
            slopes_stack: (n_wfs, n_active, 2)
        """
        if phases_wfs_stack is None:
            return None

        if self._wfs_batch_enabled and (self._wfs_batch_template is not None):
            s0 = self._wfs_batch_template
            kwargs = {"phase_map": phases_wfs_stack, "return_image": False, "method": self.sensor_method, "assume_radians": True}
            try:
                _, sl, _ = s0.measure(**kwargs)
            except TypeError:
                # older signature: no assume_radians or method
                kwargs.pop("assume_radians", None)
                kwargs.pop("method", None)
                _, sl, _ = s0.measure(**kwargs)
            return sl.astype(cp.float32, copy=False)

        # Fallback: per-sensor loop, pack into 1D vector (supports heterogeneous WFS geometry).
        if (self._wfs_slices is None) or (self._slopes_total_len <= 0):
            wfs_list = list(self._sensor_list[1:])
            self._wfs_n_active = [int(len(getattr(s0, "sub_aps_idx"))) for s0 in wfs_list]
            lens = [2 * n for n in self._wfs_n_active]
            offs = [0]
            for ln in lens:
                offs.append(offs[-1] + int(ln))
            self._wfs_slices = [(offs[i], offs[i+1]) for i in range(len(wfs_list))]
            self._slopes_total_len = int(offs[-1])

        n = int(self._slopes_total_len)
        if n <= 0:
            return None
        if (self._slopes_meas_vec is None) or (int(self._slopes_meas_vec.size) != n):
            self._slopes_meas_vec = cp.empty((n,), dtype=cp.float32)

        self._measure_wfs_slopes_vec_into(phases_wfs_stack, self._slopes_meas_vec)
        return self._slopes_meas_vec

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


        # --- dynamic pointing: disable fixed-geometry sampling fast path when angles change over time ---
        self._dynamic_thetas = any(getattr(s, "field_angle_fn", None) is not None for s in (self._sensor_list or []))


        # --- fast patch sampler for fixed sensor geometry (reduces per-frame cost) ---
        self._patch_sampler = None
        if (not self._dynamic_thetas) and hasattr(self.sim, "build_patch_sampler") and hasattr(self.sim, "sample_patches_from_sampler"):
            try:
                self._patch_sampler = self.sim.build_patch_sampler(
                    thetas_xy_rad=self.thetas_gpu,
                    ranges_m=self.ranges_gpu,
                    size_pixels=self.patch_size_px,
                    M=self.patch_M,
                    angle_deg=0.0,
                    margin=2.0,
                    autopan_init=True,
                )
                self.status_log.emit("Fast patch sampler initialized.")
            except Exception as e:
                self.status_log.emit(f"Fast patch sampler unavailable (falling back): {e}")
                self._patch_sampler = None
        elif self._dynamic_thetas:
            self.status_log.emit("Dynamic field angle functions detected; fast patch sampler disabled.")

        # Cache measure() signature support per WFS to avoid per-frame try/except overhead
        self._wfs_measure_accepts = []
        for s0 in self._sensor_list[1:]:
            try:
                ps = inspect.signature(s0.measure).parameters
                self._wfs_measure_accepts.append({
                    "assume_radians": "assume_radians" in ps,
                    "method": "method" in ps,
                    "return_image": "return_image" in ps,
                })
            except Exception:
                self._wfs_measure_accepts.append({"assume_radians": False, "method": False, "return_image": False})

        # --- batched Southwell slopes for the control loop (huge per-frame win) ---
        self._wfs_batch_enabled = False
        self._wfs_batch_template = None
        self._wfs_phase_buf = None
        self._slopes_ref_stack = None
        self._slopes_err_stack = None
        self._slopes_sub_stack = None
        self._wfs_is_lgs = None

        sm = str(getattr(self, "sensor_method", "southwell")).lower().strip()
        if sm in ("southwell", "southwell_fast") and len(self._sensor_list) > 1:
            try:
                tmpl = self._sensor_list[1]
                wfs_list = list(self._sensor_list[1:])
                n_wfs = int(len(wfs_list))
                n_active = int(len(getattr(tmpl, "sub_aps_idx")))

                # Conservative compatibility check: identical active-subap count + same grid geometry.
                ok = True
                for s0 in wfs_list:
                    if type(s0) is not type(tmpl):
                        ok = False
                        break
                    if int(getattr(s0, "grid_size", -1)) != int(getattr(tmpl, "grid_size", -2)):
                        ok = False
                        break
                    if int(getattr(s0, "n_sub", -1)) != int(getattr(tmpl, "n_sub", -2)):
                        ok = False
                        break
                    if int(len(getattr(s0, "sub_aps_idx"))) != n_active:
                        ok = False
                        break

                if ok and n_active > 0:
                    self._wfs_batch_enabled = True
                    self._wfs_batch_template = tmpl
                    self._wfs_phase_buf = cp.empty((n_wfs, self.patch_M, self.patch_M), dtype=cp.float32)
                    self._slopes_ref_stack = cp.zeros((n_wfs, n_active, 2), dtype=cp.float32)
                    self._slopes_err_stack = cp.empty_like(self._slopes_ref_stack)
                    self._slopes_sub_stack = cp.empty_like(self._slopes_ref_stack)
                    self._wfs_is_lgs = [s0.is_lgs for s0 in wfs_list]
                    self._wfs_wavelengths = [float(getattr(s0, "wavelength", 500e-9)) for s0 in wfs_list]
                    # Dedicated low-order (TT) reconstructor (separate from HO reconstructor)
                    self._tt_recon = TipTiltReconstructor_CuPy(
                        wfs_list,
                        self._wfs_is_lgs,
                        self._wfs_wavelengths,
                        weight_mode=str(self.params.get("tt_weight_mode", "pupil")),
                        sensor_method=str(getattr(self, "sensor_method", "southwell")),
                        pad=int(self.params.get("wfs_pad", 4) or 4),
                        params=self.params,
                    )

                    self.status_log.emit(f"Batched Southwell slopes enabled for {n_wfs} WFS ({n_active} active subaps each).")
                else:
                    self.status_log.emit("Batched Southwell slopes disabled (WFS geometry mismatch).")
            except Exception as e:
                self.status_log.emit(f"Batched Southwell slopes disabled: {e}")


        # --- heterogeneous WFS support: pack slopes into one vector when batching is disabled ---
        if (sm in ("southwell", "southwell_fast")) and (len(self._sensor_list) > 1) and (not self._wfs_batch_enabled):
            try:
                wfs_list = list(self._sensor_list[1:])
                self._wfs_is_lgs = [s0.is_lgs for s0 in wfs_list]
                self._wfs_wavelengths = [float(getattr(s0, "wavelength", 500e-9)) for s0 in wfs_list]
                # Dedicated low-order (TT) reconstructor (separate from HO reconstructor)
                self._tt_recon = TipTiltReconstructor_CuPy(
                    wfs_list,
                    self._wfs_is_lgs,
                    self._wfs_wavelengths,
                    weight_mode=str(self.params.get("tt_weight_mode", "pupil")),
                    sensor_method=str(getattr(self, "sensor_method", "southwell")),
                    pad=int(self.params.get("wfs_pad", 4) or 4),
                    params=self.params,
                )

                self._wfs_n_active = [int(len(getattr(s0, "sub_aps_idx"))) for s0 in wfs_list]
                lens = [2 * n for n in self._wfs_n_active]
                offs = [0]
                for ln in lens:
                    offs.append(offs[-1] + int(ln))
                self._wfs_slices = [(offs[i], offs[i+1]) for i in range(len(wfs_list))]
                self._slopes_total_len = int(offs[-1])

                # vectors are allocated lazily on first use (tick) to avoid build-time stalls
                self._slopes_meas_vec = None
                self._slopes_ref_vec = None
                self._slopes_err_vec = None
                self._slopes_sub_vec = None

                self.status_log.emit(f"Packed-slope vector mode enabled (heterogeneous WFS): total_len={self._slopes_total_len}.")
            except Exception as e:
                self.status_log.emit(f"Packed-slope vector mode init failed (falling back to legacy list): {e}")

        self.status_log.emit(f"Built sim with {len(self.layer_cfgs)} layers.")
        self.status.emit("Building tomographic reconstructor")

        self.update_r0.emit(getattr(self.sim, "r0_total", 0.0))
        self.initialized = True
        self._emit_once()
        self._emit_visible_layer()

        self.R = GLAOConjugateIM_CuPy(
            sensors=wfs_list,                 # your SH sensors
            conjugation_height_m=0.0,          # e.g. 0.0 for ground, or 200.0, 500.0, ...
            dx_world_m=float(self.sim.dx),
            M=self.patch_M,                             # your pupil grid
            slopes_aggregation="stack",  
            slope_weight_mode="pupil",    # <-- important
            slope_weight_threshold=0.20,  # ignore very partial subaps
            slope_weight_power=2.0,        
            reg_alpha=1e-2,
            reg_beta=1e-2,
        )

        self.R.build_interaction_matrix(chunk_modes=64, sensor_method="southwell")
        self.R.prepare_runtime()
        self.R.factorize(rcond=8e-2)


        self.status_log.emit("Tomographic reconstructor initialized!")


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


    @Slot(int)
    def set_dm_delay_frames(self, frames: int):
        """Set DM servo-lag in integer simulation frames.

        frames=1 matches the legacy behaviour (command computed at tick t becomes
        active at tick t+1).
        """
        try:
            frames = int(frames)
        except Exception:
            frames = 1
        frames = max(1, frames)
        if frames == getattr(self, "_dm_delay_frames", 1):
            return

        self._dm_delay_frames = frames
        # Preserve current active DM phase (if any) through the pipeline reset.
        cur = getattr(self, "_dm_phi", None)
        fill = cur if cur is not None else None
        self._dm_phi_pipeline = deque([fill] * int(frames))


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

        # --- dynamic pointing support ---
        self._dynamic_thetas = any(getattr(s, "field_angle_fn", None) is not None for s in (self._sensor_list or []))
        thetas = []
        for s in (self._sensor_list or []):
            if getattr(s, "field_angle_fn", None) is not None and hasattr(s, "update_field_angle"):
                thetas.append(s.update_field_angle(0.0, int(self._tick_count)))
            else:
                thetas.append(getattr(s, "field_angle", (float(getattr(s, "dx", 0.0)), float(getattr(s, "dy", 0.0)))))

        self.thetas_gpu = cp.asarray(thetas, dtype=cp.float32)
        S = int(self.thetas_gpu.shape[0])

        r = cp.asarray([float(getattr(s, "gs_range_m", cp.inf)) for s in (self._sensor_list or [])], dtype=cp.float32)
        if r.size != S:
            r = cp.full((S,), cp.inf, dtype=cp.float32)
        self.ranges_gpu = r

        # Sensor-dependent caches/buffers (safe defaults; rebuilt lazily on next tick)
        self._patch_sampler = None
        self._wfs_measure_accepts = None
        self._wfs_groups_vec = None
        self._wfs_slices = None
        self._wfs_n_active = None
        self._slopes_total_len = 0

        if self.sim is not None:
            # If fast sampler is enabled and the geometry is fixed, rebuild the sampler.
            if (not self._dynamic_thetas) and self.fast_sampler and hasattr(self.sim, "build_patch_sampler") and hasattr(self.sim, "sample_patches_from_sampler"):
                try:
                    self._patch_sampler = self.sim.build_patch_sampler(
                        thetas_xy_rad=self.thetas_gpu,
                        ranges_m=self.ranges_gpu,
                        size_pixels=self.patch_size_px,
                        M=self.patch_M,
                        angle_deg=0.0,
                        margin=2.0,
                        autopan_init=True,
                    )
                    self.status_log.emit("Fast patch sampler rebuilt after sensor_update.")
                except Exception as e:
                    self.status_log.emit(f"Fast patch sampler rebuild failed (falling back): {e}")
                    self._patch_sampler = None

            # Cache measure() signature support per WFS (used by batching)
            self._wfs_measure_accepts = []
            for s0 in (self._sensor_list or [])[1:]:
                try:
                    ps = inspect.signature(s0.measure).parameters
                    self._wfs_measure_accepts.append({
                        "assume_radians": "assume_radians" in ps,
                        "method": "method" in ps,
                        "return_image": "return_image" in ps,
                    })
                except Exception:
                    self._wfs_measure_accepts.append({"assume_radians": False, "method": False, "return_image": False})

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
        # Reset DM servo-lag pipeline
        try:
            n = int(getattr(self, "_dm_delay_frames", 1))
        except Exception:
            n = 1
        self._dm_phi_pipeline = deque([None] * max(1, n))

        # Reset TT controller state
        try:
            self._tt_cmd_yx = cp.zeros((2,), dtype=cp.float32)
            self._tt_phi_cmd = cp.zeros((self.patch_M, self.patch_M), dtype=cp.float32)
            self._tt_phi = cp.zeros((self.patch_M, self.patch_M), dtype=cp.float32)
            if hasattr(self, "_rebuild_tt_delay_line"):
                self._rebuild_tt_delay_line()
        except Exception:
            pass

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


    # ------------------ long-run batch API ------------------
    @Slot(dict)
    def start_long_run(self, cfg: dict):
        '''Run a long batch simulation in this worker thread.

        This pauses the realtime timer (if running), runs AOBatchRunner for the requested
        number of frames, writes memmaps / npy outputs to disk, and emits a compact results
        dict suitable for UI display.
        '''
        if self._long_run_active:
            self.long_run_failed.emit("Long run already active")
            return

        if self.sim is None:
            self.build_sim()
        if self.R is None:
            # If a reconstructor hasn't been built yet, build one now.
            try:
                self.build_reconstructor()
            except Exception as e:
                self.long_run_failed.emit(f"Failed to build reconstructor: {e}")
                return

        cfg = dict(cfg or {})

        # Pause realtime timer and suppress heavy GUI emissions during the long run.
        was_running = bool(self.timer.isActive())
        if was_running:
            self.timer.stop()
            self.status_running.emit(False)

        self._long_run_prev_flags = {
            "overview": self._overview_enabled,
            "reconstructor": self._reconstructor_enabled,
            "detailed": self._detailed_enabled,
            "loop": self._loop_enabled,
        }
        self._overview_enabled = False
        self._reconstructor_enabled = False
        self._detailed_enabled = False
        self._loop_enabled = False

        self._long_run_active = True
        self._long_run_cancel = False

        try:
            duration_s = float(cfg.get("duration_s", 60.0))
            discard_first_s = float(
                cfg.get("discard_first_s",
                cfg.get("discard_s",
                cfg.get("discard_seconds",
                cfg.get("discard", 0.0))))
            )
            n_frames = cfg.get("n_frames", None)
            if n_frames is None:
                n_frames = int(round(duration_s / max(self.dt_s, 1e-12)))
            n_frames = int(n_frames)

            chunk_frames = int(cfg.get("chunk_frames", 256))
            psf_roi = int(cfg.get("psf_roi", 128))
            record_psfs = bool(cfg.get("record_psfs", True))
            record_psfs_ttremoved = bool(cfg.get("record_psfs_ttremoved", False))
            save_tt_series = bool(cfg.get("save_tt_series", True))

            psf_tt_mode = str(cfg.get("psf_tt_mode", "hybrid")).lower().strip()

            # Optional: off-axis evaluation points (arcsec) for long-exposure PSF/FWHM
            eval_offaxis_arcsec = cfg.get("eval_offaxis_arcsec", None)
            write_eval_fits = bool(cfg.get("write_eval_fits", True))
            write_eval_uncorrected_fits = bool(cfg.get("write_eval_uncorrected_fits", True))
            eval_accumulate_uncorrected = bool(cfg.get("eval_accumulate_uncorrected", True))
            if not eval_accumulate_uncorrected:
                write_eval_uncorrected_fits = False

            reset_before = bool(cfg.get("reset_before", True))
            base_out = str(cfg.get("out_dir", str(Path.cwd() / "output" / "batch_runs")))
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path(base_out) / f"longrun_{ts}"
            out_dir.mkdir(parents=True, exist_ok=True)

            if reset_before:
                self.reset()

            # progress callback
            def _prog(done: int, total: int, fps: float, msg: str):
                self.long_run_progress.emit(int(done), int(total), float(fps), str(msg))

            def _stop() -> bool:
                return bool(self._long_run_cancel)

            self.long_run_started.emit({
                "out_dir": str(out_dir),
                "n_frames": int(n_frames),
                "discard_first_s": float(discard_first_s),
                "discard_first_frames": int(max(0, int(math.floor(discard_first_s / max(self.dt_s, 1e-12))))),
            })

            runner = AOBatchRunner(
                sim=self.sim,
                sensors=self._sensor_list,
                reconstructor=self.R,
                pupil=self._pupil_mask_patch,
                dt_s=self.dt_s,
                gain=float(cfg.get("gain", getattr(self, "_loop_gain", 0.3))),
                leak=float(cfg.get("leak", getattr(self, "_loop_leak", 0.0))),
                wfs_method=self.sensor_method,
                strip_tt_lgs=True,
                psf_tt_mode=psf_tt_mode,
                tt_wfs_indices=None,

                # Separate NGS-based tip-tilt controller (optional)
                tt_enabled=bool(cfg.get("tt_enabled", getattr(self, "_tt_enabled", False))),
                tt_gain=float(cfg.get("tt_gain", getattr(self, "_tt_gain", 0.25))),
                tt_leak=float(cfg.get("tt_leak", getattr(self, "_tt_leak", 0.0))),
                tt_rate_hz=float(cfg.get("tt_rate_hz", getattr(self, "_tt_rate_hz", 0.0))),
                tt_delay_frames=int(cfg.get("tt_delay_frames", getattr(self, "_tt_delay_frames", 1))),
                dm_delay_frames=int(getattr(self, "_dm_delay_frames", 1)),
                params=self.params,
            )

            res = runner.run(
                n_frames=n_frames,
                discard_first_s=discard_first_s,
                record_psfs=record_psfs,
                record_psfs_ttremoved=record_psfs_ttremoved,
                save_tt_series=save_tt_series,
                out_dir=str(out_dir),
                psf_roi=psf_roi,
                chunk_frames=chunk_frames,
                eval_offaxis_arcsec=eval_offaxis_arcsec,
                write_eval_fits=write_eval_fits,
                write_eval_uncorrected_fits=write_eval_uncorrected_fits,
                eval_accumulate_uncorrected=eval_accumulate_uncorrected,
                progress_cb=_prog,
                should_stop=_stop,
                progress_every=int(cfg.get("progress_every", max(256, chunk_frames))),
            )

            # Convert key arrays for UI (small) and write a summary JSON
            def _to_np(x):
                try:
                    return cp.asnumpy(x)
                except Exception:
                    return x

            summary = {
                "out_dir": str(out_dir),
                "psf_tt_mode": res.get("psf_tt_mode"),
                "n_frames_requested": int(res.get("n_frames_requested", n_frames)),
                "n_frames_done": int(res.get("n_frames_done", n_frames)),
                "discard_first_s": float(res.get("discard_first_s", discard_first_s)),
                "discard_first_frames": int(res.get("discard_first_frames", 0)),
                "n_frames_used": int(res.get("n_frames_used", 0)),
                "cancelled": bool(res.get("cancelled", False)),
                "fwhm_rad_corrected": float(res.get("fwhm_rad_corrected", 0.0)),
                "fwhm_rad_uncorrected": float(res.get("fwhm_rad_uncorrected", 0.0)),
                "fwhm_arcsec_corrected": float(res.get("fwhm_rad_corrected", 0.0)) * (180.0 * 3600.0) / math.pi,
                "fwhm_arcsec_uncorrected": float(res.get("fwhm_rad_uncorrected", 0.0)) * (180.0 * 3600.0) / math.pi,
                "fwhm_arcsec_corrected_moffat": float(res.get("fwhm_rad_corrected_moffat", 0.0)) * (180.0 * 3600.0) / math.pi,
                "fwhm_arcsec_uncorrected_moffat": float(res.get("fwhm_rad_uncorrected_moffat", 0.0)) * (180.0 * 3600.0) / math.pi,
                "fwhm_arcsec_corrected_moffat_wings": float(res.get("fwhm_rad_corrected_moffat_wings", 0.0)) * (180.0 * 3600.0) / math.pi,
                "fwhm_arcsec_uncorrected_moffat_wings": float(res.get("fwhm_rad_uncorrected_moffat_wings", 0.0)) * (180.0 * 3600.0) / math.pi,
                "fwhm_rad_corrected_contour": float(res.get("fwhm_rad_corrected_contour", float("nan"))),
                "fwhm_rad_uncorrected_contour": float(res.get("fwhm_rad_uncorrected_contour", float("nan"))),
                "fwhm_arcsec_corrected_contour": float(res.get("fwhm_rad_corrected_contour", 0.0)) * (180.0 * 3600.0) / math.pi,
                "fwhm_arcsec_uncorrected_contour": float(res.get("fwhm_rad_uncorrected_contour", 0.0)) * (180.0 * 3600.0) / math.pi,
                "fwhm_arcsec_corrected_contour_major": float(res.get("fwhm_arcsec_corrected_contour_major", float("nan"))),
                "fwhm_arcsec_corrected_contour_minor": float(res.get("fwhm_arcsec_corrected_contour_minor", float("nan"))),
                "fwhm_arcsec_uncorrected_contour_major": float(res.get("fwhm_arcsec_uncorrected_contour_major", float("nan"))),
                "fwhm_arcsec_uncorrected_contour_minor": float(res.get("fwhm_arcsec_uncorrected_contour_minor", float("nan"))),
                "fwhm_contour_corrected_ratio": float(res.get("fwhm_contour_corrected_ratio", float("nan"))),
                "fwhm_contour_uncorrected_ratio": float(res.get("fwhm_contour_uncorrected_ratio", float("nan"))),
                "fwhm_contour_corrected_theta_deg": float(res.get("fwhm_contour_corrected_theta_deg", float("nan"))),
                "fwhm_contour_uncorrected_theta_deg": float(res.get("fwhm_contour_uncorrected_theta_deg", float("nan"))),
                "plate_rad_per_pix": float(res.get("plate_rad_per_pix", 0.0) or 0.0),
                # Fit metadata for overlays (native PSF pixels)
                "gauss_corrected": res.get("gauss_corrected"),
                "gauss_uncorrected": res.get("gauss_uncorrected"),
                "moffat_corrected": res.get("moffat_corrected"),
                "moffat_uncorrected": res.get("moffat_uncorrected"),
                "moffat_wings_corrected": res.get("moffat_wings_corrected"),
                "moffat_wings_uncorrected": res.get("moffat_wings_uncorrected"),
                "contour_corrected": res.get("contour_corrected"),
                "contour_uncorrected": res.get("contour_uncorrected"),
            }

            # Off-axis evaluation (optional)
            if "offaxis_metrics" in res:
                summary["eval_offaxis_arcsec"] = res.get("eval_offaxis_arcsec")
                summary["eval_accumulate_uncorrected"] = bool(res.get("eval_accumulate_uncorrected", True))
                summary["offaxis_metrics"] = res.get("offaxis_metrics")
                summary["offaxis_stats"] = res.get("offaxis_stats")
                # file paths (if written)
                if "psf_long_corrected_offaxis_fits" in res:
                    summary["psf_long_corrected_offaxis_fits"] = res.get("psf_long_corrected_offaxis_fits")
                if "psf_long_uncorrected_offaxis_fits" in res:
                    summary["psf_long_uncorrected_offaxis_fits"] = res.get("psf_long_uncorrected_offaxis_fits")
            if "fwhm_rad_corrected_ttremoved" in res:
                summary["fwhm_rad_corrected_ttremoved"] = float(res["fwhm_rad_corrected_ttremoved"])
                summary["fwhm_arcsec_corrected_ttremoved"] = float(res["fwhm_rad_corrected_ttremoved"]) * (180.0 * 3600.0) / math.pi

            for k in ("r0_est_m", "r0_uncorrected_m", "r0_corrected_m", "r0_effective_m"):
                if k in res:
                    try:
                        summary[k] = float(res[k])
                    except Exception:
                        pass

            with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            ui_res = dict(summary)
            ui_res["psf_long_corrected"] = _to_np(res.get("psf_long_corrected"))
            ui_res["psf_long_uncorrected"] = _to_np(res.get("psf_long_uncorrected"))

            if "psf_long_corrected_offaxis" in res:
                ui_res["psf_long_corrected_offaxis"] = _to_np(res.get("psf_long_corrected_offaxis"))
            if "psf_long_uncorrected_offaxis" in res:
                ui_res["psf_long_uncorrected_offaxis"] = _to_np(res.get("psf_long_uncorrected_offaxis"))
            if "psf_long_corrected_ttremoved" in res:
                ui_res["psf_long_corrected_ttremoved"] = _to_np(res.get("psf_long_corrected_ttremoved"))

            # also pass file paths if present
            for k in (
                "psf_corrected_memmap",
                "psf_uncorrected_memmap",
                "psf_corrected_ttremoved_memmap",
                "tt_yx_series_path",
                "psf_frames_written",
                "psf_shape",
                "psf_dtype",
            ):
                if k in res:
                    ui_res[k] = res[k]

            self.long_run_finished.emit(ui_res)

        except Exception:
            tb = traceback.format_exc()

            # best-effort: save traceback into the run folder if it exists
            try:
                if "out_dir" in locals() and out_dir is not None:
                    p = Path(out_dir) if not isinstance(out_dir, Path) else out_dir
                    p.mkdir(parents=True, exist_ok=True)
                    with open(p / "traceback.txt", "w", encoding="utf-8") as f:
                        f.write(tb)
            except Exception:
                pass

            try:
                self.status_log.emit(tb)
            except Exception:
                pass

            logger.error(tb)
            self.long_run_failed.emit(tb)

        finally:
            self._long_run_active = False
            self._long_run_cancel = False
            if isinstance(self._long_run_prev_flags, dict):
                self._overview_enabled = bool(self._long_run_prev_flags.get("overview", False))
                self._reconstructor_enabled = bool(self._long_run_prev_flags.get("reconstructor", False))
                self._detailed_enabled = bool(self._long_run_prev_flags.get("detailed", True))
                self._loop_enabled = bool(self._long_run_prev_flags.get("loop", False))
            self._long_run_prev_flags = None

            if was_running:
                self.timer.start(self._interval_ms)
                self.status_running.emit(True)

    @Slot()
    def cancel_long_run(self):
        self._long_run_cancel = True


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
                psf = self._sensor_list[0].measure(phase0, fft_pad = self.patch_M)[2][0].get()
                zoomed = Analysis._zoom2d(psf, 2, cp)
                cx, cy = zoomed.shape[0]//2, zoomed.shape[1]//2
                self.sensor_psf_ready.emit(sidx, zoomed[cx-64:cx+64, cy-64:cy+64])

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
        slopes_list = self._measure_wfs_slopes(phases_res[1:])

        # --- reference slopes (flat) ---
        if self._s_ref is None:
            self._s_ref = []
            # Reuse one zero phase map (shape matches patch) where possible.
            if self._zero_phase_patch is None or self._zero_phase_patch.shape != phases_atm.shape[1:]:
                self._zero_phase_patch = cp.zeros(phases_atm.shape[1:], dtype=cp.float32)
            ph0 = self._zero_phase_patch
            acc_list = self._wfs_measure_accepts or [{} for _ in range(len(self._sensor_list) - 1)]
            for s0, acc in zip(self._sensor_list[1:], acc_list):
                kwargs = {"phase_map": ph0, "return_image": False}
                if acc.get("assume_radians", False):
                    kwargs["assume_radians"] = True
                if acc.get("method", False):
                    kwargs["method"] = self.sensor_method
                _, sl0, _ = s0.measure(**kwargs)
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

        # Update dynamic field angles (if any) for this tick before sampling.
        if getattr(self, "_dynamic_thetas", False):
            t_s = float(self._tick_count) * float(getattr(self, "dt_s", 0.0))
            thetas = []
            for s in (self._sensor_list or []):
                if getattr(s, "field_angle_fn", None) is not None and hasattr(s, "update_field_angle"):
                    thetas.append(s.update_field_angle(t_s, int(self._tick_count)))
                else:
                    thetas.append(getattr(s, "field_angle", (float(getattr(s, "dx", 0.0)), float(getattr(s, "dy", 0.0)))))
            self.thetas_gpu = cp.asarray(thetas, dtype=cp.float32)

        tc = self._tick_count
        need_visible_sensor = ((tc % self._sensor_period) == 0)
        need_visible_tab = ((tc % self._layer_tab_period) == 0)
        need_overview_psf = (self._overview_enabled and ((tc % self._overview_psf_period) == 0))
        need_overview_layer = (self._overview_enabled and ((tc % self._overview_layer_period) == 0))
        need_combined = ((tc % self._combined_period) == 0)

        t1 = tn()

        # Sample atmosphere (cached). Avoid DM subtraction/copies unless needed.
        phases_atm, _ = self._sample_frame_once(need_per_layer=False)   # (S,M,M)
        if phases_atm is None:
            return

        M = int(phases_atm.shape[-1])
        m, m_f, m_sum = self._ensure_pupil_float_cache(M)

        # IMPORTANT: WFS slopes + recon are expensive. Only compute at 1 kHz when the loop is enabled.
        want_recon = bool(self._reconstructor_enabled and (self.R is not None))
        need_recon_calc = bool(want_recon and need_combined)
        need_slopes = bool(self._loop_enabled or need_recon_calc)

        # DM phase applied for *this* measurement tick (new DM command becomes active next tick)
        dm_phi_prev = self._dm_phi
        # TT phase applied for this measurement tick (after configurable delay line)
        tt_phi_prev = self._tt_phi if (getattr(self, "_tt_enabled", False) and self._loop_enabled) else None

        t2 = tn()

        slopes_stack = None
        if need_slopes and (len(self._sensor_list) > 1):
            # Residual WFS phases into reusable buffer (WFS only) to avoid allocating phases_res every frame.
            n_wfs = len(self._sensor_list) - 1
            if (self._wfs_phase_buf is None) or (self._wfs_phase_buf.shape != (n_wfs, M, M)):
                self._wfs_phase_buf = cp.empty((n_wfs, M, M), dtype=cp.float32)
            cp.copyto(self._wfs_phase_buf, phases_atm[1:].astype(cp.float32, copy=False))
            if dm_phi_prev is not None:
                self._wfs_phase_buf -= dm_phi_prev[None, :, :]
            if tt_phi_prev is not None:
                self._wfs_phase_buf -= tt_phi_prev[None, :, :]
            slopes_stack = self._measure_wfs_slopes_stack(self._wfs_phase_buf)

            # One-time reference slopes (flat). Handle both stack and packed-vector modes.
            if slopes_stack is not None:
                if slopes_stack.ndim == 3:
                    if self._slopes_ref_stack is None:
                        zph = cp.zeros((slopes_stack.shape[0], M, M), dtype=cp.float32)
                        self._slopes_ref_stack = self._measure_wfs_slopes_stack(zph)
                        self._slopes_err_stack = cp.empty_like(self._slopes_ref_stack)
                        self._slopes_sub_stack = cp.empty_like(self._slopes_ref_stack)
                else:
                    # packed vector mode (heterogeneous WFS)
                    if self._slopes_ref_vec is None:
                        self._slopes_ref_vec = cp.empty_like(slopes_stack)
                        self._slopes_err_vec = cp.empty_like(slopes_stack)
                        self._slopes_sub_vec = cp.empty_like(slopes_stack)
                        zph = cp.zeros((len(self._sensor_list) - 1, M, M), dtype=cp.float32)
                        self._measure_wfs_slopes_vec_into(zph, self._slopes_ref_vec)

        t3 = tn()

        # ---- optional recon display (subtract ref only; TT removal already handled in recon output) ----
        rec = None
        surf_m = cp.zeros_like(phases_atm[0])
        phi_res_disp = None
        phase0 = None

        if need_recon_calc and (slopes_stack is not None) and want_recon:
            if slopes_stack.ndim == 3:
                if self._slopes_ref_stack is None:
                    # should have been initialized above, but be safe
                    zph = cp.zeros((slopes_stack.shape[0], M, M), dtype=cp.float32)
                    self._slopes_ref_stack = self._measure_wfs_slopes_stack(zph)
                    self._slopes_err_stack = cp.empty_like(self._slopes_ref_stack)
                    self._slopes_sub_stack = cp.empty_like(self._slopes_ref_stack)

                cp.subtract(slopes_stack, self._slopes_ref_stack, out=self._slopes_sub_stack)
                # pack to [sx..., sy...] ordering to match interaction-matrix convention
                nS, nA = int(self._slopes_sub_stack.shape[0]), int(self._slopes_sub_stack.shape[1])
                nvec = nS * 2 * nA
                if (self._slopes_sub_vec is None) or (int(self._slopes_sub_vec.size) != int(nvec)):
                    self._slopes_sub_vec = cp.empty((nvec,), dtype=cp.float32)
                sub2d = self._slopes_sub_vec.reshape(nS, 2 * nA)
                sub2d[:, :nA] = self._slopes_sub_stack[..., 1]  # sx
                sub2d[:, nA:] = self._slopes_sub_stack[..., 0]  # sy
                self.reconstructed, xhat_disp = self.R.reconstruct_onaxis_stack(self._slopes_sub_vec, return_coeffs=True)

            else:
                if self._slopes_ref_vec is None:
                    # should have been initialized above, but be safe
                    self._slopes_ref_vec = cp.empty_like(slopes_stack)
                    self._slopes_err_vec = cp.empty_like(slopes_stack)
                    self._slopes_sub_vec = cp.empty_like(slopes_stack)
                    zph = cp.zeros((len(self._sensor_list) - 1, M, M), dtype=cp.float32)
                    self._measure_wfs_slopes_vec_into(zph, self._slopes_ref_vec)

                cp.subtract(slopes_stack, self._slopes_ref_vec, out=self._slopes_sub_vec)
                self.reconstructed, xhat_disp = self.R.reconstruct_onaxis_stack(self._slopes_sub_vec, return_coeffs=True)

            rec = self.reconstructed - cp.mean(self.reconstructed)

            # on-axis residual (science sees this tick)
            sci_res = phases_atm[0]
            if dm_phi_prev is not None:
                sci_res = sci_res - dm_phi_prev
            if tt_phi_prev is not None:
                sci_res = sci_res - tt_phi_prev
            phase0 = self._remove_piston_tt_fast(sci_res)

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

        # ---- CLOSED LOOP (optimized) ----

        if self._closed_loop and self._loop_enabled and (slopes_stack is not None):
            if slopes_stack.ndim == 3:
                if self._slopes_ref_stack is None:
                    # should have been initialized above, but be safe
                    zph = cp.zeros((slopes_stack.shape[0], M, M), dtype=cp.float32)
                    self._slopes_ref_stack = self._measure_wfs_slopes_stack(zph)
                    self._slopes_err_stack = cp.empty_like(self._slopes_ref_stack)
                    self._slopes_sub_stack = cp.empty_like(self._slopes_ref_stack)

                # slope error relative to flat
                cp.subtract(slopes_stack, self._slopes_ref_stack, out=self._slopes_err_stack)

                # --- TT measurement from NGS WFS (before we zero NGS for DM solve) ---
                # NOTE: if TT runs slower than the main loop, we optionally average measurements across
                # the TT integration period (closer to a real WFS exposure).
                tt_meas_yx = None
                if getattr(self, "_tt_enabled", False):
                    # Dedicated low-order reconstructor (NGS-only) for TT
                    if getattr(self, "_tt_recon", None) is not None:
                        tt_meas_yx = self._tt_recon.estimate_tt_yx_500nm(self._slopes_err_stack)
                    else:
                        # Fallback: unweighted mean slopes (500nm-ref scaled)
                        if (self._wfs_is_lgs is not None) and (self._wfs_wavelengths is not None):
                            ngs_idx = [i for i, is_lgs in enumerate(self._wfs_is_lgs) if not is_lgs]
                            if ngs_idx:
                                acc = cp.zeros((2,), dtype=cp.float32)
                                for wi in ngs_idx:
                                    wl = cp.float32(self._wfs_wavelengths[wi])
                                    acc += self._slopes_err_stack[wi].mean(axis=0) * (wl / cp.float32(500e-9))
                                tt_meas_yx = acc / cp.float32(len(ngs_idx))
                            else:
                                tt_meas_yx = cp.zeros((2,), dtype=cp.float32)

                # LGS cannot sense global TT; strip mean slopes. Off-axis NGS: keep prior behaviour (ignore).
                if self._wfs_is_lgs is not None:
                    lgs_idx = [i for i, is_lgs in enumerate(self._wfs_is_lgs) if is_lgs]
                    ngs_idx = [i for i, is_lgs in enumerate(self._wfs_is_lgs) if not is_lgs]
                    if lgs_idx:
                        if getattr(self, "_tt_recon", None) is not None:
                            self._tt_recon.strip_tt_inplace(self._slopes_err_stack, lgs_idx)
                        else:
                            means = self._slopes_err_stack[lgs_idx].mean(axis=1, keepdims=True)
                            self._slopes_err_stack[lgs_idx] -= means
                    if ngs_idx:
                        self._slopes_err_stack[ngs_idx].fill(0.0)

                # --- TT controller update (runs at its own rate) ---
                if (tt_meas_yx is not None) and (getattr(self, "_tt_enabled", False)):
                    # accumulate/average TT measurements across the TT period if enabled
                    if getattr(self, "_tt_meas_acc", None) is None:
                        self._tt_meas_acc = cp.zeros((2,), dtype=cp.float32)
                        self._tt_meas_count = 0

                    if bool(getattr(self, "_tt_measure_average", True)) and int(getattr(self, "_tt_period_ticks", 1)) > 1:
                        self._tt_meas_acc = self._tt_meas_acc + tt_meas_yx
                        self._tt_meas_count = int(getattr(self, "_tt_meas_count", 0)) + 1

                    if (tc % int(getattr(self, "_tt_period_ticks", 1))) == 0:
                        if bool(getattr(self, "_tt_measure_average", True)) and int(getattr(self, "_tt_period_ticks", 1)) > 1:
                            cnt = max(1, int(getattr(self, "_tt_meas_count", 0)))
                            tt_meas_yx = self._tt_meas_acc / cp.float32(cnt)
                            # reset accumulator for next TT exposure
                            self._tt_meas_acc = cp.zeros((2,), dtype=cp.float32)
                            self._tt_meas_count = 0
                        # Guard against non-finite WFS outputs to avoid poisoning the sim.
                        try:
                            if not bool(cp.all(cp.isfinite(tt_meas_yx)).get()):
                                tt_meas_yx = cp.zeros((2,), dtype=cp.float32)
                        except Exception:
                            tt_meas_yx = cp.zeros((2,), dtype=cp.float32)

                        self._tt_cmd_yx = (1.0 - self._tt_leak) * self._tt_cmd_yx + self._tt_gain * tt_meas_yx

                        # If controller state goes non-finite, reset rather than propagating NaNs.
                        try:
                            if not bool(cp.all(cp.isfinite(self._tt_cmd_yx)).get()):
                                self._tt_cmd_yx = cp.zeros((2,), dtype=cp.float32)
                        except Exception:
                            self._tt_cmd_yx = cp.zeros((2,), dtype=cp.float32)
                        self._tt_phi_cmd = self._tt_cmd_to_phase(self._tt_cmd_yx)

                # pack to [sx..., sy...] ordering to match interaction-matrix convention
                nS, nA = int(self._slopes_err_stack.shape[0]), int(self._slopes_err_stack.shape[1])
                nvec = nS * 2 * nA
                if (self._slopes_err_vec is None) or (int(self._slopes_err_vec.size) != int(nvec)):
                    self._slopes_err_vec = cp.empty((nvec,), dtype=cp.float32)
                err2d = self._slopes_err_vec.reshape(nS, 2 * nA)
                err2d[:, :nA] = self._slopes_err_stack[..., 1]  # sx
                err2d[:, nA:] = self._slopes_err_stack[..., 0]  # sy
                xhat = self.R.solve_coeffs_stack(self._slopes_err_vec)

            else:
                if self._slopes_ref_vec is None:
                    # should have been initialized above, but be safe
                    self._slopes_ref_vec = cp.empty_like(slopes_stack)
                    self._slopes_err_vec = cp.empty_like(slopes_stack)
                    self._slopes_sub_vec = cp.empty_like(slopes_stack)
                    zph = cp.zeros((len(self._sensor_list) - 1, M, M), dtype=cp.float32)
                    self._measure_wfs_slopes_vec_into(zph, self._slopes_ref_vec)

                # slope error relative to flat
                cp.subtract(slopes_stack, self._slopes_ref_vec, out=self._slopes_err_vec)

                # --- TT measurement from NGS WFS (before we zero NGS for DM solve) ---
                if getattr(self, "_tt_enabled", False) and (self._wfs_is_lgs is not None) and (self._wfs_slices is not None) and (self._wfs_n_active is not None) and (self._wfs_wavelengths is not None):
                    acc = cp.zeros((2,), dtype=cp.float32)
                    n_tt = 0
                    for wi, is_lgs in enumerate(self._wfs_is_lgs):
                        if is_lgs:
                            continue
                        na = int(self._wfs_n_active[wi])
                        if na <= 0:
                            continue
                        a, b = self._wfs_slices[wi]
                        # vector layout per WFS is [sx(0..na-1), sy(0..na-1)]
                        sx = self._slopes_err_vec[a : a + na]
                        sy = self._slopes_err_vec[a + na : a + 2 * na]
                        wl = cp.float32(self._wfs_wavelengths[wi])
                        mu_y = sy.mean() * (wl / cp.float32(500e-9))  # sy (500nm-ref)
                        mu_x = sx.mean() * (wl / cp.float32(500e-9))  # sx (500nm-ref)
                        # If available, convert centroid-bin units -> phase-ramp slope via TT calibration.
                        try:
                            if getattr(self, "_tt_recon", None) is not None and wi < len(self._tt_recon.k500_yx):
                                k = self._tt_recon.k500_yx[wi]
                                mu_y = mu_y / cp.maximum(k[0], cp.float32(1e-20))
                                mu_x = mu_x / cp.maximum(k[1], cp.float32(1e-20))
                        except Exception:
                            pass
                        acc[0] += mu_y
                        acc[1] += mu_x
                        n_tt += 1
                    tt_meas_yx = acc / cp.float32(max(1, n_tt))
                    # accumulate/average TT measurements across the TT period if enabled
                    if getattr(self, "_tt_meas_acc", None) is None:
                        self._tt_meas_acc = cp.zeros((2,), dtype=cp.float32)
                        self._tt_meas_count = 0

                    if bool(getattr(self, "_tt_measure_average", True)) and int(getattr(self, "_tt_period_ticks", 1)) > 1:
                        self._tt_meas_acc = self._tt_meas_acc + tt_meas_yx
                        self._tt_meas_count = int(getattr(self, "_tt_meas_count", 0)) + 1

                    if (tc % int(getattr(self, "_tt_period_ticks", 1))) == 0:
                        if bool(getattr(self, "_tt_measure_average", True)) and int(getattr(self, "_tt_period_ticks", 1)) > 1:
                            cnt = max(1, int(getattr(self, "_tt_meas_count", 0)))
                            tt_meas_yx = self._tt_meas_acc / cp.float32(cnt)
                            self._tt_meas_acc = cp.zeros((2,), dtype=cp.float32)
                            self._tt_meas_count = 0
                        try:
                            if not bool(cp.all(cp.isfinite(tt_meas_yx)).get()):
                                tt_meas_yx = cp.zeros((2,), dtype=cp.float32)
                        except Exception:
                            tt_meas_yx = cp.zeros((2,), dtype=cp.float32)

                        self._tt_cmd_yx = (1.0 - self._tt_leak) * self._tt_cmd_yx + self._tt_gain * tt_meas_yx

                        try:
                            if not bool(cp.all(cp.isfinite(self._tt_cmd_yx)).get()):
                                self._tt_cmd_yx = cp.zeros((2,), dtype=cp.float32)
                        except Exception:
                            self._tt_cmd_yx = cp.zeros((2,), dtype=cp.float32)
                        self._tt_phi_cmd = self._tt_cmd_to_phase(self._tt_cmd_yx)

                # LGS cannot sense global TT; strip mean slopes per-WFS block. Off-axis NGS: ignore (fill 0).
                if (self._wfs_is_lgs is not None) and (self._wfs_slices is not None) and (self._wfs_n_active is not None):
                    for wi, is_lgs in enumerate(self._wfs_is_lgs):
                        a, b = self._wfs_slices[wi]
                        na = int(self._wfs_n_active[wi])
                        if na <= 0:
                            continue
                        # vector layout per WFS is [sx(0..na-1), sy(0..na-1)]
                        sx = self._slopes_err_vec[a : a + na]
                        sy = self._slopes_err_vec[a + na : a + 2 * na]
                        if is_lgs:
                            sx -= sx.mean()
                            sy -= sy.mean()
                        else:
                            sx.fill(0.0)
                            sy.fill(0.0)

                xhat = self.R.solve_coeffs_stack(self._slopes_err_vec)

            if self._dm_cmd is None:
                self._dm_cmd = cp.zeros_like(xhat)

            self._dm_cmd = (1.0 - self._loop_leak) * self._dm_cmd - self._loop_gain * xhat

            # DM phase already piston+TT removed in coeffs_to_onaxis_phase (remove_tt_in_output=True)
            dm_phi_new = -self.R.coeffs_to_onaxis_phase(self._dm_cmd)

            # DM servo-lag FIFO: command computed at tick t becomes active at tick t+dm_delay_frames.
            if getattr(self, "_dm_phi_pipeline", None) is None:
                try:
                    n = int(getattr(self, "_dm_delay_frames", 1))
                except Exception:
                    n = 1
                self._dm_phi_pipeline = deque([None] * max(1, n))
            self._dm_phi_pipeline.append(dm_phi_new)
            if len(self._dm_phi_pipeline) > int(getattr(self, "_dm_delay_frames", 1)):
                self._dm_phi_pipeline.popleft()
            self._dm_phi = self._dm_phi_pipeline[0]

            # --- TT delay line advance (once per simulation tick) ---
            if getattr(self, "_tt_enabled", False):
                if len(self._tt_delay_line) == 0:
                    self._tt_phi = self._tt_phi_cmd
                else:
                    self._tt_delay_line.append(self._tt_phi_cmd)
                    self._tt_phi = self._tt_delay_line.popleft()
            else:
                self._tt_phi.fill(0.0)

            # ---- loop tab outputs (decoupled from 1 kHz control) ----
            if ((tc % self._loop_disp_period) == 0) or (self._loop_t == 0):
                corr = phases_atm[0]
                if dm_phi_prev is not None:
                    corr = corr - dm_phi_prev
                if tt_phi_prev is not None:
                    corr = corr - tt_phi_prev
                phi_corr = self._remove_piston_tt_fast(corr)
                phi_unc = self._remove_piston_tt_fast(phases_atm[0])

                # r0 via phase structure function at a few lags
                # Dϕ​(ρ)=⟨(ϕ(x+ρ)−ϕ(x))^2⟩ ≈ 6.88(r0​ρ​)^5/3
                V = cp.stack([phi_corr, phi_unc], axis=0)   # (2,H,W)
                Ds = []
                rhos = []
                if self._sf_cache is None:
                    self._sf_cache = {}
                for lag in (2, 4, 8):
                    # calculate structure function Dϕ​(ρ) with separation ρ = lag
                    if lag not in self._sf_cache or self._sf_cache[lag][0].shape != (M, M - lag):
                        mx_f = (m[:, lag:] & m[:, :-lag]).astype(cp.float32)
                        my_f = (m[lag:, :] & m[:-lag, :]).astype(cp.float32)
                        self._sf_cache[lag] = (mx_f, my_f, cp.sum(mx_f), cp.sum(my_f))
                    mx_f, my_f, mx_sum, my_sum = self._sf_cache[lag]

                    dpx = V[:, :, lag:] - V[:, :, :-lag]
                    dpy = V[:, lag:, :] - V[:, :-lag, :]
                    Dx = (dpx*dpx*mx_f[None, :, :]).sum(axis=(1, 2)) / (mx_sum + 1e-12)
                    Dy = (dpy*dpy*my_f[None, :, :]).sum(axis=(1, 2)) / (my_sum + 1e-12)
                    Ds.append(0.5 * (Dx + Dy))
                    rhos.append(self.sim.dx * lag)

                Ds = cp.stack(Ds, axis=0)  # (nlag, 2)
                rhos = cp.asarray(rhos, dtype=cp.float32)[:, None]

                slope = 5.0 / 3.0

                # fit in log space (for log(r0) linearity)
                y = cp.log(Ds) - cp.log(cp.float32(6.88)) - slope * cp.log(rhos)
                log_r0 = -cp.mean(y, axis=0) / slope

                # linearize back from log
                r0_eff_corrected, r0_eff_uncorrected = cp.exp(log_r0)

                if not self._loop_r0_only:
                    psf_period = self._hz_to_period(max(1, int(self.loop_display_hz)))
                    do_psf = ((tc % psf_period) == 0) or (self._loop_t == 0) or (self.long_exposure_psf is None)
                    if do_psf:
                        fft_pad = self.patch_M // 2
                        I, I_uncorrected = self._sensor_list[0].measure(
                            pupil=None,
                            phase_map=cp.stack((phi_corr, phi_unc), axis=0),
                            pad=fft_pad,
                            assume_radians=True,
                        )[2]

                        if self.plate_rad is None:
                            self.plate_rad = (
                                float(self._sensor_list[0].wavelength) * float(self.patch_M) /
                                (float(self.params["telescope_diameter"]) * float(I.shape[0] + 2 * fft_pad))
                            )

                        if self.long_exposure_psf is None:
                            self.long_exposure_psf = Analysis.PSFRingMean(shape=I.shape, K=self.long_exposure_frames, save_label="Corrected")
                            self.long_exposure_psf_uncorrected = Analysis.PSFRingMean(shape=I_uncorrected.shape, K=self.long_exposure_frames, save_label="Uncorrected")

                        self.long_exposure_psf.add(I)
                        self.long_exposure_psf_uncorrected.add(I_uncorrected)

                        sci_long = self.long_exposure_psf.mean()
                        raw_long = self.long_exposure_psf_uncorrected.mean()

                        zoom = 2
                        zimg = Analysis._zoom2d(sci_long, zoom, cp)
                        p, (_xc, _yc), fwhm, roi, (_x0, _y0), fwhm_z = self.gaussian_fitter.fit_frame(
                            zimg, search_r=24, loss="soft_l1", max_iter=10, zoom=zoom
                        )
                        zimg_uc = Analysis._zoom2d(raw_long, zoom, cp)
                        _, _, fwhm_uncorr, _, _, _ = self.gaussian_fitter_uc.fit_frame(
                            zimg_uc, search_r=24, loss="soft_l1", max_iter=10, zoom=zoom
                        )

                        self._loop_last_roi = (roi.astype(cp.float16)).get()
                        self._loop_last_fwhm = float(self.plate_rad * fwhm[0])
                        self._loop_last_fwhm_uncorr = float(self.plate_rad * fwhm_uncorr[0])
                        self._loop_last_gauss = (float(p[1]), float(p[2]), float(fwhm_z[1] / 2))

                    self.loop_ready.emit(
                        self._loop_last_roi,
                        (phi_corr * self.pupil_mask).get(),
                        r0_eff_corrected,
                        r0_eff_uncorrected,
                        self._loop_last_fwhm,
                        self._loop_last_fwhm_uncorr,
                        self._loop_t,
                        self._loop_last_gauss,
                    )
                else:
                    self.loop_ready.emit(
                        None,
                        (phi_corr * self.pupil_mask).get(),
                        r0_eff_corrected,
                        r0_eff_uncorrected,
                        0,
                        0,
                        self._loop_t,
                        (0, 0, 1, 1, 0),
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

            if rec is not None and want_recon and phase0 is not None and phi_res_disp is not None:
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