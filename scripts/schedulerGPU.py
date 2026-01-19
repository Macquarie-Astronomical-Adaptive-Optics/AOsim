# scripts/schedulerGPU.py

import time
from typing import Any, Dict, List, Optional

import cupy as cp
from PySide6.QtCore import QObject, Signal, Slot, QTimer, Qt

from scripts.reconstructor import TomoOnAxisIM_CuPy


def profiling_safe_mean(a, axis):
    return a.mean(axis=axis)


def _u8_linear(x, vmin, vmax):
    eps = 1e-12
    x = cp.nan_to_num(x, nan=vmin, posinf=vmax, neginf=vmin)
    y = (x - vmin) * (255.0 / (vmax - vmin + eps))
    return cp.clip(y, 0.0, 255.0).astype(cp.uint8)


def _phase_to_u8(phi, k=3.0):
    mu = cp.mean(phi)
    sig = cp.std(phi)
    return _u8_linear(phi, mu - k * sig, mu + k * sig)


def _logpsf_to_u8(logI, dyn=6.0):
    vmax = cp.max(logI)
    vmin = vmax - dyn
    return _u8_linear(logI, vmin, vmax)

def time_n():
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter_ns()

class SimWorker(QObject):
    # status / lifecycle
    status_log = Signal(object)
    status = Signal(str)
    status_running = Signal(bool)

    fpsReady = Signal(float)
    finished = Signal()

    # main outputs
    combined_ready = Signal(object)       # np (N,N)
    update_r0 = Signal(float)
    sim_signal = Signal(object)
    reconstructed = Signal(object)

    # single views (for tabs)
    layer_ready = Signal(int, object)          # (layer_idx, np (H,W))
    sensor_phase_ready = Signal(int, object)   # (sensor_idx, np (M,M))
    sensor_psf_ready   = Signal(int, object)   # (sensor_idx, np (M,M))

    # overview outputs (for grid pages)
    all_sensor_psf_ready = Signal(object)      # np (S, psf_M, psf_M)
    all_sensor_phase_ready = Signal(object)    # np (S, psf_M, psf_M) or (S, patch_M, patch_M)

    all_layers_ready = Signal(object)          # np (L, S, layer_M, layer_M)

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
        fps_cap: int = 2000,

        # single-tab update rates
        layer_tab_hz: int = 10,
        sensor_hz: int = 20,

        # overview update rates
        overview_psf_hz: int = 5,
        overview_layer_hz: int = 10,

        # display sizes
        psf_M: Optional[int] = None,       # default: patch_M//2
        layer_M: Optional[int] = None,     # default: patch_M//2 (for downsample of layers)

        # PSF controls
        emit_psf: bool = True,
        pupil_mask=None,                  # for single-sensor PSF (should match patch_M x patch_M)
        parent=None,

        # combined update rate (display). None/0 -> every tick
        combined_hz: Optional[int] = None,
    ):
        super().__init__(parent)

        self.sim_factory = sim_factory
        self.sim_kwargs = dict(sim_kwargs)
        self.layer_cfgs = list(layers)

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
        self._detailed_enabled = True

        # view selection
        self._visible_layer: Optional[int] = 0
        self._visible_sensor: int = 0

        # rate limit periods (ticks)
        self._interval_ms = max(1, int(1000 / max(1, self.fps_cap)))
        self._combined_period = self._hz_to_period(int(combined_hz)) if (combined_hz is not None and combined_hz > 0) else 1

        self._layer_tab_period = self._hz_to_period(layer_tab_hz)
        self._sensor_period    = self._hz_to_period(sensor_hz)
        self._overview_psf_period   = self._hz_to_period(overview_psf_hz)
        self._overview_layer_period = self._hz_to_period(overview_layer_hz)

        self._tick_count = 0

        # stepping state
        self._stepping = False
        self._steps_left = 0
        self._emit_every = 0
        self._batch = 20

        # fps stats
        self._frames = 0
        self._t0 = None

        # sim + masks
        self.sim = None
        self.R = None

        # reusable GPU buffers (to avoid per-tick allocations)
        self._combined_buf = None
        self.pupil_mask = None if pupil_mask is None else cp.asarray(pupil_mask).astype(cp.float32)
        self._pupil_mask_psf = None
        self._pupil_mask_patch = None

        # ------------------ per-frame sampling cache ------------------
        self._frame_tick = -1
        self._frame_need_per_layer = False
        self._frame_phases_M = None        # (S, patch_M, patch_M) cp.float32
        self._frame_layers_M = None        # (L, S, patch_M, patch_M) cp.float32 or None

        # lazily computed downsampled caches
        self._frame_phases_psfM = None     # (S, psf_M, psf_M)
        self._frame_layers_layerM = None   # (L, S, layer_M, layer_M)

        # timer
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)

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

    def _invalidate_frame_cache(self):
        self._frame_tick = -1
        self._frame_need_per_layer = False
        self._frame_phases_M = None
        self._frame_layers_M = None
        self._frame_phases_psfM = None
        self._frame_layers_layerM = None

    def _downsample_stack(self, arr, M_out: int):
        M_in = int(arr.shape[-1])
        M_out = int(M_out)
        if M_out == M_in:
            return arr

        f = M_in // M_out
        if f > 1 and f * M_out == M_in:
            if arr.ndim == 3:
                S = arr.shape[0]
                return profiling_safe_mean(arr.reshape(S, M_out, f, M_out, f), axis=(2, 4))
            elif arr.ndim == 4:
                L, S = arr.shape[0], arr.shape[1]
                return profiling_safe_mean(arr.reshape(L, S, M_out, f, M_out, f), axis=(3, 5))

        return arr[..., ::f, ::f]

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
        [self.status_log.emit("     " + str((i, j))) for i, j in self.sim_kwargs.items()]
        max_theta = max([sensor.sen_angle for sensor in self._sensor_list]) if self._sensor_list else 0.0

        for k, layer in enumerate(self.layer_cfgs):
            self.status_log.emit(f"   Adding layer {k+1}...")
            [self.status_log.emit("     " + str((i, j))) for i, j in layer.items()]
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
        self._emit_once()
        self._emit_visible_layer()


        # layer_heights_m = [L["altitude_m"] for L in self.layer_cfgs]
        # self.R = TomoOnAxisIM_CuPy(
        #     sensors=self._sensor_list,
        #     layer_heights_m=layer_heights_m,
        #     dx_world_m=float(self.sim.dx),
        #     M=self.patch_M,
        #     size_pixels=self.patch_M,
        #     reg_alpha=1e-2,
        # )
        # self.status_log.emit("Tomographic reconstructor initialized!")

        # self.status.emit("Building interaction matrix")
        # self.R.build_interaction_matrix()
        # self.status_log.emit("Interaction matrix built!")

        # self.R.factorize()
        # self.status_log.emit("Interaction matrix factorized!")

        # if hasattr(self.R, "prepare_runtime"):
        #     self.R.prepare_runtime()
        #     self.status_log.emit("Runtime coeff matrix prepared!")

        self.status.emit("Sim ready")
        self.status_running.emit(False)


    @Slot(dict)
    def add_layer(self, layer_cfgs):
        if self.sim is None:
            self.build_sim()

        self.status_log.emit("Adding layer...")
        self.sim.add_layer(**layer_cfgs)
        self.layer_cfgs.append(layer_cfgs)

        # update any GUI lists
        if hasattr(self.sim, "layers"):
            self.sim_signal.emit(self.sim.layers)

        # r0 update if available
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
        r = cp.asarray([float(s.gs_range_m) for s in self._sensor_list], dtype=cp.float32)
        if r.size != S:
            r = cp.full((S,), cp.inf, dtype=cp.float32)
        self.ranges_gpu = r

        self._invalidate_frame_cache()

    @Slot()
    def start(self):
        if self.sim is None:
            self.build_sim()
        self._tick_count = 0
        self._frames = 0
        self._t0 = None
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


    @Slot()
    def reset(self):
        if self.sim is None:
            return
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
        
        self.layer_ready.emit(i, (_phase_to_u8(outL)))

    @Slot(int)
    def set_visible_sensor(self, idx: int):
        self._visible_sensor = int(idx)

    # ------------------ layer edits ------------------
    @Slot(list, bool)
    def set_active(self, names: list, active: bool):
        for i, name in enumerate(names):
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

        # altitude
        if "altitude_m" in p:
            h = float(p["altitude_m"])
            if hasattr(self.sim, "set_layer_altitude"):
                self.sim.set_layer_altitude(i, h)


        # wind
        if "wind" in p and isinstance(p["wind"], (list, tuple)) and len(p["wind"]) == 2:
            vx, vy = float(p["wind"][0]), float(p["wind"][1])
            if hasattr(self.sim, "set_layer_wind"):
                self.sim.set_layer_wind(i, vx, vy)
            else:
                self.sim.layers[i]["vx"] = vx
                self.sim.layers[i]["vy"] = vy

        # L0
        if "L0" in p:
            L0 = float(p["L0"])
            if hasattr(self.sim, "set_layer_L0"):
                self.sim.set_layer_L0(i, L0)
            else:
                self.sim.layers[i]["L0"] = L0
                self.sim.layers[i]["phi"] = None
                self.sim.layers[i]["phi_t"] = None

        # weight
        if "r0" in p:
            r0 = float(p["r0"])
            if hasattr(self.sim, "set_layer_r0"):
                self.sim.set_layer_r0(i, r0)
            else:
                self.sim.layers[i]["r0"] = r0

            self.update_r0.emit(self.sim.r0_total)

        # seed_offset: rebuild just that layer if supported; otherwise rebuild sim
        if "seed_offset" in p:
            new_off = int(p["seed_offset"])
            old_off = int(self.layer_cfgs[i].get("seed_offset", 0))
            if new_off != old_off:
                self.layer_cfgs[i]["seed_offset"] = new_off
                self._regen_single_layer(i)

        # invalidate layer cache
        if i < len(self.sim.layers):
            self.sim.layers[i]["phi"] = None
            self.sim.layers[i]["phi_t"] = None

        self.step_once(status=False)

    def _regen_single_layer(self, i: int):
        # If your sim has a regen method, use it:
        if hasattr(self.sim, "regen_layer"):
            self.sim.regen_layer(i, seed_offset=self.layer_cfgs[i].get("seed_offset", 0))
            return

        # fallback: rebuild entire sim
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

    def _emit_visible_sensor_products_from_cache(self):
        if self.sim is None or not self._detailed_enabled:
            return
        sidx = int(self._visible_sensor)
        S = int(self.thetas_gpu.shape[0])
        if not (0 <= sidx < S):
            return

        phases, _ = self._sample_frame_once(need_per_layer=False)
        if phases is None:
            return

        phase0 = phases[sidx]
        self.sensor_phase_ready.emit(sidx, (_phase_to_u8(phase0)))

        if self.emit_psf:
            pupil = self._pupil_mask_patch
            psf = self._phase_to_psf_logI(phase0, pupil)
            self.sensor_psf_ready.emit(sidx, (_logpsf_to_u8(psf)))

    def _compute_all_sensor_psfs_from_cache(self):
        phases, _ = self._sample_frame_once(need_per_layer=False)
        if phases is None:
            return None

        if self.psf_M != self.patch_M:
            if self._frame_phases_psfM is None:
                self._frame_phases_psfM = self._downsample_stack(phases, self.psf_M)
            phases_psf = self._frame_phases_psfM
        else:
            phases_psf = phases

        self.all_sensor_phase_ready.emit((_phase_to_u8(phases_psf)))

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

        self.all_layers_ready.emit((_phase_to_u8(outLd)))

    def _emit_once(self):
        """One-shot: do all emissions once, with at most one sampling call."""
        self._ensure_cache()
        self._invalidate_frame_cache()

        # combined (world-anchored) at display size
        if self._detailed_enabled:
            combined = self._combined_gpu()
            self.combined_ready.emit((_phase_to_u8(combined)))

        # visible sensor products
        self._emit_visible_sensor_products_from_cache()

        # overview PSFs
        psfs = self._compute_all_sensor_psfs_from_cache()
        if psfs is not None:
            self.all_sensor_psf_ready.emit((_logpsf_to_u8(psfs)))

        # overview layers
        was = self._overview_enabled
        self._overview_enabled = True
        self._emit_overview_from_cache()
        self._overview_enabled = was

    # ------------------ main tick ------------------
    @Slot()
    def _tick(self):
        t0 = time_n()

        self._ensure_cache()
        self._advance(1)
        self._tick_count += 1

        # decide sampling needs up front to sample call once per tick
        need_visible_sensor = ((self._tick_count % self._sensor_period) == 0)
        need_visible_tab = ((self._tick_count % self._layer_tab_period) == 0)
        need_overview_psf   = (self._overview_enabled and ((self._tick_count % self._overview_psf_period) == 0))
        need_overview_layer = (self._overview_enabled and ((self._tick_count % self._overview_layer_period) == 0))

        need_per_layer = bool(need_overview_layer)

        t1 = time_n()
        phases, _ = self._sample_frame_once(need_per_layer=need_per_layer)

        t2 = time_n()

        slopes_list = []
        if phases is not None and self.R is not None:
            for s, ph in zip(self._sensor_list, phases):
                _, slopes, _ = s.measure(phase_map=ph, return_image=True)
                slopes_list.append(slopes[0])

        t3 = time_n()
        
        if self.R is not None and slopes_list:
            reconstructed = self.R.reconstruct_onaxis(slopes_list)

        t4 = time_n()
        

        # combined at capped rate
        t_c_0 = 0
        t_c_1 = 0
        if self._detailed_enabled:
            if self._combined_period <= 1 or self._tick_count == 1 or (self._tick_count % self._combined_period) == 0:
                t_c_0 = time_n()
                combined = self._combined_gpu()
                self.combined_ready.emit((_phase_to_u8(combined)))
                t_c_1 = time_n()


        # visible sensor at its rate (uses cached sampling)
        t_s_0 = 0
        t_s_1 = 0
        if need_visible_sensor:
            t_s_0 = time_n()
            self._emit_visible_sensor_products_from_cache()
            t_s_1 = time_n()

        if need_visible_tab:
            self._emit_visible_layer()
        # overview PSFs (uses cached sampling)
        t_os_0 = 0
        t_os_1 = 0
        if need_overview_psf:
            t_os_0 = time_n()
            psfs = self._compute_all_sensor_psfs_from_cache()
            if psfs is not None:
                self.all_sensor_psf_ready.emit((_logpsf_to_u8(psfs)))
            t_os_1 = time_n()

        # overview layers (uses cached per-layer sampling)
        t_ol_0 = 0
        t_ol_1 = 0
        if need_overview_layer:
            t_ol_0 = time_n()
            self._emit_overview_from_cache()
            t_ol_1 = time_n()

        # fps
        now = time_n()
        if self._t0 is None:
            self._t0 = now
        self._frames += 1
        tdelta = (now - self._t0)/1e9
        fps = self._frames / tdelta

        if (fps < 60 and self._frames % 5 == 0) or need_visible_tab or need_visible_sensor or need_overview_psf or need_overview_layer:
            self.fpsReady.emit(fps)
            self.status.emit(
                f"Running Sim {fps:.0f} fps : {((now - self._t0)/1e6)/self._frames:.0f} ms/t measured : {(now - t0)/1e6:.0f} ms/t estimated "
                f"\nAdvanceSim {(t1-t0)/1e6:.2f}ms | ScrnSample {(t2 - t1)/1e6:.0f}ms | SensorMeasure {(t3-t2)/1e6:.0f}ms | Reconstruct {(t4-t3)/1e6:.1f}ms"
                f"\nSensorPSF {(t_s_1-t_s_0)/1e6:.0f}ms | TotalScreen {(t_c_1-t_c_0)/1e6:.0f}ms | OverviewSensors {(t_os_1-t_os_0)/1e6:.0f}ms | OverviewScreens {(t_ol_1-t_ol_0)/1e6:.0f}ms"
                f"\n"
            )
            self._frames = 0
            self._t0 = now
