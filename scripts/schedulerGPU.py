# scripts/schedulerGPU_new.py
import time
from typing import Any, Dict, List, Optional

import cupy as cp
from PySide6.QtCore import QObject, Signal, Slot, QTimer

def profiling_safe_mean(a, axis):
    return a.mean(axis=axis)


class SimWorker(QObject):
    # status / lifecycle
    status = Signal(str)
    fpsReady = Signal(float)
    finished = Signal()

    # main outputs
    combined_ready = Signal(object)       # np (N,N)
    update_r0 = Signal(float)
    sim_signal = Signal(object)

    # single views (for tabs)
    layer_ready = Signal(int, object)     # (layer_idx, np (H,W))
    sensor_phase_ready = Signal(int, object)  # (sensor_idx, np (M,M))
    sensor_psf_ready   = Signal(int, object)  # (sensor_idx, np (M,M))

    # overview outputs (for grid pages)
    all_sensor_psf_ready = Signal(object)  # np (S, psf_M, psf_M)
    all_sensor_phase_ready = Signal(object)  # (sensor_idx, np (M,M))

    all_layers_ready     = Signal(object)  # np (L, layer_M, layer_M)

    def __init__(
        self,
        sim_factory,
        sim_kwargs: Dict[str, Any],
        layers: List[Dict[str, Any]],
        thetas_xy_rad,
        ranges_m,
        patch_size_px,
        patch_M,
        dt_s: float = 0.001,
        fps_cap: int = 60,

        # single-tab update rates
        layer_tab_hz: int = 10,
        sensor_hz: int = 10,

        # overview update rates
        overview_psf_hz: int = 5,
        overview_layer_hz: int = 2,

        # display sizes
        psf_M: Optional[int] = None,       # default: patch_M//2
        layer_M: Optional[int] = None,     # default: patch_M//2 (for downsample of layers)

        # PSF controls
        emit_psf: bool = True,
        pupil_mask=None,                  # for single-sensor PSF (should match patch_M x patch_M)
        parent=None,
    ):
        super().__init__(parent)

        self.sim_factory = sim_factory
        self.sim_kwargs = dict(sim_kwargs)
        self.layer_cfgs = list(layers)

        self.dt_s = float(dt_s)
        self.fps_cap = int(fps_cap)

        # AO geometry
        self.ranges_m = ranges_m
        self.thetas_gpu = cp.asarray(thetas_xy_rad, dtype=cp.float32)  # (S,2)
        self.patch_center = (self.sim_kwargs["N"]/2, self.sim_kwargs["N"]/2)
        self.patch_size_px = float(patch_size_px)
        self.patch_M = int(patch_M)

        self.psf_M = int(psf_M) if psf_M is not None else (self.patch_M // 2)
        self.layer_M = int(layer_M) if layer_M is not None else (self.patch_M // 2)

        # controls
        self.emit_psf = bool(emit_psf)
        self._overview_enabled = False

        # view selection
        self._visible_layer: Optional[int] = 0
        self._visible_sensor: int = 0

        # rate limit periods (ticks)
        self._interval_ms = max(1, int(1000 / max(1, self.fps_cap)))

        self._layer_tab_period = self._hz_to_period(layer_tab_hz)
        self._sensor_period    = self._hz_to_period(sensor_hz)
        self._overview_psf_period   = self._hz_to_period(overview_psf_hz)
        self._overview_layer_period = self._hz_to_period(overview_layer_hz)

        self._tick_count = 0

        # step-n state
        self._stepping = False
        self._steps_left = 0
        self._emit_every = 0
        self._batch = 20

        # fps stats
        self._frames = 0
        self._t0 = None

        # sim + masks
        self.sim = None
        self.pupil_mask = None if pupil_mask is None else cp.asarray(pupil_mask).astype(cp.float32)
        self._pupil_mask_psf = None  # built on build_sim (circular mask for overview PSFs)

        # timer
        self.timer = QTimer(self)
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

    # ------------------ lifecycle ------------------
    @Slot()
    def build_sim(self):
        if self.sim is not None:
            return

        cp.cuda.Device().use()
        self.status.emit("Building LayeredPhaseScreen on worker thread...")

        self.status.emit("   Creating sim...")
        self.sim = self.sim_factory(**self.sim_kwargs)
        [self.status.emit("     "+str((i, j))) for i, j in self.sim_kwargs.items()]

        for k, layer in enumerate(self.layer_cfgs):
            self.status.emit(f"   Adding layer {k+1}...")
            [self.status.emit("     "+str((i, j))) for i, j in layer.items()]

            self.sim.add_layer(**layer)

        self.status.emit("Screen Sim built!")
        

        # overview PSF uses circular pupil unless you provide your own
        self._pupil_mask_psf = self._build_circular_pupil(self.psf_M)

        self.status.emit(f"Built sim with {len(self.layer_cfgs)} layers.")
        self.update_r0.emit(self.sim.r0_total)
        self._emit_once()

    @Slot(dict)
    def add_layer(self, layer_cfgs):

        self.status.emit(f"Adding layer...")
        self.sim.add_layer(**layer_cfgs)
        self.layer_cfgs.append(layer_cfgs) 

        self.sim_signal.emit(self.sim.layers)
        self._emit_once()

    @Slot()
    def start(self):
        if self.sim is None:
            self.build_sim()
        self._stepping = False
        self._steps_left = 0
        self._tick_count = 0
        self._frames = 0
        self._t0 = None
        self.timer.start(self._interval_ms)
        self.status.emit("Running (continuous).")

    @Slot()
    def pause(self):
        self.timer.stop()
        self._stepping = False
        self._steps_left = 0
        self.status.emit("Paused.")

    @Slot()
    def stop(self):
        self.timer.stop()
        self._stepping = False
        self._steps_left = 0
        self.finished.emit()
        self.status.emit("Stopped.")

    @Slot()
    def reset(self):
        if self.sim is None:
            return
        if hasattr(self.sim, "hard_reset"):
            self.sim.hard_reset()
            self.status.emit("Hard reset to initial frame.")
        else:
            self.status.emit("Unable to hard reset! ")

        self._emit_once()

    # ------------------ stepping ------------------
    @Slot()
    def step_once(self):
        if self.sim is None:
            self.build_sim()
        self.timer.stop()
        self._ensure_cache()
        self._advance(1)
        self._emit_once()

    @Slot(int, int)
    def step_n(self, n: int, emit_every: int = 0):
        if self.sim is None:
            self.build_sim()
        self.timer.stop()
        self._stepping = True
        self._steps_left = max(0, int(n))
        self._emit_every = max(0, int(emit_every))
        QTimer.singleShot(0, self._tick_step_n)

    # ------------------ view selection ------------------
    @Slot(bool)
    def set_overview_enabled(self, en: bool):
        self._overview_enabled = bool(en)
        if self._overview_enabled:
            self.emit_overview_once()


    @Slot(int, int)
    def set_overview_rates(self, psf_hz: int, layer_hz: int):
        self._overview_psf_period = self._hz_to_period(psf_hz)
        self._overview_layer_period = self._hz_to_period(layer_hz)

    @Slot(int)
    def set_visible_layer(self, idx: int):
        self._visible_layer = int(idx)
        self._emit_visible_layer()

    @Slot(int)
    def set_visible_sensor(self, idx: int):
        self._visible_sensor = int(idx)
        self._emit_visible_sensor_products()

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

        self.step_once()

    def _regen_single_layer(self, i: int):
        # If your sim has a regen method, use it:
        if hasattr(self.sim, "regen_layer"):
            self.sim.regen_layer(i, seed_offset=self.layer_cfgs[i].get("seed_offset", 0))
            return

        # fallback: rebuild entire sim
        self.status.emit(f"Rebuilding sim to apply seed_offset for layer {i}...")
        sim_kwargs = dict(self.sim_kwargs)
        self.sim = self.sim_factory(**sim_kwargs)
        for layer in self.layer_cfgs:
            self.sim.add_layer(**layer)
        self._pupil_mask_psf = self._build_circular_pupil(self.psf_M)
        self.status.emit("Rebuilt sim.")

    # ------------------ internals ------------------
    def _advance(self, n_steps: int):
        if hasattr(self.sim, "advance"):
            for _ in range(n_steps):
                self.sim.advance(self.dt_s)
        else:
            if hasattr(self.sim, "time"):
                self.sim.time += self.dt_s * n_steps

    def _ensure_cache(self):
        if hasattr(self.sim, "_ensure_layer_cache"):
            self.sim._ensure_layer_cache()

    def _combined_gpu(self):
        """
        Return combined phase (sum of layers) as a 2D cupy array.
        Uses world-anchored layer views so display doesn't jump even if caches pan.
        """
        N = int(self.sim_kwargs["N"])  # display size (or patch_M)

        L = len(self.sim.layers)
        if L == 0:
            return cp.zeros((N, N), dtype=cp.float32)

        # (L, N, N)
        stack = cp.stack(
            [ 
                self.sim.get_layer_view(
                    i,
                    center_xy_pix=((N - 1) / 2.0, (N - 1) / 2.0),
                    size_pixels=N,
                    M=N,
                    angle_deg=0.0,
                    return_gpu=True,
                )
                if self.sim._layers_obj[i].active 
                else cp.zeros((N, N,), dtype=cp.float32)
                for i in range(L) 
            ],
            axis=0,
        )

        # sum over layers -> (N, N)
        combined = cp.sum(stack, axis=0)

        # remove piston
        combined -= cp.mean(combined)

        return combined

    # ---------- PSF + phase emissions ----------
    def _phase_to_psf_logI(self, phase, pupil):
        E = pupil * cp.exp(1j * phase)
        F = cp.fft.fftshift(cp.fft.fft2(E))
        I = (F.real * F.real + F.imag * F.imag).astype(cp.float32)
        return cp.log10(I + 1e-12)

    def _emit_visible_sensor_products(self):
        if self.sim is None:
            return
        
        self._ensure_cache()
        s = int(self._visible_sensor)
        S = int(self.thetas_gpu.shape[0])
        if not (0 <= s < S):
            return

        th = self.thetas_gpu[s:s+1]  # (1,2)
        rng = self.ranges_m[s:s+1]

        phases = self.sim.sample_patches_batched(
            thetas_xy_rad=th,
            center_xy_pix=self.patch_center,
            size_pixels=self.patch_size_px,
            M=self.patch_M,
            angle_deg=0.0,
            ranges_m = rng,
            remove_piston=True,
            return_gpu=True,
        )
        phase0 = phases[0]
        self.sensor_phase_ready.emit(s, cp.asnumpy(phase0.astype(cp.float16)))

        if self.emit_psf:
            if self.pupil_mask is None:
                # fallback: circular pupil matching patch_M
                pupil = self._build_circular_pupil(self.patch_M)
            else:
                pupil = self.pupil_mask
                if pupil.shape != phase0.shape:
                    # fallback if mismatch
                    pupil = self._build_circular_pupil(self.patch_M)

            psf = self._phase_to_psf_logI(phase0, pupil)
            self.sensor_psf_ready.emit(s, cp.asnumpy(psf.astype(cp.float16)))

    # ---------- overview emissions ----------
    def _compute_all_sensor_psfs(self):
        phases = self.sim.sample_patches_batched(
            thetas_xy_rad=self.thetas_gpu,
            center_xy_pix=self.patch_center,
            size_pixels=self.patch_size_px,
            M=self.psf_M,
            angle_deg=0.0,
            ranges_m = self.ranges_m,
            remove_piston=True,
            return_gpu=True,
        )  # (S,M,M)
        self.all_sensor_phase_ready.emit(cp.asnumpy(phases.astype(cp.float16)))

        pupil = self._pupil_mask_psf  # (M,M)
        E = pupil[None, :, :] * cp.exp(1j * phases)
        F = cp.fft.fftshift(cp.fft.fft2(E, axes=(-2, -1)), axes=(-2, -1))
        I = (F.real * F.real + F.imag * F.imag).astype(cp.float32)
        return cp.log10(I + 1e-12)

    def _downsample_layers_stack(self, stack):
        # (L,N,N) -> (L,layer_M,layer_M) using box-average if integer factor
        L, N, _ = stack.shape
        M = self.layer_M
        if M == N:
            return stack
        f = N // M
        if f * M != N:
            return stack[:, ::f, ::f]
        return profiling_safe_mean(stack.reshape(L, M, f, M, f), axis=(2, 4))

    # ---------- layer emissions ----------
    def _emit_visible_layer(self):
        if self.sim is None or self._visible_layer is None:
            return
        i = int(self._visible_layer)
        if not (0 <= i < len(self.sim.layers)):
            return
        self._ensure_cache()
        phi = self.sim.get_layer_view(i, M=self.sim_kwargs["N"], return_gpu=True)

        if phi is not None:
            self.layer_ready.emit(i, cp.asnumpy(phi.astype(cp.float16)))

    def _emit_once(self):
        self._ensure_cache()
        self.combined_ready.emit(cp.asnumpy(self._combined_gpu().astype(cp.float16)))
        self._emit_visible_layer()
        self._emit_visible_sensor_products()
        psfs = self._compute_all_sensor_psfs()
        self.all_sensor_psf_ready.emit(cp.asnumpy(psfs.astype(cp.float16)))
        
        self._overview_enabled = True # Temporarily override once
        self._emit_overview()
        self._overview_enabled = False

    @Slot()
    def emit_overview_once(self):
        """Compute and emit overview outputs once (works for step_once / step_n)."""
        if self.sim is None:
            self.build_sim()
        self._ensure_cache()
        self._emit_overview()

    def _emit_overview(self):
        if not self._overview_enabled:
            return

        # all sensor PSFs
        self._ensure_cache()
        phis = [L.get("phi", None) for L in self.sim.layers]
        phis = [p for p in phis if p is not None]
        if len(phis) != len(self.sim.layers):
            self.status.emit("Overview skipped: some layer phis not ready yet.")
            return
        
        psfs = self._compute_all_sensor_psfs()
        self.all_sensor_psf_ready.emit(cp.asnumpy(psfs.astype(cp.float16)))

        # all layers (downsampled)
        phases = self.sim.sample_patches_batched(
            thetas_xy_rad=self.thetas_gpu,
            center_xy_pix=self.patch_center,
            size_pixels=self.patch_size_px,
            M=self.psf_M,
            angle_deg=0.0,
            ranges_m = self.ranges_m,
            remove_piston=True,
            return_gpu=True,
            return_per_layer=True,
            layer_first=True,
        )  # (S,M,M)
        self.all_layers_ready.emit(cp.asnumpy(phases.astype(cp.float16)))


    # ------------------ main tick ------------------
    @Slot()
    def _tick(self):
        self._ensure_cache()
        self._advance(1)

        self._tick_count += 1

        # always emit combined (cheap-ish; still a copy)
        self.combined_ready.emit(cp.asnumpy(self._combined_gpu().astype(cp.float16)))

        # visible layer at layer_tab rate
        if self._visible_layer is not None and (self._tick_count % self._layer_tab_period == 0):
            self._emit_visible_layer()

        # visible sensor at sensor rate
        if (self._tick_count % self._sensor_period) == 0:
            self._emit_visible_sensor_products()

        # overview at its own rates
        if self._overview_enabled:
            if (self._tick_count % self._overview_psf_period) == 0:
                psfs = self._compute_all_sensor_psfs()
                self.all_sensor_psf_ready.emit(cp.asnumpy(psfs.astype(cp.float16)))

            if (self._tick_count % self._overview_layer_period) == 0:
                self._emit_overview()

        # fps
        now = time.perf_counter()
        if self._t0 is None:
            self._t0 = now
        self._frames += 1
        if now - self._t0 >= 0.5:
            self.fpsReady.emit(self._frames / (now - self._t0))
            self._frames = 0
            self._t0 = now

    def _tick_step_n(self):
        if not self._stepping or self._steps_left <= 0:
            self._stepping = False
            self._emit_once()
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
