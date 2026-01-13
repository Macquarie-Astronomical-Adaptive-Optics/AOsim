import aotools
import cupy as cp
import numpy as np
from pathlib import Path
import json
import threading

# -----------------------------
# Config / parameter management
# -----------------------------
with open(Path(__file__).parent.parent / "config_default.json", "r") as f:
    config = json.load(f)

params = config.copy()

def set_params(new_params):
    """Update module-level params (used as defaults when explicit args are None)."""
    global params
    params = new_params

# -----------------------------
# Small GPU caches (keyed)
# -----------------------------
_XY_CACHE = {}          # (device, grid_size, dtype) -> (y_col, x_row)
_PUPIL_CACHE = {}       # (device, grid_size, obsc, dtype) -> pupil
_SCIENCE_I0PEAK = {}    # (device, grid_size, obsc, pad, science_lambda, dtype) -> float
_RADIAL_CACHE = {}      # (backend, shape) -> (r_int, radial_cnt)
_COORD_CACHE = {}       # (device, h_p, w_p, dtype) -> (y_coords, x_coords)
_SUBAP_GEOM_CACHE = {}  # (device, n_sub, grid_size, obsc) -> dict geometry (for SH)

def _device_id():
    try:
        return int(cp.cuda.runtime.getDevice())
    except Exception:
        return 0

def _get_xy(grid_size: int, dtype=cp.float32):
    """Return (y[:,1], x[1,:]) grids on GPU for a given grid_size."""
    key = (_device_id(), int(grid_size), dtype)
    out = _XY_CACHE.get(key)
    if out is None:
        y = cp.arange(grid_size, dtype=dtype)[:, None]
        x = cp.arange(grid_size, dtype=dtype)[None, :]
        out = (y, x)
        _XY_CACHE[key] = out
    return out

def _as_float32(x):
    return x.astype(cp.float32, copy=False) if isinstance(x, cp.ndarray) else x

# -----------------------------
# Phase screen / phase maps
# -----------------------------
class PhaseMap_tools:
    @staticmethod
    def generate_phase_map(
        grid_size=None, telescope_diameter=None, r0=None, L0=None, Vwind=None,
        px_scale=None, random_seed=None, **kwargs
    ):
        """
        From AOtools:
        A "Phase Screen" for use in AO simulation using the Fried method for Kolmogorov turbulence.

        Parameters:
        nx_size (int): Size of phase screen (NxN)
        pixel_scale(float): Size of each phase pixel in metres
        r0 (float): fried parameter (metres)
        L0 (float): Outer scale (metres)
        random_seed (int, optional): seed for the random number generator
        stencil_length_factor (int, optional): How much longer is the stencil than the desired phase? default is 4
        """
        # read current params when values not provided
        if grid_size is None:
            grid_size = params.get("grid_size")
        if telescope_diameter is None:
            telescope_diameter = params.get("telescope_diameter")
        if r0 is None:
            r0 = params.get("r0")
        if L0 is None:
            L0 = params.get("L0")
        if Vwind is None:
            Vwind = params.get("Vwind")
        if random_seed is None:
            random_seed = params.get("random_seed")
        if px_scale is None:
            px_scale = telescope_diameter / grid_size

        # aotools phase screen is CPU-based; keep it that way (init cost is amortized)
        phase_screen = aotools.infinitephasescreen.PhaseScreenKolmogorov(
            int(np.ceil(grid_size)), px_scale, r0, L0, random_seed=random_seed, **kwargs
        )

        add_row = phase_screen.add_row

        # generate next rows based on wind vel and frame_rate
        def advance_phase_map(frame_rate=500):
            # shift (m) over dt, then to pixels
            shift_m = float(Vwind) / float(frame_rate)
            shift_pix = shift_m / px_scale
            n_rows = int(np.ceil(abs(shift_pix)))
            # advance at least one row per call
            for _ in range(max(1, n_rows)):
                add_row()
            return add_row()

        return phase_screen, advance_phase_map

    @staticmethod
    def gaussian(r0, c0, amp=1e-6, sigma=None, grid_size=None, dtype=cp.float32):
        """Single 2D Gaussian on GPU."""
        if grid_size is None:
            grid_size = params.get("grid_size")
        if sigma is None:
            sigma = max(1.0, float(grid_size) * 0.05)

        y, x = _get_xy(int(grid_size), dtype=dtype)
        # compute in float32 to reduce bandwidth
        dy = y - cp.asarray(r0, dtype=dtype)
        dx = x - cp.asarray(c0, dtype=dtype)
        inv_2s2 = cp.asarray(1.0 / (2.0 * float(sigma) * float(sigma)), dtype=dtype)
        surf = cp.asarray(amp, dtype=dtype) * cp.exp(-(dx * dx + dy * dy) * inv_2s2)
        return surf

# -----------------------------
# Pupil / DM helpers
# -----------------------------
class Pupil_tools:
    @staticmethod
    def circle(radius, grid_size, center=None, dtype=cp.float32):
        """
        Generate a grid_size by grid_size px image of with a circle with px radius.

        radius (float): circle radius
        grid_size (int/float): image px by px size
        center (float, float): circle center coordinates
        """
        if center is None:
            # Use pixel-center convention for even-sized grids to keep the pupil perfectly symmetric.
            # For N even, (N-1)/2 is the true center of the discrete pixel grid.
            center = ((grid_size - 1) / 2.0, (grid_size - 1) / 2.0)

        y, x = _get_xy(int(grid_size), dtype=dtype)
        dy = y - cp.asarray(center[0], dtype=dtype)
        dx = x - cp.asarray(center[1], dtype=dtype)
        dist2 = dy * dy + dx * dx
        r2 = cp.asarray(float(radius) * float(radius), dtype=dtype)
        return (dist2 <= r2).astype(dtype, copy=False)

    @staticmethod
    def generate_pupil(grid_size=None, telescope_center_obscuration=None, dtype=cp.float32):
        """
        Generate circular pupil with central obscuration fraction.
        
        grid_size (int/float): px by px size of pupil
        telescope_center_obscuration (float, 0.0 to 1.0): size of central obscuration
        """
        if grid_size is None:
            grid_size = params.get("grid_size")
        if telescope_center_obscuration is None:
            telescope_center_obscuration = params.get("telescope_center_obscuration")

        key = (_device_id(), int(grid_size), float(telescope_center_obscuration), dtype)
        pup = _PUPIL_CACHE.get(key)
        if pup is not None:
            return pup

        outer = Pupil_tools.circle(grid_size / 2.0, grid_size, dtype=dtype)
        inner = Pupil_tools.circle((grid_size / 2.0) * float(telescope_center_obscuration), grid_size, dtype=dtype)
        pup = (outer - inner).astype(dtype, copy=False)
        _PUPIL_CACHE[key] = pup
        return pup

    @staticmethod
    def generate_actuators(pupil=None, actuators=None, grid_size=None):
        """
        Finds actuator coordinates seen by pupil.

        pupil (N, N): image/mask representing telescope pupil
        actuators (float/int): number of actuators per row/col
        """

        if grid_size is None:
            grid_size = params.get("grid_size")
        if actuators is None:
            actuators = params.get("actuators")
        if pupil is None:
            pupil = Pupil_tools.generate_pupil(grid_size=grid_size)

        # grid of candidate centers
        step = float(grid_size) / float(actuators)
        coords = cp.arange(step / 2.0, float(grid_size), step, dtype=cp.float32)
        rr, cc = cp.meshgrid(coords, coords, indexing="ij")

        rr_idx = rr.astype(cp.int32)
        cc_idx = cc.astype(cp.int32)

        # return actuators seen by pupil
        mask = pupil[rr_idx, cc_idx] > 0
        act_centers = cp.stack([rr_idx[mask], cc_idx[mask]], axis=1)
        return act_centers

    @staticmethod
    def actuator_influence_single(
        act_center_rc,
        pupil=None,
        actuators=None,
        poke_amplitude=None,
        sigma=None,
        grid_size=None,
        dtype=cp.float32,
    ):
        """
        Influence map for one actuator center (r,c).
        """
        if grid_size is None:
            grid_size = params.get("grid_size")
        if actuators is None:
            actuators = params.get("actuators")
        if poke_amplitude is None:
            poke_amplitude = params.get("poke_amplitude")
        if pupil is None:
            pupil = Pupil_tools.generate_pupil(grid_size=grid_size, dtype=dtype)

        if sigma is None:
            sigma = (float(grid_size) / float(actuators)) * 0.6

        r0, c0 = int(act_center_rc[0]), int(act_center_rc[1])

        y, x = _get_xy(int(grid_size), dtype=dtype)
        dy = y - cp.asarray(r0, dtype=dtype)
        dx = x - cp.asarray(c0, dtype=dtype)
        inv_2s2 = cp.asarray(1.0 / (2.0 * float(sigma) * float(sigma)), dtype=dtype)
        surf = cp.exp(-(dx * dx + dy * dy) * inv_2s2)

        # normalize (peak is 1.0 for this gaussian discretization)
        surf *= cp.asarray(float(poke_amplitude), dtype=dtype)
        surf *= pupil
        return surf

    @staticmethod
    def generate_actuator_influence_map(
        act_centers=None,
        pupil=None,
        actuators=None,
        poke_amplitude=None,
        grid_size=None,
        sigma=None,
        return_mode="full",
        chunk_size=64,
        dtype=cp.float32,
    ):
        """
        Influence maps for each actuator, returns an image with values 0.0 to 1.0
        representing actuator influence on that pixel.

        act_centers: actuator coordinates, from generate_actuators
        sigma: strength/size of actuator influence
        return_mode:
          - "lazy": returns a callable get_map(i) -> (N,N) influence map, plus act_centers
          - "full": returns (n_act, N, N) array (!may use a lot of memory!)
          - "chunked": returns generator yielding (start, stop, block) blocks
        """
        if grid_size is None:
            grid_size = params.get("grid_size")
        if actuators is None:
            actuators = params.get("actuators")
        if poke_amplitude is None:
            poke_amplitude = params.get("poke_amplitude")
        if pupil is None:
            pupil = Pupil_tools.generate_pupil(grid_size=grid_size, dtype=dtype)
        if act_centers is None:
            act_centers = Pupil_tools.generate_actuators(pupil=pupil, actuators=actuators, grid_size=grid_size)
        if sigma is None:
            sigma = (float(grid_size) / float(actuators)) * 0.6

        if return_mode == "lazy":
            # keep small closures; no big allocations
            def get_map(i):
                return Pupil_tools.actuator_influence_single(
                    act_centers[int(i)], pupil=pupil, actuators=actuators,
                    poke_amplitude=poke_amplitude, sigma=sigma, grid_size=grid_size, dtype=dtype
                )
            return get_map, act_centers

        n_act = int(act_centers.shape[0])

        if return_mode == "chunked":
            def gen():
                for s in range(0, n_act, int(chunk_size)):
                    e = min(n_act, s + int(chunk_size))
                    block = cp.empty((e - s, int(grid_size), int(grid_size)), dtype=dtype)
                    for j in range(s, e):
                        block[j - s] = Pupil_tools.actuator_influence_single(
                            act_centers[j], pupil=pupil, actuators=actuators,
                            poke_amplitude=poke_amplitude, sigma=sigma, grid_size=grid_size, dtype=dtype
                        )
                    yield s, e, block
            return gen(), act_centers

        # "full": 
        influence_maps = cp.empty((n_act, int(grid_size), int(grid_size)), dtype=dtype)
        for j in range(n_act):
            influence_maps[j] = Pupil_tools.actuator_influence_single(
                act_centers[j], pupil=pupil, actuators=actuators,
                poke_amplitude=poke_amplitude, sigma=sigma, grid_size=grid_size, dtype=dtype
            )
        return influence_maps

    @staticmethod
    def dm_surface_from_commands(
        commands,
        act_centers=None,
        pupil=None,
        actuators=None,
        grid_size=None,
        sigma=None,
        poke_amplitude=None,
        dtype=cp.float32,
    ):
        """
        Builds a continuous DM surface from actuator commands without storing (n_act,N,N).

        """
        if grid_size is None:
            grid_size = params.get("grid_size")
        if actuators is None:
            actuators = params.get("actuators")
        if poke_amplitude is None:
            poke_amplitude = params.get("poke_amplitude")
        if pupil is None:
            pupil = Pupil_tools.generate_pupil(grid_size=grid_size, dtype=dtype)
        if act_centers is None:
            act_centers = Pupil_tools.generate_actuators(pupil=pupil, actuators=actuators, grid_size=grid_size)

        if sigma is None:
            sigma = (float(grid_size) / float(actuators)) * 0.6

        cmd = cp.asarray(commands, dtype=dtype)
        cmd_grid = cp.zeros((int(grid_size), int(grid_size)), dtype=dtype)

        rr = act_centers[:, 0].astype(cp.int32)
        cc = act_centers[:, 1].astype(cp.int32)

        # accumulate in case of duplicate indices (shouldn't happen, but safe)
        cp.add.at(cmd_grid, (rr, cc), cmd)

        # gaussian blur on GPU
        try:
            import cupyx.scipy.ndimage as ndi
            surf = ndi.gaussian_filter(cmd_grid, sigma=float(sigma), mode="constant")
        except Exception:
            # fallback: sum a few largest commands with explicit gaussians (still avoids n_act cube)
            # This fallback is slower, but keeps correctness if cupyx isn't available.
            surf = cp.zeros_like(cmd_grid)
            # take up to 256 strongest commands
            k = int(min(256, cmd.size))
            idx = cp.argsort(cp.abs(cmd))[-k:]
            for ii in idx.tolist():
                surf += (cmd[ii] / float(poke_amplitude)) * Pupil_tools.actuator_influence_single(
                    act_centers[int(ii)], pupil=cp.ones_like(pupil), actuators=actuators,
                    poke_amplitude=poke_amplitude, sigma=sigma, grid_size=grid_size, dtype=dtype
                )

        surf *= pupil
        return surf

# -----------------------------
# Analysis utilities
# -----------------------------
class Analysis:
    @staticmethod
    def _i0_peak_for_pupil(pupil, pad, science_lambda):
        """
        Calculate peak intensity for a given pupil and wavelength.
        """

        # cache only when pupil matches default params (most common)
        grid_size = int(pupil.shape[0])
        obsc = float(params.get("telescope_center_obscuration"))
        key = (_device_id(), int(getattr(pupil, "data", type("x",(object,),{"ptr":0})()).ptr), grid_size, int(pad), float(science_lambda), pupil.dtype)
        peak = _SCIENCE_I0PEAK.get(key)
        if peak is not None:
            return peak

        # compute unaberrated PSF peak
        field0 = pupil.astype(cp.complex64, copy=False)
        if pad:
            N = grid_size + 2 * int(pad)
            field_p = cp.zeros((N, N), dtype=cp.complex64)
            field_p[int(pad):int(pad)+grid_size, int(pad):int(pad)+grid_size] = field0
        else:
            field_p = field0

        F0 = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(field_p)))
        I0 = F0.real * F0.real + F0.imag * F0.imag
        peak = float(I0.max().item())
        _SCIENCE_I0PEAK[key] = peak
        return peak

    @staticmethod
    def generate_science_image(pupil=None, phase_map=None, science_lambda=None, pad=None):
        """
        Compute science image for a given pupil, phase_map, and wavelength.
        """
        if pad is None:
            pad = params.get("field_padding")
        if science_lambda is None:
            science_lambda = params.get("science_lambda")
        if pupil is None:
            pupil = Pupil_tools.generate_pupil()
        if phase_map is None:
            phase_map = cp.zeros_like(pupil, dtype=cp.float32)

        pupil = pupil.astype(cp.float32, copy=False)
        phase_map = phase_map.astype(cp.float32, copy=False)

        # build complex field
        field = pupil * cp.exp(1j * phase_map)  # complex64

        # manual pad (faster than cp.pad for repeated calls)
        grid_size = int(pupil.shape[0])
        pad = int(pad)
        if pad:
            N = grid_size + 2 * pad
            field_p = cp.zeros((N, N), dtype=cp.complex64)
            field_p[pad:pad+grid_size, pad:pad+grid_size] = field.astype(cp.complex64, copy=False)
        else:
            field_p = field.astype(cp.complex64, copy=False)

        F = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(field_p)))
        I = F.real * F.real + F.imag * F.imag  # float32/float64 depending on backend

        i0_peak = Analysis._i0_peak_for_pupil(pupil, pad, science_lambda)
        strehl = float(I.max().item()) / float(i0_peak)

        return I, strehl

    @staticmethod
    def fwhm_radial(I):
        """
        Compute Full Width Half Max of a given image.
        """
        xp = cp.get_array_module(I)
        shape = tuple(I.shape)
        backend = "cupy" if xp is cp else "numpy"

        cached = _RADIAL_CACHE.get((backend, shape))
        if cached is None:
            ny, nx = shape
            cy, cx = ny // 2, nx // 2
            y, x = xp.indices(shape)
            r = xp.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            r_int = r.ravel().astype(xp.int32)
            max_r = int(r_int.max()) + 1

            radial_cnt = xp.bincount(r_int, minlength=max_r)
            _RADIAL_CACHE[(backend, shape)] = (r_int, radial_cnt)
            cached = (r_int, radial_cnt)

        r_int, radial_cnt = cached

        I_flat = I.ravel()
        max_r = int(radial_cnt.shape[0])

        radial_sum = xp.bincount(r_int, weights=I_flat, minlength=max_r)
        # avoid div-by-zero (shouldn't happen but safe)
        radial_prof = radial_sum / xp.where(radial_cnt == 0, 1, radial_cnt)

        # normalize to peak
        radial_prof = radial_prof / radial_prof[0]

        # find half-maximum
        half = 0.5
        idx = int(xp.where(radial_prof <= half)[0][0])

        # linear interpolation
        r1, r2 = idx - 1, idx
        y1, y2 = radial_prof[r1], radial_prof[r2]
        r_half = r1 + (half - y1) / (y2 - y1)

        return 2.0 * float(r_half)

# -----------------------------
# WFS tools
# -----------------------------
gpu_lock = threading.Lock()

class WFSensor_tools:
    ARCSEC2RAD = cp.pi / (180.0 * 3600.0)
    RAD2ARCSEC = (180.0 * 3600.0) / cp.pi

    class ShackHartmann:
        def __init__(self, gs_range_m=None, n_sub=None, wavelength=None, dx=None, dy=None, noise=0.0, pupil=None, grid_size=None):
            if gs_range_m is None:
                gs_range_m = cp.inf
            if n_sub is None:
                n_sub = params.get("sub_apertures")
            if wavelength is None:
                wavelength = params.get("wfs_lambda")
            if grid_size is None:
                grid_size = params.get("grid_size")
            if pupil is None:
                pupil = Pupil_tools.generate_pupil(grid_size=grid_size)
            if dx is None:
                dx = 0.0
            if dy is None:
                dy = 0.0

            self.n_sub = int(n_sub)
            self.wavelength = float(wavelength)
            self.noise = float(noise)
            self.pupil = pupil
            self.grid_size = int(grid_size)

            self.dx = float(dx) * WFSensor_tools.ARCSEC2RAD
            self.dy = float(dy) * WFSensor_tools.ARCSEC2RAD
            self.field_angle = (self.dx, self.dy)
            self.sen_angle = cp.sqrt(self.field_angle[0] ** 2 + self.field_angle[1] ** 2)
            self.gs_range_m = gs_range_m

            self._build_subap_geometry()

        def recompute(self, n_sub=None, wavelength=None, dx=None, dy=None, noise=0.0, pupil=None, grid_size=None):
            if n_sub is not None:
                self.n_sub = int(n_sub)
            if wavelength is not None:
                self.wavelength = float(wavelength)
            if noise is not None:
                self.noise = float(noise)
            if pupil is not None:
                self.pupil = pupil
            if grid_size is not None:
                self.grid_size = int(grid_size)
            if dx is not None:
                self.dx = float(dx) * WFSensor_tools.ARCSEC2RAD
            if dy is not None:
                self.dy = float(dy) * WFSensor_tools.ARCSEC2RAD

            self.field_angle = (self.dx, self.dy)
            self.sen_angle = cp.sqrt(self.field_angle[0] ** 2 + self.field_angle[1] ** 2)

            self._build_subap_geometry()


        # --- Compatibility with older codebase ---
        def regen_sub_apertures(self):
            """DEPRECATED. Rebuild sub-aperture geometry."""
            self._build_subap_geometry()

        def generate_sub_apertures(self, pupil=None, grid_size=None):
            """DEPRECATED. Returns sensor geometry as tuple."""
            if pupil is not None:
                self.pupil = pupil
            if grid_size is not None:
                self.grid_size = int(grid_size)
            self._build_subap_geometry()
            return (
                self.active_sub_aps,
                self.sub_aps,
                self.sub_aps_idx,
                self.sub_ap_width,
                self.h,
                self.w,
                None,
                None,
                self.ys,
                self.xs,
                getattr(self, "sub_slice", None),
                self.sub_pupils,
            )

        def _build_subap_geometry(self):
            """
            One-time (or recompute-time) geometry build:
            - finds active subaps (CPU call)
            - precomputes GPU index grids for extracting subap windows
            - caches sub-pupil masks per subap
            """
            grid_size = self.grid_size
            pupil = self.pupil

            cache_key = (_device_id(), self.n_sub, grid_size, int(pupil.data.ptr))

            cached = _SUBAP_GEOM_CACHE.get(cache_key)
            if cached is not None:
                self.active_sub_aps = cached["active_sub_aps"]
                self.sub_aps = cached["sub_aps"]
                self.sub_aps_idx = cached["sub_aps_idx"]
                self.sub_ap_width = cached["sub_ap_width"]
                self.h = cached["h"]
                self.w = cached["w"]
                self.ys = cached["ys"]
                self.xs = cached["xs"]
                self.sub_pupils = cached["sub_pupils"]
                self.grid_ny = cached["grid_ny"]
                self.grid_nx = cached["grid_nx"]
                return

            # aotools wfs helper is CPU (expects numpy)
            pupil_np = cp.asnumpy(pupil)
            sub_aps_np = aotools.wfs.findActiveSubaps(self.n_sub, pupil_np, 0.6)

            # Keep a copy of the aotools-returned subap coordinates for plotting/compatibility.
            # Use float64 for index math to avoid float32 rounding pushing values just below integers
            # (which can create duplicated indices -> "missing rows/cols" and occasional misplacements).
            sub_aps64 = cp.asarray(sub_aps_np, dtype=cp.float64)
            sub_aps = sub_aps64.astype(cp.float32, copy=False)
            active_sub_aps = int(sub_aps64.shape[0])

            sub_ap_width = float(grid_size) / float(self.n_sub)  # may be non-integer
            w64 = cp.asarray(sub_ap_width, dtype=cp.float64)

            # Robust index computation:
            # - use floor with a tiny epsilon to protect against 2.999999999 -> 2
            # - clip to [0, n_sub-1] to avoid boundary spill
            eps = cp.asarray(1e-9, dtype=cp.float64)
            sub_aps_idx = cp.floor(sub_aps64 / w64 + eps).astype(cp.int32)
            sub_aps_idx = cp.clip(sub_aps_idx, 0, int(self.n_sub) - 1)

            # Subap extraction window size (kept consistent with the original codebase).
            h = int(sub_ap_width) + 1
            w = int(sub_ap_width) + 1

            yy = cp.arange(h, dtype=cp.int32)[:, None]
            xx = cp.arange(w, dtype=cp.int32)[None, :]

            # Compute pixel top-left corners from indices, rounding to nearest pixel boundary.
            top_left = cp.rint(sub_aps_idx.astype(cp.float64) * w64).astype(cp.int32)

            # Clip to stay in-bounds (python slicing used to clip implicitly; fancy indexing does not).
            top_left_y = cp.clip(top_left[:, 0], 0, grid_size - h)
            top_left_x = cp.clip(top_left[:, 1], 0, grid_size - w)
            top_left = cp.stack((top_left_y, top_left_x), axis=1)

            ys = top_left[:, 0, None, None] + yy[None, :, :]
            xs = top_left[:, 1, None, None] + xx[None, :, :]

            # Legacy: keep a CPU list of slice windows per active subap (used by older visualization code).
            try:
                _iy = cp.asnumpy(top_left[:, 0]).astype(np.int32, copy=False)
                _ix = cp.asnumpy(top_left[:, 1]).astype(np.int32, copy=False)

                self.sub_slice = [(slice(int(y0), int(y0) + h), slice(int(x0), int(x0) + w)) for y0, x0 in zip(_iy, _ix)]

            except Exception:
                self.sub_slice = None

            sub_pupils = pupil[ys, xs].astype(cp.float32, copy=False)

            # Use the full nominal SH grid for stitched sensor-image layout (even if edge rows/cols are inactive).
            grid_ny = int(self.n_sub)
            grid_nx = int(self.n_sub)

            self.active_sub_aps = active_sub_aps
            self.sub_aps = sub_aps
            self.sub_aps_idx = sub_aps_idx
            self.sub_ap_width = sub_ap_width
            self.h = h
            self.w = w
            self.ys = ys
            self.xs = xs
            self.sub_pupils = sub_pupils
            self.grid_ny = grid_ny
            self.grid_nx = grid_nx

            _SUBAP_GEOM_CACHE[cache_key] = {
                "active_sub_aps": active_sub_aps,
                "sub_aps": sub_aps,
                "sub_aps_idx": sub_aps_idx,
                "sub_ap_width": sub_ap_width,
                "h": h,
                "w": w,
                "ys": ys,
                "xs": xs,
                "sub_pupils": sub_pupils,
                "grid_ny": grid_ny,
                "grid_nx": grid_nx,
            }

        def measure(self, pupil=None, phase_map=None, phase_map_alt=None, poke_amplitude=None, pad=None, return_image=True):
            """
            Measure Shack-Hartmann slopes (and optionally a stitched sensor image).
            """
            if pad is None:
                pad = params.get("field_padding")
            if pupil is not None:
                self.pupil = pupil
                self._build_subap_geometry()

            if phase_map is None:
                phase_map = cp.zeros((self.grid_size, self.grid_size), dtype=cp.float32)

            # accept (N,N) or (n_map,N,N)
            if phase_map.ndim == 2:
                phase_map = phase_map[None, :, :]
            phase_map = phase_map.astype(cp.float32, copy=False)

            n_map = int(phase_map.shape[0])
            n_sub = int(self.sub_aps_idx.shape[0])
            pad = int(pad)

            # Extract subap phases directly
            sub_phase = phase_map[:, self.ys, self.xs]  # (n_map, n_sub, h, w)
            sub_phase *= self.sub_pupils[None, :, :, :]                   # zero outside pupil in subap

            scale = cp.asarray(4.0 * cp.pi / float(self.wavelength), dtype=cp.float32)
            sub_phase *= scale

            # complex field per subap
            field = cp.exp(1j * sub_phase).astype(cp.complex64, copy=False)
            field *= self.sub_pupils[None, :, :, :].astype(cp.complex64, copy=False)

            # manual pad
            h, w = self.h, self.w
            h_p, w_p = h + 2 * pad, w + 2 * pad
            field_p = cp.zeros((n_map, n_sub, h_p, w_p), dtype=cp.complex64)
            field_p[:, :, pad:pad+h, pad:pad+w] = field

            # FFT batch on last axes
            F = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(field_p, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
            I = (F.real * F.real + F.imag * F.imag).astype(cp.float32, copy=False)

            # normalize per subap per map
            denom = cp.sum(I, axis=(-2, -1), keepdims=True)
            denom = cp.where(denom == 0, 1.0, denom)
            I = I / denom

            # cached coordinate vectors for centroiding
            ckey = (_device_id(), h_p, w_p, cp.float32)
            coords = _COORD_CACHE.get(ckey)
            if coords is None:
                y_coords = cp.arange(h_p, dtype=cp.float32)
                x_coords = cp.arange(w_p, dtype=cp.float32)
                coords = (y_coords, x_coords)
                _COORD_CACHE[ckey] = coords
            y_coords, x_coords = coords

            # centroid (y,x)
            cx = cp.sum(I * x_coords[None, None, None, :], axis=(-2, -1))
            cy = cp.sum(I * y_coords[None, None, :, None], axis=(-2, -1))
            centroids = cp.stack((cy, cx), axis=-1)  # (n_map, n_sub, 2)

            center = cp.asarray([(h_p - 1) * 0.5, (w_p - 1) * 0.5], dtype=cp.float32)
            slopes = centroids - center[None, None, :]

            sensor_image = None
            if return_image:
                # Cheap(ish) placement loop for display use-cases (n_sub is typically small)
                H = self.grid_ny * h_p
                W = self.grid_nx * w_p
                sensor_image = cp.zeros((n_map, H, W), dtype=I.dtype)
                iy = cp.asnumpy(self.sub_aps_idx[:, 0]).astype(np.int32)
                ix = cp.asnumpy(self.sub_aps_idx[:, 1]).astype(np.int32)

                # python loop is OK here: number of active subaps is limited (tens to hundreds)
                for j in range(n_sub):
                    y0 = int(iy[j]) * h_p
                    x0 = int(ix[j]) * w_p
                    sensor_image[:, y0:y0+h_p, x0:x0+w_p] = I[:, j, :, :]

            return centroids, slopes, sensor_image

    class PyramidWFS:
        # TODO unimplemented
        def __init__(self, n_pixels, wavelength, modulation=1.0):
            self.n_pixels = n_pixels
            self.wavelength = wavelength
            self.modulation = modulation
            self.n_slopes = 2 * n_pixels**2

        def measure(self, phase):
            # TODO
            return

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse
    import inspect
    import sys

    classes = {
        name: cls
        for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
        if not name.startswith("_")
    }

    functions = {}
    for cls_name, cls in classes.items():
        for name, func in inspect.getmembers(cls, inspect.isfunction):
            if func.__module__ == __name__ and not name.startswith("_"):
                functions[f"{cls_name}.{name}"] = func

    parser = argparse.ArgumentParser(description="Run any function from this file")
    parser.add_argument("--list-functions", action="store_true", help="List available functions")
    parser.add_argument("function", nargs="?", choices=functions.keys(), help="Function to run")

    # AO parameters with defaults from config
    parser.add_argument("--telescope_diameter", type=float, default=config.get("telescope_diameter"))
    parser.add_argument("--telescope_center_obscuration", type=float, default=config.get("telescope_center_obscuration"))
    parser.add_argument("--wfs_lambda", type=float, default=config.get("wfs_lambda"))
    parser.add_argument("--science_lambda", type=float, default=config.get("science_lambda"))
    parser.add_argument("--r0", type=float, default=config.get("r0"))
    parser.add_argument("--L0", type=float, default=config.get("L0"))
    parser.add_argument("--Vwind", type=float, default=config.get("Vwind"))
    parser.add_argument("--actuators", type=int, default=config.get("actuators"))
    parser.add_argument("--sub_apertures", type=int, default=config.get("sub_apertures"))
    parser.add_argument("--frame_rate", type=int, default=config.get("frame_rate"))
    parser.add_argument("--grid_size", type=int, default=config.get("grid_size"))
    parser.add_argument("--field_padding", type=int, default=config.get("field_padding"))
    parser.add_argument("--poke_amplitude", type=float, default=config.get("poke_amplitude"))
    parser.add_argument("--random_seed", type=int, default=config.get("random_seed"))
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    parser.set_defaults(use_gpu=config.get("use_gpu", False))
    parser.add_argument("--data_path", type=str, default=config.get("data_path"))

    parser.add_argument("--args", nargs="*", help="Additional key=value arguments for the function")

    args, unknown = parser.parse_known_args()

    if args.list_functions:
        print("Available functions:")
        for name in functions.keys():
            print("-", name)
        sys.exit()

    for key, value in vars(args).items():
        if value is not None and key not in ("function", "args"):
            params[key] = value

    kwargs = {}
    if args.args:
        for item in args.args:
            key, value = item.split("=")
            try:
                value = eval(value)
            except Exception:
                pass
            kwargs[key] = value

    result = functions[args.function](**kwargs)
    print("Result:", result)
