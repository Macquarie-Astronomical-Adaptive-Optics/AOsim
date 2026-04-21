import aotools
import cupy as cp
import numpy as np
from pathlib import Path
import threading
import math
import hashlib
import logging
logger = logging.getLogger(__name__)
from typing import Callable, Optional, Tuple

# -----------------------------
# Config / parameter management
# -----------------------------
class Params(dict):
    """Lightweight params wrapper (dict-compatible).

    Adds small typed helpers while staying fully compatible with existing code.
    """

    def get_int(self, key: str, default: int = 0) -> int:
        try:
            return int(self.get(key, default))
        except Exception:
            return int(default)

    def get_float(self, key: str, default: float = 0.0) -> float:
        try:
            return float(self.get(key, default))
        except Exception:
            return float(default)

    def get_bool(self, key: str, default: bool = False) -> bool:
        try:
            return bool(self.get(key, default))
        except Exception:
            return bool(default)


# Single module-level params object.
# It starts empty and must be populated by main.py via set_params(...).
params = Params()

rootdir = Path(__file__).parent.parent


def set_params(new_params):
    """Update module-level params in place.

    Important:
    - We mutate the existing Params object instead of rebinding it.
    - This keeps any imported aliases (e.g. `from scripts.utilities import params`)
      pointing at the same live object.
    """
    if new_params is None:
        raise ValueError("set_params(new_params): new_params must not be None")

    if isinstance(new_params, Params):
        src = dict(new_params)
    else:
        src = dict(new_params)

    params.clear()
    params.update(src)
    
# import aotools
# import cupy as cp
# import numpy as np
# from pathlib import Path
# import json
# import threading
# import math
# import hashlib
# import logging
# logger = logging.getLogger(__name__)
# from typing import Callable, Optional, Tuple

# # -----------------------------
# # Config / parameter management
# # -----------------------------
# _DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config_ultimate_testbench.json" #"config_ultimate.json"


# try:
#     with open(_DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
#         config = json.load(f)
# except FileNotFoundError:
#     # Allow importing utilities in isolation (e.g., unit tests) without the repo config file.
#     config = {}
# except json.JSONDecodeError as e:
#     raise RuntimeError(f"Failed to parse {_DEFAULT_CONFIG_PATH}: {e}") from e

# class Params(dict):
#     """Lightweight params wrapper (dict-compatible).

#     Adds small typed helpers while staying fully compatible with existing code.
#     """

#     def get_int(self, key: str, default: int = 0) -> int:
#         try:
#             return int(self.get(key, default))
#         except Exception:
#             return int(default)

#     def get_float(self, key: str, default: float = 0.0) -> float:
#         try:
#             return float(self.get(key, default))
#         except Exception:
#             return float(default)

#     def get_bool(self, key: str, default: bool = False) -> bool:
#         try:
#             return bool(self.get(key, default))
#         except Exception:
#             return bool(default)


# params = Params(config)


# rootdir = Path(__file__).parent.parent

# def set_params(new_params):
#     """Update module-level params (used as defaults when explicit args are None)."""
#     global params
#     if isinstance(new_params, Params):
#         params = new_params
#     else:
#         # Accept any mapping/dict-like object.
#         params = Params(dict(new_params))


# -----------------------------
# Small GPU caches (keyed)
# -----------------------------
_XY_CACHE = {}          # (device, grid_size, dtype) -> (y_col, x_row)
_PUPIL_CACHE = {}       # (device, grid_size, obsc, dtype) -> pupil
_SCIENCE_I0PEAK = {}    # (device, grid_size, obsc, pad, science_lambda, dtype) -> float
_SUBAP_GEOM_CACHE = {}  # (device, n_sub, grid_size, obsc) -> dict geometry (for SH)
_CENTROID_WIN_CACHE = {}  # (device, h_p, w_p, win_px, dtype) -> window mask (fftshifted coords)

def _device_id():
    try:
        return int(cp.cuda.runtime.getDevice())
    except Exception:
        return 0

def _array_ptr(a) -> int:
    """Best-effort stable-ish pointer/id for caching.

    For CuPy arrays this is the device pointer; for NumPy arrays it's the Python id.
    """
    try:
        # CuPy: a.data.ptr exists.
        return int(a.data.ptr)  # type: ignore[attr-defined]
    except Exception:
        return int(id(a))


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


def _get_centroid_window_mask(h_p: int, w_p: int, win_px: int, dtype=cp.float32):
    """Return a 2D square centroid window mask in *fftshifted* pixel coordinates.

    The returned mask is centered on (0,0) after fftshift, i.e. its center pixel is
    at (h_p//2, w_p//2). It is intended to be applied to `cp.fft.fftshift(I)`.
    """
    win_px = int(win_px)
    if win_px <= 0:
        return None
    win_px = int(min(win_px, h_p, w_p))
    key = (_device_id(), int(h_p), int(w_p), int(win_px), dtype)
    m = _CENTROID_WIN_CACHE.get(key)
    if m is None:
        hy = h_p // 2
        hx = w_p // 2
        y = cp.arange(h_p, dtype=cp.int32) - hy
        x = cp.arange(w_p, dtype=cp.int32) - hx
        # Define a symmetric square window of size win_px (approx; even sizes include both sides).
        half = win_px // 2
        m2d = (cp.abs(y)[:, None] <= half) & (cp.abs(x)[None, :] <= half)
        m = m2d.astype(dtype)
        _CENTROID_WIN_CACHE[key] = m
    return m


def _is_lgs_range(gs_range_m: object, lgs_max_range_m: float = 1.0e7) -> bool:
    """Heuristic LGS/NGS classifier.

    We classify as LGS if the range is finite and <= lgs_max_range_m (~10,000 km).
    This keeps typical sodium LGS (90 km) as LGS while allowing configurations that
    represent NGS range as a huge finite number.
    """
    try:
        r = float(gs_range_m)
    except Exception:
        return False
    if not math.isfinite(r):
        return False
    return r <= float(lgs_max_range_m)

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
        key = (_device_id(), _array_ptr(pupil), grid_size, int(pad), float(science_lambda), str(pupil.dtype))
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
        """Compute PSF FWHM (in pixels) from the azimuthally-averaged radial profile.
        """
        xp = cp.get_array_module(I)
        I = xp.asarray(I)

        # center on peak
        p = int(xp.argmax(I))
        y0, x0 = xp.unravel_index(p, I.shape)
        I0 = float(I[y0, x0])
        if I0 <= 0:
            return float("nan")

        y, x = xp.indices(I.shape)
        r = xp.sqrt((x - x0) ** 2 + (y - y0) ** 2).astype(xp.float32)

        r0 = xp.floor(r).astype(xp.int32)
        w1 = (r - r0).astype(xp.float32)      # weight to r0+1
        w0 = (1.0 - w1).astype(xp.float32)    # weight to r0

        max_r = int(r0.max()) + 2
        I_flat = I.ravel().astype(xp.float32)
        r0f = r0.ravel()
        w0f = w0.ravel()
        w1f = w1.ravel()

        # weighted sums to bins r0 and r0+1
        sum0 = xp.bincount(r0f, weights=I_flat * w0f, minlength=max_r)
        sum1 = xp.bincount(r0f + 1, weights=I_flat * w1f, minlength=max_r)
        cnt0 = xp.bincount(r0f, weights=w0f, minlength=max_r)
        cnt1 = xp.bincount(r0f + 1, weights=w1f, minlength=max_r)

        radial_sum = sum0 + sum1
        radial_cnt = cnt0 + cnt1
        prof = radial_sum / xp.maximum(radial_cnt, 1e-12)

        # normalize to peak (use I0, not prof[0], to avoid bin artifacts)
        prof = prof / I0

        # light smoothing to avoid ring-induced early crossings
        prof = (prof + xp.roll(prof, 1) + xp.roll(prof, -1)) / 3.0

        # enforce a non-increasing envelope (robust to rings)
        out = prof.copy()
        n = int(out.size)
        stride = 1
        while stride < n:
            shifted = xp.empty_like(out)
            shifted[:stride] = xp.inf
            shifted[stride:] = out[:-stride]
            out = xp.minimum(out, shifted)
            stride <<= 1

        prof_mono = out
        idxs = xp.where(prof_mono <= 0.5)[0]
        if idxs.size == 0:
            return float("nan")
        idx = int(idxs[0])
        if idx == 0:
            return 0.0

        r1, r2 = idx - 1, idx
        y1, y2 = float(prof_mono[r1]), float(prof_mono[r2])
        if y2 == y1:
            r_half = float(r2)
        else:
            r_half = r1 + (0.5 - y1) / (y2 - y1)

        return 2.0 * float(r_half)

    @staticmethod
    def _zoom2d(I, factor, xp):
        """Upsample by 'factor' using bilinear zoom if available, else nearest-neighbor repeat."""
        if factor == 1:
            return I
        try:
            if xp is cp:
                from cupyx.scipy.ndimage import zoom
            else:
                from scipy.ndimage import zoom
            return zoom(I, factor, order=1)  # bilinear
        except Exception:
            f = int(factor)
        return xp.repeat(xp.repeat(I, f, axis=0), f, axis=1)

    def _fwhm_contour_region_pca(
        psf,
        pixelScale,          # e.g. mas/pix (recommended)
        rebin=1,             # upsample factor
        min_pts=30,          # required number of half-max pixels
        bg_percentile=1.0,   # background estimate from low percentile of image (robust for cropped PSFs)
        use_bg=False,
        return_ratio_theta=True,
        return_ellipse: bool = False,
    ):
        """
        Contour FWHM using the half-maximum region.
        Uses upsampling (rebin) to reduce discretization.

        Returns:
        - If return_ratio_theta and not return_ellipse:
            (fwhm_major, fwhm_minor, aRatio, theta_deg)
        - If return_ratio_theta and return_ellipse:
            (fwhm_major, fwhm_minor, aRatio, theta_deg, ellipse)
        - If not return_ratio_theta and not return_ellipse:
            (fwhm_major, fwhm_minor)
        - If not return_ratio_theta and return_ellipse:
            (fwhm_major, fwhm_minor, ellipse)

        Where ellipse is a dict in **native PSF pixels**:
            {"x0_px", "y0_px", "rx_px", "ry_px", "angle_deg"}
        """
        xp = cp.get_array_module(psf)
        I0 = xp.asarray(psf).astype(xp.float32, copy=False)

        # Upsample
        r = int(rebin) if rebin is not None else 1
        r = max(r, 1)
        I = Analysis._zoom2d(I0, r, xp)

        peak = float(xp.max(I).item())
        if peak <= 0:
            if return_ratio_theta:
                out = (float("nan"), float("nan"), float("nan"), float("nan"))
            else:
                out = (float("nan"), float("nan"))
            if return_ellipse:
                out = tuple(out) + (None,)
            return out

        # Background estimate (low percentile is safer than "outer ring" for cropped long-PSFs)
        if use_bg:
            try:
                B = float(xp.percentile(I, bg_percentile).item())
            except Exception:
                # percentile can be slow/unsupported in some builds; fallback to min
                B = float(xp.min(I).item())
            level = B + 0.5 * (peak - B)
            mask = I >= level
            if int(mask.sum()) < min_pts:
                # fallback to absolute half-peak
                level = 0.5 * peak
                mask = I >= level
        else:
            level = 0.5 * peak
            mask = I >= level

        n = int(mask.sum())
        if n < min_pts:
            if return_ratio_theta:
                out = (float("nan"), float("nan"), float("nan"), float("nan"))
            else:
                out = (float("nan"), float("nan"))
            if return_ellipse:
                out = tuple(out) + (None,)
            return out

        ys, xs = xp.nonzero(mask)
        xs = xs.astype(xp.float32)
        ys = ys.astype(xp.float32)

        # center at centroid (more stable than bbox midpoint for asymmetric PSFs)
        cx = float(xp.mean(xs).item())
        cy = float(xp.mean(ys).item())
        dx0 = xs - cx
        dy0 = ys - cy

        # PCA for orientation
        cxx = xp.mean(dx0 * dx0)
        cyy = xp.mean(dy0 * dy0)
        cxy = xp.mean(dx0 * dy0)

        cov = xp.array([[cxx, cxy],
                        [cxy, cyy]], dtype=xp.float32)
        evals, evecs = xp.linalg.eigh(cov)

        # major axis direction = eigenvector with larger eigenvalue
        u = evecs[:, 1]
        v = evecs[:, 0]

        pu = dx0 * u[0] + dy0 * u[1]
        pv = dx0 * v[0] + dy0 * v[1]

        # semi-axes in UPSAMPLED pixels
        a = 0.5 * float((pu.max() - pu.min()).item())
        b = 0.5 * float((pv.max() - pv.min()).item())
        if a < b:
            a, b = b, a
            u = v

        # Convert back to ORIGINAL pixels by dividing by rebin, then to angular units
        a0 = float(a) / float(r)  # semi-major in native pixels
        b0 = float(b) / float(r)  # semi-minor in native pixels
        fwhm_major = (2.0 * a0) * float(pixelScale)
        fwhm_minor = (2.0 * b0) * float(pixelScale)

        ellipse = None
        if return_ellipse:
            ellipse = {
                "x0_px": float(cx) / float(r),
                "y0_px": float(cy) / float(r),
                "rx_px": float(a0),
                "ry_px": float(b0),
                "angle_deg": float((xp.arctan2(u[1], u[0]) * 180.0 / xp.pi).item()),
            }

        if not return_ratio_theta:
            out = (fwhm_major, fwhm_minor)
            if return_ellipse:
                out = tuple(out) + (ellipse,)
            return out

        aRatio = fwhm_major / max(fwhm_minor, 1e-12)
        theta_deg = float((xp.arctan2(u[1], u[0]) * 180.0 / xp.pi).item())
        if return_ellipse and isinstance(ellipse, dict):
            ellipse["angle_deg"] = float(theta_deg)
            return fwhm_major, fwhm_minor, aRatio, theta_deg, ellipse
        return fwhm_major, fwhm_minor, aRatio, theta_deg

    def _find_contour_points_tiptop(image, level):
        """
        Find contour points at a given level using the TIPTOP-style radial interpolation
        from the peak. Returns an (N,2) array of [x, y] points in image pixel coords.
        If no points are found, returns [0] (to match TIPTOP's caller behavior).
        """
        # NOTE: this implementation is intentionally close to TIPTOP's reference code.
        import numpy as _np
        try:
            from scipy import ndimage as _ndimage
        except Exception:
            _ndimage = None

        image = _np.asarray(image, dtype=_np.float32)

        # Peak position
        peak_y, peak_x = _np.unravel_index(_np.argmax(image), image.shape)

        # Above threshold
        above_threshold = image > float(level)

        # Boundary points via dilation
        if _ndimage is None:
            return [0]
        kernel = _np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=bool)
        dilated = _ndimage.binary_dilation(above_threshold, structure=kernel)
        boundary_points = dilated & ~above_threshold

        boundary_y, boundary_x = _np.where(boundary_points)
        if boundary_x.size == 0:
            return [0]

        # Precompute above-threshold coordinates
        above_y, above_x = _np.where(above_threshold)
        if above_x.size == 0:
            return [0]

        contour_points = []
        # For every edge point
        for x, y in zip(boundary_x, boundary_y):
            # Find closest points above threshold
            dists_to_above = _np.sqrt((above_y - y) ** 2 + (above_x - x) ** 2)

            if dists_to_above.size > 4:
                nearest_indices = _np.argpartition(dists_to_above, 4)[:4]
            else:
                nearest_indices = _np.argsort(dists_to_above)[:dists_to_above.size]

            nearest_points = _np.column_stack((above_x[nearest_indices], above_y[nearest_indices]))
            dists_from_peak = _np.sqrt(_np.sum((nearest_points - [peak_x, peak_y]) ** 2, axis=1))

            # Farthest (among nearest) from peak
            farthest_idx = int(_np.argmax(dists_from_peak))
            far_point = nearest_points[farthest_idx]
            far_val = float(image[int(far_point[1]), int(far_point[0])])

            curr_val = float(image[int(y), int(x)])

            # Direction vector from peak to farthest point
            direction = _np.array([far_point[0] - peak_x, far_point[1] - peak_y], dtype=_np.float32)
            direction_norm = float(_np.sqrt(_np.sum(direction ** 2)))
            if direction_norm > 0:
                direction = direction / direction_norm

            # Interpolation on the line
            if curr_val < float(level) < far_val:
                t = (float(level) - curr_val) / (far_val - curr_val)

                curr_dist = float(_np.sqrt((x - peak_x) ** 2 + (y - peak_y) ** 2))
                far_dist = float(_np.sqrt(_np.sum((far_point - [peak_x, peak_y]) ** 2)))

                interp_dist = curr_dist + t * (far_dist - curr_dist)

                px = float(peak_x + direction[0] * interp_dist)
                py = float(peak_y + direction[1] * interp_dist)
                contour_points.append([px, py])

        if not contour_points:
            return [0]

        contour_points = _np.array(contour_points, dtype=_np.float32)

        # Filter outliers (TIPTOP: within 2*std of median distance from peak)
        distances_from_peak = _np.sqrt(_np.sum((contour_points - [peak_x, peak_y]) ** 2, axis=1))
        med_distance = float(_np.median(distances_from_peak))
        std_distance = float(_np.std(distances_from_peak))
        if std_distance > 0:
            mask = _np.abs(distances_from_peak - med_distance) < 2.0 * std_distance
            contour_points = contour_points[mask]

        return contour_points

    def fwhm_contour(
        psf,
        pixelScale,          # e.g. arcsec/pix (recommended)
        rebin=1,             # upsample factor
        min_pts=30,          # kept for backward-compat; not used by TIPTOP method
        bg_percentile=1.0,   # kept for backward-compat; not used by TIPTOP method
        use_bg=False,        # kept for backward-compat; not used by TIPTOP method
        return_ratio_theta=True,
        return_ellipse: bool = False,
    ):
        """
        TIPTOP-style contour FWHM:
          1) Upsample by `rebin`
          2) Find interpolated contour points at half-maximum
          3) Estimate ellipse center from bounding box midpoint
          4) Major/minor radii from max/min distance to center
          5) Orientation from the farthest contour point

        Returns (in angular units):
          - (fwhm_major, fwhm_minor, aRatio, theta_deg)  if return_ratio_theta
          - plus `ellipse` dict if return_ellipse=True

        ellipse dict is in **native PSF pixels**:
          {"x0_px","y0_px","rx_px","ry_px","angle_deg"}
        """
        xp = cp.get_array_module(psf)
        I0 = xp.asarray(psf).astype(xp.float32, copy=False)

        r = int(rebin) if rebin is not None else 1
        r = max(r, 1)
        I_hr = Analysis._zoom2d(I0, r, xp)

        # move to CPU for the TIPTOP contour-point search (Python loop)
        if xp is cp:
            im_hr = cp.asnumpy(I_hr)
        else:
            im_hr = np.asarray(I_hr)

        peak_val = float(np.max(im_hr))
        if not np.isfinite(peak_val) or peak_val <= 0:
            if return_ratio_theta:
                out = (float("nan"), float("nan"), float("nan"), float("nan"))
            else:
                out = (float("nan"), float("nan"))
            if return_ellipse:
                out = tuple(out) + (None,)
            return out

        contour_points = Analysis._find_contour_points_tiptop(im_hr, peak_val / 2.0)

        # TIPTOP fallback behavior: if not enough contour points, fall back to the older region-PCA method
        if not isinstance(contour_points, np.ndarray) or contour_points.ndim != 2 or contour_points.shape[0] < 3:
            return Analysis._fwhm_contour_region_pca(
                psf,
                pixelScale,
                rebin=rebin,
                min_pts=min_pts,
                bg_percentile=bg_percentile,
                use_bg=use_bg,
                return_ratio_theta=return_ratio_theta,
                return_ellipse=return_ellipse,
            )

        xC = contour_points[:, 0]
        yC = contour_points[:, 1]

        # Centering (TIPTOP: bbox midpoint)
        mx = np.array([float(xC.max()), float(yC.max())], dtype=np.float32)
        mn = np.array([float(xC.min()), float(yC.min())], dtype=np.float32)
        cent = (mx + mn) / 2.0

        wx = xC - cent[0]
        wy = yC - cent[1]

        # Radii (native pixels) and in angular units
        rr_hr = np.hypot(wx, wy).astype(np.float32)  # high-res pixels
        # TIPTOP: wr = rr_hr/rebin*pixelScale
        wr = (rr_hr / float(r)) * float(pixelScale)

        # FWHM major/minor (TIPTOP: max/min radius)
        r_major = float(np.max(wr))
        r_minor = float(np.min(wr))
        fwhm_major = 2.0 * r_major
        fwhm_minor = 2.0 * r_minor

        # Orientation from farthest point
        idx = int(np.argmax(wr))
        xm = float(wx[idx])
        ym = float(wy[idx])
        theta_deg = float(np.mean(180.0 * np.arctan2(ym, xm) / np.pi))

        ellipse = None
        if return_ellipse:
            # convert center/radii back to native pixels (divide by rebin)
            rx_px = float(np.max(rr_hr) / float(r))
            ry_px = float(np.min(rr_hr) / float(r))
            ellipse = {
                "x0_px": float(cent[0]) / float(r),
                "y0_px": float(cent[1]) / float(r),
                "rx_px": rx_px,
                "ry_px": ry_px,
                "angle_deg": float(theta_deg),
            }

        if not return_ratio_theta:
            out = (float(fwhm_major), float(fwhm_minor))
            if return_ellipse:
                out = tuple(out) + (ellipse,)
            return out

        aRatio = float(fwhm_major) / max(float(fwhm_minor), 1e-12)
        if return_ellipse:
            return float(fwhm_major), float(fwhm_minor), aRatio, float(theta_deg), ellipse
        return float(fwhm_major), float(fwhm_minor), aRatio, float(theta_deg)

# ---------- Long Exposure PSF ----------
    class PSFRingMean:
        def __init__(self, shape, K, dtype=cp.float32, save_path=rootdir / "output" / "psf", save_label = ""):
            self.K = int(K)
            self.shape = shape
            self.dtype = dtype
            self.buf = cp.zeros((self.K, *shape), dtype=dtype)
            self.sum = cp.zeros(shape, dtype=dtype)
            self.i = 0
            self.n = 0

            self.first_save = False
            self.save_path = save_path / (str(id(self)) + save_label)

        def add(self, I):
            I = cp.asarray(I, dtype=self.buf.dtype)
            if self.n < self.K:
                self.buf[self.i] = I
                self.sum += I
                self.n += 1
            else:
                old = self.buf[self.i].copy()
                self.buf[self.i] = I
                self.sum += I - old

            if not self.first_save and self.n == self.K:
                self.save_frames_to_file()
                self.first_save = True

            self.i = (self.i + 1) % self.K

        def mean(self):
            return self.sum / max(self.n, 1)
        
        def reset(self):
            self.buf = cp.zeros((self.K, * self.shape), dtype=self.dtype)
            self.sum = cp.zeros(self.shape, dtype=self.dtype)
            self.i = 0
            self.n = 0

        def _ordered_indices(self, last=None):
            """Indices into self.buf that yield frames in chronological order (oldest -> newest)."""
            n = self.n
            if n == 0:
                return []
            m = n if last is None else int(min(max(last, 1), n))

            # self.i points to next write position.
            # The oldest frame among the last m frames starts at (self.i - m) modulo K.
            start = (self.i - m) % self.K
            return [(start + k) % self.K for k in range(m)]

        @staticmethod
        def _to_uint8(frame_cp, vmin, vmax):
            """Scale float frame to uint8 [0,255] on GPU, then return numpy."""
            denom = (vmax - vmin)
            if denom <= 0:
                u = cp.zeros(frame_cp.shape, dtype=cp.uint8)
            else:
                g = (frame_cp - vmin) / denom
                g = cp.clip(g, 0.0, 1.0)
                u = (g * 255.0 + 0.5).astype(cp.uint8)
            return cp.asnumpy(u)

        def _display(self, I_cp, log10=True, eps=1e-12):
            I_cp = I_cp.astype(cp.float32, copy=False)
            if log10:
                return cp.log10(cp.maximum(I_cp, eps))
            return I_cp
        
        def pad_to_16(self, u8):
            H, W = u8.shape[:2]
            H2 = (H + 15) // 16 * 16
            W2 = (W + 15) // 16 * 16
            if (H2, W2) == (H, W):
                return u8
            if u8.ndim == 2:
                out = np.zeros((H2, W2), dtype=u8.dtype)
                out[:H, :W] = u8
            else:
                out = np.zeros((H2, W2, u8.shape[2]), dtype=u8.dtype)
                out[:H, :W, :] = u8
            return out
        
        @staticmethod
        def _get_writer(path, fps, codec="libx264"):
            import imageio.v2 as imageio

            path_l = path.lower()
            if path_l.endswith(".gif"):
                return imageio.get_writer(Path(path), mode="I", duration=1.0 / float(fps))
            elif path_l.endswith(".mp4"):
                return imageio.get_writer(Path(path), fps=fps, codec=codec)
            else:
                raise ValueError("Unsupported extension. Use .gif or .mp4")

        def save_frames_to_file(
            self,
            raw_path = None,
            mean_path = None,
            last=None,
            fps=30,
            log10=True,
            norm="global",              # "global" or "per_frame"
            clip_percentiles=(1.0, 99.7),
            eps=1e-12,
            codec="libx264",
            save_type = ".mp4",
            mean_mode="ring",           # "ring" (sliding K) or "cumulative"
            rgb_for_mp4=True,
        ):
            """
            Writes two animations:
            - raw_path: raw frames
            - mean_path: mean frames over time (sliding window by default)

            """
            if raw_path is None:
                raw_path = str(self.save_path)+ "_raw"
            if mean_path is None:
                mean_path = str(self.save_path)+ "_mean"


            idxs = self._ordered_indices(last=last)
            if not idxs:
                raise ValueError("No frames to save (buffer is empty).")

            p_lo, p_hi = clip_percentiles

            def frame_percentiles(F_cp):
                vmin = float(cp.percentile(F_cp, p_lo).item())
                vmax = float(cp.percentile(F_cp, p_hi).item())
                return vmin, vmax

            # ---------- Pass 1: determine scaling if norm="global" ----------
            if norm == "global":
                # RAW scaling
                raw_vmin = +np.inf
                raw_vmax = -np.inf
                for j in idxs:
                    F = self._display(self.buf[j], log10=log10, eps=eps)
                    vmin, vmax = frame_percentiles(F)
                    raw_vmin = min(raw_vmin, vmin)
                    raw_vmax = max(raw_vmax, vmax)

                # MEAN scaling
                mean_vmin = +np.inf
                mean_vmax = -np.inf
                sum_cp = cp.zeros(self.shape, dtype=cp.float32)
                for t, j in enumerate(idxs):
                    sum_cp += self.buf[j].astype(cp.float32, copy=False)
                    if mean_mode == "ring" and t >= self.K:
                        sum_cp -= self.buf[idxs[t - self.K]].astype(cp.float32, copy=False)
                        denom = float(self.K)
                    else:
                        denom = float(t + 1)
                    mean_cp = sum_cp / denom
                    Fm = self._display(mean_cp, log10=log10, eps=eps)
                    vmin, vmax = frame_percentiles(Fm)
                    mean_vmin = min(mean_vmin, vmin)
                    mean_vmax = max(mean_vmax, vmax)
            elif norm == "per_frame":
                raw_vmin = raw_vmax = mean_vmin = mean_vmax = None
            else:
                raise ValueError('norm must be "global" or "per_frame"')

            # ---------- Pass 2: write both animations ----------
            w_raw  = self._get_writer(str(raw_path)  + save_type, fps=fps, codec=codec)
            w_mean = self._get_writer(str(mean_path) + save_type, fps=fps, codec=codec)

            try:
                sum_cp = cp.zeros(self.shape, dtype=cp.float32)

                for t, j in enumerate(idxs):
                    # --- RAW frame ---
                    Fr = self._display(self.buf[j], log10=log10, eps=eps)
                    if norm == "per_frame":
                        rvmin, rvmax = frame_percentiles(Fr)
                    else:
                        rvmin, rvmax = raw_vmin, raw_vmax
                    raw_u8 = self._to_uint8(Fr, rvmin, rvmax)

                    # --- MEAN frame ---
                    sum_cp += self.buf[j].astype(cp.float32, copy=False)
                    if mean_mode == "ring" and t >= self.K:
                        sum_cp -= self.buf[idxs[t - self.K]].astype(cp.float32, copy=False)
                        denom = float(self.K)
                    else:
                        denom = float(t + 1)
                    mean_cp = sum_cp / denom

                    Fm = self._display(mean_cp, log10=log10, eps=eps)
                    if norm == "per_frame":
                        mvmin, mvmax = frame_percentiles(Fm)
                    else:
                        mvmin, mvmax = mean_vmin, mean_vmax
                    mean_u8 = self._to_uint8(Fm, mvmin, mvmax)

                    # MP4 generally expects RGB; GIF is fine with grayscale too
                    if rgb_for_mp4 and raw_path.lower().endswith(".mp4"):
                        raw_u8 = np.stack([raw_u8, raw_u8, raw_u8], axis=-1)
                    if rgb_for_mp4 and mean_path.lower().endswith(".mp4"):
                        mean_u8 = np.stack([mean_u8, mean_u8, mean_u8], axis=-1)

                    w_raw.append_data(self.pad_to_16(raw_u8))
                    w_mean.append_data(self.pad_to_16(mean_u8))
            finally:
                w_raw.close()
                w_mean.close()


# -----------------------------
# WFS tools
# -----------------------------
gpu_lock = threading.Lock()
import cupyx.scipy.ndimage as ndi

class WFSensor_tools:
    ARCSEC2RAD = cp.pi / (180.0 * 3600.0)
    RAD2ARCSEC = (180.0 * 3600.0) / cp.pi

    @staticmethod
    def _to_float_or_inf(x):
        """Convert JSON-y values to a float, defaulting to +inf.

        JSON cannot represent NaN/Inf; configs commonly encode these as
        null/"inf"/"infinity". We interpret any non-finite / non-numeric
        value as +inf.
        """
        if x is None:
            return float("inf")
        if isinstance(x, str):
            s = x.strip().lower()
            if s in {"", "none", "null", "inf", "+inf", "infinity", "+infinity", "nan"}:
                return float("inf")
            try:
                xf = float(s)
            except Exception:
                return float("inf")
            return xf if math.isfinite(xf) else float("inf")
        try:
            xf = float(x)
        except Exception:
            return float("inf")
        return xf if math.isfinite(xf) else float("inf")

    class ShackHartmann:
        """
        Shack–Hartmann WFS with LGS spot-elongation model.

        Key ideas:
        - Geometry (active subaps, ys/xs, sub_pupils) is cached per (device, n_sub, grid_size, pupil_ptr).
        - LGS elongation is applied after forming subap PSF intensities I, before centroiding.
        """
        # small local cache for centroid coordinate vectors
        _coord_cache = {}  # (device, h_p, w_p) -> (y_coords, x_coords)

        def __init__(
            self,
            gs_range_m=np.inf,                 # np.inf for NGS, finite for LGS
            n_sub=None,
            wavelength=None,
            dx=None,
            dy=None,
            pupil=None,
            grid_size=None,
            lenslet_f_m=None,
            pixel_pitch_m=None,
            # --- guide-star / LGS ---
            lgs_thickness_m=0.0,               # sodium thickness (m). 0 disables elongation
            lgs_remove_tt=True,                # remove global TT from LGS slopes
            lgs_launch_offset_px=(0.0, 0.0),   # (off_x_px, off_y_px) relative to pupil center
            lgs_dir_bins=64,                   
            lgs_rad_bins=16,
            lgs_elong_scale=2.0,             
            lgs_half_max=24,                   # cap kernel half-length (pixels)

            # --- dynamic pointing ---
            # If provided, called as: field_angle_fn(t_s, tick, sensor=self) -> (theta_x_rad, theta_y_rad)
            field_angle_fn=None,
        ):
            if n_sub is None:
                n_sub = params.get("sub_apertures")
            if wavelength is None:
                wavelength = params.get("wfs_lambda")
            if grid_size is None:
                grid_size = params.get("grid_size")
            if pupil is None:
                pupil = Pupil_tools.generate_pupil(grid_size=grid_size)
            if lenslet_f_m is None:
                self.lenslet_f_m   = 5e-3    # e.g. 5 mm
            if pixel_pitch_m is None:
                self.pixel_pitch_m = 6.5e-6  # e.g. sCMOS

            if dx is None:
                dx = 0.0
            if dy is None:
                dy = 0.0

            self.n_sub = int(n_sub)
            self.wavelength = float(wavelength)

            self.grid_size = int(grid_size)
            self.pupil = pupil

            # store angles in radians internally
            self.dx = float(dx) * WFSensor_tools.ARCSEC2RAD
            self.dy = float(dy) * WFSensor_tools.ARCSEC2RAD

            # Optional dynamic pointing function.
            self.field_angle_fn = field_angle_fn
            self.field_angle = (self.dx, self.dy)
            self.sen_angle = float(np.sqrt(self.dx * self.dx + self.dy * self.dy))
            

            # LGS
            # NOTE: gs_range_m may come from JSON (null/"inf"/etc.).
            self.gs_range_m = WFSensor_tools._to_float_or_inf(gs_range_m)
            self.is_lgs = not (self.gs_range_m == float("inf"))
            self.lgs_thickness_m = float(lgs_thickness_m) if lgs_thickness_m is not None else 0.0
            self.lgs_remove_tt = bool(lgs_remove_tt)
            if lgs_launch_offset_px is None:
                self.lgs_launch_offset_px = (0.0, 0.0)
            else:
                self.lgs_launch_offset_px = (float(lgs_launch_offset_px[0]), float(lgs_launch_offset_px[1]))
            self.lgs_dir_bins = int(lgs_dir_bins)
            self.lgs_rad_bins = int(lgs_rad_bins)
            self.lgs_elong_scale = float(lgs_elong_scale)
            self.lgs_half_max = int(lgs_half_max)

            # LGS bin caches 
            self._lgs_dirbin_cpu = None
            self._lgs_rbin_cpu = None
            self._lgs_group_order_cpu = None
            self._lgs_group_starts_cpu = None
            self._lgs_group_ends_cpu = None
            self._lgs_group_gid_cpu = None

            self._lgs_group_order_gpu = None
            self._lgs_group_starts_gpu = None
            self._lgs_group_ends_gpu = None
            self._lgs_group_gid_gpu = None

            # kernel cache: (ndir, dirbin, half) -> cp.ndarray (k,k)
            self._lgs_kernel_cache = {}

            self._build_subap_geometry()

        # ------------------------
        # pointings + batching API
        # ------------------------
        def update_field_angle(self, t_s: float, tick: int = 0):
            """Update the current field angle from field_angle_fn (if provided).

            field_angle_fn is expected to return (theta_x_rad, theta_y_rad) in **radians**.
            This method mutates self.dx/self.dy/self.field_angle/self.sen_angle.
            """
            fn = getattr(self, "field_angle_fn", None)
            if fn is None:
                return self.field_angle

            try:
                th = fn(float(t_s), int(tick), self)
            except TypeError:
                # Back-compat: allow callables that only accept (t_s) or (t_s, tick)
                try:
                    th = fn(float(t_s), int(tick))
                except TypeError:
                    th = fn(float(t_s))

            thx = float(th[0])
            thy = float(th[1])
            self.dx = thx
            self.dy = thy
            self.field_angle = (thx, thy)
            self.sen_angle = float(np.sqrt(thx * thx + thy * thy))
            return self.field_angle

        def batch_geometry_key(self):
            """Return an immutable key that identifies the *measurement* geometry.

            Notes:
            - Excludes field angle and gs_range_m; those affect sampling, not the SHWFS measurement geometry.
            - Includes LGS elongation parameters because they can affect the spot shape/centroiding.
            """
            # subap layout can differ if pupil differs; include a stable signature of active indices
            try:
                idx = getattr(self, "sub_aps_idx", None)
                if idx is None:
                    idx_sig = 0
                else:
                    # idx is small (~n_active) so a tuple is OK
                    # convert to python ints for hash stability
                    if hasattr(idx, "get"):
                        idx = idx.get()
                    idx_sig = hash(tuple(int(x) for x in np.asarray(idx).reshape(-1)))
            except Exception:
                idx_sig = 0

            return (
                "ShackHartmann",
                int(getattr(self, "grid_size", -1)),
                int(getattr(self, "n_sub", -1)),
                int(len(getattr(self, "sub_aps_idx", []))),
                # LGS elongation-related
                float(getattr(self, "lgs_thickness_m", 0.0)),
                bool(getattr(self, "lgs_remove_tt", True)),
                tuple(float(x) for x in getattr(self, "lgs_launch_offset_px", (0.0, 0.0))),
                int(getattr(self, "lgs_dir_bins", 0)),
                int(getattr(self, "lgs_rad_bins", 0)),
                float(getattr(self, "lgs_elong_scale", 0.0)),
                int(getattr(self, "lgs_half_max", 0)),
                # optics
                float(getattr(self, "lenslet_f_m", 0.0)),
                float(getattr(self, "pixel_pitch_m", 0.0)),
                # active layout signature
                int(idx_sig),
            )

        def recompute(
            self,
            n_sub=None,
            wavelength=None,
            dx=None,
            dy=None,
            pupil=None,
            grid_size=None,
            gs_range_m=None,
            lgs_thickness_m=None,
            lgs_remove_tt=None,
            lgs_launch_offset_px=None,
            lgs_dir_bins=None,
            lgs_rad_bins=None,
            lgs_elong_scale=None,
            lgs_half_max=None,
            field_angle_fn=None,
            **kwargs
        ):
            if n_sub is not None:
                self.n_sub = int(n_sub)
            if wavelength is not None:
                self.wavelength = float(wavelength)
            if grid_size is not None:
                self.grid_size = int(grid_size)
            if pupil is not None:
                self.pupil = pupil

            if dx is not None:
                self.dx = float(dx) * WFSensor_tools.ARCSEC2RAD
            if dy is not None:
                self.dy = float(dy) * WFSensor_tools.ARCSEC2RAD

            if field_angle_fn is not None:
                self.field_angle_fn = field_angle_fn
            self.field_angle = (self.dx, self.dy)
            self.sen_angle = float(np.sqrt(self.dx * self.dx + self.dy * self.dy))

            if gs_range_m is not None:
                self.gs_range_m = WFSensor_tools._to_float_or_inf(gs_range_m)
            if lgs_thickness_m is not None:
                self.lgs_thickness_m = float(lgs_thickness_m)
            if lgs_remove_tt is not None:
                self.lgs_remove_tt = bool(lgs_remove_tt)
            if lgs_launch_offset_px is not None:
                self.lgs_launch_offset_px = (float(lgs_launch_offset_px[0]), float(lgs_launch_offset_px[1]))
            if lgs_dir_bins is not None:
                self.lgs_dir_bins = int(lgs_dir_bins)
            if lgs_rad_bins is not None:
                self.lgs_rad_bins = int(lgs_rad_bins)
            if lgs_elong_scale is not None:
                self.lgs_elong_scale = float(lgs_elong_scale)
            if lgs_half_max is not None:
                self.lgs_half_max = int(lgs_half_max)

            # invalidate LGS GPU group caches (they depend on bins)
            self._lgs_group_order_gpu = None
            self._lgs_group_starts_gpu = None
            self._lgs_group_ends_gpu = None
            self._lgs_group_gid_gpu = None

            # kernel cache depends on ndir too
            self._lgs_kernel_cache = {}

            self._build_subap_geometry()
        def _device_id(self):
            try:
                return int(cp.cuda.runtime.getDevice())
            except Exception:
                return 0

        def _compute_lgs_bins_and_groups_from_top_left(self, top_left_y_cpu, top_left_x_cpu, h, w):
            if top_left_y_cpu is None or top_left_x_cpu is None:
                self._lgs_dirbin_cpu = None
                self._lgs_rbin_cpu = None
                self._lgs_group_order_cpu = None
                self._lgs_group_starts_cpu = None
                self._lgs_group_ends_cpu = None
                self._lgs_group_gid_cpu = None
                return

            nsub = int(top_left_y_cpu.shape[0])
            ndir = int(max(1, self.lgs_dir_bins))
            nrad = int(max(1, self.lgs_rad_bins))

            # subap centers
            cx = top_left_x_cpu.astype(np.float32, copy=False) + 0.5 * float(w)
            cy = top_left_y_cpu.astype(np.float32, copy=False) + 0.5 * float(h)

            pc = (float(self.grid_size) - 1.0) * 0.5
            lx = pc + float(self.lgs_launch_offset_px[0])
            ly = pc - float(self.lgs_launch_offset_px[1]) # original is y-up, but calculations are in y-down so negate

            vx = cx - lx
            vy = cy - ly

            ang = (np.arctan2(vy, vx) + 2.0 * np.pi) % (2.0 * np.pi)  # [0,2pi)
            dirbin = np.floor(ang * (ndir / (2.0 * np.pi))).astype(np.int16, copy=False)
            dirbin = np.clip(dirbin, 0, ndir - 1)

            r = np.sqrt(vx * vx + vy * vy)
            rnorm = r / (0.5 * float(self.grid_size))
            rbin = np.floor(rnorm * nrad).astype(np.int16, copy=False)
            rbin = np.clip(rbin, 0, nrad - 1)

            self._lgs_dirbin_cpu = dirbin
            self._lgs_rbin_cpu = rbin

            # group id per subap
            gid = dirbin.astype(np.int32) + ndir * rbin.astype(np.int32)  # (nsub,)
            order = np.argsort(gid, kind="mergesort").astype(np.int32, copy=False)
            gid_sorted = gid[order]

            if nsub == 0:
                self._lgs_group_order_cpu = order
                self._lgs_group_starts_cpu = np.zeros((0,), dtype=np.int32)
                self._lgs_group_ends_cpu = np.zeros((0,), dtype=np.int32)
                self._lgs_group_gid_cpu = np.zeros((0,), dtype=np.int32)
                return

            cuts = np.flatnonzero(np.diff(gid_sorted)) + 1
            starts = np.r_[0, cuts].astype(np.int32, copy=False)
            ends = np.r_[cuts, gid_sorted.size].astype(np.int32, copy=False)
            uniq_gid = gid_sorted[starts].astype(np.int32, copy=False)

            self._lgs_group_order_cpu = order
            self._lgs_group_starts_cpu = starts
            self._lgs_group_ends_cpu = ends
            self._lgs_group_gid_cpu = uniq_gid

            # invalidate GPU copies
            self._lgs_group_order_gpu = None
            self._lgs_group_starts_gpu = None
            self._lgs_group_ends_gpu = None
            self._lgs_group_gid_gpu = None

        def _ensure_lgs_groups_gpu(self):
            if self._lgs_group_order_cpu is None:
                return
            if self._lgs_group_order_gpu is not None:
                return

            self._lgs_group_order_gpu = cp.asarray(self._lgs_group_order_cpu, dtype=cp.int32)
            self._lgs_group_starts_gpu = cp.asarray(self._lgs_group_starts_cpu, dtype=cp.int32)
            self._lgs_group_ends_gpu = cp.asarray(self._lgs_group_ends_cpu, dtype=cp.int32)
            self._lgs_group_gid_gpu = cp.asarray(self._lgs_group_gid_cpu, dtype=cp.int32)

        def _get_line_kernel(self, dir_i: int, half: int):
            """
            Return normalized 2D line kernel of size (2*half+1) oriented by dir_i/ndir.
            Cached on (ndir, dir_i, half).
            """
            ndir = int(max(1, self.lgs_dir_bins))
            half = int(half)
            key = (ndir, int(dir_i), half)
            ker = self._lgs_kernel_cache.get(key, None)
            if ker is not None:
                return ker

            k = 2 * half + 1
            if k <= 1:
                ker = cp.asarray([[1.0]], dtype=cp.float32)
                self._lgs_kernel_cache[key] = ker
                return ker

            theta = ( (dir_i + 0.5) * (2.0 * np.pi / float(ndir)) )
            c0 = float(half)

            arr = np.zeros((k, k), dtype=np.float32)
            for t in range(-half, half + 1):
                x = int(round(c0 + float(t) * np.cos(theta)))
                y = int(round(c0 + float(t) * np.sin(theta)))
                if 0 <= x < k and 0 <= y < k:
                    arr[y, x] += 1.0

            s = float(arr.sum())
            if s <= 0:
                arr[half, half] = 1.0
                s = 1.0
            arr /= s

            ker = cp.asarray(arr, dtype=cp.float32)
            self._lgs_kernel_cache[key] = ker
            return ker

        def _ensure_lgs_elong_plan(self):
            # Quick reject
            if (not _is_lgs_range(getattr(self, "gs_range_m", float("inf")))) or float(self.lgs_thickness_m) <= 0.0:
                self._lgs_elong_plan = None
                self._lgs_elong_plan_key = None
                return

            self._ensure_lgs_groups_gpu()
            if self._lgs_group_gid_gpu is None or int(self._lgs_group_gid_gpu.size) == 0:
                self._lgs_elong_plan = None
                self._lgs_elong_plan_key = None
                return

            ndir = int(max(1, self.lgs_dir_bins))
            nrad = int(max(1, self.lgs_rad_bins))

            # Key: rebuild plan only if these change
            D = float(params.get("telescope_diameter"))
            key = (
                self._device_id(),
                ndir, nrad,
                float(self.gs_range_m),
                float(self.lgs_thickness_m),
                float(self.lgs_elong_scale),
                int(self.lgs_half_max),
                float(self.lenslet_f_m),
                float(self.pixel_pitch_m),
                D,
            )
            if getattr(self, "_lgs_elong_plan_key", None) == key and getattr(self, "_lgs_elong_plan", None) is not None:
                return

            self._lgs_elong_plan_key = key

            if getattr(self, "_lgs_group_starts_cpu", None) is None or getattr(self, "_lgs_group_key_for_cpu", None) != (self._device_id(), ndir, nrad):
                self._lgs_group_starts_cpu = cp.asnumpy(self._lgs_group_starts_gpu).astype("int32", copy=False)
                self._lgs_group_ends_cpu   = cp.asnumpy(self._lgs_group_ends_gpu).astype("int32", copy=False)
                self._lgs_group_gids_cpu   = cp.asnumpy(self._lgs_group_gid_gpu).astype("int32", copy=False)
                self._lgs_group_key_for_cpu = (self._device_id(), ndir, nrad)

            self._lgs_group_order_gpu = cp.asarray(self._lgs_group_order_cpu, dtype=cp.int32)

            inv = cp.empty_like(self._lgs_group_order_gpu)
            inv[self._lgs_group_order_gpu] = cp.arange(self._lgs_group_order_gpu.size, dtype=cp.int32)
            self._lgs_group_inv_order_gpu = inv

            order = self._lgs_group_order_gpu  # cp.int32, permutation of subaps

            # Precompute half per radial bin (CPU, once)
            H = float(self.gs_range_m)
            thick = float(self.lgs_thickness_m)
            px_per_rad = float(self.lenslet_f_m) / float(self.pixel_pitch_m)
            scale_elong = float(self.lgs_elong_scale)
            half_max = int(self.lgs_half_max)

            # rb -> half
            half_by_rb = [0] * nrad
            H2 = H * H + 1e-30
            for rb in range(nrad):
                r_mean = (rb + 0.5) / float(nrad)
                rho_m = r_mean * 0.5 * D
                dtheta = (rho_m * thick) / H2
                L_px = dtheta * px_per_rad
                half = int(round(0.5 * L_px) * scale_elong) 
                if half < 0: half = 0
                if half > half_max: half = half_max
                half_by_rb[rb] = half

            # Kernel cache (GPU), store expanded weights (1,1,k,k)
            if not hasattr(self, "_lgs_kernel_cache_gpu"):
                self._lgs_kernel_cache_gpu = {}

            # Build selections grouped by (db, half) to reduce convolve calls
            sels_by_key = {}  # (db,half) -> list of cp arrays (views)
            gids = self._lgs_group_gids_cpu
            starts = self._lgs_group_starts_cpu
            ends = self._lgs_group_ends_cpu

            # Final plan: list of (sel_gpu, wts_gpu)
            segs_by_key = {}  # (db,half) -> list[(s,e)] in ORDERED space

            for gi in range(gids.size):
                gid = int(gids[gi])
                rb = gid // ndir
                db = gid - rb * ndir
                half = half_by_rb[rb]
                if half == 0:
                    continue
                s = int(starts[gi]); e = int(ends[gi])
                if e <= s:
                    continue

                segs_by_key.setdefault((db, half), []).append((s, e))

            plan = []
            for (db, half), segs in segs_by_key.items():
                wts = self._lgs_kernel_cache_gpu.get((db, half))
                if wts is None:
                    ker2 = cp.asarray(self._get_line_kernel(db, half), dtype=cp.float32)
                    wts = ker2[None, None, :, :]
                    self._lgs_kernel_cache_gpu[(db, half)] = wts
                plan.append((segs, wts))

            self._lgs_elong_plan = plan


        def _build_subap_geometry(self):
            """
            Build/cached:
            - active_sub_aps, sub_aps, sub_aps_idx
            - sub_ap_width, h,w
            - ys,xs and sub_pupils (GPU)
            - CPU top_left_y/x (for LGS bins/groups)
            """
            grid_size = int(self.grid_size)
            pupil = self.pupil
            cache_key = (self._device_id(), int(self.n_sub), grid_size, pupil.data.ptr)

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
                self.sub_slice = cached.get("sub_slice", None)

                # southwell helpers
                self._southwell_nsub = cached.get("_southwell_nsub", int(self.n_sub))
                self._southwell_pitch = cached.get("_southwell_pitch", float(self.grid_size) / float(self.n_sub))
                self._southwell_rr = cached.get("_southwell_rr", None)
                self._southwell_cc = cached.get("_southwell_cc", None)
                self._southwell_wsum_inv = cached.get("_southwell_wsum_inv", None)
                # Optional neighbor-LUT for O(n_active) Southwell slopes
                self._southwell_left = cached.get("_southwell_left", None)
                self._southwell_right = cached.get("_southwell_right", None)
                self._southwell_up = cached.get("_southwell_up", None)
                self._southwell_down = cached.get("_southwell_down", None)

                # recompute LGS bins/groups from cached top_left
                top_left_y_cpu = cached.get("top_left_y_cpu", None)
                top_left_x_cpu = cached.get("top_left_x_cpu", None)
                self._compute_lgs_bins_and_groups_from_top_left(top_left_y_cpu, top_left_x_cpu, self.h, self.w)
                return

            pupil_np = cp.asnumpy(pupil)
            sub_aps_np = aotools.wfs.findActiveSubaps(self.n_sub, pupil_np, 0.6)

            sub_aps64 = cp.asarray(sub_aps_np, dtype=cp.float64)
            sub_aps = sub_aps64.astype(cp.float32, copy=False)
            active_sub_aps = int(sub_aps64.shape[0])

            sub_ap_width = float(grid_size) / float(self.n_sub)
            w64 = cp.asarray(sub_ap_width, dtype=cp.float64)

            eps = cp.asarray(1e-9, dtype=cp.float64)
            sub_aps_idx = cp.floor(sub_aps64 / w64 + eps).astype(cp.int32)
            sub_aps_idx = cp.clip(sub_aps_idx, 0, int(self.n_sub) - 1)

            h = int(sub_ap_width) + 1
            w = int(sub_ap_width) + 1

            yy = cp.arange(h, dtype=cp.int32)[:, None]
            xx = cp.arange(w, dtype=cp.int32)[None, :]

            top_left = cp.rint(sub_aps_idx.astype(cp.float64) * w64).astype(cp.int32)
            top_left_y = cp.clip(top_left[:, 0], 0, grid_size - h)
            top_left_x = cp.clip(top_left[:, 1], 0, grid_size - w)
            top_left = cp.stack((top_left_y, top_left_x), axis=1)

            ys = top_left[:, 0, None, None] + yy[None, :, :]
            xs = top_left[:, 1, None, None] + xx[None, :, :]

            # legacy CPU list-of-slices
            try:
                _iy = cp.asnumpy(top_left[:, 0]).astype(np.int32, copy=False)
                _ix = cp.asnumpy(top_left[:, 1]).astype(np.int32, copy=False)
                sub_slice = [(slice(int(y0), int(y0) + h), slice(int(x0), int(x0) + w)) for y0, x0 in zip(_iy, _ix)]
            except Exception:
                sub_slice = None
                _iy = _ix = None

            sub_pupils = pupil[ys, xs].astype(cp.float32, copy=False)

            # --- helpers for fast southwell slopes (cached; avoids per-call cp.max syncs) ---
            nsub = int(self.n_sub)
            idx_flat = sub_aps_idx[:, 0] * nsub + sub_aps_idx[:, 1]
            rr = (idx_flat // nsub).astype(cp.int32, copy=False)
            cc = (idx_flat - rr * nsub).astype(cp.int32, copy=False)

            wsum = cp.sum(sub_pupils, axis=(1, 2)).astype(cp.float32, copy=False)
            wsum_inv = (1.0 / (wsum + cp.float32(1e-12))).astype(cp.float32, copy=False)

            # full nominal grid for stitching
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
            self.sub_slice = sub_slice

            self._southwell_nsub = nsub
            self._southwell_pitch = float(grid_size) / float(nsub)
            self._southwell_rr = rr
            self._southwell_cc = cc
            self._southwell_wsum_inv = wsum_inv

            # ---- neighbor LUT for fast Southwell slopes (O(n_active)) ----
            # Build on CPU (n_active is modest) then upload to GPU.
            try:
                rr_cpu = cp.asnumpy(rr).astype(np.int32, copy=False)
                cc_cpu = cp.asnumpy(cc).astype(np.int32, copy=False)
                idx_of = {(int(r0), int(c0)): k for k, (r0, c0) in enumerate(zip(rr_cpu, cc_cpu))}

                left = np.full((active_sub_aps,), -1, dtype=np.int32)
                right = np.full((active_sub_aps,), -1, dtype=np.int32)
                up = np.full((active_sub_aps,), -1, dtype=np.int32)
                down = np.full((active_sub_aps,), -1, dtype=np.int32)

                for k, (r0, c0) in enumerate(zip(rr_cpu, cc_cpu)):
                    left[k] = idx_of.get((int(r0), int(c0) - 1), -1)
                    right[k] = idx_of.get((int(r0), int(c0) + 1), -1)
                    up[k] = idx_of.get((int(r0) - 1, int(c0)), -1)
                    down[k] = idx_of.get((int(r0) + 1, int(c0)), -1)

                self._southwell_left = cp.asarray(left, dtype=cp.int32)
                self._southwell_right = cp.asarray(right, dtype=cp.int32)
                self._southwell_up = cp.asarray(up, dtype=cp.int32)
                self._southwell_down = cp.asarray(down, dtype=cp.int32)
            except Exception:
                # LUT not critical; fall back to dense (nsub,nsub) approach
                self._southwell_left = None
                self._southwell_right = None
                self._southwell_up = None
                self._southwell_down = None

            # LGS bins/groups based on CPU top_left arrays
            if _iy is not None and _ix is not None:
                self._compute_lgs_bins_and_groups_from_top_left(_iy, _ix, h, w)
            else:
                self._compute_lgs_bins_and_groups_from_top_left(None, None, h, w)

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
                "sub_slice": sub_slice,
                "top_left_y_cpu": _iy,
                "top_left_x_cpu": _ix,
                "_southwell_nsub": self._southwell_nsub,
                "_southwell_pitch": self._southwell_pitch,
                "_southwell_rr": rr,
                "_southwell_cc": cc,
                "_southwell_wsum_inv": wsum_inv,
                "_southwell_left": getattr(self, "_southwell_left", None),
                "_southwell_right": getattr(self, "_southwell_right", None),
                "_southwell_up": getattr(self, "_southwell_up", None),
                "_southwell_down": getattr(self, "_southwell_down", None),
            }

        def _ensure_freq_coords(self, h_p, w_p):
            key = (self._device_id(), h_p, w_p)
            item = self._coord_cache.get(key)
            if item is not None:
                return item  # (yrel, xrel)

            # signed pixel coords centered at 0 using fftfreq (no fftshift needed)
            yrel = (cp.fft.fftfreq(h_p).astype(cp.float32) * cp.float32(h_p))
            xrel = (cp.fft.fftfreq(w_p).astype(cp.float32) * cp.float32(w_p))
            self._coord_cache[key] = (yrel, xrel)
            return yrel, xrel


        def measure(
            self,
            pupil=None,
            phase_map=None,
            pad=None,
            return_image=True,
            assume_radians: bool | None = None,
            method: str = "fft",
        ):
            """
            Returns:
            centroids: (n_map, n_sub_active, 2) in subap-padded pixel coords
            slopes:    (n_map, n_sub_active, 2) relative to subap center
            sensor_image: stitched mosaic (n_map, n_sub*n_p, n_sub*n_p) if return_image else None
            """
            if pad is None:
                pad = int(4)

            if pupil is not None:
                self.pupil = pupil
                self._build_subap_geometry()

            if phase_map is None:
                phase_map = cp.zeros((self.grid_size, self.grid_size), dtype=cp.float32)

            if phase_map.ndim == 2:
                phase_map = phase_map[None, :, :]

            # Keep phase_map as a lightweight view; do *not* copy the full (N,N) array.
            # Instead, apply wavelength scaling on the gathered subaperture phases.
            phase_map = cp.asarray(phase_map, dtype=cp.float32)
            n_map = int(phase_map.shape[0])

            # (n_map, n_active, h, w)
            sub_phase = phase_map[:, self.ys, self.xs].astype(cp.float32, copy=False)

            # If the turbulence phase is defined at the 500nm reference, scale to sensor wavelength.
            # (This matches the previous behaviour: phase_map *= 500e-9/self.wavelength)
            sub_phase *= cp.float32(500e-9 / float(self.wavelength))
            sub_phase *= self.sub_pupils[None, :, :, :]
            # radians (WFS wavelength)
            # NOTE: cp.max(...) forces a device sync. Avoid it by passing assume_radians explicitly.
            if assume_radians is None:
                # Preserve previous heuristic but run it on the smaller sub_phase buffer.
                if float(cp.max(sub_phase).item()) == float(params.get("poke_amplitude")):
                    scale = cp.asarray(4.0 * cp.pi / float(self.wavelength), dtype=cp.float32)
                    sub_phase *= scale
            elif not assume_radians:
                scale = cp.asarray(4.0 * cp.pi / float(self.wavelength), dtype=cp.float32)
                sub_phase *= scale

            method = str(method).lower().strip()

            if method in ("southwell", "southwell_fast"):
                def _apply_southwell_sensor_noise(slopes_in: cp.ndarray) -> cp.ndarray:
                    """Approximate WFS photon/read noise for the Southwell (phase-difference) slope model.

                    Southwell does not form an intensity image, so we cannot apply Poisson/read noise on pixels
                    directly. Instead we add an *equivalent* Gaussian noise to the measured slopes.

                    The scaling uses the same config keys as the FFT centroiding path:
                      - wfs_photons_per_subap
                      - wfs_read_noise_e
                      - wfs_centroid_window_px (used only to size the read-noise moment window)

                    If wfs_photons_per_subap <= 0, noise injection is disabled for Southwell (no photon budget).
                    """
                    try:
                        photons_per_subap = float(params.get("wfs_photons_per_subap", 0) or 0)
                    except Exception:
                        photons_per_subap = 0.0
                    if photons_per_subap <= 0:
                        return slopes_in

                    try:
                        read_noise_e = float(params.get("wfs_read_noise_e", 0.0) or 0.0)
                    except Exception:
                        read_noise_e = 0.0
                    try:
                        win_px = int(params.get("wfs_centroid_window_px", 0) or 0)
                    except Exception:
                        win_px = 0

                    # FFT size used by the full SH model (also sets the tilt->centroid scaling in bins)
                    h_loc = int(self.h)
                    w_loc = int(self.w)
                    h_p_loc = int(h_loc + 2 * int(pad))
                    w_p_loc = int(w_loc + 2 * int(pad))
                    N_loc = float(max(h_p_loc, w_p_loc))

                    # Estimate focal-plane spot RMS (in "centroid bins") from the diffraction-limited scaling:
                    # spot width ~ N / D_subap. Here D_subap ~ h_loc (pupil pixels).
                    # This is a lightweight approximation intended for fast Southwell runs.
                    D_eff = max(1.0, float(min(h_loc, w_loc)))
                    sigma_spot = max(0.5, 0.30 * (N_loc / D_eff))

                    # Photon term: Var(centroid) ≈ sigma_spot^2 / Nph
                    var_c = (sigma_spot * sigma_spot) / max(photons_per_subap, 1.0)

                    # Read-noise term (CoM approximation): Var(centroid) scales like
                    # (sigma_r^2 / Nph^2) * Sum(x^2) over the moment window.
                    if read_noise_e > 0.0:
                        W = int(win_px) if (win_px and win_px > 0) else int(max(h_p_loc, w_p_loc))
                        # Sum(x^2) for a square window: N_pix * rms_coord^2
                        N_pix = float(W * W)
                        rms_coord = float(W) / math.sqrt(12.0)
                        var_c += (read_noise_e * read_noise_e) * N_pix * (rms_coord * rms_coord) / (
                            max(photons_per_subap, 1.0) ** 2
                        )

                    sigma_c = math.sqrt(max(0.0, var_c))

                    # Convert centroid-bin noise to phase-slope noise (rad per pupil pixel).
                    # For a DFT of size N, a phase ramp slope (rad/pix) produces a centroid shift of
                    # approximately slope * N / (2π). => slope_noise ≈ (2π/N) * centroid_noise.
                    sigma_slope = (2.0 * math.pi / max(N_loc, 1.0)) * sigma_c
                    if sigma_slope <= 0.0:
                        return slopes_in

                    return slopes_in + cp.random.normal(
                        loc=0.0,
                        scale=sigma_slope,
                        size=slopes_in.shape,
                    ).astype(cp.float32, copy=False)

                # weighted mean phase at each active lenslet center
                wsum_inv = getattr(self, "_southwell_wsum_inv", None)
                if wsum_inv is None:
                    wsum = cp.sum(self.sub_pupils, axis=(1, 2)).astype(cp.float32, copy=False)
                    wsum_inv = (1.0 / (wsum + cp.float32(1e-12))).astype(cp.float32, copy=False)

                phi_act = cp.sum(sub_phase, axis=(2, 3)) * wsum_inv[None, :]  # (n_map,n_active)

                # active lenslet indices (row,col) on the lenslet grid
                nsub = int(getattr(self, "_southwell_nsub", int(self.n_sub)))
                rr = getattr(self, "_southwell_rr", None)
                cc = getattr(self, "_southwell_cc", None)
                if rr is None or cc is None:
                    idx2 = cp.asarray(self.sub_aps_idx)
                    rr = idx2[:, 0].astype(cp.int32, copy=False)
                    cc = idx2[:, 1].astype(cp.int32, copy=False)

                pitch = float(getattr(self, "_southwell_pitch", float(self.grid_size) / float(nsub)))

                # ---- FAST PATH: neighbor LUT on active lenslets (O(n_active)) ----
                # If the neighbor index LUT is available, avoid building the dense (nsub,nsub)
                # grid and expensive roll/isfinite operations.
                left = getattr(self, "_southwell_left", None)
                right = getattr(self, "_southwell_right", None)
                up = getattr(self, "_southwell_up", None)
                down = getattr(self, "_southwell_down", None)

                if left is not None and right is not None and up is not None and down is not None:
                    n_active = int(phi_act.shape[1])
                    # clip indices for safe gather; masks keep invalid entries from affecting result
                    left_c = cp.clip(left, 0, n_active - 1)
                    right_c = cp.clip(right, 0, n_active - 1)
                    up_c = cp.clip(up, 0, n_active - 1)
                    down_c = cp.clip(down, 0, n_active - 1)

                    phi_cur = phi_act
                    phi_l = phi_act[:, left_c]
                    phi_r = phi_act[:, right_c]
                    phi_u = phi_act[:, up_c]
                    phi_d = phi_act[:, down_c]

                    v_l = (left >= 0)[None, :]
                    v_r = (right >= 0)[None, :]
                    v_u = (up >= 0)[None, :]
                    v_d = (down >= 0)[None, :]

                    both_lr = v_l & v_r
                    only_r = v_r & ~v_l
                    only_l = v_l & ~v_r

                    both_ud = v_u & v_d
                    only_d = v_d & ~v_u
                    only_u = v_u & ~v_d

                    inv2p = cp.float32(1.0 / (2.0 * pitch))
                    invp = cp.float32(1.0 / pitch)

                    sx_act = cp.where(both_lr, (phi_r - phi_l) * inv2p,
                                      cp.where(only_r, (phi_r - phi_cur) * invp,
                                               cp.where(only_l, (phi_cur - phi_l) * invp, cp.float32(0.0))))

                    sy_act = cp.where(both_ud, (phi_d - phi_u) * inv2p,
                                      cp.where(only_d, (phi_d - phi_cur) * invp,
                                               cp.where(only_u, (phi_cur - phi_u) * invp, cp.float32(0.0))))

                    slopes = cp.stack([sy_act, sx_act], axis=-1)  # (n_map,n_active,2)

                    # Match FFT path behaviour for LGS: remove global TT if requested.
                    if _is_lgs_range(getattr(self, "gs_range_m", float("inf"))) and bool(self.lgs_remove_tt):
                        slopes = slopes - cp.mean(slopes, axis=1, keepdims=True)

                    # Optional approximate sensor noise for Southwell slopes.
                    slopes = _apply_southwell_sensor_noise(slopes)

                    centroids = cp.zeros_like(slopes)
                    return centroids, slopes, None

                # build full lenslet-center phase grid (NaN where inactive)
                phi_g = cp.full((n_map, nsub, nsub), cp.nan, dtype=cp.float32)
                phi_g[:, rr, cc] = phi_act

                finite = cp.isfinite(phi_g)

                # x neighbors
                cur = phi_g
                left = cp.roll(phi_g, 1, axis=2)
                right = cp.roll(phi_g, -1, axis=2)
                left_ok = finite & cp.roll(finite, 1, axis=2)
                right_ok = finite & cp.roll(finite, -1, axis=2)

                c_idx = cp.arange(nsub, dtype=cp.int32)[None, None, :]
                left_ok &= (c_idx != 0)                 # prevent wrap
                right_ok &= (c_idx != (nsub - 1))

                both = left_ok & right_ok
                only_r = right_ok & ~left_ok
                only_l = left_ok & ~right_ok

                sx = cp.zeros_like(phi_g, dtype=cp.float32)
                sx = cp.where(both, (right - left) / (cp.float32(2.0 * pitch)), sx)
                sx = cp.where(only_r, (right - cur) / (cp.float32(pitch)), sx)
                sx = cp.where(only_l, (cur - left) / (cp.float32(pitch)), sx)

                # y neighbors
                up = cp.roll(phi_g, 1, axis=1)          # (r-1)
                down = cp.roll(phi_g, -1, axis=1)       # (r+1)
                up_ok = finite & cp.roll(finite, 1, axis=1)
                down_ok = finite & cp.roll(finite, -1, axis=1)

                r_idx = cp.arange(nsub, dtype=cp.int32)[None, :, None]
                up_ok &= (r_idx != 0)
                down_ok &= (r_idx != (nsub - 1))

                bothy = up_ok & down_ok
                only_d = down_ok & ~up_ok
                only_u = up_ok & ~down_ok

                sy = cp.zeros_like(phi_g, dtype=cp.float32)
                sy = cp.where(bothy, (down - up) / (cp.float32(2.0 * pitch)), sy)
                sy = cp.where(only_d, (down - cur) / (cp.float32(pitch)), sy)
                sy = cp.where(only_u, (cur - up) / (cp.float32(pitch)), sy)

                # extract only active lenslets back into (n_map,n_active)
                sy_act = sy[:, rr, cc]
                sx_act = sx[:, rr, cc]
                slopes = cp.stack([sy_act, sx_act], axis=-1)  # (n_map,n_active,2)

                # Match FFT path behaviour for LGS: remove global TT if requested.
                if _is_lgs_range(getattr(self, "gs_range_m", float("inf"))) and bool(self.lgs_remove_tt):
                    slopes = slopes - cp.mean(slopes, axis=1, keepdims=True)

                # Optional approximate sensor noise for Southwell slopes.
                slopes = _apply_southwell_sensor_noise(slopes)

                centroids = cp.zeros_like(slopes)
                return centroids, slopes, None

            # ---------------- full FFT model ----------------
            # complex pupil field
            field = cp.exp(1j * sub_phase).astype(cp.complex64, copy=False)
            field *= self.sub_pupils[None, :, :, :]

            h, w = int(self.h), int(self.w)
            h_p, w_p = h + 2 * pad, w + 2 * pad

            # FFT
            F = cp.fft.fft2(field, s=(h_p, w_p), axes=(-2, -1))  
            I = (F.real * F.real + F.imag * F.imag).astype(cp.float32, copy=False)


            # --- LGS elongation here ---
            if _is_lgs_range(getattr(self, "gs_range_m", float("inf"))) and self.lgs_thickness_m > 0.0:
                self._ensure_lgs_elong_plan()
                plan = getattr(self, "_lgs_elong_plan", None)
                if plan:
                    order = self._lgs_group_order_gpu
                    inv   = self._lgs_group_inv_order_gpu

                    I = I[:, order, :, :]  # ONE gather

                    for segs, wts in plan:
                        for s, e in segs:
                            I[:, s:e, :, :] = ndi.convolve(I[:, s:e, :, :], wts, mode="constant", cval=0.0)

                    # compute slopes/centroids from this reordered I ...
                    # then at the end reorder outputs back:
                    centroids = centroids[:, inv, :]
                    slopes    = slopes[:, inv, :]


            # --- optional WFS noise + centroid windowing ---
            # Photon noise model: if wfs_photons_per_subap>0, intensity in each subap is normalized
            # to that photon budget and a Poisson draw is applied.
            try:
                photons_per_subap = float(params.get("wfs_photons_per_subap", 0) or 0)
            except Exception:
                photons_per_subap = 0.0
            try:
                read_noise_e = float(params.get("wfs_read_noise_e", 0.0) or 0.0)
            except Exception:
                read_noise_e = 0.0
            try:
                win_px = int(params.get("wfs_centroid_window_px", 0) or 0)
            except Exception:
                win_px = 0

            I_mom = I
            if (photons_per_subap > 0) or (read_noise_e > 0):
                if photons_per_subap > 0:
                    # Normalize each subap to a fixed photon budget
                    sumI = cp.sum(I_mom, axis=(-2, -1), keepdims=True)
                    sumI = cp.maximum(sumI, cp.float32(1e-20))
                    I_norm = I_mom / sumI
                    lam = I_norm * cp.float32(photons_per_subap)
                    I_mom = cp.random.poisson(lam).astype(cp.float32, copy=False)
                # Add Gaussian read noise (in electrons) in the same "count" units
                if read_noise_e > 0:
                    I_mom = I_mom + cp.random.normal(
                        loc=0.0,
                        scale=read_noise_e,
                        size=I_mom.shape,
                    ).astype(cp.float32, copy=False)
                    I_mom = cp.maximum(I_mom, cp.float32(0.0))

            # moments without normalizing I (optionally windowed)
            if win_px and win_px > 0:
                I_shift = cp.fft.fftshift(I_mom, axes=(-2, -1))
                mwin = _get_centroid_window_mask(h_p, w_p, win_px, dtype=cp.float32)
                if mwin is not None:
                    I_shift = I_shift * mwin[None, None, :, :]

                ypix = (cp.arange(h_p, dtype=cp.float32) - cp.float32(h_p // 2))
                xpix = (cp.arange(w_p, dtype=cp.float32) - cp.float32(w_p // 2))

                Ix = I_shift.sum(axis=-2)                         # (n_map,n_sub,w_p)
                denom = Ix.sum(axis=-1)                           # (n_map,n_sub)
                denom = cp.maximum(denom, cp.float32(1e-20))
                num_x = (Ix * xpix[None, None, :]).sum(axis=-1)

                Iy = I_shift.sum(axis=-1)                         # (n_map,n_sub,h_p)
                num_y = (Iy * ypix[None, None, :]).sum(axis=-1)
            else:
                yrel, xrel = self._ensure_freq_coords(h_p, w_p)
                Ix = I_mom.sum(axis=-2)                           # (n_map,n_sub,w_p)
                denom = Ix.sum(axis=-1)                           # (n_map,n_sub)
                denom = cp.maximum(denom, cp.float32(1e-20))
                num_x = (Ix * xrel[None, None, :]).sum(axis=-1)

                Iy = I_mom.sum(axis=-1)                           # (n_map,n_sub,h_p)
                num_y = (Iy * yrel[None, None, :]).sum(axis=-1)

            cx_rel = num_x / denom
            cy_rel = num_y / denom

            centroids = cp.stack(
                (cy_rel + cp.float32((h_p - 1) * 0.5),
                cx_rel + cp.float32((w_p - 1) * 0.5)),
                axis=-1
            )

            slopes = cp.stack((cy_rel, cx_rel), axis=-1)

            if _is_lgs_range(getattr(self, "gs_range_m", float("inf"))) and bool(self.lgs_remove_tt):
                slopes = slopes - cp.mean(slopes, axis=1, keepdims=True)

            sensor_image = None
            if return_image:
                # normalize only for display
                den_disp = cp.sum(I_mom, axis=(-2, -1), keepdims=True)
                den_disp = cp.maximum(den_disp, cp.float32(1e-20))
                I_norm = I_mom / den_disp
                I_norm = cp.fft.fftshift(I_norm, axes=(-2, -1))
                I_norm = I_norm[:, :, pad:-pad -1, pad:-pad -1]

                h += -1
                w += -1

                # TODO: replace tiles approach with LUT+kernel 
                gy = int(self.grid_ny); gx = int(self.grid_nx)
                tiles = cp.zeros((n_map, gy, gx, h, w), dtype=I_norm.dtype)
                iy = self.sub_aps_idx[:, 0].astype(cp.int32, copy=False)
                ix = self.sub_aps_idx[:, 1].astype(cp.int32, copy=False)
                tiles[:, iy, ix, :, :] = I_norm
                sensor_image = tiles.transpose(0, 1, 3, 2, 4).reshape(n_map, gy * h, gx * w)

                                # tile size used in stitching
                H = int(h)
                W = int(w)

                iy = self.sub_aps_idx[:, 0].astype(cp.int32, copy=False)  # (n_sub,)
                ix = self.sub_aps_idx[:, 1].astype(cp.int32, copy=False)  # (n_sub,)

                # centroids: (n_map, n_sub, 2)  (y,x)
                cy = centroids[..., 0]
                cx = centroids[..., 1]

                # mosaic coords: (n_map, n_sub)
                mosaic_y = cy + iy[None, :] * H - pad
                mosaic_x = cx + ix[None, :] * W - pad

                centroids = cp.stack((mosaic_y, mosaic_x), axis=-1)  # (n_map, n_sub, 2)


            return centroids, slopes, sensor_image



    class ScienceImager:
        """Dedicated science-channel sensor for generating science images (PSFs).

        This class focuses on producing science images and does **not** implement
        SHWFS centroid/slope logic.

        Compatibility surface used elsewhere in the codebase:
        - .wavelength, .grid_size, .pupil
        - .field_angle = (theta_x, theta_y) in radians
        - .gs_range_m (typically inf)
        - .update_field_angle(t_s, tick)
        - .batch_geometry_key()
        - .measure(...) -> (None, None, (I0, I1, ...)) intensities
        """

        def __init__(
            self,
            wavelength=None,
            dx=None,
            dy=None,
            pupil=None,
            grid_size=None,
            gs_range_m=np.inf,
            field_angle_fn=None,
        ):
            if wavelength is None:
                wavelength = params.get("science_lambda", params.get("wfs_lambda"))
            if grid_size is None:
                grid_size = params.get("grid_size")
            if pupil is None:
                pupil = Pupil_tools.generate_pupil(grid_size=grid_size)

            if dx is None:
                dx = 0.0
            if dy is None:
                dy = 0.0

            self.wavelength = float(wavelength)
            self.grid_size = int(grid_size)
            self.pupil = pupil

            self.gs_range_m = WFSensor_tools._to_float_or_inf(gs_range_m)

            # store angles in radians internally
            self.dx = float(dx) * WFSensor_tools.ARCSEC2RAD
            self.dy = float(dy) * WFSensor_tools.ARCSEC2RAD
            self.field_angle_fn = field_angle_fn
            self.field_angle = (self.dx, self.dy)
            self.sen_angle = float(np.sqrt(self.dx * self.dx + self.dy * self.dy))

            # marker used by GUI to avoid treating this like a SHWFS
            self.is_science_sensor = True

        def update_field_angle(self, t_s: float, tick: int = 0):
            """Update the current field angle from field_angle_fn (if provided).

            field_angle_fn is expected to return (theta_x_rad, theta_y_rad) in **radians**.
            """
            fn = getattr(self, "field_angle_fn", None)
            if fn is None:
                return self.field_angle
            try:
                th = fn(float(t_s), int(tick), self)
            except TypeError:
                try:
                    th = fn(float(t_s), int(tick))
                except TypeError:
                    th = fn(float(t_s))
            thx = float(th[0])
            thy = float(th[1])
            self.dx = thx
            self.dy = thy
            self.field_angle = (thx, thy)
            self.sen_angle = float(np.sqrt(thx * thx + thy * thy))
            return self.field_angle

        def batch_geometry_key(self):
            # science image geometry does not participate in WFS batching
            return (
                "ScienceImager",
                int(getattr(self, "grid_size", -1)),
                float(getattr(self, "wavelength", 0.0)),
            )

        def recompute(
            self,
            wavelength=None,
            dx=None,
            dy=None,
            pupil=None,
            grid_size=None,
            gs_range_m=None,
            field_angle_fn=None,
            **kwargs,
        ):
            if wavelength is not None:
                self.wavelength = float(wavelength)
            if grid_size is not None:
                self.grid_size = int(grid_size)
            if pupil is not None:
                self.pupil = pupil
            if gs_range_m is not None:
                self.gs_range_m = WFSensor_tools._to_float_or_inf(gs_range_m)
            if dx is not None:
                self.dx = float(dx) * WFSensor_tools.ARCSEC2RAD
            if dy is not None:
                self.dy = float(dy) * WFSensor_tools.ARCSEC2RAD
            if field_angle_fn is not None:
                self.field_angle_fn = field_angle_fn

            self.field_angle = (self.dx, self.dy)
            self.sen_angle = float(np.sqrt(self.dx * self.dx + self.dy * self.dy))

        def measure(
            self,
            phase_map,
            pupil=None,
            pad=None,
            assume_radians: bool = True,
            return_image: bool = True,
            fft_pad: int = 0,
            normalize: bool = True,
            return_log: bool = False,
            **kwargs,
        ):
            """Generate science PSF intensity images.

            phase_map:
              - (M,M) or (B,M,M). Convention matches ShackHartmann: phase is treated as
                radians at 500 nm reference by default and scaled to this sensor's wavelength.

            Returns:
              (None, None, (I0, I1, ...)) where each Ii is (M',M') float32 (or log10 if return_log).
            """
            if pupil is None:
                pupil = self.pupil
            pupil = cp.asarray(pupil).astype(cp.float32, copy=False)

            ph = cp.asarray(phase_map).astype(cp.float32, copy=False)
            if ph.ndim == 2:
                ph = ph[None, :, :]
            if ph.ndim != 3:
                raise ValueError(f"ScienceImager.measure expects (M,M) or (B,M,M); got {tuple(ph.shape)}")

            # If caller supplied OPD (meters), convert to radians at *this* wavelength.
            if not bool(assume_radians):
                ph = ph * (4.0 * np.pi / float(self.wavelength))

            # Convert 500nm-reference phase to this wavelength (keeps convention consistent with SHWFS code).
            ph = ph * (float(500e-9) / float(self.wavelength))

            if pad is None:
                pad = int(fft_pad) if (fft_pad and int(fft_pad) > 0) else 0
            pad = int(pad)

            if pad > 0:
                ph = cp.pad(ph, ((0, 0), (pad, pad), (pad, pad)), mode="constant", constant_values=0.0)
                pup = cp.pad(pupil, ((pad, pad), (pad, pad)), mode="constant", constant_values=0.0)
            else:
                pup = pupil

            E = pup[None, :, :] * cp.exp(1j * ph)
            F = cp.fft.fftshift(cp.fft.fft2(E, axes=(-2, -1)), axes=(-2, -1))
            I = (F.real * F.real + F.imag * F.imag).astype(cp.float32)

            if normalize:
                denom = cp.sum(I, axis=(-2, -1), keepdims=True)
                I = I / cp.maximum(denom, cp.float32(1e-20))

            if return_log:
                I = cp.log10(I + cp.float32(1e-12))

            imgs = tuple(I[i] for i in range(int(I.shape[0])))
            return None, None, imgs




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

class Gaussian2DFitter:
    """
    Robust LM (IRLS) fit of 2D Gaussian + constant background on GPU,
    Optional tracking that recenters ROI from the brightest pixel.

    fit_frame() returns:
    p = [A, x0, y0, sx, sy, theta, b0]   - Fitted Gaussian Parameters
    (xc, yc)   - Gaussian coordinates in full image
    (fwhm_eq, fwhm_x, fwhm_y)   -  FWHM parameters in full image
    roi   - zoomed / roi image
    (x0, y0)   - zoomed / roi origin coordinates in full image
    (fwhm_eq_zoom, fwhm_x_zoom, fwhm_y_zoom)   - FWHM parameters in zoomed / roi image
    """

    def __init__(self, roi_half=32, dtype=cp.float32):
        self.half = int(roi_half)
        self.H = self.W = 2 * self.half
        self.dtype = dtype

        yy = cp.arange(self.H, dtype=dtype)
        xx = cp.arange(self.W, dtype=dtype)
        self.X, self.Y = cp.meshgrid(xx, yy)

        self.I = cp.eye(5, dtype=dtype)
        self.ones = cp.ones(self.H * self.W, dtype=dtype)

        self.xc = None
        self.yc = None
        self.p_prev = None

    @staticmethod
    def _rho_prime(s, loss):
        if loss == "linear":
            return cp.ones_like(s)
        if loss == "huber":
            return cp.where(s <= 1.0, 1.0, 1.0 / cp.sqrt(s))
        if loss == "soft_l1":
            return 1.0 / cp.sqrt(1.0 + s)
        if loss == "cauchy":
            return 1.0 / (1.0 + s)
        raise ValueError(loss)

    @staticmethod
    def _robust_scale_mad(r):
        med = cp.median(r)
        mad = cp.median(cp.abs(r - med)) + 1e-12
        return (1.4826 * mad).astype(r.dtype) + 1e-6

    def _model_and_jac(self, p):
        A, x0, y0, s, b0 = p
        s = cp.maximum(s, cp.asarray(1e-6, self.dtype))

        dx = self.X - x0
        dy = self.Y - y0
        R2 = dx*dx + dy*dy

        invs2 = 1.0 / (s*s)
        E = cp.exp(-0.5 * R2 * invs2)
        M = A * E + b0

        # dE/dx0 = E * (dx / s^2), dE/dy0 = E * (dy / s^2)
        dE_dx0 = E * (dx * invs2)
        dE_dy0 = E * (dy * invs2)
        # dE/ds = E * (R2 / s^3)
        dE_ds  = E * (R2 / (s*s*s))

        N = self.H * self.W
        J = cp.empty((N, 5), dtype=self.dtype)
        J[:, 0] = E.ravel()              # dM/dA
        J[:, 1] = (A * dE_dx0).ravel()    # dM/dx0
        J[:, 2] = (A * dE_dy0).ravel()    # dM/dy0
        J[:, 3] = (A * dE_ds ).ravel()    # dM/ds
        J[:, 4] = self.ones              # dM/db0
        return M, J

    def init_from_roi(self, Z):
        Z = Z.astype(self.dtype, copy=False)
        border = cp.concatenate([Z[0, :], Z[-1, :], Z[:, 0], Z[:, -1]])
        b0 = cp.median(border)

        Zp = cp.maximum(Z - b0, 0)
        clip = cp.maximum(cp.max(Zp) * 0.2, cp.asarray(0, self.dtype))
        Wgt = cp.minimum(Zp, clip)

        Wsum = cp.sum(Wgt) + 1e-12
        x0 = cp.sum(self.X * Wgt) / Wsum
        y0 = cp.sum(self.Y * Wgt) / Wsum

        dx = self.X - x0
        dy = self.Y - y0
        s = cp.sqrt(cp.sum(Wgt * (dx*dx + dy*dy)) / (2.0 * Wsum)) + 1e-6

        A = cp.maximum(cp.max(Z) - b0, 0).astype(self.dtype)
        return cp.array([A, x0, y0, s, b0], dtype=self.dtype)

    def fit_roi(self, roi, p0=None, loss="soft_l1", max_iter=10, tol=1e-4, lam0=1e-2):
        Z = cp.asarray(roi, dtype=self.dtype)
        p = self.init_from_roi(Z) if p0 is None else cp.asarray(p0, dtype=self.dtype).copy()
        lam = cp.asarray(lam0, dtype=self.dtype)

        M, _ = self._model_and_jac(p)
        fs = self._robust_scale_mad((M - Z).ravel())

        for _ in range(max_iter):
            M, J = self._model_and_jac(p)
            r = (M - Z).ravel()

            s2 = (r / fs) ** 2
            rho_p = self._rho_prime(s2, loss)
            w2 = rho_p / (fs * fs)

            JW = J * w2[:, None]
            Hn = J.T @ JW
            gn = J.T @ (w2 * r)

            Hn = Hn + lam * self.I
            dp = -cp.linalg.solve(Hn, gn)

            p_new = p + dp
            p_new[3] = cp.maximum(p_new[3], cp.asarray(1e-6, self.dtype))  # s>0

            M2, _ = self._model_and_jac(p_new)
            r2 = (M2 - Z).ravel()

            cost  = cp.sum(self._rho_prime((r / fs) ** 2, loss)  * (r  * r))
            cost2 = cp.sum(self._rho_prime((r2 / fs) ** 2, loss) * (r2 * r2))

            if cost2 < cost:
                p = p_new
                lam = cp.maximum(lam * 0.3, cp.asarray(1e-8, self.dtype))
                if cp.linalg.norm(dp) < tol * (cp.linalg.norm(p) + tol):
                    break
            else:
                lam = cp.minimum(lam * 5.0, cp.asarray(1e6, self.dtype))

        return p


    # ---------------- tracking helpers (brightest pixel) ----------------
    def _brightest_px(self, img, search_r=24):
        """
        Returns (xc, yc) in img coordinates.
        If we have a previous center, search in a (2r+1)x(2r+1) window around it.
        Otherwise, use global argmax.
        """
        H, W = img.shape

        # Ignore NaNs/Infs when searching for the brightest pixel.
        img2 = cp.where(cp.isfinite(img), img, cp.asarray(-cp.inf, dtype=img.dtype))

        # If we don't have a valid prior estimate, use global argmax.
        if self.xc is None or self.yc is None:
            idx = int(cp.argmax(img2).get())
            yc = idx // W
            xc = idx %  W
            return xc, yc

        # Prior center might become NaN if a previous fit failed; reset in that case.
        try:
            xc0 = float(self.xc)
            yc0 = float(self.yc)
            if (not math.isfinite(xc0)) or (not math.isfinite(yc0)):
                raise ValueError("non-finite prior")
        except Exception:
            self.xc = None
            self.yc = None
            idx = int(cp.argmax(img2).get())
            yc = idx // W
            xc = idx % W
            return xc, yc
        r = int(search_r)

        x1 = max(0, int(round(xc0 - r))); x2 = min(W, int(round(xc0 + r + 1)))
        y1 = max(0, int(round(yc0 - r))); y2 = min(H, int(round(yc0 + r + 1)))

        patch = img2[y1:y2, x1:x2]
        if patch.size == 0:
            idx = int(cp.argmax(img2).get())
            yc = idx // W
            xc = idx %  W
            return xc, yc

        idx = int(cp.argmax(patch).get())
        ph, pw = patch.shape
        dy = idx // pw
        dx = idx %  pw
        return (x1 + dx), (y1 + dy)

    def _crop_roi(self, img, xc, yc):
        """
        Crop a fixed-size ROI centered at (xc,yc). Returns (roi, x0, y0)
        where x0,y0 are the ROI origin in img coords.
        """
        H, W = img.shape
        half = self.half

        x0 = max(0, min(W - 2*half, int(round(xc - half))))
        y0 = max(0, min(H - 2*half, int(round(yc - half))))

        roi = img[y0:y0 + 2*half, x0:x0 + 2*half]
        return roi, x0, y0

    # ---------------- one-call tracking fit ----------------
    def fit_frame(self, img, search_r=24, loss="soft_l1", max_iter=10, zoom=1):
        """
        img: full image (already zoomed if you use zooming)
        zoom: set to your upsample factor for FWHM conversion back to native pixels

        Returns:
            p = [A, x0, y0, sx, sy, theta, b0]   - Fitted Gaussian Parameters,
        \n    (xc, yc)   - Gaussian coordinates in full image,
        \n    (fwhm_eq, fwhm)   -  FWHM parameters in full image,
        \n    roi   - zoomed / roi image,
        \n    (x0, y0)   - zoomed / roi origin coordinates in full image,
        \n    (fwhm_eq_zoom, fwhm_zoom)   - FWHM parameters in zoomed / roi image
        """
        img = cp.asarray(img, dtype=self.dtype)

        # 1) start from brightest pixel (global first time, local thereafter)
        xc, yc = self._brightest_px(img, search_r=search_r)

        # 2) crop ROI around brughtest pixel
        roi, x0, y0 = self._crop_roi(img, xc, yc)

        # 3) warm start: shift previous fit into this ROI coord frame
        p0 = None
        if self.p_prev is not None:
            p0 = self.p_prev.copy()
            # p = [A, x0, y0, ...] in ROI coords
            p0[1] = xc - x0  # x in ROI coords
            p0[2] = yc - y0  # y in ROI coords

        # 4) fit
        p = self.fit_roi(roi, p0=p0, loss=loss, max_iter=max_iter)

        # If the fit diverged (NaNs/Infs), fall back to a moment-based init instead
        # of poisoning the tracker state.
        try:
            p_host = cp.asnumpy(p)
            if not np.all(np.isfinite(p_host)):
                raise ValueError("non-finite fit")
        except Exception:
            p = self.init_from_roi(roi)

        self.p_prev = p

        # 5) update global center for next frame (guard against NaNs)
        try:
            xc_new = x0 + float(p[1].get())  # p[1]=x (ROI coords)
            yc_new = y0 + float(p[2].get())  # p[2]=y (ROI coords)
            if (not math.isfinite(xc_new)) or (not math.isfinite(yc_new)):
                raise ValueError("non-finite center")
            self.xc = xc_new
            self.yc = yc_new
        except Exception:
            self.xc = float(xc)
            self.yc = float(yc)

        # 6) FWHM (area-equivalent), convert back to native pixels if zoom>1
        k = 2.354820045
        fwhm_eq_zoom = k * cp.sqrt(p[3] * p[3])
        fwhm = k * p[3]

        fwhm_eq_native = fwhm_eq_zoom / float(zoom)
        fwhm_native = fwhm / zoom

        return p, (self.xc, self.yc), (fwhm_eq_native, fwhm_native), roi, (x0, y0), (fwhm_eq_zoom, fwhm)

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
        logger.info("Available functions:")
        for name in functions.keys():
            logger.info("- %s", name)
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
    logger.info("Result: %s", result)