import os
import time
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cupy as cp
import numpy as np

from scripts.reconstructor import TomoOnAxisIM_CuPy
from scripts.fits_writer import PsfFITSRecorder
from scripts.utilities import Analysis, Gaussian2DFitter, params
from astropy.modeling import models, fitting

def remove_tt(phi: cp.ndarray, pupil: cp.ndarray) -> cp.ndarray:
    """Remove best-fit tip/tilt plane from phi over the pupil.

    Uses the same masked least-squares plane fit used in schedulerGPU.py.
    """
    m = (pupil > 0)
    yy, xx = cp.indices(phi.shape, dtype=cp.float32)
    xx = xx - (phi.shape[1] - 1) * 0.5
    yy = yy - (phi.shape[0] - 1) * 0.5

    z = phi[m].ravel()
    X = cp.stack([xx[m].ravel(), yy[m].ravel(), cp.ones_like(z)], axis=1)
    beta, *_ = cp.linalg.lstsq(X, z, rcond=None)
    plane = (beta[0] * xx + beta[1] * yy + beta[2]) * m
    return phi - plane


def strip_tt_from_slopes(sl_yx: cp.ndarray) -> cp.ndarray:
    """Remove mean slope (global TT) from a (n_sub,2) slope array."""
    return sl_yx - cp.mean(sl_yx, axis=0, keepdims=True)


def _phase_to_psf_batch(
    phase_batch: cp.ndarray,
    pupil: cp.ndarray,
    fft_pad: int,
    roi: Optional[int] = None,
    phase_scale: float = 1.0,
    tt_yx: Optional[cp.ndarray] = None,
    xx: Optional[cp.ndarray] = None,
    yy: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    """Compute normalized PSF intensity for a batch of phase maps.

    phase_batch: (B,N,N) radians
    pupil: (N,N) float/bool
    fft_pad: integer padding on each side (FFT size = N + 2*fft_pad)
    roi: if not None, crop central roi x roi after fftshift
    tt_yx: optional per-frame (B,2) [sy,sx] tip/tilt (radians / pixel) to remove
           from the *phase* before PSF formation. This is equivalent to recentring
           PSFs via shift-and-add, but avoids extra FFTs.
    xx,yy: optional centered coordinate grids (N,N) float32 used with tt_yx.

    returns: (B,roi,roi) or (B,Nfft,Nfft) float32
    """
    assert phase_batch.ndim == 3
    N = int(phase_batch.shape[-1])
    Nfft = N + 2 * int(fft_pad)

    P = pupil.astype(cp.float32, copy=False)

    # Optional wavelength scaling.
    # In this simulator, phase screens are treated as "radians at 500 nm reference".
    # Most sensors (including WFS) apply a factor (500e-9 / wavelength) internally.
    # For PSFs we must do the same (using the science wavelength).
    ps = cp.float32(phase_scale)
    if float(ps) != 1.0:
        phase_batch = phase_batch * ps

    if tt_yx is not None:
        # Remove a linear phase ramp: phi' = phi - (sx*xx + sy*yy)
        if xx is None or yy is None:
            yy0, xx0 = cp.indices((N, N), dtype=cp.float32)
            xx0 = xx0 - (N - 1) * 0.5
            yy0 = yy0 - (N - 1) * 0.5
            xx = xx0
            yy = yy0

        tt_yx = cp.asarray(tt_yx, dtype=cp.float32)
        if tt_yx.ndim != 2 or tt_yx.shape[0] != phase_batch.shape[0] or tt_yx.shape[1] != 2:
            raise ValueError("tt_yx must have shape (B,2) matching phase_batch")

        # If the phase was scaled (e.g., 500 nm -> science λ), the TT coefficients
        # must be scaled the same way.
        if float(ps) != 1.0:
            tt_yx = tt_yx * ps

        sy = tt_yx[:, 0][:, None, None]
        sx = tt_yx[:, 1][:, None, None]
        phase_batch = phase_batch - (sx * xx[None, :, :] + sy * yy[None, :, :])

    E = cp.exp(1j * phase_batch).astype(cp.complex64, copy=False)
    E *= P[None, :, :]

    F = cp.fft.fft2(E, s=(Nfft, Nfft), axes=(-2, -1))
    I = (F.real * F.real + F.imag * F.imag).astype(cp.float32, copy=False)
    I = cp.fft.fftshift(I, axes=(-2, -1))

    # Normalize to unit energy (matches display normalization)
    denom = cp.sum(I, axis=(-2, -1), keepdims=True)
    I = I / cp.maximum(denom, cp.float32(1e-20))

    if roi is not None:
        r = int(roi)
        c = Nfft // 2
        h = r // 2
        I = I[:, c - h : c - h + r, c - h : c - h + r]
    return I

import numpy as np
from astropy.modeling import models, fitting

def fit_moffat2d(psf: np.ndarray, fit_box: int = 128, bkg: float | None = None, zoom: float = 2):
    """
    Fit astropy.modeling.models.Moffat2D to the PSF core.

    Parameters
    ----------
    psf : (H,W) ndarray
        Long-exposure PSF image (intensity).
    fit_box : int
        Size of the square window (pixels) around the peak to fit.
    bkg : float or None
        Background level to subtract before fitting. If None, estimate from border.
    float : float
        Divide final FWHM by image zoom level

    Returns
    -------
    model_fit : astropy Moffat2D model (fitted)
    fwhm_px : float
        FWHM in pixels from the fitted Moffat parameters.
    """
    psf = np.asarray(psf, dtype=float)

    # center on peak
    y0p, x0p = np.unravel_index(np.argmax(psf), psf.shape)
    half = fit_box // 2
    y1, y2 = max(0, y0p - half), min(psf.shape[0], y0p + half)
    x1, x2 = max(0, x0p - half), min(psf.shape[1], x0p + half)

    cut = psf[y1:y2, x1:x2].copy()

    # background estimate from the cutout border if not provided
    if bkg is None:
        border = np.concatenate([
            cut[0, :], cut[-1, :], cut[:, 0], cut[:, -1]
        ])
        bkg = np.median(border)

    cut -= bkg
    cut[cut < 0] = 0.0  # keep fitter stable

    yy, xx = np.mgrid[0:cut.shape[0], 0:cut.shape[1]]

    # initial guesses
    amp0 = float(np.max(cut))
    x0 = float(np.argmax(np.sum(cut, axis=0)))  # crude centroid
    y0 = float(np.argmax(np.sum(cut, axis=1)))
    gamma0 = 2.0          # core width (pixels) initial
    alpha0 = 2.5          # wing slope initial

    m0 = models.Moffat2D(amplitude=amp0, x_0=x0, y_0=y0, gamma=gamma0, alpha=alpha0)

    # bounds 
    m0.amplitude.min = 0.0
    m0.gamma.min = 0.2
    m0.gamma.max = max(cut.shape)  
    m0.alpha.min = 1.01            # alpha<=1 gives infinite variance; avoid
    m0.alpha.max = 20.0
    m0.x_0.min, m0.x_0.max = 0, cut.shape[1] - 1
    m0.y_0.min, m0.y_0.max = 0, cut.shape[0] - 1

    fitter = fitting.TRFLSQFitter()
    mfit = fitter(m0, xx, yy, cut, inplace=True)
    fwhm_px = float(mfit.fwhm) / float(zoom)
    # Convert fitted center (in cutout coords) back to full image coords, then to native pixels.
    x0_full = float(x1) + float(getattr(mfit.x_0, 'value', mfit.x_0))
    y0_full = float(y1) + float(getattr(mfit.y_0, 'value', mfit.y_0))
    x0_px = x0_full / float(zoom)
    y0_px = y0_full / float(zoom)
    r_px = 0.5 * float(fwhm_px)
    return mfit, {"fwhm_px": float(fwhm_px), "x0_px": float(x0_px), "y0_px": float(y0_px), "r_px": float(r_px)}


@dataclass
class PsfMemmapRecorder:
    """Chunked PSF recorder.

    Writes normalized PSFs to disk as float16 using a numpy memmap.
    """

    out_path: str
    n_frames: int
    psf_shape: Tuple[int, int]
    chunk: int = 256
    dtype: np.dtype = np.float16

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        self._mm = np.memmap(
            self.out_path,
            mode="w+",
            dtype=self.dtype,
            shape=(self.n_frames, self.psf_shape[0], self.psf_shape[1]),
        )
        self._i0 = 0

    def write_chunk(self, psf_batch: cp.ndarray):
        """Write (B,H,W) batch to memmap."""
        B = int(psf_batch.shape[0])
        i1 = self._i0 + B
        if i1 > self.n_frames:
            raise ValueError("PSF recorder overflow")
        self._mm[self._i0 : i1] = cp.asnumpy(psf_batch).astype(self.dtype, copy=False)
        self._i0 = i1

    def flush(self):
        self._mm.flush()


class StructureFunctionAccumulator:
    """Accumulate spatial structure function over time for a few pixel lags."""

    def __init__(self, pupil: cp.ndarray, dx_m: float, lags_px: Sequence[int] = (2, 4, 8)):
        self.pupil = pupil.astype(cp.float32, copy=False)
        self.dx_m = float(dx_m)
        self.lags_px = tuple(int(l) for l in lags_px)
        K = len(self.lags_px)
        self._sumDx = cp.zeros(K, dtype=cp.float64)
        self._sumDy = cp.zeros(K, dtype=cp.float64)
        self._sumDx0 = cp.zeros(K, dtype=cp.float64)
        self._sumDy0 = cp.zeros(K, dtype=cp.float64)
        self._countDx = cp.zeros(K, dtype=cp.float64)
        self._countDy = cp.zeros(K, dtype=cp.float64)
        self._countDx0 = cp.zeros(K, dtype=cp.float64)
        self._countDy0 = cp.zeros(K, dtype=cp.float64)

    def add(self, phi_corr: cp.ndarray, phi_uncorr: cp.ndarray):
        m = self.pupil
        for k, lag in enumerate(self.lags_px):
            # x
            mx = m[:, lag:] * m[:, :-lag]
            dxc = (phi_corr[:, lag:] - phi_corr[:, :-lag])
            dx0 = (phi_uncorr[:, lag:] - phi_uncorr[:, :-lag])
            self._sumDx[k] += cp.sum((dxc * dxc) * mx)
            self._sumDx0[k] += cp.sum((dx0 * dx0) * mx)
            c = cp.sum(mx)
            self._countDx[k] += c
            self._countDx0[k] += c

            # y
            my = m[lag:, :] * m[:-lag, :]
            dyc = (phi_corr[lag:, :] - phi_corr[:-lag, :])
            dy0 = (phi_uncorr[lag:, :] - phi_uncorr[:-lag, :])
            self._sumDy[k] += cp.sum((dyc * dyc) * my)
            self._sumDy0[k] += cp.sum((dy0 * dy0) * my)
            c = cp.sum(my)
            self._countDy[k] += c
            self._countDy0[k] += c

    def estimate_r0(self) -> Dict[str, float]:
        # Average structure functions
        Dx = (self._sumDx / cp.maximum(self._countDx, 1.0)).astype(cp.float64)
        Dy = (self._sumDy / cp.maximum(self._countDy, 1.0)).astype(cp.float64)
        Dx0 = (self._sumDx0 / cp.maximum(self._countDx0, 1.0)).astype(cp.float64)
        Dy0 = (self._sumDy0 / cp.maximum(self._countDy0, 1.0)).astype(cp.float64)
        D = 0.5 * (Dx + Dy)
        D0 = 0.5 * (Dx0 + Dy0)

        rhos = cp.asarray([lag * self.dx_m for lag in self.lags_px], dtype=cp.float64)
        slope = 5.0 / 3.0
        log_rhos = cp.log(rhos)
        log_6p88 = cp.log(cp.float64(6.88))

        def _fit_r0(Dv: cp.ndarray) -> float:
            y = cp.log(cp.maximum(Dv, 1e-20)) - log_6p88 - slope * log_rhos
            log_r0 = -cp.mean(y) / slope
            return float(cp.exp(log_r0).get())

        return {
            "r0_corrected_m": _fit_r0(D),
            "r0_uncorrected_m": _fit_r0(D0),
        }


class AOBatchRunner:
    """Run a long AO simulation and record PSFs + summary stats."""

    def __init__(
        self,
        sim,
        sensors: Sequence,
        reconstructor: TomoOnAxisIM_CuPy,
        pupil: cp.ndarray,
        dt_s: float = 0.001,
        gain: float = 0.35,
        leak: float = 0.00,
        wfs_method: str = "southwell_fast",
        strip_tt_lgs: bool = True,
        remove_tt_for_psf: bool = True,
        psf_tt_mode: Optional[str] = None,
        tt_wfs_indices: Optional[Sequence[int]] = None,
    ):
        self.sim = sim
        self.sensors = list(sensors)
        self.R = reconstructor
        self.pupil = pupil.astype(cp.float32, copy=False)
        self.dt_s = float(dt_s)
        self.gain = float(gain)
        self.leak = float(leak)
        self.wfs_method = str(wfs_method)
        self.strip_tt_lgs = bool(strip_tt_lgs)
        # Tip/tilt handling for PSF/metrics.
        #  - "fit":   masked least-squares TT removal per-frame (slowest; reference)
        #  - "hybrid": estimate TT from WFS mean slopes, then remove that tilt only
        #              when forming PSFs (equivalent to shift-and-add recentring)
        #  - "none":  keep residual TT (PSF includes jitter / motion blur)
        if psf_tt_mode is None:
            self.psf_tt_mode = "fit" if bool(remove_tt_for_psf) else "none"
        else:
            self.psf_tt_mode = str(psf_tt_mode).lower().strip()
        if self.psf_tt_mode == "hybrid":
            raise ValueError("psf_tt_mode='hybrid' has been removed; use 'fit' or 'none'")
        if self.psf_tt_mode not in ("fit", "none"):
            raise ValueError("psf_tt_mode must be one of: 'fit', 'none'")

        # Which WFS indices (relative to sensors[1:]) to use for TT estimation in hybrid mode.
        # If None, prefer NGS WFS (infinite range), otherwise fall back to all WFS.
        self.tt_wfs_indices = None if tt_wfs_indices is None else [int(i) for i in tt_wfs_indices]

        # Build the fixed sampler for all sensors (science + WFS).
        # NOTE: build_patch_sampler expects patch side length in *world pixels* (same convention as schedulerGPU).
        thetas = []
        for s in self.sensors:
            fa = getattr(s, "field_angle", None)
            if fa is None:
                fa = (float(getattr(s, "dx", 0.0)), float(getattr(s, "dy", 0.0)))
            thetas.append((float(fa[0]), float(fa[1])))
        ranges = [float(getattr(s, "gs_range_m", float("inf"))) for s in self.sensors]
        patch_M = int(getattr(self.sensors[0], "grid_size"))
        size_pixels = float(patch_M)

        self.sampler = self.sim.build_patch_sampler(
            thetas_xy_rad=thetas,
            ranges_m=ranges,
            size_pixels=size_pixels,
            M=patch_M,
            angle_deg=0.0,
            autopan_init=True,
        )

        # WFS list (exclude science) and packed slope-vector layout.
        if len(self.sensors) < 2:
            raise ValueError("Need at least one WFS sensor in sensors[1:]")

        self.wfs_list = list(self.sensors[1:])
        self.n_wfs = int(len(self.wfs_list))

        # LGS/NGS classification: finite range -> LGS (cannot sense global TT).
        self._wfs_is_lgs = [np.isfinite(float(getattr(s, "gs_range_m", np.inf))) for s in self.wfs_list]

        # Cache WFS wavelengths. ShackHartmann.measure assumes phase screens are
        # in "radians at 500 nm" and internally applies a factor (500e-9 / wl).
        # For hybrid TT we want TT coefficients in the *same units as the phase*
        # (i.e. 500 nm reference), so we may need to undo that scaling.
        self._wfs_wavelengths = [float(getattr(s, "wavelength", 500e-9)) for s in self.wfs_list]

        # Active subap counts + packed offsets per WFS (supports heterogeneous geometries).
        self._n_active = [int(len(getattr(s, "sub_aps_idx"))) for s in self.wfs_list]
        offs = [0]
        for na in self._n_active:
            offs.append(offs[-1] + 2 * int(na))
        self._slices = [(offs[i], offs[i + 1]) for i in range(self.n_wfs)]
        self._slopes_total_len = int(offs[-1])

        # Reference slopes per WFS (nearly always ~0, but keep for consistency).
        self._slopes_ref_list = []
        z2 = cp.zeros((patch_M, patch_M), dtype=cp.float32)
        for s in self.wfs_list:
            _, sl, _ = s.measure(
                pupil=None,
                phase_map=z2,
                return_image=False,
                assume_radians=True,
                method=self.wfs_method,
            )
            self._slopes_ref_list.append(sl[0].astype(cp.float32, copy=False))

        # Group WFS by identical geometry so we can measure in batches where possible.
        groups = {}
        for i, s in enumerate(self.wfs_list):
            key = (
                type(s),
                int(getattr(s, "grid_size", -1)),
                int(getattr(s, "n_sub", -1)),
                int(self._n_active[i]),
            )
            groups.setdefault(key, []).append(i)

        self._wfs_groups = []
        for _, inds in groups.items():
            tmpl = self.wfs_list[inds[0]]
            G = int(len(inds))
            self._wfs_groups.append(
                {
                    "inds": inds,
                    "tmpl": tmpl,
                    "phase_buf": cp.empty((G, patch_M, patch_M), dtype=cp.float32),
                }
            )

        # Packed slope error vector (layout: per WFS block is [sx(na), sy(na)]).
        self._slopes_err_vec = cp.empty((self._slopes_total_len,), dtype=cp.float32)

        # Resolve default TT estimation WFS indices if not provided (prefer NGS).
        if self.tt_wfs_indices is None:
            ngs = [i for i, s in enumerate(self.wfs_list) if np.isinf(float(getattr(s, "gs_range_m", np.inf)))]
            self.tt_wfs_indices = ngs if len(ngs) > 0 else list(range(self.n_wfs))
        self._tt_wfs_set = set(int(i) for i in self.tt_wfs_indices)

        # DM state
        self.dm_cmd = cp.zeros((self.R.n_modes,), dtype=cp.float32)
        self.dm_phi = cp.zeros((patch_M, patch_M), dtype=cp.float32)

        # Centered coordinate grids for fast TT ramp removal (hybrid mode)
        yy0, xx0 = cp.indices((patch_M, patch_M), dtype=cp.float32)
        self._xx = xx0 - (patch_M - 1) * 0.5
        self._yy = yy0 - (patch_M - 1) * 0.5

    def run(
        self,
        n_frames: int,
        # Discard the first N seconds worth of frames from *final statistics* (FWHM,
        # long-exposure PSFs, r0 estimates), while still recording all per-frame
        # data products (memmaps, TT series) if enabled.
        discard_first_s: float = 0.0,
        # Optional explicit override for warm-up discard expressed in frames.
        # If provided, it takes precedence over discard_first_s.
        discard_first_frames: Optional[int] = None,
        record_psfs: bool = True,
        record_psfs_ttremoved: bool = False,
        save_tt_series: bool = True,
        out_dir: str = "run_out",
        fft_pad: Optional[int] = None,
        psf_roi: int = 128,
        chunk_frames: int = 256,
        structure_lags_px: Sequence[int] = (2, 4, 8),
        progress_cb: Optional[Callable[[int, int, float, str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        progress_every: int = 512,
        dt_s : float = 0.001,
    ) -> Dict[str, object]:
        n_frames = int(n_frames)
        patch_M = int(self.sensors[0].grid_size)
        if fft_pad is None:
            fft_pad = patch_M // 2

        discard_first_s = float(discard_first_s)
        if discard_first_frames is None:
            # Use floor so small warm-up windows don't round to 0 unexpectedly.
            discard_first_frames = int(math.floor(discard_first_s / max(self.dt_s, 1e-12) + 1e-9))
        else:
            discard_first_frames = int(discard_first_frames)
            discard_first_s = float(discard_first_frames) * float(self.dt_s)
        discard_first_frames = max(0, min(int(discard_first_frames), int(n_frames)))

        # Plate scale for PSF pixels (radians / pixel)
        D = float(params.get("telescope_diameter"))
        # Prefer the science sensor's wavelength, fall back to params.
        lam_sci = float(getattr(self.sensors[0], "wavelength", params.get("science_lambda")))
        Nfft = patch_M + 2 * int(fft_pad)
        plate_rad = lam_sci * patch_M / (D * Nfft)

        # Convert 500 nm-reference phase screens -> science wavelength phase.
        # This matches the WFS scaling convention used elsewhere.
        phase_scale = float(500e-9 / max(lam_sci, 1e-30))

        os.makedirs(out_dir, exist_ok=True)
        rec_corr = None
        rec_unc = None
        rec_corr_tt = None

        # Per-frame PSF recording (optional)
        if record_psfs:
            # Stream per-frame PSFs to FITS (multiple IMAGE extensions), float32.
            hdr = {
                "DT": float(dt_s),
                "LAMSCI": float(lam_sci),
                "ROI": int(psf_roi),
                "CHUNK": int(chunk_frames),
                "TTMODE": str(self.psf_tt_mode),
            }
            rec_corr = PsfFITSRecorder(
                out_path=os.path.join(out_dir, "psf_corrected.fits"),
                psf_shape=(psf_roi, psf_roi),
                extra_header=hdr,
                extname_chunks="PSF",
            )
            rec_unc = PsfFITSRecorder(
                out_path=os.path.join(out_dir, "psf_uncorrected.fits"),
                psf_shape=(psf_roi, psf_roi),
                extra_header=hdr,
                extname_chunks="PSF",
            )

        # Optional: record TT-removed corrected PSFs (hybrid mode only)
        if record_psfs_ttremoved and self.psf_tt_mode == "hybrid":
            hdr_tt = {
                "DT": float(dt_s),
                "LAMSCI": float(lam_sci),
                "ROI": int(psf_roi),
                "CHUNK": int(chunk_frames),
                "TTMODE": "hybrid",
                "TTREM": True,
            }
            rec_corr_tt = PsfFITSRecorder(
                out_path=os.path.join(out_dir, "psf_corrected_ttremoved.fits"),
                psf_shape=(psf_roi, psf_roi),
                extra_header=hdr_tt,
                extname_chunks="PSF",
            )

        # Accumulators
        dx_m = float(params.get("telescope_diameter")) / float(patch_M)
        sf = StructureFunctionAccumulator(self.pupil, dx_m=dx_m, lags_px=structure_lags_px)

        sum_psf_corr = cp.zeros((psf_roi, psf_roi), dtype=cp.float64)
        sum_psf_unc = cp.zeros((psf_roi, psf_roi), dtype=cp.float64)
        sum_psf_corr_tt = None
        if self.psf_tt_mode == "hybrid":
            sum_psf_corr_tt = cp.zeros((psf_roi, psf_roi), dtype=cp.float64)

        # Phase buffers for chunked PSF FFT
        buf_corr = cp.empty((chunk_frames, patch_M, patch_M), dtype=cp.float32)
        buf_unc = cp.empty((chunk_frames, patch_M, patch_M), dtype=cp.float32)
        buf_tt = None
        tt_series = None
        if self.psf_tt_mode == "hybrid":
            buf_tt = cp.empty((chunk_frames, 2), dtype=cp.float32)
            if save_tt_series:
                tt_series = cp.empty((n_frames, 2), dtype=cp.float32)

        b = 0

        n_done = 0
        n_used = 0
        cancelled = False
        wfs_phase_buf = cp.empty((self.n_wfs, patch_M, patch_M), dtype=cp.float32)
        tt_acc = cp.zeros((2,), dtype=cp.float32) if self.psf_tt_mode == "hybrid" else None
        t0 = time.perf_counter()
        last_prog = 0

        for t in range(n_frames):
            if should_stop is not None and should_stop():
                cancelled = True
                break

            # 1) Advance turbulence
            self.sim.advance(self.dt_s)

            # 2) Sample all sensor phase patches (radians @ 500nm reference)
            phases_atm = self.sim.sample_patches_from_sampler(
                self.sampler,
                remove_piston=True,
                return_gpu=True,
            )  # (n_sensors, M, M)
            # 3) Science residual phase (DM from previous tick is conjugate to ground)
            phi_unc = phases_atm[0]
            phi_cor = phi_unc - self.dm_phi

            if self.psf_tt_mode == "fit":
                phi_unc = remove_tt(phi_unc, self.pupil)
                phi_cor = remove_tt(phi_cor, self.pupil)

            # 4) Measure slopes on WFS phases (batched by geometry) and pack a slope-error vector.
            # Reuse buffers to avoid per-frame allocations.
            cp.copyto(wfs_phase_buf, phases_atm[1:])
            wfs_phase_buf -= self.dm_phi[None, :, :]

            if self.psf_tt_mode == "hybrid":
                tt_acc.fill(0.0)
                tt_n = 0

            # NGS are ignored for recon; leave zeros in their blocks.
            self._slopes_err_vec.fill(0.0)

            for g in self._wfs_groups:
                inds = g["inds"]
                phb = g["phase_buf"]
                for jj, wi in enumerate(inds):
                    cp.copyto(phb[jj], wfs_phase_buf[wi])

                _, sl_stack, _ = g["tmpl"].measure(
                    pupil=None,
                    phase_map=phb,
                    return_image=False,
                    assume_radians=True,
                    method=self.wfs_method,
                )

                for jj, wi in enumerate(inds):
                    sl = sl_stack[jj].astype(cp.float32, copy=False)  # (n_active,2) [sy,sx]
                    ref = self._slopes_ref_list[wi]
                    sl_err = sl - ref

                    # Hybrid TT estimate from mean slopes (before LGS TT stripping / NGS ignore)
                    if self.psf_tt_mode == "hybrid" and (wi in self._tt_wfs_set):
                        # sl_err is a gradient of the phase *after* ShackHartmann has
                        # applied its internal 500nm->wl scaling (sub_phase *= 500e-9/wl).
                        # Convert back to 500nm-reference radians so it matches phi_* buffers.
                        wl = cp.float32(self._wfs_wavelengths[wi])
                        tt_acc += cp.mean(sl_err, axis=0) * (wl / cp.float32(500e-9))
                        tt_n += 1

                    # Reconstruction input: LGS only; optionally strip TT
                    if not self._wfs_is_lgs[wi]:
                        continue
                    if self.strip_tt_lgs:
                        sl_err = sl_err - cp.mean(sl_err, axis=0, keepdims=True)

                    na = self._n_active[wi]
                    if na <= 0:
                        continue
                    a, _ = self._slices[wi]
                    self._slopes_err_vec[a : a + na] = sl_err[:, 1]
                    self._slopes_err_vec[a + na : a + 2 * na] = sl_err[:, 0]

            if self.psf_tt_mode == "hybrid":
                if tt_n > 0:
                    tt_yx = tt_acc / cp.float32(tt_n)
                else:
                    tt_yx = cp.zeros((2,), dtype=cp.float32)
                buf_tt[b] = tt_yx
                if tt_series is not None:
                    tt_series[t] = tt_yx

            # 5) Reconstruct + controller
            xhat = self.R.solve_coeffs_stack(self._slopes_err_vec)
            self.dm_cmd = (1.0 - self.leak) * self.dm_cmd - self.gain * xhat
            self.dm_phi = -self.R.coeffs_to_onaxis_phase(self.dm_cmd)
            self.dm_phi = (self.dm_phi - cp.mean(self.dm_phi)) * self.pupil

            if self.psf_tt_mode == "fit":
                self.dm_phi = remove_tt(self.dm_phi, self.pupil)

            # 6) Record structure function stats (cheap; no FFT)
            # (Apply discard window only to final statistics.)
            if t >= discard_first_frames:
                sf.add(phi_cor, phi_unc)

            # 7) Buffer phases for PSF batches
            buf_corr[b] = phi_cor
            buf_unc[b] = phi_unc
            b += 1
            n_done = t + 1
            if b == chunk_frames or t == (n_frames - 1):
                B = b
                # Global indices for this chunk
                chunk_end = int(t + 1)
                chunk_start = int(chunk_end - B)
                off_used = max(0, int(discard_first_frames) - int(chunk_start))
                B_used = int(B - off_used)

                # Decide what to compute:
                #  - If recording PSFs, we must compute the full chunk.
                #  - Otherwise, only compute the used subset (post-discard) for stats.
                compute_full = bool(record_psfs)
                compute_stats = bool(B_used > 0)

                Icorr = None
                Iunc = None
                if compute_full or compute_stats:
                    if compute_full:
                        Icorr = _phase_to_psf_batch(
                            buf_corr[:B],
                            self.pupil,
                            fft_pad=fft_pad,
                            roi=psf_roi,
                            phase_scale=phase_scale,
                        )
                        Iunc = _phase_to_psf_batch(
                            buf_unc[:B],
                            self.pupil,
                            fft_pad=fft_pad,
                            roi=psf_roi,
                            phase_scale=phase_scale,
                        )
                    else:
                        # Stats-only: compute only the used tail of the chunk.
                        Icorr = _phase_to_psf_batch(
                            buf_corr[off_used:B],
                            self.pupil,
                            fft_pad=fft_pad,
                            roi=psf_roi,
                            phase_scale=phase_scale,
                        )
                        Iunc = _phase_to_psf_batch(
                            buf_unc[off_used:B],
                            self.pupil,
                            fft_pad=fft_pad,
                            roi=psf_roi,
                            phase_scale=phase_scale,
                        )

                # Accumulate long-exposure PSFs only for frames >= discard_first_frames.
                if B_used > 0 and Icorr is not None and Iunc is not None:
                    if compute_full:
                        sum_psf_corr += cp.sum(Icorr[off_used:], axis=0)
                        sum_psf_unc += cp.sum(Iunc[off_used:], axis=0)
                    else:
                        sum_psf_corr += cp.sum(Icorr, axis=0)
                        sum_psf_unc += cp.sum(Iunc, axis=0)
                    n_used += int(B_used)

                if self.psf_tt_mode == "hybrid":
                    # TT-removed corrected PSFs via phase-ramp compensation.
                    # If recording TT-removed PSFs, compute full chunk; otherwise
                    # compute only used frames for stats.
                    if rec_corr_tt is not None:
                        Icorr_tt = _phase_to_psf_batch(
                            buf_corr[:B],
                            self.pupil,
                            fft_pad=fft_pad,
                            roi=psf_roi,
                            phase_scale=phase_scale,
                            tt_yx=buf_tt[:B],
                            xx=self._xx,
                            yy=self._yy,
                        )
                        if B_used > 0:
                            sum_psf_corr_tt += cp.sum(Icorr_tt[off_used:], axis=0)
                        rec_corr_tt.write_chunk(Icorr_tt)
                    elif B_used > 0:
                        Icorr_tt = _phase_to_psf_batch(
                            buf_corr[off_used:B],
                            self.pupil,
                            fft_pad=fft_pad,
                            roi=psf_roi,
                            phase_scale=phase_scale,
                            tt_yx=buf_tt[off_used:B],
                            xx=self._xx,
                            yy=self._yy,
                        )
                        sum_psf_corr_tt += cp.sum(Icorr_tt, axis=0)

                if record_psfs:
                    # When recording PSFs, Icorr/Iunc are always full-chunk.
                    rec_corr.write_chunk(Icorr)
                    rec_unc.write_chunk(Iunc)
                b = 0

                # Progress callback (rate-limited)
                if progress_cb is not None:
                    done = int(n_done)
                    if (done - last_prog) >= int(progress_every) or done == int(n_frames):
                        dt = max(time.perf_counter() - t0, 1e-9)
                        fps = float(done) / float(dt)
                        progress_cb(done, int(n_frames), fps, "running")
                        last_prog = done

                # (FITS recorders are finalized after long-exposure PSFs are computed)


        if self.psf_tt_mode == "hybrid" and tt_series is not None:
            np.save(os.path.join(out_dir, "tt_yx_from_wfs.npy"), cp.asnumpy(tt_series[:n_done]))

        # Long-exposure PSFs (mean over frames >= discard_first_frames)
        denom_used = float(max(int(n_used), 1))
        psf_long_corr = (sum_psf_corr / denom_used).astype(cp.float32)
        psf_long_unc = (sum_psf_unc / denom_used).astype(cp.float32)

        psf_long_corr_tt = None
        if self.psf_tt_mode == "hybrid":
            psf_long_corr_tt = (sum_psf_corr_tt / denom_used).astype(cp.float32)

        # --- Write long-exposure PSFs to FITS ---
        # Always write 2D long-exposure PSFs as separate FITS files for convenience.
        # If recording per-frame PSFs, also append LONGEXP extensions to those files.
        from scripts.fits_writer import write_primary_empty, append_image_hdu

        # Separate files
        with open(os.path.join(out_dir, "psf_long_corrected.fits"), "wb") as f:
            write_primary_empty(f, {"DT": float(dt_s), "LAMSCI": float(lam_sci), "ROI": int(psf_roi), "USED": int(n_used)})
            append_image_hdu(f, cp.asnumpy(psf_long_corr), extname="LONGEXP", extver=1)
        with open(os.path.join(out_dir, "psf_long_uncorrected.fits"), "wb") as f:
            write_primary_empty(f, {"DT": float(dt_s), "LAMSCI": float(lam_sci), "ROI": int(psf_roi), "USED": int(n_used)})
            append_image_hdu(f, cp.asnumpy(psf_long_unc), extname="LONGEXP", extver=1)
        if psf_long_corr_tt is not None:
            with open(os.path.join(out_dir, "psf_long_corrected_ttremoved.fits"), "wb") as f:
                write_primary_empty(f, {"DT": float(dt_s), "LAMSCI": float(lam_sci), "ROI": int(psf_roi), "USED": int(n_used), "TTREM": True})
                append_image_hdu(f, cp.asnumpy(psf_long_corr_tt), extname="LONGEXP", extver=1)

        # Append LONGEXP extensions into the per-frame PSF FITS files
        if record_psfs and rec_corr is not None and rec_unc is not None:
            try:
                rec_corr.write_longexp(psf_long_corr, extname="LONGEXP", extra_header={"USED": int(n_used), "DISCARD": int(discard_first_frames)})
                rec_unc.write_longexp(psf_long_unc, extname="LONGEXP", extra_header={"USED": int(n_used), "DISCARD": int(discard_first_frames)})
            except Exception:
                pass
        if rec_corr_tt is not None and psf_long_corr_tt is not None:
            try:
                rec_corr_tt.write_longexp(psf_long_corr_tt, extname="LONGEXP", extra_header={"USED": int(n_used), "DISCARD": int(discard_first_frames)})
            except Exception:
                pass

        # Finalize recorders
        if record_psfs and rec_corr is not None and rec_unc is not None:
            rec_corr.close()
            rec_unc.close()
        if rec_corr_tt is not None:
            rec_corr_tt.close()

        # FWHM estimates via Gaussian fit (uses GPU LM fit)
        zoom = int(params.get("fwhm_zoom", 2))
        fitter = Gaussian2DFitter(roi_half=int(params.get("fwhm_roi_half", 32)))

        # Fit metadata (native PSF pixels; for overlays)
        gauss_corr = {"fwhm_px": float("nan"), "x0_px": float("nan"), "y0_px": float("nan"), "r_px": float("nan")}
        gauss_unc  = {"fwhm_px": float("nan"), "x0_px": float("nan"), "y0_px": float("nan"), "r_px": float("nan")}
        moff_corr  = {"fwhm_px": float("nan"), "x0_px": float("nan"), "y0_px": float("nan"), "r_px": float("nan")}
        moff_unc   = {"fwhm_px": float("nan"), "x0_px": float("nan"), "y0_px": float("nan"), "r_px": float("nan")}

        Icorr_moffat = float("nan")
        Iunc_moffat = float("nan")

        if int(n_used) > 0:
            Icorr_zoom = Analysis._zoom2d(psf_long_corr, zoom, cp)
            Iunc_zoom = Analysis._zoom2d(psf_long_unc, zoom, cp)

            # Gaussian (GPU). Note: centers are returned in zoomed pixel coordinates.
            _p, (xc_z, yc_z), (fwhm_eq_native, _), *_rest = fitter.fit_frame(
                Icorr_zoom, search_r=24, loss="soft_l1", max_iter=25, zoom=zoom
            )
            fwhm_px_corr = float(fwhm_eq_native.get())
            gauss_corr = {
                "fwhm_px": float(fwhm_px_corr),
                "x0_px": float(xc_z) / float(zoom),
                "y0_px": float(yc_z) / float(zoom),
                "r_px": 0.5 * float(fwhm_px_corr),
            }

            _p, (xc_z, yc_z), (fwhm_eq_native, _), *_rest = fitter.fit_frame(
                Iunc_zoom, search_r=24, loss="soft_l1", max_iter=25, zoom=zoom
            )
            fwhm_px_unc = float(fwhm_eq_native.get())
            gauss_unc = {
                "fwhm_px": float(fwhm_px_unc),
                "x0_px": float(xc_z) / float(zoom),
                "y0_px": float(yc_z) / float(zoom),
                "r_px": 0.5 * float(fwhm_px_unc),
            }

            # Moffat (CPU on zoomed cutout)
            _mfit, moff_corr = fit_moffat2d(Icorr_zoom.get(), zoom=float(zoom))
            _mfit, moff_unc = fit_moffat2d(Iunc_zoom.get(), zoom=float(zoom))
            Icorr_moffat = float(moff_corr.get("fwhm_px", float("nan")))
            Iunc_moffat = float(moff_unc.get("fwhm_px", float("nan")))
        else:
            fwhm_px_corr = float("nan")
            fwhm_px_unc = float("nan")

        fwhm_px_corr_tt = None
        if psf_long_corr_tt is not None:
            if int(n_used) > 0:
                Icorr_tt_zoom = Analysis._zoom2d(psf_long_corr_tt, zoom, cp)
                fwhm_px_corr_tt = float(fitter.fit_frame(Icorr_tt_zoom, search_r=24, loss="soft_l1", max_iter=25, zoom=zoom)[2][0].get())
            else:
                fwhm_px_corr_tt = float("nan")
        out: Dict[str, object] = {
            "plate_rad_per_pix": plate_rad,
            "psf_long_corrected": psf_long_corr,
            "psf_long_uncorrected": psf_long_unc,
            "fwhm_rad_corrected": fwhm_px_corr * plate_rad,
            "fwhm_rad_uncorrected": fwhm_px_unc * plate_rad,
            "fwhm_rad_corrected_moffat": Icorr_moffat * plate_rad,
            "fwhm_rad_uncorrected_moffat": Iunc_moffat * plate_rad,
            # Fit metadata for overlays (native PSF pixels)
            "gauss_corrected": gauss_corr,
            "gauss_uncorrected": gauss_unc,
            "moffat_corrected": moff_corr,
            "moffat_uncorrected": moff_unc,
            "psf_tt_mode": self.psf_tt_mode,
            "n_frames_requested": int(n_frames),
            "n_frames_done": int(n_done),
            "discard_first_s": float(discard_first_s),
            "discard_first_frames": int(discard_first_frames),
            "n_frames_used": int(n_used),
            "cancelled": bool(cancelled),
        }
        if psf_long_corr_tt is not None:
            out.update(
                {
                    "psf_long_corrected_ttremoved": psf_long_corr_tt,
                    "fwhm_rad_corrected_ttremoved": float(fwhm_px_corr_tt) * plate_rad,
                    "tt_yx_series_path": os.path.join(out_dir, "tt_yx_from_wfs.npy"),
                }
            )
        out.update(sf.estimate_r0())
        if record_psfs:
            out.update(
                {
                    # Backward-compatible keys (used by older UI code)
                    "psf_corrected_memmap": os.path.join(out_dir, "psf_corrected.fits"),
                    "psf_uncorrected_memmap": os.path.join(out_dir, "psf_uncorrected.fits"),
                    # Preferred keys
                    "psf_corrected_fits": os.path.join(out_dir, "psf_corrected.fits"),
                    "psf_uncorrected_fits": os.path.join(out_dir, "psf_uncorrected.fits"),
                    "psf_format": "fits_image_extensions",
                    "psf_shape": (n_frames, psf_roi, psf_roi),
                    "psf_frames_written": int(n_done),
                    "psf_dtype": "float32",
                }
            )        
            if rec_corr_tt is not None:
                out["psf_corrected_ttremoved_memmap"] = os.path.join(out_dir, "psf_corrected_ttremoved.fits")
                out["psf_corrected_ttremoved_fits"] = os.path.join(out_dir, "psf_corrected_ttremoved.fits")
        return out