import os
import time
import math
from dataclasses import dataclass
from collections import deque
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cupy as cp
import numpy as np

from scripts.reconstructor import TomoOnAxisIM_CuPy, TipTiltReconstructor_CuPy
from scripts.fits_writer import PsfFITSRecorder
from scripts.utilities import Analysis, Gaussian2DFitter, params
from astropy.modeling import models, fitting


class TTPlaneRemover:
    """Fast masked least-squares plane (tip/tilt + piston) remover.

    Precomputes a pseudo-inverse for the fixed pupil mask so we can remove TT
    from many phase maps with a single GEMM per batch (much faster than calling
    cp.linalg.lstsq per frame / per field point).
    """

    def __init__(self, pupil: cp.ndarray):
        m = (pupil > 0)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError("pupil must be a square 2D array")
        self.mask = m
        N = int(m.shape[0])

        yy, xx = cp.indices((N, N), dtype=cp.float32)
        xx = xx - (N - 1) * 0.5
        yy = yy - (N - 1) * 0.5
        self.xx = xx
        self.yy = yy
        self.m_float = m.astype(cp.float32, copy=False)

        xm = xx[m].ravel()
        ym = yy[m].ravel()
        ones = cp.ones_like(xm)

        X = cp.stack([xm, ym, ones], axis=1)  # (K,3)
        X64 = X.astype(cp.float64, copy=False)
        pinv = cp.linalg.inv(X64.T @ X64) @ X64.T  # (3,K)
        self.pinv = pinv.astype(cp.float32, copy=False)

    def remove_inplace(self, phi: cp.ndarray) -> cp.ndarray:
        """Remove best-fit plane from phi.

        phi can be (N,N) or (B,N,N). If (N,N), the operation is in-place and the
        same 2D array is returned.
        """
        if phi.ndim == 2:
            phi_b = phi[None, :, :]
            squeeze = True
        elif phi.ndim == 3:
            phi_b = phi
            squeeze = False
        else:
            raise ValueError("phi must be 2D or 3D")

        # z: (B,K) values under the pupil mask
        z = phi_b[:, self.mask]
        beta = z @ self.pinv.T  # (B,3)

        bx = beta[:, 0][:, None, None]
        by = beta[:, 1][:, None, None]
        bc = beta[:, 2][:, None, None]

        plane = (bx * self.xx[None, :, :] + by * self.yy[None, :, :] + bc) * self.m_float[None, :, :]
        phi_b -= plane

        return phi if squeeze else phi_b



def _phase_to_psf_batch(
    phase_batch: cp.ndarray,
    pupil: cp.ndarray,
    fft_pad: int,
    roi: Optional[int] = None,
    phase_scale: float = 1.0,
    norm_inv: Optional[cp.ndarray] = None,
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

        sy = tt_yx[:, 0][:, None, None]
        sx = tt_yx[:, 1][:, None, None]
        phase_batch = phase_batch - (sx * xx[None, :, :] + sy * yy[None, :, :])

    # Build complex field. Avoid materializing scaled phase when possible.
    if float(ps) == 1.0:
        E = cp.exp(1j * phase_batch).astype(cp.complex64, copy=False)
    else:
        # fused multiply inside exp
        E = cp.exp(1j * (phase_batch * ps)).astype(cp.complex64, copy=False)
    E *= P[None, :, :]

    F = cp.fft.fft2(E, s=(Nfft, Nfft), axes=(-2, -1))

    # Normalization: energy is invariant to phase (|E|^2 = P^2), so
    # sum(|F|^2) is constant for fixed pupil + FFT size (Parseval).
    if norm_inv is None:
        p2 = cp.sum(P * P)
        norm_inv = cp.float32(1.0) / cp.maximum(cp.float32(Nfft * Nfft) * p2, cp.float32(1e-20))

    if roi is None:
        I = (F.real * F.real + F.imag * F.imag).astype(cp.float32, copy=False)
        I = cp.fft.fftshift(I, axes=(-2, -1))
        I *= norm_inv
        return I

    # ROI-only intensity without allocating full-size I or doing fftshift.
    r = int(roi)
    h = r // 2
    if r <= 0 or r > Nfft:
        raise ValueError(f"roi must be in [1, {Nfft}], got {r}")

    def _wrap_segments(N: int, start: int, length: int):
        start = int(start) % int(N)
        length = int(length)
        if start + length <= N:
            return [(start, start + length, 0, length)]
        l1 = N - start
        l2 = length - l1
        return [(start, N, 0, l1), (0, l2, l1, length)]

    # Shifted ROI around center corresponds to unshifted indices around 0.
    start = (-h) % int(Nfft)  # == Nfft-h for even Nfft
    row_segs = _wrap_segments(Nfft, start, r)
    col_segs = _wrap_segments(Nfft, start, r)

    out = cp.empty((phase_batch.shape[0], r, r), dtype=cp.float32)
    for rs0, rs1, rd0, rd1 in row_segs:
        for cs0, cs1, cd0, cd1 in col_segs:
            blk = F[:, rs0:rs1, cs0:cs1]
            out[:, rd0:rd1, cd0:cd1] = (blk.real * blk.real + blk.imag * blk.imag).astype(cp.float32, copy=False)

    out *= norm_inv
    return out

import numpy as np
from astropy.modeling import models, fitting

def fit_moffat2d(psf: np.ndarray, fit_box: int = 1024, bkg: float | None = None, zoom: float = 2):
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
    gamma_full = float(getattr(mfit.gamma, "value", mfit.gamma))
    alpha = float(getattr(mfit.alpha, "value", mfit.alpha))
    amp = float(getattr(mfit.amplitude, "value", mfit.amplitude))
    gamma_px = gamma_full / float(zoom)
    return mfit, {
        "fwhm_px": float(fwhm_px),
        "x0_px": float(x0_px),
        "y0_px": float(y0_px),
        "r_px": float(r_px),
        "amp": amp,
        "bkg": float(bkg),
        "gamma_px": float(gamma_px),
        "alpha": float(alpha),
    }


def fit_moffat2d_wings(
    psf: np.ndarray,
    fit_box: int = 1024,
    bkg: float | None = None,
    zoom: float = 2,
    weight_q: float = 0.5,
    weight_p: float = 1.0,
    r0_frac: float = 0.25,
    w_max: float = 100.0,
    core_exclude_frac: float = 0.0,
):
    """
    Fit a Moffat2D with weights that emphasize the wings/skirts more than the peak.

    This is a metric option intended to bias the least-squares fit away from the very bright core.
    The weighting is heuristic but robust:

        w = (1 + r/r0)^p / (I + eps)^q

    where r0 is a fraction of the cutout size and eps prevents blow-ups near zero.

    Parameters
    ----------
    psf : (H,W) ndarray
        Long-exposure PSF image (intensity).
    fit_box : int
        Size of the square window (pixels) around the peak to fit.
    bkg : float or None
        Background level to subtract before fitting. If None, estimate from border.
    zoom : float
        Divide final FWHM by image zoom level.
    weight_q : float
        Intensity down-weight exponent for the core (q>0 emphasizes wings).
    weight_p : float
        Radial up-weight exponent for the outskirts (p>0 emphasizes wings).
    r0_frac : float
        r0 = r0_frac * min(cutout.shape) (in cutout pixels).
    w_max : float
        Cap weights to avoid ill-conditioning.
    core_exclude_frac : float
        Optionally exclude a small core radius: r < core_exclude_frac*min(shape) ⇒ w=0.

    Returns
    -------
    model_fit : astropy Moffat2D model (fitted)
    info : dict
        Fit info compatible with fit_moffat2d.
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
        border = np.concatenate([cut[0, :], cut[-1, :], cut[:, 0], cut[:, -1]])
        bkg = np.median(border)

    cut -= bkg
    cut[cut < 0] = 0.0  # keep fitter stable

    yy, xx = np.mgrid[0:cut.shape[0], 0:cut.shape[1]]

    # initial guesses
    amp0 = float(np.max(cut))
    x0 = float(np.argmax(np.sum(cut, axis=0)))
    y0 = float(np.argmax(np.sum(cut, axis=1)))
    gamma0 = 2.0
    alpha0 = 2.5

    m0 = models.Moffat2D(amplitude=amp0, x_0=x0, y_0=y0, gamma=gamma0, alpha=alpha0)

    # bounds
    m0.amplitude.min = 0.0
    m0.gamma.min = 0.2
    m0.gamma.max = max(cut.shape)
    m0.alpha.min = 1.01
    m0.alpha.max = 20.0
    m0.x_0.min, m0.x_0.max = 0, cut.shape[1] - 1
    m0.y_0.min, m0.y_0.max = 0, cut.shape[0] - 1

    # Wing-emphasizing weights
    if amp0 <= 0 or not np.isfinite(amp0):
        weights = None
    else:
        eps = max(1e-12, 1e-2 * amp0)
        rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
        r0w = max(1e-6, float(r0_frac) * float(min(cut.shape)))
        w = (1.0 + (rr / r0w)) ** float(weight_p)
        w *= (1.0 / (cut + eps)) ** float(weight_q)
        if core_exclude_frac and core_exclude_frac > 0:
            r_ex = float(core_exclude_frac) * float(min(cut.shape))
            w[rr < r_ex] = 0.0
        if w_max and w_max > 0:
            w = np.minimum(w, float(w_max))
        # keep finite
        w[~np.isfinite(w)] = 0.0
        weights = w

    fitter = fitting.TRFLSQFitter()
    mfit = fitter(m0, xx, yy, cut, weights=weights, inplace=True)

    fwhm_px = float(mfit.fwhm) / float(zoom)

    x0_full = float(x1) + float(getattr(mfit.x_0, "value", mfit.x_0))
    y0_full = float(y1) + float(getattr(mfit.y_0, "value", mfit.y_0))
    x0_px = x0_full / float(zoom)
    y0_px = y0_full / float(zoom)
    r_px = 0.5 * float(fwhm_px)

    gamma_full = float(getattr(mfit.gamma, "value", mfit.gamma))
    alpha = float(getattr(mfit.alpha, "value", mfit.alpha))
    amp = float(getattr(mfit.amplitude, "value", mfit.amplitude))
    gamma_px = gamma_full / float(zoom)

    return mfit, {
        "fwhm_px": float(fwhm_px),
        "x0_px": float(x0_px),
        "y0_px": float(y0_px),
        "r_px": float(r_px),
        "amp": amp,
        "bkg": float(bkg),
        "gamma_px": float(gamma_px),
        "alpha": float(alpha),
        # weight params (for transparency/debug)
        "weight_q": float(weight_q),
        "weight_p": float(weight_p),
        "r0_frac": float(r0_frac),
        "w_max": float(w_max),
        "core_exclude_frac": float(core_exclude_frac),
    }

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
        dm_delay_frames: int = 1,

        # Separate NGS-based tip-tilt controller
        tt_enabled: bool = False,
        tt_gain: float = 0.25,
        tt_leak: float = 0.0,
        tt_rate_hz: float = 0.0,
        tt_delay_frames: int = 1,
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


        # --- DM servo-lag ---
        try:
            self.dm_delay_frames = max(1, int(dm_delay_frames))
        except Exception:
            self.dm_delay_frames = 1

        self.tt_enabled = bool(tt_enabled)
        # Physically consistent behaviour (and TIPTOP-like): when a dedicated NGS TT loop is enabled,
        # the high-order reconstructor must *not* try to chase global TT from LGS slopes, otherwise
        # the DM loop can wind up in an unobservable mode (NaNs/instability). Therefore we enforce
        # LGS TT stripping in the recon path whenever the TT loop is active.
        if self.tt_enabled:
            self.strip_tt_lgs = True
        self.tt_gain = float(tt_gain)
        self.tt_leak = float(tt_leak)
        self.tt_rate_hz = float(tt_rate_hz)
        try:
            self.tt_delay_frames = int(tt_delay_frames)
        except Exception:
            self.tt_delay_frames = 1
        self.tt_period_frames = 1
        if self.tt_rate_hz and self.tt_rate_hz > 0:
            try:
                self.tt_period_frames = max(1, int(round((1.0 / self.tt_rate_hz) / max(self.dt_s, 1e-12))))
            except Exception:
                self.tt_period_frames = 1

        # --- Dynamic pointing support ---
        # If any sensor provides a field_angle_fn, we must update thetas each frame and
        # cannot use the fast fixed sampler.
        self._dynamic_thetas = any(getattr(s, "field_angle_fn", None) is not None for s in (self.sensors or []))

        # Initial thetas (t=0). If dynamic, call update_field_angle once to seed.
        thetas = []
        for s in self.sensors:
            if getattr(s, "field_angle_fn", None) is not None and hasattr(s, "update_field_angle"):
                fa = s.update_field_angle(0.0, 0)
            else:
                fa = getattr(s, "field_angle", None)
                if fa is None:
                    fa = (float(getattr(s, "dx", 0.0)), float(getattr(s, "dy", 0.0)))
            thetas.append((float(fa[0]), float(fa[1])))

        ranges = [float(getattr(s, "gs_range_m", float("inf"))) for s in self.sensors]
        self._ranges_gpu = cp.asarray(ranges, dtype=cp.float32)

        # Patch size (assume simulation and sensors are consistent; fall back sensibly).
        patch_M = int(getattr(self.sim, "N", getattr(self.sensors[0], "grid_size", self.pupil.shape[0])))
        self.patch_M = patch_M
        # Fast TT remover (only used when psf_tt_mode == 'fit')
        self._tt_remover = TTPlaneRemover(self.pupil) if self.psf_tt_mode == "fit" else None
        self._thetas_gpu = cp.asarray(thetas, dtype=cp.float32)
        size_pixels = float(patch_M)

        self.sampler = None
        if (not self._dynamic_thetas) and hasattr(self.sim, "build_patch_sampler") and hasattr(self.sim, "sample_patches_from_sampler"):
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
        # Classify WFS as LGS/NGS (robust to "NGS at huge finite range" configs)
        self._wfs_is_lgs = [s.is_lgs for s in self.wfs_list]
        print(self._wfs_is_lgs)

        # Cache WFS wavelengths. ShackHartmann.measure assumes phase screens are
        # in "radians at 500 nm" and internally applies a factor (500e-9 / wl).
        # For hybrid TT we want TT coefficients in the *same units as the phase*
        # (i.e. 500 nm reference), so we may need to undo that scaling.
        self._wfs_wavelengths = [float(getattr(s, "wavelength", 500e-9)) for s in self.wfs_list]
        # Dedicated low-order (TT) reconstructor (separate from HO reconstructor)
        self._tt_recon = TipTiltReconstructor_CuPy(self.wfs_list, self._wfs_is_lgs, self._wfs_wavelengths, weight_mode=str(params.get("tt_weight_mode", "pupil")))


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
            # Use the sensor's own batching key when available (captures LGS elongation
            # and other geometry-specific settings).
            if hasattr(s, "batch_geometry_key"):
                try:
                    key = s.batch_geometry_key()
                except Exception:
                    key = None
            else:
                key = None

            if key is None:
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
        # FIFO pipeline: command computed at tick t becomes active at tick t+dm_delay_frames
        self._dm_phi_pipeline = deque([cp.zeros((patch_M, patch_M), dtype=cp.float32) for _ in range(self.dm_delay_frames)])
        self.dm_phi = self._dm_phi_pipeline[0]

        # Separate NGS-based tip-tilt loop state
        self.tt_cmd_yx = cp.zeros((2,), dtype=cp.float32)  # (sy, sx) in 500nm-reference rad/pixel
        self.tt_phi_cmd = cp.zeros((patch_M, patch_M), dtype=cp.float32)
        self.tt_phi = cp.zeros((patch_M, patch_M), dtype=cp.float32)  # active TT phase (after delay)
        # Delay line: length is (d-1) where d=1 corresponds to the inherent 1-frame control latency.
        d = max(1, int(self.tt_delay_frames))
        qlen = max(0, d - 1)
        self._tt_delay_line = deque([cp.zeros((patch_M, patch_M), dtype=cp.float32) for _ in range(qlen)])

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
        # Evaluate long-exposure PSFs (and FWHM) at additional *fixed* field points
        # for the full duration of the run (post-discard window). Angles are in
        # arcseconds on-sky: [(dx_arcsec, dy_arcsec), ...].
        eval_offaxis_arcsec: Optional[Sequence[Tuple[float, float]]] = None,
        # Write long-exposure off-axis PSFs to FITS (one file per field point).
        write_eval_fits: bool = True,
        # Also write uncorrected off-axis long-exposure PSFs.
        write_eval_uncorrected_fits: bool = True,
        # If False, skip computing/accumulating uncorrected off-axis PSFs (speed).
        eval_accumulate_uncorrected: bool = True,
    ) -> Dict[str, object]:
        n_frames = int(n_frames)
        patch_M = int(getattr(self, "patch_M", getattr(self.sensors[0], "grid_size", self.pupil.shape[0])))
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

        # PSF normalization constant (Parseval): for fixed pupil + FFT size,
        # sum(|FFT(E)|^2) is invariant to phase when |E| = pupil.
        P0 = self.pupil.astype(cp.float32, copy=False)
        p2sum = cp.sum(P0 * P0)
        psf_norm_inv = cp.float32(1.0) / cp.maximum(cp.float32(Nfft * Nfft) * p2sum, cp.float32(1e-20))

        # Convert 500 nm-reference phase screens -> science wavelength phase.
        # This matches the WFS scaling convention used elsewhere.
        phase_scale = float(500e-9 / max(lam_sci, 1e-30))
        # --- Optional: off-axis long-exposure evaluation for fixed field points ---
        ARCSEC2RAD = np.pi / (180.0 * 3600.0)
        eval_points_arcsec: List[Tuple[float, float]] = []
        if eval_offaxis_arcsec is not None:
            for p in eval_offaxis_arcsec:
                if p is None:
                    continue
                if len(p) != 2:
                    raise ValueError("eval_offaxis_arcsec items must be (dx_arcsec, dy_arcsec)")
                eval_points_arcsec.append((float(p[0]), float(p[1])))

        n_eval = int(len(eval_points_arcsec))
        eval_thetas_gpu = None
        eval_ranges_gpu = None

        # Accumulate normalized PSFs per eval direction for frames >= discard_first_frames.
        sum_psf_corr_eval = None
        sum_psf_unc_eval = None
        if n_eval > 0:
            eval_xy_rad = np.asarray(eval_points_arcsec, dtype=np.float32) * np.float32(ARCSEC2RAD)
            eval_thetas_gpu = cp.asarray(eval_xy_rad)
            eval_ranges_gpu = cp.full((n_eval,), cp.inf, dtype=cp.float32)
            sum_psf_corr_eval = cp.zeros((n_eval, psf_roi, psf_roi), dtype=cp.float64)
            if eval_accumulate_uncorrected:
                sum_psf_unc_eval = cp.zeros((n_eval, psf_roi, psf_roi), dtype=cp.float64)
        # --- Off-axis evaluation (fixed field points) performance path ---
        # We only need *long-exposure* off-axis PSFs, so we:
        #   1) sample off-axis phases using the fixed-geometry sampler when possible
        #   2) buffer phases and run batched FFTs at chunk boundaries
        # This avoids small per-frame FFT batches (which are launch/overhead bound).
        eval_sampler = None
        eval_chunk_frames = 0
        eval_buf_cor = None
        eval_buf_unc = None
        eval_buf_t = 0  # number of post-discard time frames currently buffered

        def _flush_eval_buf(n_time: int):
            return  # replaced below when n_eval > 0

        if n_eval > 0:
            # Prefer fixed-geometry sampler (no per-frame X/Y grid construction).
            if (not getattr(self, "_dynamic_thetas", False)) and hasattr(self.sim, "build_patch_sampler") and hasattr(self.sim, "sample_patches_from_sampler"):
                try:
                    eval_sampler = self.sim.build_patch_sampler(
                        thetas_xy_rad=eval_thetas_gpu,
                        ranges_m=eval_ranges_gpu,
                        size_pixels=float(patch_M),
                        M=int(patch_M),
                        angle_deg=0.0,
                        autopan_init=True,
                    )
                except Exception:
                    eval_sampler = None

            # Choose an eval batch size that stays within VRAM.
            # The limiting factor is usually the temporary FFT output for PSFs, which scales as:
            #   B * (Nfft*Nfft) complex64
            # where B = n_time * n_eval.
            # We estimate an upper bound on B from current free VRAM, leaving headroom for
            # the main on-axis chunk buffers and other simulator state.
            free_b, _total_b = cp.cuda.runtime.memGetInfo()
            # main chunk PSF phase buffers are allocated later; reserve space for them now
            reserve_b = int(2 * int(chunk_frames) * int(patch_M) * int(patch_M) * 4)  # buf_corr + buf_unc
            # plus a safety headroom (avoid going into shared VRAM on 8GB cards)
            headroom_b = int(192 * 1024 * 1024)
            free_eff = max(0, int(free_b) - int(reserve_b) - int(headroom_b))

            # bytes per PSF sample (dominant: complex FFT output) for roi path
            bytes_per_sample = int(8 * int(Nfft) * int(Nfft) + 12 * int(patch_M) * int(patch_M) + 4 * int(psf_roi) * int(psf_roi))
            # allow up to 30% of effective free memory for off-axis PSF temporaries
            budget_b = max(int(64 * 1024 * 1024), int(0.30 * float(free_eff)))
            max_B = max(1, int(budget_b) // max(1, int(bytes_per_sample)))
            max_time_fft = max(1, int(max_B) // max(1, int(n_eval)))
            eval_chunk_frames = int(min(int(chunk_frames), int(max_time_fft)))

            # Flat buffers: (eval_chunk_frames * n_eval, M, M)
            eval_buf_cor = cp.empty((eval_chunk_frames * n_eval, patch_M, patch_M), dtype=cp.float32)
            if eval_accumulate_uncorrected:
                eval_buf_unc = cp.empty((eval_chunk_frames * n_eval, patch_M, patch_M), dtype=cp.float32)

            def _flush_eval_buf(n_time: int):
                if n_time <= 0:
                    return
                # Process in time-chunks to cap peak FFT workspace.
                # Slice boundaries are multiples of n_eval so we can reshape cheaply.
                t0 = 0
                while t0 < int(n_time):
                    tstep = int(min(int(n_time) - t0, int(eval_chunk_frames)))
                    b0 = int(t0) * int(n_eval)
                    b1 = int(t0 + tstep) * int(n_eval)

                    cor_view = eval_buf_cor[b0:b1]
                    if self.psf_tt_mode == "fit" and self._tt_remover is not None:
                        self._tt_remover.remove_inplace(cor_view)

                    Icorr = _phase_to_psf_batch(
                        cor_view,
                        self.pupil,
                        fft_pad=fft_pad,
                        roi=psf_roi,
                        phase_scale=phase_scale,
                        norm_inv=psf_norm_inv,
                    )
                    Icorr = Icorr.reshape(int(tstep), int(n_eval), int(psf_roi), int(psf_roi))
                    cp.add(sum_psf_corr_eval, cp.sum(Icorr, axis=0).astype(cp.float64, copy=False), out=sum_psf_corr_eval)

                    if eval_accumulate_uncorrected and (sum_psf_unc_eval is not None) and (eval_buf_unc is not None):
                        unc_view = eval_buf_unc[b0:b1]
                        if self.psf_tt_mode == "fit" and self._tt_remover is not None:
                            self._tt_remover.remove_inplace(unc_view)

                        Iunc = _phase_to_psf_batch(
                            unc_view,
                            self.pupil,
                            fft_pad=fft_pad,
                            roi=psf_roi,
                            phase_scale=phase_scale,
                            norm_inv=psf_norm_inv,
                        )
                        Iunc = Iunc.reshape(int(tstep), int(n_eval), int(psf_roi), int(psf_roi))
                        cp.add(sum_psf_unc_eval, cp.sum(Iunc, axis=0).astype(cp.float64, copy=False), out=sum_psf_unc_eval)

                    t0 += tstep




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

            # 1b) Update dynamic field angles (if any) before sampling.
            if getattr(self, "_dynamic_thetas", False):
                t_s = float(t + 1) * float(self.dt_s)
                tick = int(t + 1)
                thetas_dyn = []
                for s in self.sensors:
                    if getattr(s, "field_angle_fn", None) is not None and hasattr(s, "update_field_angle"):
                        fa = s.update_field_angle(t_s, tick)
                    else:
                        fa = getattr(s, "field_angle", None)
                        if fa is None:
                            fa = (float(getattr(s, "dx", 0.0)), float(getattr(s, "dy", 0.0)))
                    thetas_dyn.append((float(fa[0]), float(fa[1])))
                self._thetas_gpu = cp.asarray(thetas_dyn, dtype=cp.float32)

            # 2) Sample all sensor phase patches (radians @ 500nm reference)
            if self.sampler is not None:
                phases_atm = self.sim.sample_patches_from_sampler(
                    self.sampler,
                    remove_piston=True,
                    return_gpu=True,
                )  # (n_sensors, M, M)
            else:
                phases_atm = self.sim.sample_patches_batched(
                    thetas_xy_rad=self._thetas_gpu,
                    ranges_m=self._ranges_gpu,
                    size_pixels=float(patch_M),
                    M=int(patch_M),
                    angle_deg=0.0,
                    remove_piston=True,
                    return_gpu=True,
                    return_per_layer=False,
                )
            # 3) Science residual phase (DM from previous tick is conjugate to ground)
            phi_unc = phases_atm[0]
            if self.tt_enabled:
                phi_cor = phi_unc - self.dm_phi - self.tt_phi
            else:
                phi_cor = phi_unc - self.dm_phi

            if self.psf_tt_mode == "fit" and self._tt_remover is not None:
                self._tt_remover.remove_inplace(phi_unc)
                self._tt_remover.remove_inplace(phi_cor)

            # Off-axis evaluation (fixed field points): buffer phases and process with batched FFTs.
            if n_eval > 0 and t >= discard_first_frames and (eval_buf_cor is not None):
                # Sample off-axis phase patches (radians @ 500nm reference)
                if eval_sampler is not None:
                    eval_phi_unc = self.sim.sample_patches_from_sampler(
                        eval_sampler,
                        remove_piston=True,
                        return_gpu=True,
                    )  # (n_eval,M,M)
                else:
                    eval_phi_unc = self.sim.sample_patches_batched(
                        thetas_xy_rad=eval_thetas_gpu,
                        ranges_m=eval_ranges_gpu,
                        size_pixels=float(patch_M),
                        M=int(patch_M),
                        angle_deg=0.0,
                        remove_piston=True,
                        return_gpu=True,
                        return_per_layer=False,
                    )

                j0 = int(eval_buf_t) * int(n_eval)
                j1 = j0 + int(n_eval)

                # Uncorrected (optional)
                if eval_accumulate_uncorrected and (eval_buf_unc is not None):
                    cp.copyto(eval_buf_unc[j0:j1], eval_phi_unc)

                # Corrected: subtract current DM and global TT (apply to all eval points)
                if self.tt_enabled:
                    cp.subtract(eval_phi_unc, (self.dm_phi + self.tt_phi)[None, :, :], out=eval_buf_cor[j0:j1])
                else:
                    cp.subtract(eval_phi_unc, self.dm_phi[None, :, :], out=eval_buf_cor[j0:j1])

                eval_buf_t += 1
                if eval_buf_t == eval_chunk_frames:
                    _flush_eval_buf(eval_buf_t)
                    eval_buf_t = 0

            # 4) Measure slopes on WFS phases (batched by geometry) and pack a slope-error vector.
            # Reuse buffers to avoid per-frame allocations.
            cp.copyto(wfs_phase_buf, phases_atm[1:])
            wfs_phase_buf -= self.dm_phi[None, :, :]
            if self.tt_enabled:
                wfs_phase_buf -= self.tt_phi[None, :, :]

            if self.psf_tt_mode == "hybrid":
                tt_acc.fill(0.0)
                tt_n = 0

            # NGS are ignored for recon; leave zeros in their blocks.
            self._slopes_err_vec.fill(0.0)

            # TT controller update uses NGS mean slopes at its own rate.
            do_tt_update = bool(
                self.tt_enabled
                and (self.tt_period_frames > 0)
                and ((t % int(self.tt_period_frames)) == 0)
            )
            if do_tt_update:
                tt_ctrl_acc = cp.zeros((2,), dtype=cp.float32)  # (sy, sx)
                tt_ctrl_n = 0

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

                    # Separate TT loop: use NGS mean slopes (before LGS TT stripping / NGS ignore for recon)
                    if do_tt_update and (not self._wfs_is_lgs[wi]):
                        wl = cp.float32(self._wfs_wavelengths[wi])
                        tt_ctrl_acc += self._tt_recon._weighted_mean_yx(sl_err, wi) * (wl / cp.float32(500e-9))
                        tt_ctrl_n += 1

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
                        if getattr(self, "_tt_recon", None) is not None:
                            # Strip TT using the LO reconstructor projector (weighted mean)
                            mu = self._tt_recon._weighted_mean_yx(sl_err, wi)[None, :]
                            sl_err = sl_err - mu
                        else:
                            sl_err = sl_err - cp.mean(sl_err, axis=0, keepdims=True)

                    na = self._n_active[wi]
                    if na <= 0:
                        continue
                    a, _ = self._slices[wi]
                    self._slopes_err_vec[a : a + na] = sl_err[:, 1]
                    self._slopes_err_vec[a + na : a + 2 * na] = sl_err[:, 0]

            # ---- TT controller update (NGS mean slopes) ----
            if self.tt_enabled and do_tt_update:
                if tt_ctrl_n > 0:
                    tt_meas_yx = tt_ctrl_acc / cp.float32(tt_ctrl_n)
                else:
                    tt_meas_yx = cp.zeros((2,), dtype=cp.float32)

                # Guard against NaNs/Infs (e.g., if a WFS measurement returns non-finite slopes)
                try:
                    if not bool(cp.all(cp.isfinite(tt_meas_yx)).get()):
                        tt_meas_yx = cp.zeros((2,), dtype=cp.float32)
                except Exception:
                    tt_meas_yx = cp.zeros((2,), dtype=cp.float32)

                # Integrator with optional leak
                self.tt_cmd_yx = (1.0 - cp.float32(self.tt_leak)) * self.tt_cmd_yx - cp.float32(self.tt_gain) * tt_meas_yx

                # If the controller state ever goes non-finite, reset to zero rather than poisoning the sim.
                try:
                    if not bool(cp.all(cp.isfinite(self.tt_cmd_yx)).get()):
                        self.tt_cmd_yx = cp.zeros((2,), dtype=cp.float32)
                except Exception:
                    self.tt_cmd_yx = cp.zeros((2,), dtype=cp.float32)

                # Convert command to pupil-masked phase ramp (500nm-ref radians)
                phi = (self.tt_cmd_yx[0] * self._yy + self.tt_cmd_yx[1] * self._xx)
                phi = phi * self.pupil
                mu = cp.sum(phi) / cp.maximum(cp.sum(self.pupil), cp.float32(1e-20))
                self.tt_phi_cmd = (phi - mu) * self.pupil

            # Advance TT delay line once per simulation tick
            if self.tt_enabled:
                if len(self._tt_delay_line) == 0:
                    self.tt_phi = self.tt_phi_cmd
                else:
                    self._tt_delay_line.append(self.tt_phi_cmd)
                    self.tt_phi = self._tt_delay_line.popleft()
            else:
                self.tt_phi.fill(0.0)

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
            dm_phi_new = -self.R.coeffs_to_onaxis_phase(self.dm_cmd)
            dm_phi_new = (dm_phi_new - cp.mean(dm_phi_new)) * self.pupil

            # If PSF TT mode is "fit", remove any residual TT from the DM command phase before it enters the delay line.
            if self.psf_tt_mode == "fit" and self._tt_remover is not None:
                self._tt_remover.remove_inplace(dm_phi_new)

            # DM servo-lag FIFO: command computed at tick t becomes active at tick t+dm_delay_frames.
            if getattr(self, "_dm_phi_pipeline", None) is None:
                self._dm_phi_pipeline = deque([cp.zeros_like(dm_phi_new) for _ in range(max(1, int(getattr(self, "dm_delay_frames", 1))))])
            self._dm_phi_pipeline.append(dm_phi_new)
            if len(self._dm_phi_pipeline) > int(getattr(self, "dm_delay_frames", 1)):
                self._dm_phi_pipeline.popleft()
            self.dm_phi = self._dm_phi_pipeline[0]

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
                            norm_inv=psf_norm_inv,
                        )
                        Iunc = _phase_to_psf_batch(
                            buf_unc[:B],
                            self.pupil,
                            fft_pad=fft_pad,
                            roi=psf_roi,
                            phase_scale=phase_scale,
                            norm_inv=psf_norm_inv,
                        )
                    else:
                        # Stats-only: compute only the used tail of the chunk.
                        Icorr = _phase_to_psf_batch(
                            buf_corr[off_used:B],
                            self.pupil,
                            fft_pad=fft_pad,
                            roi=psf_roi,
                            phase_scale=phase_scale,
                            norm_inv=psf_norm_inv,
                        )
                        Iunc = _phase_to_psf_batch(
                            buf_unc[off_used:B],
                            self.pupil,
                            fft_pad=fft_pad,
                            roi=psf_roi,
                            phase_scale=phase_scale,
                            norm_inv=psf_norm_inv,
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
                            norm_inv=psf_norm_inv,
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
                            norm_inv=psf_norm_inv,
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



        # Flush any remaining off-axis evaluation phases (post-discard).
        if n_eval > 0 and (eval_buf_cor is not None) and eval_buf_t > 0:
            _flush_eval_buf(eval_buf_t)
            eval_buf_t = 0

        if self.psf_tt_mode == "hybrid" and tt_series is not None:
            np.save(os.path.join(out_dir, "tt_yx_from_wfs.npy"), cp.asnumpy(tt_series[:n_done]))

        # Long-exposure PSFs (mean over frames >= discard_first_frames)
        denom_used = float(max(int(n_used), 1))
        psf_long_corr = (sum_psf_corr / denom_used).astype(cp.float32)
        psf_long_unc = (sum_psf_unc / denom_used).astype(cp.float32)

        # Optional: off-axis long-exposure PSFs
        psf_long_corr_eval = None
        psf_long_unc_eval = None
        if n_eval > 0 and sum_psf_corr_eval is not None:
            psf_long_corr_eval = (sum_psf_corr_eval / denom_used).astype(cp.float32)
            if sum_psf_unc_eval is not None:
                psf_long_unc_eval = (sum_psf_unc_eval / denom_used).astype(cp.float32)

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


        # Off-axis long-exposure PSFs: one FITS file per field point (corrected
        # and optionally uncorrected).
        eval_long_corr_fits: List[str] = []
        eval_long_unc_fits: List[str] = []
        if write_eval_fits and psf_long_corr_eval is not None and int(n_used) > 0:
            def _tag(x: float) -> str:
                # Safe token for filenames: -1.25 -> m1p25, 0.0 -> 0p00, 10 -> 10p00
                s = f"{x:+.3f}"
                s = s.replace("+", "")
                s = s.replace("-", "m")
                s = s.replace(".", "p")
                return s

            for k, (dx_as, dy_as) in enumerate(eval_points_arcsec):
                hdr0 = {
                    "DT": float(dt_s),
                    "LAMSCI": float(lam_sci),
                    "ROI": int(psf_roi),
                    "USED": int(n_used),
                    "DXAS": float(dx_as),
                    "DYAS": float(dy_as),
                }
                fbase = f"psf_long_corrected_offaxis_{k:03d}_dx{_tag(dx_as)}_dy{_tag(dy_as)}.fits"
                fpath = os.path.join(out_dir, fbase)
                with open(fpath, "wb") as f:
                    write_primary_empty(f, hdr0)
                    append_image_hdu(f, cp.asnumpy(psf_long_corr_eval[k]), extname="LONGEXP", extver=1)
                eval_long_corr_fits.append(fpath)

                if write_eval_uncorrected_fits and psf_long_unc_eval is not None:
                    fbase = f"psf_long_uncorrected_offaxis_{k:03d}_dx{_tag(dx_as)}_dy{_tag(dy_as)}.fits"
                    fpath_u = os.path.join(out_dir, fbase)
                    with open(fpath_u, "wb") as f:
                        write_primary_empty(f, hdr0)
                        append_image_hdu(f, cp.asnumpy(psf_long_unc_eval[k]), extname="LONGEXP", extver=1)
                    eval_long_unc_fits.append(fpath_u)

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
        moffw_corr = {"fwhm_px": float("nan"), "x0_px": float("nan"), "y0_px": float("nan"), "r_px": float("nan")}
        moffw_unc  = {"fwhm_px": float("nan"), "x0_px": float("nan"), "y0_px": float("nan"), "r_px": float("nan")}

        Icorr_moffat = float("nan")
        Iunc_moffat = float("nan")

        Icorr_moffat_wings = float("nan")
        Iunc_moffat_wings = float("nan")

        fwhm_px_corr = float("nan")
        fwhm_px_unc = float("nan")

        # Contour FWHM (half-maximum region; ellipse approximation). Values are in arcsec.
        fwhm_arcsec_corrected_contour = float("nan")
        fwhm_arcsec_uncorrected_contour = float("nan")
        fwhm_arcsec_corrected_contour_major = float("nan")
        fwhm_arcsec_corrected_contour_minor = float("nan")
        fwhm_arcsec_uncorrected_contour_major = float("nan")
        fwhm_arcsec_uncorrected_contour_minor = float("nan")
        fwhm_contour_corrected_ratio = float("nan")
        fwhm_contour_uncorrected_ratio = float("nan")
        fwhm_contour_corrected_theta_deg = float("nan")
        fwhm_contour_uncorrected_theta_deg = float("nan")

        # Ellipse fits for contour method (native PSF pixels; for UI overlays)
        contour_corr = None
        contour_unc = None

        if int(n_used) > 0:
            Icorr_zoom = Analysis._zoom2d(psf_long_corr, zoom, cp)
            Iunc_zoom = Analysis._zoom2d(psf_long_unc, zoom, cp)

            # Contour FWHM (arcsec) from the half-maximum region.
            # Use the native-resolution PSF but upsample internally by rebin=zoom to reduce discretization.
            pixelScale_as_per_pix = float(plate_rad) * (180.0 / np.pi) * 3600.0
            try:
                cmaj, cmin, cr, ct, cell = Analysis.fwhm_contour(
                    psf_long_corr,
                    pixelScale_as_per_pix,
                    rebin=zoom,
                    return_ellipse=True,
                )
                fwhm_arcsec_corrected_contour_major = float(cmaj)
                fwhm_arcsec_corrected_contour_minor = float(cmin)
                fwhm_contour_corrected_ratio = float(cr)
                fwhm_contour_corrected_theta_deg = float(ct)
                contour_corr = cell
                if np.isfinite(fwhm_arcsec_corrected_contour_major) and np.isfinite(fwhm_arcsec_corrected_contour_minor):
                    fwhm_arcsec_corrected_contour = float(np.sqrt(fwhm_arcsec_corrected_contour_major * fwhm_arcsec_corrected_contour_minor))
            except Exception:
                pass
            try:
                cmaj, cmin, cr, ct, cell = Analysis.fwhm_contour(
                    psf_long_unc,
                    pixelScale_as_per_pix,
                    rebin=zoom,
                    return_ellipse=True,
                )
                fwhm_arcsec_uncorrected_contour_major = float(cmaj)
                fwhm_arcsec_uncorrected_contour_minor = float(cmin)
                fwhm_contour_uncorrected_ratio = float(cr)
                fwhm_contour_uncorrected_theta_deg = float(ct)
                contour_unc = cell
                if np.isfinite(fwhm_arcsec_uncorrected_contour_major) and np.isfinite(fwhm_arcsec_uncorrected_contour_minor):
                    fwhm_arcsec_uncorrected_contour = float(np.sqrt(fwhm_arcsec_uncorrected_contour_major * fwhm_arcsec_uncorrected_contour_minor))
            except Exception:
                pass

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
                "amp": float(_p[0].get()),
                "sigma_px": float(_p[3].get()) / float(zoom),
                "bkg": float(_p[4].get()),
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
                "amp": float(_p[0].get()),
                "sigma_px": float(_p[3].get()) / float(zoom),
                "bkg": float(_p[4].get()),
            }

            # Moffat (CPU on zoomed cutout)
            # Moffat (CPU on zoomed cutout)
            _mfit, moff_corr = fit_moffat2d(Icorr_zoom.get(), zoom=float(zoom))
            _mfit, moff_unc  = fit_moffat2d(Iunc_zoom.get(), zoom=float(zoom))
            _mfit, moffw_corr = fit_moffat2d_wings(Icorr_zoom.get(), zoom=float(zoom))
            _mfit, moffw_unc  = fit_moffat2d_wings(Iunc_zoom.get(), zoom=float(zoom))
            Icorr_moffat = float(moff_corr.get("fwhm_px", float("nan")))
            Iunc_moffat  = float(moff_unc.get("fwhm_px", float("nan")))
            Icorr_moffat_wings = float(moffw_corr.get("fwhm_px", float("nan")))
            Iunc_moffat_wings  = float(moffw_unc.get("fwhm_px", float("nan")))

        # Off-axis FWHM evaluation (full-duration long-exposure, post-discard)
        offaxis_metrics: List[Dict[str, Any]] = []
        if n_eval > 0 and int(n_used) > 0 and psf_long_corr_eval is not None:
            for k, (dx_as, dy_as) in enumerate(eval_points_arcsec):
                # Corrected
                Icor_k = psf_long_corr_eval[k]
                Icor_k_zoom = Analysis._zoom2d(Icor_k, zoom, cp)
                _p_c, (xc_z, yc_z), (fwhm_eq_native, _), *_rest = fitter.fit_frame(
                    Icor_k_zoom, search_r=24, loss="soft_l1", max_iter=25, zoom=zoom
                )
                fwhm_px_c = float(fwhm_eq_native.get())
                gauss_c = {
                    "fwhm_px": float(fwhm_px_c),
                    "x0_px": float(xc_z) / float(zoom),
                    "y0_px": float(yc_z) / float(zoom),
                    "r_px": 0.5 * float(fwhm_px_c),
                    "amp": float(_p_c[0].get()),
                    "sigma_px": float(_p_c[3].get()) / float(zoom),
                    "bkg": float(_p_c[4].get()),
                }
                _mfit, moff_c = fit_moffat2d(Icor_k_zoom.get(), zoom=float(zoom))
                fwhm_px_c_m = float(moff_c.get("fwhm_px", float("nan")))
                _mfit, moffw_c = fit_moffat2d_wings(Icor_k_zoom.get(), zoom=float(zoom))
                fwhm_px_c_mw = float(moffw_c.get("fwhm_px", float("nan")))


                # Contour FWHM (arcsec) for this off-axis point
                fwhm_as_c_contour = float("nan")
                fwhm_as_c_contour_major = float("nan")
                fwhm_as_c_contour_minor = float("nan")
                fwhm_c_contour_ratio = float("nan")
                fwhm_c_contour_theta_deg = float("nan")
                contour_c = None
                try:
                    cmaj, cmin, cr, ct, cell = Analysis.fwhm_contour(
                        Icor_k, pixelScale_as_per_pix, rebin=zoom, return_ellipse=True
                    )
                    fwhm_as_c_contour_major = float(cmaj)
                    fwhm_as_c_contour_minor = float(cmin)
                    fwhm_c_contour_ratio = float(cr)
                    fwhm_c_contour_theta_deg = float(ct)
                    contour_c = cell
                    if np.isfinite(fwhm_as_c_contour_major) and np.isfinite(fwhm_as_c_contour_minor):
                        fwhm_as_c_contour = float(np.sqrt(fwhm_as_c_contour_major * fwhm_as_c_contour_minor))
                except Exception:
                    pass

                # Uncorrected (optional; computed if we also accumulated it)
                fwhm_px_u = float("nan")
                fwhm_px_u_m = float("nan")
                fwhm_px_u_mw = float("nan")
                fwhm_as_u_contour = float("nan")
                fwhm_as_u_contour_major = float("nan")
                fwhm_as_u_contour_minor = float("nan")
                fwhm_u_contour_ratio = float("nan")
                fwhm_u_contour_theta_deg = float("nan")
                contour_u = None
                gauss_u: Optional[Dict[str, Any]] = None
                moff_u: Optional[Dict[str, Any]] = None
                moffw_u: Optional[Dict[str, Any]] = None
                contour_u = None
                if psf_long_unc_eval is not None:
                    Iunc_k = psf_long_unc_eval[k]
                    Iunc_k_zoom = Analysis._zoom2d(Iunc_k, zoom, cp)
                    _p_u, (xc_z_u, yc_z_u), (fwhm_eq_native_u, _), *_rest = fitter.fit_frame(
                        Iunc_k_zoom, search_r=24, loss="soft_l1", max_iter=25, zoom=zoom
                    )
                    fwhm_px_u = float(fwhm_eq_native_u.get())
                    gauss_u = {
                        "fwhm_px": float(fwhm_px_u),
                        "x0_px": float(xc_z_u) / float(zoom),
                        "y0_px": float(yc_z_u) / float(zoom),
                        "r_px": 0.5 * float(fwhm_px_u),
                        "amp": float(_p_u[0].get()),
                        "sigma_px": float(_p_u[3].get()) / float(zoom),
                        "bkg": float(_p_u[4].get()),
                    }
                    _mfit, moff_u = fit_moffat2d(Iunc_k_zoom.get(), zoom=float(zoom))
                    fwhm_px_u_m = float(moff_u.get("fwhm_px", float("nan")))
                    _mfit, moffw_u = fit_moffat2d_wings(Iunc_k_zoom.get(), zoom=float(zoom))
                    fwhm_px_u_mw = float(moffw_u.get("fwhm_px", float("nan")))


                    # Contour FWHM (arcsec) for this off-axis point (uncorrected)
                    fwhm_as_u_contour = float("nan")
                    fwhm_as_u_contour_major = float("nan")
                    fwhm_as_u_contour_minor = float("nan")
                    fwhm_u_contour_ratio = float("nan")
                    fwhm_u_contour_theta_deg = float("nan")
                    try:
                        cmaj, cmin, cr, ct, cell = Analysis.fwhm_contour(
                            Iunc_k, pixelScale_as_per_pix, rebin=zoom, return_ellipse=True
                        )
                        fwhm_as_u_contour_major = float(cmaj)
                        fwhm_as_u_contour_minor = float(cmin)
                        fwhm_u_contour_ratio = float(cr)
                        fwhm_u_contour_theta_deg = float(ct)
                        contour_u = cell
                        if np.isfinite(fwhm_as_u_contour_major) and np.isfinite(fwhm_as_u_contour_minor):
                            fwhm_as_u_contour = float(np.sqrt(fwhm_as_u_contour_major * fwhm_as_u_contour_minor))
                    except Exception:
                        pass

                # Convert to arcsec for convenience
                fwhm_as_c = fwhm_px_c * plate_rad * (180.0 / np.pi) * 3600.0
                fwhm_as_c_m = fwhm_px_c_m * plate_rad * (180.0 / np.pi) * 3600.0
                fwhm_as_u = fwhm_px_u * plate_rad * (180.0 / np.pi) * 3600.0
                fwhm_as_c_mw = fwhm_px_c_mw * plate_rad * (180.0 / np.pi) * 3600.0
                fwhm_as_u_m = fwhm_px_u_m * plate_rad * (180.0 / np.pi) * 3600.0
                fwhm_as_u_mw = fwhm_px_u_mw * plate_rad * (180.0 / np.pi) * 3600.0

                offaxis_metrics.append(
                    {
                        "dx_arcsec": float(dx_as),
                        "dy_arcsec": float(dy_as),
                        "fwhm_arcsec_corrected": float(fwhm_as_c),
                        "fwhm_arcsec_corrected_moffat": float(fwhm_as_c_m),
                        "fwhm_arcsec_corrected_moffat_wings": float(fwhm_as_c_mw),
                        "fwhm_arcsec_uncorrected": float(fwhm_as_u),
                        "fwhm_arcsec_uncorrected_moffat": float(fwhm_as_u_m),
                        "fwhm_arcsec_uncorrected_moffat_wings": float(fwhm_as_u_mw),
                        "fwhm_arcsec_corrected_contour": float(fwhm_as_c_contour),
                        "fwhm_arcsec_corrected_contour_major": float(fwhm_as_c_contour_major),
                        "fwhm_arcsec_corrected_contour_minor": float(fwhm_as_c_contour_minor),
                        "fwhm_contour_corrected_ratio": float(fwhm_c_contour_ratio),
                        "fwhm_contour_corrected_theta_deg": float(fwhm_c_contour_theta_deg),
                        "fwhm_arcsec_uncorrected_contour": float(fwhm_as_u_contour),
                        "fwhm_arcsec_uncorrected_contour_major": float(fwhm_as_u_contour_major),
                        "fwhm_arcsec_uncorrected_contour_minor": float(fwhm_as_u_contour_minor),
                        "fwhm_contour_uncorrected_ratio": float(fwhm_u_contour_ratio),
                        "fwhm_contour_uncorrected_theta_deg": float(fwhm_u_contour_theta_deg),
                        "fwhm_px_corrected": float(fwhm_px_c),
                        "fwhm_px_corrected_moffat": float(fwhm_px_c_m),
                        "fwhm_px_corrected_moffat_wings": float(fwhm_px_c_mw),
                        "fwhm_px_uncorrected": float(fwhm_px_u),
                        "fwhm_px_uncorrected_moffat": float(fwhm_px_u_m),
                        "fwhm_px_uncorrected_moffat_wings": float(fwhm_px_u_mw),
                        # Fit metadata for overlays (native PSF pixels)
                        "gauss_corrected": gauss_c,
                        "moffat_corrected": moff_c,
                        "moffat_wings_corrected": moffw_c,
                        "gauss_uncorrected": gauss_u,
                        "moffat_uncorrected": moff_u,
                        "moffat_wings_uncorrected": moffw_u,
                        "contour_corrected": contour_c,
                        "contour_uncorrected": contour_u,
                    }
                )

        # Off-axis summary stats (Gaussian/Moffat, corrected/uncorrected)
        offaxis_stats: Dict[str, Any] = {}
        if len(offaxis_metrics) > 0:
            def _stats(values: list[float]) -> Dict[str, float]:
                v = np.asarray(values, dtype=float)
                v = v[np.isfinite(v)]
                if v.size == 0:
                    return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "median": float("nan")}
                return {
                    "n": int(v.size),
                    "mean": float(np.mean(v)),
                    "std": float(np.std(v, ddof=0)),
                    "min": float(np.min(v)),
                    "max": float(np.max(v)),
                    "median": float(np.median(v)),
                }

            offaxis_stats["gauss_corrected_arcsec"] = _stats([m.get("fwhm_arcsec_corrected", float("nan")) for m in offaxis_metrics])
            offaxis_stats["moffat_corrected_arcsec"] = _stats([m.get("fwhm_arcsec_corrected_moffat", float("nan")) for m in offaxis_metrics])
            offaxis_stats["moffat_wings_corrected_arcsec"] = _stats([m.get("fwhm_arcsec_corrected_moffat_wings", float("nan")) for m in offaxis_metrics])
            offaxis_stats["contour_corrected_arcsec"] = _stats([m.get("fwhm_arcsec_corrected_contour", float("nan")) for m in offaxis_metrics])

            # Uncorrected only if present (not skipped)
            offaxis_stats["gauss_uncorrected_arcsec"] = _stats([m.get("fwhm_arcsec_uncorrected", float("nan")) for m in offaxis_metrics])
            offaxis_stats["moffat_uncorrected_arcsec"] = _stats([m.get("fwhm_arcsec_uncorrected_moffat", float("nan")) for m in offaxis_metrics])
            offaxis_stats["moffat_wings_uncorrected_arcsec"] = _stats([m.get("fwhm_arcsec_uncorrected_moffat_wings", float("nan")) for m in offaxis_metrics])
            offaxis_stats["contour_uncorrected_arcsec"] = _stats([m.get("fwhm_arcsec_uncorrected_contour", float("nan")) for m in offaxis_metrics])

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
            "fwhm_rad_corrected_moffat_wings": Icorr_moffat_wings * plate_rad,
            "fwhm_rad_uncorrected_moffat_wings": Iunc_moffat_wings * plate_rad,
            "fwhm_rad_corrected_contour": float(fwhm_arcsec_corrected_contour) * (np.pi / (180.0 * 3600.0)),
            "fwhm_rad_uncorrected_contour": float(fwhm_arcsec_uncorrected_contour) * (np.pi / (180.0 * 3600.0)),
            "fwhm_arcsec_corrected_contour": float(fwhm_arcsec_corrected_contour),
            "fwhm_arcsec_uncorrected_contour": float(fwhm_arcsec_uncorrected_contour),
            "fwhm_arcsec_corrected_contour_major": float(fwhm_arcsec_corrected_contour_major),
            "fwhm_arcsec_corrected_contour_minor": float(fwhm_arcsec_corrected_contour_minor),
            "fwhm_arcsec_uncorrected_contour_major": float(fwhm_arcsec_uncorrected_contour_major),
            "fwhm_arcsec_uncorrected_contour_minor": float(fwhm_arcsec_uncorrected_contour_minor),
            "fwhm_contour_corrected_ratio": float(fwhm_contour_corrected_ratio),
            "fwhm_contour_uncorrected_ratio": float(fwhm_contour_uncorrected_ratio),
            "fwhm_contour_corrected_theta_deg": float(fwhm_contour_corrected_theta_deg),
            "fwhm_contour_uncorrected_theta_deg": float(fwhm_contour_uncorrected_theta_deg),
            "fwhm_arcsec_corrected_moffat_wings": Icorr_moffat_wings * plate_rad * (180.0 / np.pi) * 3600.0,
            "fwhm_arcsec_uncorrected_moffat_wings": Iunc_moffat_wings * plate_rad * (180.0 / np.pi) * 3600.0,
            # Fit metadata for overlays (native PSF pixels)
            "gauss_corrected": gauss_corr,
            "gauss_uncorrected": gauss_unc,
            "moffat_corrected": moff_corr,
            "moffat_uncorrected": moff_unc,
            "moffat_wings_corrected": moffw_corr,
            "moffat_wings_uncorrected": moffw_unc,
            "contour_corrected": contour_corr,
            "contour_uncorrected": contour_unc,
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
        # Off-axis evaluation outputs
        if n_eval > 0:
            out["eval_offaxis_arcsec"] = list(eval_points_arcsec)
            out["eval_accumulate_uncorrected"] = bool(eval_accumulate_uncorrected)
            out["offaxis_metrics"] = offaxis_metrics
            out["offaxis_stats"] = offaxis_stats
            out["psf_long_corrected_offaxis"] = psf_long_corr_eval
            out["psf_long_corrected_offaxis_fits"] = eval_long_corr_fits
            if psf_long_unc_eval is not None:
                out["psf_long_uncorrected_offaxis"] = psf_long_unc_eval
                out["psf_long_uncorrected_offaxis_fits"] = eval_long_unc_fits

        return out
