#!/usr/bin/env python3
"""
Validate PSF/OTF impact of covariance models for GLAO PSF-R.

This script converts residual modal covariance models into Monte-Carlo
long-exposure PSFs using the saved modal basis maps.

It compares PSFs from:

  1. truth residual covariance
  2. telemetry-only
  3. regularization-aware block model
  4. H telemetry/POL
  5. H telemetry/POL with scalar trusted calibration
  6. H oracle diagonal transfer

Important requirement
---------------------
This step needs the spatial modal basis saved in control_bundle.npz.

If the script exits with "No basis maps found", rerun the headless simulation with:

    "control_bundle_include_basis": True

in the run configuration.

Typical use
-----------
RUN_DIR=$(ls -td experiments/ultimate_wfi_psf_reco/outputs/telemetry_truth_10s/longrun_* | head -1)
export RUN_DIR

python experiments/ultimate_wfi_psf_reco/validate_psf_from_covariance_models.py \
    "$RUN_DIR" \
    --ground-altitude-max-m 500 \
    --gain-floor 0.12 \
    --use-gram-ls-truth \
    --gram-rcond 1e-6 \
    --target total \
    --n-mc 128 \
    --fft-size 1024 \
    --alpha auto \
    --outdir "$RUN_DIR/psf_from_covariance_models" \
    --verbose

Notes
-----
- The default reference is the Monte-Carlo PSF from the truth residual covariance.
- If you pass --reference-psf-fits, the script also compares to that FITS PSF.
- The default assumes basis maps and modal coefficients already combine into phase
  in radians. If the basis is OPD in metres, use --basis-units opd_m and provide
  --wavelength-m.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt


def load_json(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)


def first_present(npz: np.lib.npyio.NpzFile, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in npz.files:
            return name
    return None


def finite_clip(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def covariance(x: np.ndarray) -> np.ndarray:
    x = finite_clip(x)
    x = x - np.mean(x, axis=0, keepdims=True)
    n = x.shape[0]
    if n < 2:
        raise ValueError("Need at least two samples")
    C = (x.T @ x) / float(n - 1)
    return 0.5 * (C + C.T)


def covariance_in_basis(x: np.ndarray, V: np.ndarray) -> np.ndarray:
    return covariance(finite_clip(x) @ finite_clip(V))


def rel_fro(C_true: np.ndarray, C_est: np.ndarray) -> float:
    den = np.linalg.norm(C_true)
    return float(np.linalg.norm(C_true - C_est) / den) if den > 0 else float("nan")


def rel_l2_image(ref: np.ndarray, img: np.ndarray) -> float:
    ref = normalize_psf(ref)
    img = normalize_psf(img)
    den = np.linalg.norm(ref)
    return float(np.linalg.norm(img - ref) / den) if den > 0 else float("nan")


def l1_image(ref: np.ndarray, img: np.ndarray) -> float:
    ref = normalize_psf(ref)
    img = normalize_psf(img)
    return float(np.sum(np.abs(img - ref)))


def peak_rel_error(ref: np.ndarray, img: np.ndarray) -> float:
    ref = normalize_psf(ref)
    img = normalize_psf(img)
    p = float(np.max(ref))
    return float((np.max(img) - p) / p) if p > 0 else float("nan")


def normalize_psf(psf: np.ndarray) -> np.ndarray:
    psf = np.nan_to_num(np.asarray(psf, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    psf = np.maximum(psf, 0.0)
    s = float(np.sum(psf))
    return psf / s if s > 0 else psf


def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    img = np.asarray(img)
    h, w = img.shape[-2], img.shape[-1]
    size = int(min(size, h, w))
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return img[..., y0:y0+size, x0:x0+size]


def radial_profile(img: np.ndarray, nbins: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    img = normalize_psf(img)
    h, w = img.shape
    y, x = np.indices((h, w))
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    r = np.sqrt((y - cy)**2 + (x - cx)**2)
    if nbins is None:
        nbins = int(np.floor(np.max(r))) + 1
    edges = np.linspace(0, np.max(r), nbins + 1)
    prof = np.zeros(nbins)
    rr = 0.5 * (edges[:-1] + edges[1:])
    for i in range(nbins):
        m = (r >= edges[i]) & (r < edges[i+1])
        prof[i] = np.mean(img[m]) if np.any(m) else np.nan
    return rr, prof


def gram_ls_transform_coeffs(
    coeff_diag: np.ndarray,
    G: np.ndarray,
    rcond: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    C = finite_clip(coeff_diag)
    G = finite_clip(G)
    G = 0.5 * (G + G.T)
    diag = np.diag(G)
    b = C * diag[None, :]

    evals, evecs = np.linalg.eigh(G)
    emax = float(np.max(evals)) if evals.size else 0.0
    keep = evals > float(rcond) * emax
    if not np.any(keep):
        raise ValueError("No Gram eigenmodes kept; reduce --gram-rcond")

    inv = np.zeros_like(evals)
    inv[keep] = 1.0 / evals[keep]
    G_pinv = (evecs * inv[None, :]) @ evecs.T
    out = b @ G_pinv.T
    info = {
        "truth_projection": "Gram-LS",
        "gram_rcond": float(rcond),
        "gram_eig_min": float(np.min(evals)),
        "gram_eig_max": emax,
        "gram_n_eig": int(len(evals)),
        "gram_n_kept": int(np.sum(keep)),
    }
    return out, info


def maybe_gram_ls(
    arrays: dict[str, np.ndarray],
    control: np.lib.npyio.NpzFile,
    use_gram: bool,
    rcond: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not use_gram:
        return arrays, {"truth_projection": "diagonal_inner_product_on_dm_basis"}
    if "basis_gram" not in control.files:
        raise KeyError("control_bundle.npz missing basis_gram; rerun with control_bundle_include_gram=True")

    G = finite_clip(control["basis_gram"])
    out = {}
    info_total = None
    for key, arr in arrays.items():
        out[key], info = gram_ls_transform_coeffs(arr, G, rcond)
        if info_total is None:
            info_total = info
    return out, info_total or {"truth_projection": "Gram-LS"}


def collapse_layers(
    layers: np.ndarray,
    telemetry_summary: dict[str, Any],
    ground_altitude_max_m: float,
    ground_layer_indices: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    layers = finite_clip(layers)
    n_layer = layers.shape[1]

    names = telemetry_summary.get("layer_name")
    if not isinstance(names, list) or len(names) != n_layer:
        names = [f"L{i}" for i in range(n_layer)]
    names = [str(x) for x in names]

    alt = telemetry_summary.get("layer_altitude_m")
    if isinstance(alt, list) and len(alt) == n_layer:
        alt_m = np.asarray(alt, dtype=float)
    else:
        alt_m = np.full(n_layer, np.nan)

    if ground_layer_indices:
        ground_idx = np.asarray([int(p.strip()) for p in ground_layer_indices.split(",") if p.strip()], dtype=int)
        ground_idx = ground_idx[(ground_idx >= 0) & (ground_idx < n_layer)]
    elif np.all(np.isfinite(alt_m)):
        ground_idx = np.where(alt_m <= float(ground_altitude_max_m))[0].astype(int)
    else:
        ground_idx = np.asarray([0], dtype=int)

    ground_set = set(ground_idx.tolist())
    free_idx = np.asarray([i for i in range(n_layer) if i not in ground_set], dtype=int)

    ground = np.sum(layers[:, ground_idx, :], axis=1) if ground_idx.size else np.zeros((layers.shape[0], layers.shape[2]))
    free = np.sum(layers[:, free_idx, :], axis=1) if free_idx.size else np.zeros((layers.shape[0], layers.shape[2]))

    labels = []
    for i in range(n_layer):
        if np.isfinite(alt_m[i]):
            labels.append(f"{names[i]} ({alt_m[i]/1000:.2f} km)")
        else:
            labels.append(names[i])
    return ground, free, ground_idx, free_idx, labels


def control_basis(control: np.lib.npyio.NpzFile, gain_floor: float, n_modes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    T_name = first_present(control, ("T_mode_to_mode", "T_mode_to_mode_pre_waffle"))
    if T_name is None:
        raise KeyError("control_bundle.npz missing T_mode_to_mode")
    T = finite_clip(control[T_name])[:n_modes, :n_modes]
    _, s, Vt = np.linalg.svd(T, full_matrices=True)
    V = Vt.T
    trusted = s >= float(gain_floor)
    return V, s, trusted, T_name


def safe_sqrt_ratio(num: np.ndarray, den: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    num = np.maximum(finite_clip(num), 0.0)
    den = np.maximum(finite_clip(den), floor)
    h = np.sqrt(num / den)
    h[~np.isfinite(h)] = 0.0
    return h


def apply_diag_transfer(C_prior: np.ndarray, h: np.ndarray) -> np.ndarray:
    h = finite_clip(h)
    C = (h[:, None] * C_prior) * h[None, :]
    return 0.5 * (C + C.T)


def make_block_model(C_xhat: np.ndarray, C_prior: np.ndarray, trusted: np.ndarray) -> np.ndarray:
    s = np.where(trusted)[0]
    m = np.where(~trusted)[0]
    C = np.zeros_like(C_prior)
    C[np.ix_(s, s)] = C_xhat[np.ix_(s, s)]
    C[np.ix_(m, m)] = C_prior[np.ix_(m, m)]
    return 0.5 * (C + C.T)


def find_basis_key(control: np.lib.npyio.NpzFile, requested: str) -> str:
    if requested != "auto":
        if requested not in control.files:
            raise KeyError(f"Requested basis key {requested!r} not present in control_bundle.npz")
        return requested

    candidates = (
        "basis",
        "basis_maps",
        "modal_basis",
        "dm_basis",
        "phase_basis",
        "basis_cube",
        "basis_flat",
        "basis_flattened",
        "modal_basis_flat",
    )
    for k in candidates:
        if k in control.files:
            return k

    basis_like = [k for k in control.files if "basis" in k.lower()]
    raise KeyError(
        "No basis maps found in control_bundle.npz. "
        "This PSF step needs the spatial modal basis. "
        "Rerun with control_bundle_include_basis=True. "
        f"Basis-like keys present: {basis_like}"
    )


def extract_basis_maps(
    control: np.lib.npyio.NpzFile,
    basis_key: str,
    n_modes: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    key = find_basis_key(control, basis_key)
    arr = finite_clip(control[key])
    pupil = finite_clip(control["pupil"]) if "pupil" in control.files else None

    info = {"basis_key": key, "raw_shape": list(arr.shape)}

    if arr.ndim == 3:
        # Accept n_modes,H,W or H,W,n_modes.
        if arr.shape[0] >= n_modes:
            cube = arr[:n_modes]
        elif arr.shape[-1] >= n_modes:
            cube = np.moveaxis(arr[..., :n_modes], -1, 0)
        else:
            raise ValueError(f"3D basis array shape {arr.shape} does not contain n_modes={n_modes}")

        if pupil is None:
            pupil2d = np.any(np.abs(cube) > 0, axis=0).astype(float)
        else:
            pupil2d = np.asarray(pupil, dtype=float)
            if pupil2d.shape != cube.shape[1:]:
                raise ValueError(f"pupil shape {pupil2d.shape} incompatible with basis cube shape {cube.shape}")

        mask = pupil2d > 0
        B_flat = cube[:, mask].T  # n_pix x n_modes
        info.update({"basis_format": "cube", "map_shape": list(cube.shape[1:]), "n_pupil_pixels": int(mask.sum())})
        return B_flat, pupil2d, info

    if arr.ndim == 2:
        # Accept n_modes,n_pix or n_pix,n_modes.
        if arr.shape[0] >= n_modes:
            flat = arr[:n_modes].T  # n_pix x n_modes
        elif arr.shape[1] >= n_modes:
            flat = arr[:, :n_modes]
        else:
            raise ValueError(f"2D basis array shape {arr.shape} does not contain n_modes={n_modes}")

        if pupil is None:
            # No 2D map available. Use a square-ish artificial pupil only for debugging.
            n_pix = flat.shape[0]
            side = int(np.ceil(np.sqrt(n_pix)))
            pupil2d = np.zeros((side, side), dtype=float)
            pupil2d.ravel()[:n_pix] = 1.0
            info["warning"] = "No pupil was present; made artificial square pupil. PSF geometry is not physical."
            B_pad = np.zeros((side * side, flat.shape[1]), dtype=float)
            B_pad[:n_pix] = flat
            mask = pupil2d > 0
            B_flat = B_pad[mask]
        else:
            pupil2d = np.asarray(pupil, dtype=float)
            mask = pupil2d > 0
            if flat.shape[0] != int(mask.sum()):
                if flat.shape[0] == pupil2d.size:
                    flat = flat[mask.ravel()]
                else:
                    raise ValueError(
                        f"Flat basis has {flat.shape[0]} pixels, but pupil has "
                        f"{int(mask.sum())} active pixels / {pupil2d.size} total pixels"
                    )
            B_flat = flat

        info.update({"basis_format": "flat", "map_shape": list(pupil2d.shape), "n_pupil_pixels": int((pupil2d > 0).sum())})
        return B_flat, pupil2d, info

    raise ValueError(f"Unsupported basis array ndim={arr.ndim}; shape={arr.shape}")


def phase_scale_factor(basis_units: str, wavelength_m: float) -> float:
    if basis_units == "phase":
        return 1.0
    if basis_units == "opd_m":
        return 2.0 * np.pi / float(wavelength_m)
    raise ValueError(f"Unknown basis units: {basis_units}")


def psf_from_modal_covariance(
    C: np.ndarray,
    B_control_flat: np.ndarray,
    pupil2d: np.ndarray,
    fft_size: int,
    n_mc: int,
    rng: np.random.Generator,
    phase_scale: float = 1.0,
    batch_size: int = 16,
) -> np.ndarray:
    C = finite_clip(C)
    C = 0.5 * (C + C.T)

    evals, evecs = np.linalg.eigh(C)
    evals_clip = np.clip(evals, 0.0, None)
    A = evecs * np.sqrt(evals_clip)[None, :]

    mask = pupil2d > 0
    h, w = pupil2d.shape
    N = int(fft_size)
    if N < max(h, w):
        raise ValueError(f"fft_size={N} must be >= pupil map size {pupil2d.shape}")

    y0 = (N - h) // 2
    x0 = (N - w) // 2

    psf_sum = np.zeros((N, N), dtype=np.float64)
    n_done = 0

    while n_done < n_mc:
        b = min(batch_size, n_mc - n_done)
        z = rng.standard_normal((b, C.shape[0])) @ A.T  # modal samples in control basis
        phase_flat = (z @ B_control_flat.T) * float(phase_scale)

        for j in range(b):
            phase2d = np.zeros((h, w), dtype=np.float64)
            phase2d[mask] = phase_flat[j]
            field_small = pupil2d * np.exp(1j * phase2d)

            field = np.zeros((N, N), dtype=np.complex128)
            field[y0:y0+h, x0:x0+w] = field_small

            amp = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
            psf_sum += np.abs(amp) ** 2

        n_done += b

    psf = psf_sum / float(n_mc)
    return normalize_psf(psf)


def diffraction_limited_psf(pupil2d: np.ndarray, fft_size: int) -> np.ndarray:
    h, w = pupil2d.shape
    N = int(fft_size)
    y0 = (N - h) // 2
    x0 = (N - w) // 2
    field = np.zeros((N, N), dtype=np.complex128)
    field[y0:y0+h, x0:x0+w] = pupil2d
    amp = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    return normalize_psf(np.abs(amp) ** 2)


def load_fits_image(path: Path) -> np.ndarray:
    try:
        from astropy.io import fits  # type: ignore
    except Exception as exc:
        raise RuntimeError("Reading FITS requires astropy") from exc

    data = fits.getdata(path)
    data = np.asarray(data, dtype=float)
    while data.ndim > 2:
        data = data[0]
    return normalize_psf(data)


def image_metrics(ref: np.ndarray, img: np.ndarray, crop_size: int | None = None) -> dict[str, float]:
    refn = normalize_psf(ref)
    imgn = normalize_psf(img)
    if crop_size is not None and crop_size > 0:
        refn = normalize_psf(center_crop(refn, crop_size))
        imgn = normalize_psf(center_crop(imgn, crop_size))
    return {
        "rel_l2": rel_l2_image(refn, imgn),
        "l1": l1_image(refn, imgn),
        "peak_rel_error": peak_rel_error(refn, imgn),
        "peak_ref": float(np.max(refn)),
        "peak_model": float(np.max(imgn)),
    }


def plot_psf_grid(psfs: dict[str, np.ndarray], ref_name: str, outdir: Path, crop: int) -> None:
    names = list(psfs.keys())
    n = len(names)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3*ncols, 4.0*nrows), dpi=150)
    axes = np.asarray(axes).ravel()

    vmax = np.nanmax(center_crop(psfs[ref_name], crop))
    vmin = max(vmax * 1e-5, 1e-30)

    for ax, name in zip(axes, names):
        img = center_crop(psfs[name], crop)
        ax.imshow(np.log10(np.maximum(img, vmin)), origin="lower", vmin=np.log10(vmin), vmax=np.log10(vmax))
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(names):]:
        ax.axis("off")

    fig.suptitle("PSFs from covariance models, log10 scale")
    fig.tight_layout()
    fig.savefig(outdir / "psf_model_grid.png")
    plt.close(fig)


def plot_psf_residuals(psfs: dict[str, np.ndarray], ref_name: str, outdir: Path, crop: int) -> None:
    ref = normalize_psf(psfs[ref_name])
    names = [n for n in psfs.keys() if n != ref_name]
    n = len(names)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3*ncols, 4.0*nrows), dpi=150)
    axes = np.asarray(axes).ravel()

    refc = center_crop(ref, crop)
    scale = np.nanmax(refc)

    for ax, name in zip(axes, names):
        diff = center_crop(normalize_psf(psfs[name]) - ref, crop)
        ax.imshow(diff / scale, origin="lower", vmin=-0.1, vmax=0.1)
        ax.set_title(f"{name} - {ref_name}")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(names):]:
        ax.axis("off")

    fig.suptitle("PSF residuals relative to reference, normalized by reference peak")
    fig.tight_layout()
    fig.savefig(outdir / "psf_model_residuals.png")
    plt.close(fig)


def plot_radial_profiles(psfs: dict[str, np.ndarray], outdir: Path, crop: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for name, psf in psfs.items():
        rr, prof = radial_profile(center_crop(psf, crop))
        ax.semilogy(rr, np.maximum(prof, 1e-30), label=name)
    ax.set_xlabel("Radius [pixels in cropped PSF]")
    ax.set_ylabel("Azimuthal mean PSF")
    ax.set_title("PSF radial profiles")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "psf_radial_profiles.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate PSFs produced from covariance models.")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--telemetry", type=Path, default=None)
    p.add_argument("--control-bundle", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)

    p.add_argument("--ground-altitude-max-m", type=float, default=500.0)
    p.add_argument("--ground-layer-indices", type=str, default=None)
    p.add_argument("--gain-floor", type=float, default=0.12)

    p.add_argument("--use-gram-ls-truth", action="store_true")
    p.add_argument("--gram-rcond", type=float, default=1e-6)

    p.add_argument("--target", choices=("ground", "total"), default="total")
    p.add_argument("--pol-delay", type=int, default=0)

    p.add_argument("--basis-key", type=str, default="auto")
    p.add_argument("--basis-units", choices=("phase", "opd_m"), default="phase")
    p.add_argument("--wavelength-m", type=float, default=2.15e-6)

    p.add_argument("--alpha", type=str, default="auto", help="'auto' loads alpha_full from previous summary, or give a float")
    p.add_argument("--h-clip-max", type=float, default=3.0)

    p.add_argument("--n-mc", type=int, default=128)
    p.add_argument("--fft-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--plot-crop", type=int, default=96)
    p.add_argument("--metric-crop", type=int, default=96)

    p.add_argument("--reference-psf-fits", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def load_alpha(run_dir: Path, alpha_arg: str) -> tuple[float, str]:
    if alpha_arg != "auto":
        return float(alpha_arg), "user"

    summary_path = run_dir / "telemetry_transfer_calibration" / "telemetry_transfer_calibration_summary.json"
    if summary_path.exists():
        s = load_json(summary_path)
        alpha = s.get("alpha_diagnostics", {}).get("alpha_full_grid_oracle")
        if alpha is not None:
            return float(alpha), str(summary_path)

    return 1.0, "auto_missing_default_1"


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    telemetry_path = args.telemetry.expanduser().resolve() if args.telemetry else run_dir / "telemetry.npz"
    control_path = args.control_bundle.expanduser().resolve() if args.control_bundle else run_dir / "control_bundle.npz"
    outdir = args.outdir.expanduser().resolve() if args.outdir else run_dir / "psf_from_covariance_models"
    outdir.mkdir(parents=True, exist_ok=True)

    tel = np.load(telemetry_path)
    control = np.load(control_path)
    telemetry_summary = load_json(run_dir / "telemetry_summary.json")
    control_summary = load_json(run_dir / "control_bundle_summary.json")

    for key in ("truth_layer_modal_atm", "truth_modal_dm_applied", "xhat", "dm_cmd_applied"):
        if key not in tel.files:
            raise KeyError(f"telemetry.npz missing required {key}")

    ground, free, ground_idx, free_idx, labels = collapse_layers(
        tel["truth_layer_modal_atm"],
        telemetry_summary,
        args.ground_altitude_max_m,
        args.ground_layer_indices,
    )
    truth_dm = finite_clip(tel["truth_modal_dm_applied"])
    xhat = finite_clip(tel["xhat"])
    dm = finite_clip(tel["dm_cmd_applied"])

    n_samples = min(len(ground), len(free), len(truth_dm), len(xhat), len(dm))
    n_modes = min(ground.shape[1], free.shape[1], truth_dm.shape[1], xhat.shape[1], dm.shape[1])
    ground = ground[:n_samples, :n_modes]
    free = free[:n_samples, :n_modes]
    truth_dm = truth_dm[:n_samples, :n_modes]
    xhat = xhat[:n_samples, :n_modes]
    dm = dm[:n_samples, :n_modes]

    d = int(args.pol_delay)
    if d > 0:
        xhat_pol = xhat[d:]
        dm_pol = dm[:-d]
        ground = ground[d:]
        free = free[d:]
        truth_dm = truth_dm[d:]
        xhat = xhat[d:]
    elif d < 0:
        dabs = abs(d)
        xhat_pol = xhat[:-dabs]
        dm_pol = dm[dabs:]
        ground = ground[:-dabs]
        free = free[:-dabs]
        truth_dm = truth_dm[:-dabs]
        xhat = xhat[:-dabs]
    else:
        xhat_pol = xhat
        dm_pol = dm
    pol = xhat_pol - dm_pol

    n_samples = min(len(ground), len(free), len(truth_dm), len(xhat), len(pol))
    ground = ground[:n_samples]
    free = free[:n_samples]
    truth_dm = truth_dm[:n_samples]
    xhat = xhat[:n_samples]
    pol = pol[:n_samples]

    arrays, gram_info = maybe_gram_ls(
        {"ground": ground, "free": free, "truth_dm": truth_dm},
        control,
        use_gram=args.use_gram_ls_truth,
        rcond=args.gram_rcond,
    )
    ground = arrays["ground"]
    free = arrays["free"]
    truth_dm = arrays["truth_dm"]

    ground_res_true = ground - truth_dm
    total_res_true = ground_res_true + free

    V, svals, trusted, T_name = control_basis(control, args.gain_floor, n_modes)
    sidx = np.where(trusted)[0]
    midx = np.where(~trusted)[0]

    C_ground = covariance_in_basis(ground, V)
    C_free = covariance_in_basis(free, V)
    C_ground_res_true = covariance_in_basis(ground_res_true, V)
    C_total_res_true = covariance_in_basis(total_res_true, V)
    C_xhat = covariance_in_basis(xhat, V)
    C_pol = covariance_in_basis(pol, V)

    var_ground = np.diag(C_ground)
    var_xhat = np.diag(C_xhat)
    var_pol = np.diag(C_pol)
    var_ground_res_true = np.diag(C_ground_res_true)

    h_tel_pol = safe_sqrt_ratio(var_xhat, var_pol)
    h_tel_pol[midx] = 1.0
    h_tel_pol = np.clip(h_tel_pol, 0.0, args.h_clip_max)

    alpha, alpha_source = load_alpha(run_dir, args.alpha)
    h_tel_pol_alpha = h_tel_pol.copy()
    h_tel_pol_alpha[sidx] *= float(alpha)
    h_tel_pol_alpha[midx] = 1.0
    h_tel_pol_alpha = np.clip(h_tel_pol_alpha, 0.0, args.h_clip_max)

    h_oracle = safe_sqrt_ratio(var_ground_res_true, var_ground)
    h_oracle = np.clip(h_oracle, 0.0, args.h_clip_max)

    C_tel_ground = np.zeros_like(C_ground)
    C_tel_ground[np.ix_(sidx, sidx)] = C_xhat[np.ix_(sidx, sidx)]
    C_tel_ground = 0.5 * (C_tel_ground + C_tel_ground.T)

    C_block_ground = make_block_model(C_xhat, C_ground, trusted)
    C_h_tel_ground = apply_diag_transfer(C_ground, h_tel_pol)
    C_h_tel_alpha_ground = apply_diag_transfer(C_ground, h_tel_pol_alpha)
    C_h_oracle_ground = apply_diag_transfer(C_ground, h_oracle)

    if args.target == "ground":
        C_truth = C_ground_res_true
        C_models = {
            "truth covariance": C_truth,
            "telemetry-only": C_tel_ground,
            "reg-aware block": C_block_ground,
            "H tel/POL": C_h_tel_ground,
            f"H tel/POL alpha={alpha:.2f}": C_h_tel_alpha_ground,
            "H oracle diag": C_h_oracle_ground,
        }
    else:
        # Add free-atmosphere prior covariance to each ground-residual model.
        C_truth = C_total_res_true
        C_models = {
            "truth covariance": C_truth,
            "telemetry-only + free": C_tel_ground + C_free,
            "reg-aware block + free": C_block_ground + C_free,
            "H tel/POL + free": C_h_tel_ground + C_free,
            f"H tel/POL alpha={alpha:.2f} + free": C_h_tel_alpha_ground + C_free,
            "H oracle diag + free": C_h_oracle_ground + C_free,
        }

    # Load basis and convert to control-basis spatial maps.
    B_flat_orig, pupil2d, basis_info = extract_basis_maps(control, args.basis_key, n_modes)
    phase_scale = phase_scale_factor(args.basis_units, args.wavelength_m)
    B_control_flat = B_flat_orig @ V

    # Remove piston-like mean over pupil from each basis direction for numerical stability.
    B_control_flat = B_control_flat - np.mean(B_control_flat, axis=0, keepdims=True)

    rng_master = np.random.default_rng(args.seed)
    psfs: dict[str, np.ndarray] = {}
    for name, C in C_models.items():
        if args.verbose:
            print(f"Generating PSF for {name} ...")
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        psfs[name] = psf_from_modal_covariance(
            C,
            B_control_flat,
            pupil2d,
            fft_size=args.fft_size,
            n_mc=args.n_mc,
            rng=rng,
            phase_scale=phase_scale,
            batch_size=args.batch_size,
        )

    psfs["diffraction limited"] = diffraction_limited_psf(pupil2d, args.fft_size)

    ref_name = "truth covariance"
    metrics: dict[str, Any] = {}
    for name, psf in psfs.items():
        if name == ref_name:
            continue
        metrics[name] = image_metrics(psfs[ref_name], psf, crop_size=args.metric_crop)

    fits_metrics: dict[str, Any] = {}
    if args.reference_psf_fits is not None:
        ref_fits = load_fits_image(args.reference_psf_fits.expanduser().resolve())
        ref_fits = center_crop(ref_fits, min(ref_fits.shape[-2], ref_fits.shape[-1], args.fft_size))
        # If FITS size differs, center-crop model to common size.
        common = min(ref_fits.shape[-2], psfs[ref_name].shape[-2], args.metric_crop)
        for name, psf in psfs.items():
            fits_metrics[name] = image_metrics(ref_fits, center_crop(psf, common), crop_size=common)
        psfs["reference FITS"] = center_crop(ref_fits, args.fft_size if ref_fits.shape[0] >= args.fft_size else ref_fits.shape[0])

    plot_crop = min(args.plot_crop, args.fft_size)
    plot_psf_grid(psfs, ref_name=ref_name, outdir=outdir, crop=plot_crop)
    plot_psf_residuals(psfs, ref_name=ref_name, outdir=outdir, crop=plot_crop)
    plot_radial_profiles(psfs, outdir=outdir, crop=plot_crop)

    # Save PSF arrays.
    np.savez_compressed(
        outdir / "psf_from_covariance_models.npz",
        **{name.replace(" ", "_").replace("+", "plus").replace("=", "_"): psf for name, psf in psfs.items()},
    )

    cov_errors = {
        name: rel_fro(C_truth, C)
        for name, C in C_models.items()
        if name != "truth covariance"
    }

    summary = {
        "run_dir": str(run_dir),
        "telemetry_npz": str(telemetry_path),
        "control_bundle_npz": str(control_path),
        "outdir": str(outdir),
        "target": args.target,
        "truth_projection": gram_info.get("truth_projection"),
        "gram_info": gram_info,
        "basis_info": basis_info,
        "basis_units": args.basis_units,
        "wavelength_m": float(args.wavelength_m),
        "phase_scale": float(phase_scale),
        "n_samples": int(n_samples),
        "n_modes": int(n_modes),
        "n_mc": int(args.n_mc),
        "fft_size": int(args.fft_size),
        "seed": int(args.seed),
        "ground_altitude_max_m": float(args.ground_altitude_max_m),
        "ground_layer_indices": ground_idx.tolist(),
        "free_layer_indices": free_idx.tolist(),
        "layer_labels": labels,
        "control_operator": {
            "T_name": T_name,
            "gain_floor": float(args.gain_floor),
            "n_trusted": int(np.sum(trusted)),
            "n_missing": int(np.sum(~trusted)),
            "s_min": float(np.min(svals)),
            "s_median": float(np.median(svals)),
            "s_max": float(np.max(svals)),
        },
        "alpha": {
            "value": float(alpha),
            "source": alpha_source,
        },
        "covariance_errors_vs_truth": cov_errors,
        "psf_metrics_vs_truth_covariance_psf": metrics,
        "psf_metrics_vs_reference_fits": fits_metrics,
        "notes": {
            "reference": "Default PSF reference is Monte-Carlo PSF from truth residual covariance, not necessarily the saved delivered FITS.",
            "monte_carlo": "PSF estimates have Monte-Carlo noise; increase --n-mc for publication-quality comparisons.",
            "basis_requirement": "Requires spatial modal basis in control_bundle.npz. If missing, rerun with control_bundle_include_basis=True.",
        },
        "telemetry_summary_subset": {
            "dt_s": telemetry_summary.get("dt_s"),
            "n_frames_requested": telemetry_summary.get("n_frames_requested"),
            "discard_first_frames": telemetry_summary.get("discard_first_frames"),
            "stride": telemetry_summary.get("stride"),
            "gain": telemetry_summary.get("gain"),
            "leak": telemetry_summary.get("leak"),
            "dm_delay_frames": telemetry_summary.get("dm_delay_frames"),
            "truth_modal_projection": telemetry_summary.get("truth_modal_projection"),
        },
        "control_summary_subset": {
            "reconstructor": control_summary.get("reconstructor", {}),
            "runner": control_summary.get("runner", {}),
        },
    }

    summary_path = outdir / "psf_from_covariance_models_summary.json"
    save_json(summary, summary_path)

    print("\n=== PSF-from-covariance validation ===")
    print(f"Run dir: {run_dir}")
    print(f"Output:  {outdir}")
    print(f"Target:  {args.target}")
    print(f"Truth projection: {gram_info.get('truth_projection')}")
    if args.use_gram_ls_truth:
        print(f"Gram rcond: {gram_info.get('gram_rcond')}; kept {gram_info.get('gram_n_kept')}/{gram_info.get('gram_n_eig')} Gram eigenmodes")
    print(f"Basis key: {basis_info['basis_key']}  format={basis_info['basis_format']}  active pixels={basis_info['n_pupil_pixels']}")
    print(f"Basis units: {args.basis_units}; phase scale={phase_scale:.6g}")
    print(f"Control trusted/missing: {int(np.sum(trusted))}/{int(np.sum(~trusted))}")
    print(f"Alpha: {alpha:.4g}  source={alpha_source}")
    print(f"Monte Carlo: n={args.n_mc}, fft_size={args.fft_size}")

    print("\nCovariance errors vs truth:")
    for name, val in cov_errors.items():
        print(f"  {name:30s} {100*val:7.2f}%")

    print("\nPSF metrics vs truth-covariance PSF:")
    for name, m in metrics.items():
        print(
            f"  {name:30s} "
            f"L2={100*m['rel_l2']:7.2f}%  "
            f"L1={m['l1']:.4f}  "
            f"peak_err={100*m['peak_rel_error']:+7.2f}%"
        )

    if fits_metrics:
        print("\nPSF metrics vs reference FITS:")
        for name, m in fits_metrics.items():
            print(
                f"  {name:30s} "
                f"L2={100*m['rel_l2']:7.2f}%  "
                f"L1={m['l1']:.4f}  "
                f"peak_err={100*m['peak_rel_error']:+7.2f}%"
            )

    print(f"\nWrote: {summary_path}")
    if args.verbose:
        print("\nKey plots:")
        for p in sorted(outdir.glob("*.png")):
            print(f"  {p}")


if __name__ == "__main__":
    main()
