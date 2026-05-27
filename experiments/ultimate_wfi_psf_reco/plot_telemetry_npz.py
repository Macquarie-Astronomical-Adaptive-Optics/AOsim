#!/usr/bin/env python3
"""
Quick-look plots for AOsim headless telemetry outputs.

This script loads a ``telemetry.npz`` file written by the headless
PSF-reconstruction experiment pipeline and produces diagnostic plots for:

- basic channel/shape/finite-value checks,
- AO telemetry RMS time series,
- WFE proxy time series,
- tip/tilt telemetry,
- modal spectra,
- pseudo-open-loop modal candidates,
- optional simulator-truth modal comparisons, including total, layer-wise, and collapsed ground/free atmospheric truth if available.

Use --plot-level basic/standard/full to control how many figures are generated.

The pseudo-open-loop candidates are diagnostic only until the command sign and
latency convention are validated against truth telemetry.

Example
-------
From the AOsim repository root:

    python experiments/ultimate_wfi_psf_reco/plot_telemetry_npz.py \
        experiments/ultimate_wfi_psf_reco/outputs/telemetry_smoke/longrun_YYYYMMDD_HHMMSS/telemetry.npz \
        --show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def nan_rms(a: np.ndarray, axis=None) -> np.ndarray:
    """RMS ignoring NaNs."""
    return np.sqrt(np.nanmean(np.asarray(a, dtype=np.float64) ** 2, axis=axis))


def finite_fraction(a: np.ndarray) -> float:
    """Fraction of finite entries in an array."""
    a = np.asarray(a)
    if a.size == 0:
        return np.nan
    return float(np.isfinite(a).sum() / a.size)


def nan_corr(a: np.ndarray, b: np.ndarray, axis=None) -> np.ndarray:
    """Pearson-like correlation ignoring NaNs along a given axis."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    aa = np.where(mask, a, np.nan)
    bb = np.where(mask, b, np.nan)
    aa = aa - np.nanmean(aa, axis=axis, keepdims=True)
    bb = bb - np.nanmean(bb, axis=axis, keepdims=True)
    num = np.nansum(aa * bb, axis=axis)
    den = np.sqrt(np.nansum(aa * aa, axis=axis) * np.nansum(bb * bb, axis=axis))
    return num / np.maximum(den, 1e-30)


def load_json_if_exists(path: Path) -> dict:
    if path.exists():
        try:
            with path.open("r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def load_dt(telemetry_path: Path, default_dt: float = 0.002) -> float:
    """Read dt_s from telemetry_summary.json or headless_summary.json if available."""
    for name in ("telemetry_summary.json", "headless_summary.json"):
        d = load_json_if_exists(telemetry_path.parent / name)
        if "dt_s" in d:
            return float(d["dt_s"])
        if "dt" in d:
            return float(d["dt"])
    return default_dt


def savefig(out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    print(f"saved: {path}")
    plt.close()


def print_summary(data: np.lib.npyio.NpzFile, meta: dict | None = None) -> None:
    print("\n=== Telemetry arrays ===")
    for k in data.files:
        arr = data[k]
        msg = (
            f"{k:22s} shape={str(arr.shape):18s} "
            f"dtype={str(arr.dtype):10s} finite={finite_fraction(arr):.3f}"
        )
        if np.issubdtype(arr.dtype, np.number):
            with np.errstate(all="ignore"):
                msg += (
                    f" min={np.nanmin(arr): .4e}"
                    f" max={np.nanmax(arr): .4e}"
                    f" rms={nan_rms(arr): .4e}"
                )
        print(msg)

    if meta:
        print("\n=== Telemetry metadata ===")
        for k in (
            "dt_s",
            "n_frames_requested",
            "discard_first_frames",
            "stride",
            "n_samples",
            "psf_tt_mode",
            "n_wfs",
            "n_slopes",
            "n_modes",
            "tt_enabled",
            "gain",
            "leak",
            "dm_delay_frames",
            "truth_modal_projection",
        ):
            if k in meta:
                print(f"{k:28s}: {meta[k]}")


def get_time_axis(data: np.lib.npyio.NpzFile, dt: float) -> tuple[np.ndarray, str]:
    if "frame_index" in data.files:
        frames = np.asarray(data["frame_index"])
        return frames * dt, "Time [s]"
    n = len(data[data.files[0]])
    return np.arange(n) * dt, "Time [s]"

def first_available(data: np.lib.npyio.NpzFile, names: Iterable[str]) -> str | None:
    """Return the first array name present in an NPZ file."""
    for name in names:
        if name in data.files:
            return name
    return None


def dm_for_pol(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray | None, str]:
    """Prefer the delayed command actually applied to the phase for POL checks."""
    name = first_available(data, ("dm_cmd_applied", "dm_cmd", "dm_cmd_new"))
    if name is None:
        return None, ""
    return np.asarray(data[name], dtype=float), name


def truth_atm_total_name(data: np.lib.npyio.NpzFile) -> str | None:
    return first_available(data, ("truth_modal_atm_total", "truth_modal_atm"))


def truth_residual_total_name(data: np.lib.npyio.NpzFile) -> str | None:
    return first_available(data, ("truth_modal_residual_total", "truth_modal_residual"))


def truth_dm_name(data: np.lib.npyio.NpzFile) -> str | None:
    return first_available(data, ("truth_modal_dm_applied", "truth_modal_dm"))


def plot_overview_panels(data: np.lib.npyio.NpzFile, t: np.ndarray, out_dir: Path) -> None:
    """Less misleading than putting all RMS channels on one y-axis."""
    series = []
    if "slopes_err_vec" in data.files:
        series.append(("WFS slope residual RMS", nan_rms(data["slopes_err_vec"], axis=1)))
    dm_name = first_available(data, ("dm_cmd_applied", "dm_cmd", "dm_cmd_new"))
    if dm_name is not None:
        series.append((f"DM command RMS ({dm_name})", nan_rms(data[dm_name], axis=1)))
    if "xhat" in data.files:
        series.append(("Reconstructed modal estimate RMS", nan_rms(data["xhat"], axis=1)))
    if "dm_phi_rms" in data.files:
        series.append(("DM phase RMS", np.asarray(data["dm_phi_rms"], dtype=float)))

    if not series:
        return

    fig, axes = plt.subplots(len(series), 1, figsize=(10, 2.2 * len(series)), sharex=True)
    if len(series) == 1:
        axes = [axes]
    for ax, (name, y) in zip(axes, series):
        ax.plot(t[: len(y)], y)
        ax.set_ylabel("RMS")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("AO telemetry RMS overview", y=1.01)
    savefig(out_dir, "telemetry_overview_panels.png")


def plot_rms_timeseries(data: np.lib.npyio.NpzFile, t: np.ndarray, out_dir: Path) -> None:
    plt.figure(figsize=(10, 5))

    if "slopes_err_vec" in data.files:
        plt.plot(t, nan_rms(data["slopes_err_vec"], axis=1), label="WFS slope residual RMS")
    dm_name = first_available(data, ("dm_cmd_applied", "dm_cmd", "dm_cmd_new"))
    if dm_name is not None:
        plt.plot(t, nan_rms(data[dm_name], axis=1), label=f"DM command RMS ({dm_name})")
    if "xhat" in data.files:
        plt.plot(t, nan_rms(data["xhat"], axis=1), label="Reconstructed modal estimate RMS")
    if "dm_phi_rms" in data.files:
        plt.plot(t, np.asarray(data["dm_phi_rms"]), label="DM phase RMS")

    plt.xlabel("Time [s]")
    plt.ylabel("RMS [native units]")
    plt.title("AO telemetry RMS versus time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(out_dir, "telemetry_rms_timeseries.png")


def plot_wfe(data: np.lib.npyio.NpzFile, t: np.ndarray, out_dir: Path) -> None:
    if "wfe_rms" not in data.files:
        return

    wfe = np.asarray(data["wfe_rms"])

    plt.figure(figsize=(10, 5))
    if wfe.ndim == 2 and wfe.shape[1] >= 2:
        ratio = wfe[:, 1] / np.maximum(wfe[:, 0], 1e-12)
        plt.plot(t, wfe[:, 0], label="Uncorrected phase RMS")
        plt.plot(t, wfe[:, 1], label="Corrected residual phase RMS")
        plt.plot(t, ratio, label="Corrected / uncorrected")
    else:
        plt.plot(t, np.ravel(wfe), label="WFE RMS")

    plt.xlabel("Time [s]")
    plt.ylabel("RMS [rad or internal phase units]")
    plt.title("Wavefront-error proxy versus time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(out_dir, "telemetry_wfe_rms.png")

    if wfe.ndim == 2 and wfe.shape[1] >= 2:
        plt.figure(figsize=(10, 4))
        plt.plot(t, ratio)
        plt.xlabel("Time [s]")
        plt.ylabel("Corrected / uncorrected RMS")
        plt.title("Closed-loop residual ratio")
        plt.grid(True, alpha=0.3)
        savefig(out_dir, "telemetry_wfe_ratio.png")


def plot_tt(data: np.lib.npyio.NpzFile, t: np.ndarray, out_dir: Path) -> None:
    has_cmd = "tt_cmd_yx" in data.files
    has_meas = "tt_meas_yx" in data.files
    if not (has_cmd or has_meas):
        return

    plt.figure(figsize=(10, 5))

    if has_cmd:
        tt = np.asarray(data["tt_cmd_yx"])
        if tt.ndim == 2 and tt.shape[1] >= 2:
            plt.plot(t, tt[:, 0], label="TT command y")
            plt.plot(t, tt[:, 1], label="TT command x")

    if has_meas:
        tm = np.asarray(data["tt_meas_yx"])
        if tm.ndim == 2 and tm.shape[1] >= 2:
            plt.plot(t, tm[:, 0], ".", label="TT measurement y")
            plt.plot(t, tm[:, 1], ".", label="TT measurement x")
            valid = np.isfinite(tm).all(axis=1)
            print(f"\nTT measurement valid samples: {valid.sum()} / {len(valid)}")

    plt.xlabel("Time [s]")
    plt.ylabel("TT [native units]")
    plt.title("Tip/tilt telemetry")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(out_dir, "telemetry_tt.png")


def plot_modal_spectra(data: np.lib.npyio.NpzFile, out_dir: Path, max_modes: int | None = None) -> None:
    plt.figure(figsize=(10, 5))

    if "dm_cmd" in data.files:
        dm = np.asarray(data["dm_cmd"])
        if dm.ndim == 2:
            y = nan_rms(dm, axis=0)
            if max_modes is not None:
                y = y[:max_modes]
            plt.plot(np.arange(len(y)), y, label="DM command modal RMS")

    if "xhat" in data.files:
        xhat = np.asarray(data["xhat"])
        if xhat.ndim == 2:
            y = nan_rms(xhat, axis=0)
            if max_modes is not None:
                y = y[:max_modes]
            plt.plot(np.arange(len(y)), y, label="Reconstructed mode RMS")

    plt.xlabel("Mode index")
    plt.ylabel("RMS [native units]")
    plt.title("Modal RMS spectrum")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(out_dir, "telemetry_modal_rms.png")


def plot_slope_statistics(data: np.lib.npyio.NpzFile, out_dir: Path) -> None:
    if "slopes_err_vec" not in data.files:
        return

    slopes = np.asarray(data["slopes_err_vec"])
    if slopes.ndim != 2:
        return

    slope_rms_by_channel = nan_rms(slopes, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(slopes.shape[1]), slope_rms_by_channel)
    plt.xlabel("Packed slope index")
    plt.ylabel("RMS [native slope units]")
    plt.title("WFS slope residual RMS by packed slope index")
    plt.grid(True, alpha=0.3)
    savefig(out_dir, "telemetry_slope_rms_by_channel.png")

    plt.figure(figsize=(8, 5))
    finite = np.ravel(slopes[np.isfinite(slopes)])
    plt.hist(finite, bins=100)
    plt.xlabel("Slope residual [native units]")
    plt.ylabel("Count")
    plt.title("Distribution of WFS slope residuals")
    plt.grid(True, alpha=0.3)
    savefig(out_dir, "telemetry_slope_histogram.png")


def align_for_delay(a: np.ndarray, b: np.ndarray, t: np.ndarray, delay_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a[k], b[k-delay], t[k] if delay_samples > 0."""
    n = min(len(a), len(b), len(t))
    a = a[:n]
    b = b[:n]
    t = t[:n]
    d = int(delay_samples)
    if d > 0:
        return a[d:], b[:-d], t[d:]
    if d < 0:
        return a[:d], b[-d:], t[:d]
    return a, b, t


def plot_pseudo_open_loop(data: np.lib.npyio.NpzFile, t: np.ndarray, out_dir: Path, delay_samples: int = 0) -> None:
    if "xhat" not in data.files:
        return
    dm, dm_name = dm_for_pol(data)
    if dm is None:
        return

    xhat = np.asarray(data["xhat"], dtype=float)
    x, d, tt = align_for_delay(xhat, dm, t, delay_samples)

    pol_plus = x + d
    pol_minus = x - d

    rms_x = nan_rms(x, axis=1)
    rms_d = nan_rms(d, axis=1)
    rms_plus = nan_rms(pol_plus, axis=1)
    rms_minus = nan_rms(pol_minus, axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(tt, rms_x, label="Residual modal estimate xhat RMS")
    plt.plot(tt, rms_d, label=f"DM command RMS ({dm_name})")
    plt.plot(tt, rms_plus, label=f"POL candidate: xhat + {dm_name}")
    plt.plot(tt, rms_minus, label=f"POL candidate: xhat - {dm_name}")
    plt.xlabel("Time [s]")
    plt.ylabel("Modal RMS [native units]")
    plt.title(f"Pseudo-open-loop modal candidates, delay={delay_samples} sample(s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(out_dir, f"telemetry_pseudo_open_loop_rms_delay{delay_samples}.png")

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(pol_plus.shape[1]), nan_rms(pol_plus, axis=0), label=f"xhat + {dm_name}")
    plt.plot(np.arange(pol_minus.shape[1]), nan_rms(pol_minus, axis=0), label=f"xhat - {dm_name}")
    plt.xlabel("Mode index")
    plt.ylabel("RMS [native units]")
    plt.title(f"Pseudo-open-loop modal RMS spectrum, delay={delay_samples} sample(s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(out_dir, f"telemetry_pseudo_open_loop_modal_spectrum_delay{delay_samples}.png")

    if len(dm) > 1:
        ddm = np.diff(dm, axis=0)
        x_mid = xhat[1 : 1 + len(ddm)]
        corr = nan_corr(ddm, x_mid, axis=0)

        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(corr)), corr)
        plt.axhline(0, linewidth=1)
        plt.xlabel("Mode index")
        plt.ylabel("Correlation")
        plt.title(f"Correlation between {dm_name} increment and xhat")
        plt.grid(True, alpha=0.3)
        savefig(out_dir, "telemetry_dm_increment_xhat_correlation.png")


def plot_layer_truth_summary(data: np.lib.npyio.NpzFile, t: np.ndarray, out_dir: Path, meta: dict | None = None) -> None:
    if "truth_layer_modal_atm" not in data.files:
        return
    layers = np.asarray(data["truth_layer_modal_atm"], dtype=float)  # (T, L, M)
    if layers.ndim != 3:
        return
    n_time, n_layer, _ = layers.shape
    tt = t[:n_time]
    layer_rms_t = nan_rms(layers, axis=2)  # (T, L)
    layer_rms = nan_rms(layers, axis=(0, 2))

    names = None
    if meta and isinstance(meta.get("layer_name"), list) and len(meta["layer_name"]) == n_layer:
        names = [str(x) for x in meta["layer_name"]]
    else:
        names = [f"L{i}" for i in range(n_layer)]

    alt = None
    if meta and isinstance(meta.get("layer_altitude_m"), list) and len(meta["layer_altitude_m"]) == n_layer:
        alt = np.asarray(meta["layer_altitude_m"], dtype=float)
        labels = [f"{names[i]} ({alt[i]/1000:.1f} km)" for i in range(n_layer)]
    else:
        labels = names

    plt.figure(figsize=(10, 5))
    for i in range(n_layer):
        plt.plot(tt, layer_rms_t[:, i], label=labels[i])
    plt.xlabel("Time [s]")
    plt.ylabel("Modal RMS [native units]")
    plt.title("Layer-wise atmospheric truth modal RMS")
    plt.grid(True, alpha=0.3)
    if n_layer <= 12:
        plt.legend(fontsize=8)
    savefig(out_dir, "telemetry_truth_layer_modal_rms_timeseries.png")

    plt.figure(figsize=(10, 5))
    x = np.arange(n_layer)
    plt.bar(x, layer_rms)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Modal RMS [native units]")
    plt.title("Layer-wise atmospheric truth modal RMS, time-averaged")
    plt.grid(True, axis="y", alpha=0.3)
    savefig(out_dir, "telemetry_truth_layer_modal_rms_by_layer.png")


def plot_truth_modal_comparison(data: np.lib.npyio.NpzFile, t: np.ndarray, out_dir: Path, delay_samples: int = 0) -> None:
    """Compare pseudo-open-loop candidates to optional simulator-truth modal channels."""
    if "xhat" not in data.files:
        return
    dm, dm_name = dm_for_pol(data)
    if dm is None:
        return

    atm_name = truth_atm_total_name(data)
    res_name = truth_residual_total_name(data)
    if atm_name is None and res_name is None:
        return

    xhat = np.asarray(data["xhat"], dtype=float)
    x, d, tt = align_for_delay(xhat, dm, t, delay_samples)
    pol_plus = x + d
    pol_minus = x - d

    if atm_name is not None:
        truth = np.asarray(data[atm_name], dtype=float)
        truth = truth[-len(pol_plus) :]
        n_modes = min(truth.shape[1], pol_plus.shape[1])
        truth = truth[:, :n_modes]
        pp = pol_plus[:, :n_modes]
        pm = pol_minus[:, :n_modes]

        plt.figure(figsize=(10, 5))
        plt.plot(tt[-len(truth):], nan_rms(truth, axis=1), label=f"Truth total atmosphere ({atm_name})")
        plt.plot(tt[-len(truth):], nan_rms(pp, axis=1), label=f"POL xhat + {dm_name}")
        plt.plot(tt[-len(truth):], nan_rms(pm, axis=1), label=f"POL xhat - {dm_name}")
        plt.xlabel("Time [s]")
        plt.ylabel("Modal RMS [native units]")
        plt.title(f"Pseudo-open-loop vs truth total modal atmosphere, delay={delay_samples}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        savefig(out_dir, f"telemetry_pol_vs_truth_atm_delay{delay_samples}.png")

        corr_plus = nan_corr(pp, truth, axis=0)
        corr_minus = nan_corr(pm, truth, axis=0)
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(n_modes), corr_plus, label=f"corr(truth, xhat + {dm_name})")
        plt.plot(np.arange(n_modes), corr_minus, label=f"corr(truth, xhat - {dm_name})")
        plt.axhline(0, linewidth=1)
        plt.xlabel("Mode index")
        plt.ylabel("Correlation")
        plt.title("Pseudo-open-loop sign check against truth total modal atmosphere")
        plt.grid(True, alpha=0.3)
        plt.legend()
        savefig(out_dir, f"telemetry_pol_truth_correlation_delay{delay_samples}.png")

        print("\nPseudo-open-loop truth-total-atmosphere correlation summary:")
        print(f"  median corr(truth, xhat + {dm_name}) = {np.nanmedian(corr_plus):+.3f}")
        print(f"  median corr(truth, xhat - {dm_name}) = {np.nanmedian(corr_minus):+.3f}")

        if "truth_layer_modal_atm" in data.files:
            layers = np.asarray(data["truth_layer_modal_atm"], dtype=float)
            if layers.ndim == 3:
                layers = layers[-len(pm):, :, :n_modes]
                corr_layers = []
                for li in range(layers.shape[1]):
                    corr_layers.append(np.nanmedian(nan_corr(pm, layers[:, li, :], axis=0)))
                plt.figure(figsize=(10, 5))
                plt.bar(np.arange(len(corr_layers)), corr_layers)
                plt.axhline(0, linewidth=1)
                plt.xlabel("Layer index")
                plt.ylabel("Median modal correlation")
                plt.title(f"Correlation of xhat - {dm_name} with each truth layer, delay={delay_samples}")
                plt.grid(True, axis="y", alpha=0.3)
                savefig(out_dir, f"telemetry_pol_layer_correlation_delay{delay_samples}.png")
                print("  layer median corr(truth_layer, xhat - command):", [f"{c:+.3f}" for c in corr_layers])

    if res_name is not None:
        truth_res = np.asarray(data[res_name], dtype=float)
        n = min(len(truth_res), len(xhat), len(t))
        truth_res = truth_res[:n]
        xr = xhat[:n, : truth_res.shape[1]]
        corr_res = nan_corr(xr, truth_res, axis=0)

        plt.figure(figsize=(10, 5))
        plt.plot(t[:n], nan_rms(truth_res, axis=1), label=f"Truth total modal residual ({res_name})")
        plt.plot(t[:n], nan_rms(xr, axis=1), label="xhat residual estimate")
        plt.xlabel("Time [s]")
        plt.ylabel("Modal RMS [native units]")
        plt.title("Residual modal estimate versus simulator total truth")
        plt.grid(True, alpha=0.3)
        plt.legend()
        savefig(out_dir, "telemetry_xhat_vs_truth_residual_rms.png")

        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(corr_res)), corr_res)
        plt.axhline(0, linewidth=1)
        plt.xlabel("Mode index")
        plt.ylabel("Correlation")
        plt.title("Correlation between xhat and truth total modal residual")
        plt.grid(True, alpha=0.3)
        savefig(out_dir, "telemetry_xhat_truth_residual_correlation.png")

        print(f"  median corr(truth total residual, xhat) = {np.nanmedian(corr_res):+.3f}")


def parse_index_list(s: str | None) -> list[int] | None:
    """Parse comma-separated indices such as '0,1,2'."""
    if s is None or str(s).strip() == "":
        return None
    out = []
    for part in str(s).split(','):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def layer_group_indices(meta: dict | None, n_layer: int, ground_altitude_max_m: float, ground_indices: str | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return ground/free layer indices and human-readable layer labels."""
    names = [f"L{i}" for i in range(n_layer)]
    if meta and isinstance(meta.get("layer_name"), list) and len(meta["layer_name"]) == n_layer:
        names = [str(x) for x in meta["layer_name"]]

    alt = None
    if meta and isinstance(meta.get("layer_altitude_m"), list) and len(meta["layer_altitude_m"]) == n_layer:
        alt = np.asarray(meta["layer_altitude_m"], dtype=float)
        labels = [f"{names[i]} ({alt[i]/1000:.1f} km)" for i in range(n_layer)]
    else:
        labels = names

    explicit = parse_index_list(ground_indices)
    if explicit is not None:
        ground = np.asarray([i for i in explicit if 0 <= i < n_layer], dtype=int)
    elif alt is not None:
        ground = np.where(alt <= float(ground_altitude_max_m))[0].astype(int)
    else:
        # Conservative fallback: without altitude metadata, only layer 0 is called ground.
        ground = np.asarray([0], dtype=int)

    all_idx = np.arange(n_layer, dtype=int)
    free = np.asarray([i for i in all_idx if i not in set(ground.tolist())], dtype=int)
    return ground, free, labels


def plot_collapsed_layer_truth(
    data: np.lib.npyio.NpzFile,
    t: np.ndarray,
    out_dir: Path,
    meta: dict | None = None,
    *,
    ground_altitude_max_m: float = 500.0,
    ground_indices: str | None = None,
    delay_samples: int = 0,
) -> None:
    """Collapse layer truth into ground/free components and compare with POL telemetry."""
    if "truth_layer_modal_atm" not in data.files:
        return
    if "xhat" not in data.files:
        return
    dm, dm_name = dm_for_pol(data)
    if dm is None:
        return

    layers = np.asarray(data["truth_layer_modal_atm"], dtype=float)  # (T, L, M)
    if layers.ndim != 3:
        return
    n_time, n_layer, n_modes = layers.shape
    ground_idx, free_idx, labels = layer_group_indices(meta, n_layer, ground_altitude_max_m, ground_indices)

    ground = np.nansum(layers[:, ground_idx, :], axis=1) if len(ground_idx) else np.zeros((n_time, n_modes))
    free = np.nansum(layers[:, free_idx, :], axis=1) if len(free_idx) else np.zeros((n_time, n_modes))
    total_from_layers = ground + free

    total_name = truth_atm_total_name(data)
    total = np.asarray(data[total_name], dtype=float) if total_name is not None else total_from_layers

    xhat = np.asarray(data["xhat"], dtype=float)
    x, d, tt = align_for_delay(xhat, dm, t, delay_samples)
    pol = x - d  # sign convention favoured by previous truth checks

    # Align truth arrays to the same convention used by align_for_delay.
    if delay_samples > 0:
        ground_a = ground[delay_samples:]
        free_a = free[delay_samples:]
        total_a = total[delay_samples:]
    elif delay_samples < 0:
        ground_a = ground[:delay_samples]
        free_a = free[:delay_samples]
        total_a = total[:delay_samples]
    else:
        ground_a = ground
        free_a = free
        total_a = total

    n = min(len(pol), len(ground_a), len(free_a), len(total_a), len(tt))
    pol = pol[:n]
    ground_a = ground_a[:n, :pol.shape[1]]
    free_a = free_a[:n, :pol.shape[1]]
    total_a = total_a[:n, :pol.shape[1]]
    tt = tt[:n]

    corr_ground = np.nanmedian(nan_corr(pol, ground_a, axis=0)) if len(ground_idx) else np.nan
    corr_free = np.nanmedian(nan_corr(pol, free_a, axis=0)) if len(free_idx) else np.nan
    corr_total = np.nanmedian(nan_corr(pol, total_a, axis=0))

    print("\nCollapsed layer truth comparison:")
    print(f"  ground_altitude_max_m = {ground_altitude_max_m:g}")
    print(f"  ground layers = {ground_idx.tolist()} -> {[labels[i] for i in ground_idx]}")
    print(f"  free layers   = {free_idx.tolist()} -> {[labels[i] for i in free_idx]}")
    print(f"  median corr(POL=xhat-{dm_name}, truth_ground) = {corr_ground:+.3f}")
    print(f"  median corr(POL=xhat-{dm_name}, truth_free)   = {corr_free:+.3f}")
    print(f"  median corr(POL=xhat-{dm_name}, truth_total)  = {corr_total:+.3f}")

    plt.figure(figsize=(10, 5))
    plt.plot(tt, nan_rms(total_a, axis=1), label="Truth total atmosphere")
    if len(ground_idx):
        plt.plot(tt, nan_rms(ground_a, axis=1), label="Truth collapsed ground")
    if len(free_idx):
        plt.plot(tt, nan_rms(free_a, axis=1), label="Truth collapsed free atmosphere")
    plt.plot(tt, nan_rms(pol, axis=1), label=f"POL xhat - {dm_name}")
    plt.xlabel("Time [s]")
    plt.ylabel("Modal RMS [native units]")
    plt.title(f"Collapsed layer truth versus POL, delay={delay_samples}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(out_dir, f"telemetry_collapsed_truth_vs_pol_delay{delay_samples}.png")

    plt.figure(figsize=(7, 5))
    labels_bar = ["Ground", "Free", "Total"]
    values_bar = [corr_ground, corr_free, corr_total]
    plt.bar(np.arange(3), values_bar)
    plt.axhline(0, linewidth=1)
    plt.xticks(np.arange(3), labels_bar)
    plt.ylabel("Median modal correlation")
    plt.title(f"POL correlation with collapsed truth, delay={delay_samples}")
    plt.grid(True, axis="y", alpha=0.3)
    savefig(out_dir, f"telemetry_collapsed_truth_correlation_delay{delay_samples}.png")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("telemetry_npz", type=Path, help="Path to telemetry.npz")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for plots")
    parser.add_argument("--dt", type=float, default=None, help="Override timestep in seconds")
    parser.add_argument("--show", action="store_true", help="Display plots interactively after saving")
    parser.add_argument(
        "--delay-samples",
        type=int,
        default=0,
        help="Delay to use for pseudo-open-loop candidates: xhat[k] with dm_cmd[k-delay]",
    )
    parser.add_argument(
        "--also-delay-one",
        action="store_true",
        help="Also make pseudo-open-loop/collapsed-truth plots with delay_samples=1",
    )
    parser.add_argument(
        "--max-modes",
        type=int,
        default=None,
        help="Optionally truncate modal spectra to the first N modes",
    )
    parser.add_argument(
        "--plot-level",
        choices=("basic", "standard", "full"),
        default="standard",
        help=(
            "How many quicklook plots to generate. "
            "basic: overview, WFE, collapsed layer comparison. "
            "standard: basic plus TT/modal/layer/POL plots. "
            "full: standard plus slope histograms, packed-slope plots, and residual truth plots."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed array summaries and metadata. Without this, only concise diagnostics are printed.",
    )
    parser.add_argument(
        "--ground-altitude-max-m",
        type=float,
        default=500.0,
        help="Altitude threshold used to collapse truth_layer_modal_atm into ground/free components.",
    )
    parser.add_argument(
        "--ground-layer-indices",
        type=str,
        default=None,
        help="Optional comma-separated layer indices to treat as ground, overriding --ground-altitude-max-m.",
    )
    args = parser.parse_args()

    telemetry_path = args.telemetry_npz.expanduser().resolve()
    if not telemetry_path.exists():
        raise FileNotFoundError(telemetry_path)

    out_dir = args.out_dir.resolve() if args.out_dir else telemetry_path.parent / "telemetry_quicklook"
    data = np.load(telemetry_path)
    meta = load_json_if_exists(telemetry_path.parent / "telemetry_summary.json")
    dt = args.dt if args.dt is not None else load_dt(telemetry_path)

    print(f"Telemetry file: {telemetry_path}")
    print(f"Output directory: {out_dir}")
    print(f"Using dt = {dt:.6g} s")
    print(f"Plot level = {args.plot_level}")

    if args.verbose:
        print_summary(data, meta=meta)
    else:
        print("Available channels:", ", ".join(data.files))
        for key in ("n_samples", "stride", "dt_s", "dm_delay_frames", "truth_modal_projection"):
            if key in meta:
                print(f"{key}: {meta[key]}")

    t, _ = get_time_axis(data, dt)

    # Always make the key run-health and collapsed truth plots.
    plot_overview_panels(data, t, out_dir)
    plot_wfe(data, t, out_dir)
    plot_collapsed_layer_truth(
        data,
        t,
        out_dir,
        meta,
        ground_altitude_max_m=args.ground_altitude_max_m,
        ground_indices=args.ground_layer_indices,
        delay_samples=args.delay_samples,
    )

    if args.plot_level in ("standard", "full"):
        plot_tt(data, t, out_dir)
        plot_modal_spectra(data, out_dir, max_modes=args.max_modes)
        plot_layer_truth_summary(data, t, out_dir, meta)
        plot_pseudo_open_loop(data, t, out_dir, delay_samples=args.delay_samples)
        plot_truth_modal_comparison(data, t, out_dir, delay_samples=args.delay_samples)

        if args.also_delay_one and args.delay_samples != 1:
            plot_collapsed_layer_truth(
                data,
                t,
                out_dir,
                meta,
                ground_altitude_max_m=args.ground_altitude_max_m,
                ground_indices=args.ground_layer_indices,
                delay_samples=1,
            )
            plot_pseudo_open_loop(data, t, out_dir, delay_samples=1)
            plot_truth_modal_comparison(data, t, out_dir, delay_samples=1)

    if args.plot_level == "full":
        plot_rms_timeseries(data, t, out_dir)
        plot_slope_statistics(data, out_dir)

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()



#python3 experiments/ultimate_wfi_psf_reco/plot_telemetry_npz.py \
#experiments/ultimate_wfi_psf_reco/outputs/telemetry_truth_smoke/longrun_20260527_144538/telemetry.npz \
#  --plot-level basic


"""

python3 experiments/ultimate_wfi_psf_reco/plot_telemetry_npz.py \
  experiments/ultimate_wfi_psf_reco/outputs/telemetry_truth_smoke/longrun_20260527_144538/telemetry.npz \
  --plot-level basic

python3 experiments/ultimate_wfi_psf_reco/plot_telemetry_npz.py \
  experiments/ultimate_wfi_psf_reco/outputs/telemetry_truth_smoke/longrun_20260527_144538/telemetry.npz \
  --plot-level standard \
  --also-delay-one \
  --verbose \
  --show

"""
# #!/usr/bin/env python3
# """
# Quick-look plots for AOsim headless telemetry outputs.

# This script loads a ``telemetry.npz`` file written by the headless
# PSF-reconstruction experiment pipeline and produces diagnostic plots for:

# - basic channel/shape/finite-value checks,
# - AO telemetry RMS time series,
# - WFE proxy time series,
# - tip/tilt telemetry,
# - modal spectra,
# - pseudo-open-loop modal candidates,
# - optional simulator-truth modal comparisons, including total and layer-wise atmospheric truth if available.

# The pseudo-open-loop candidates are diagnostic only until the command sign and
# latency convention are validated against truth telemetry.

# Example
# -------
# From the AOsim repository root:

#     python experiments/ultimate_wfi_psf_reco/plot_telemetry_npz.py \
#         experiments/ultimate_wfi_psf_reco/outputs/telemetry_smoke/longrun_YYYYMMDD_HHMMSS/telemetry.npz \
#         --show
# """

# from __future__ import annotations

# import argparse
# import json
# from pathlib import Path
# from typing import Iterable

# import matplotlib.pyplot as plt
# import numpy as np


# def nan_rms(a: np.ndarray, axis=None) -> np.ndarray:
#     """RMS ignoring NaNs."""
#     return np.sqrt(np.nanmean(np.asarray(a, dtype=np.float64) ** 2, axis=axis))


# def finite_fraction(a: np.ndarray) -> float:
#     """Fraction of finite entries in an array."""
#     a = np.asarray(a)
#     if a.size == 0:
#         return np.nan
#     return float(np.isfinite(a).sum() / a.size)


# def nan_corr(a: np.ndarray, b: np.ndarray, axis=None) -> np.ndarray:
#     """Pearson-like correlation ignoring NaNs along a given axis."""
#     a = np.asarray(a, dtype=np.float64)
#     b = np.asarray(b, dtype=np.float64)
#     mask = np.isfinite(a) & np.isfinite(b)
#     aa = np.where(mask, a, np.nan)
#     bb = np.where(mask, b, np.nan)
#     aa = aa - np.nanmean(aa, axis=axis, keepdims=True)
#     bb = bb - np.nanmean(bb, axis=axis, keepdims=True)
#     num = np.nansum(aa * bb, axis=axis)
#     den = np.sqrt(np.nansum(aa * aa, axis=axis) * np.nansum(bb * bb, axis=axis))
#     return num / np.maximum(den, 1e-30)


# def load_json_if_exists(path: Path) -> dict:
#     if path.exists():
#         try:
#             with path.open("r") as f:
#                 return json.load(f)
#         except Exception:
#             return {}
#     return {}


# def load_dt(telemetry_path: Path, default_dt: float = 0.002) -> float:
#     """Read dt_s from telemetry_summary.json or headless_summary.json if available."""
#     for name in ("telemetry_summary.json", "headless_summary.json"):
#         d = load_json_if_exists(telemetry_path.parent / name)
#         if "dt_s" in d:
#             return float(d["dt_s"])
#         if "dt" in d:
#             return float(d["dt"])
#     return default_dt


# def savefig(out_dir: Path, name: str) -> None:
#     out_dir.mkdir(parents=True, exist_ok=True)
#     path = out_dir / name
#     plt.tight_layout()
#     plt.savefig(path, dpi=160)
#     print(f"saved: {path}")
#     plt.close()


# def print_summary(data: np.lib.npyio.NpzFile, meta: dict | None = None) -> None:
#     print("\n=== Telemetry arrays ===")
#     for k in data.files:
#         arr = data[k]
#         msg = (
#             f"{k:22s} shape={str(arr.shape):18s} "
#             f"dtype={str(arr.dtype):10s} finite={finite_fraction(arr):.3f}"
#         )
#         if np.issubdtype(arr.dtype, np.number):
#             with np.errstate(all="ignore"):
#                 msg += (
#                     f" min={np.nanmin(arr): .4e}"
#                     f" max={np.nanmax(arr): .4e}"
#                     f" rms={nan_rms(arr): .4e}"
#                 )
#         print(msg)

#     if meta:
#         print("\n=== Telemetry metadata ===")
#         for k in (
#             "dt_s",
#             "n_frames_requested",
#             "discard_first_frames",
#             "stride",
#             "n_samples",
#             "psf_tt_mode",
#             "n_wfs",
#             "n_slopes",
#             "n_modes",
#             "tt_enabled",
#             "gain",
#             "leak",
#             "dm_delay_frames",
#             "truth_modal_projection",
#         ):
#             if k in meta:
#                 print(f"{k:28s}: {meta[k]}")


# def get_time_axis(data: np.lib.npyio.NpzFile, dt: float) -> tuple[np.ndarray, str]:
#     if "frame_index" in data.files:
#         frames = np.asarray(data["frame_index"])
#         return frames * dt, "Time [s]"
#     n = len(data[data.files[0]])
#     return np.arange(n) * dt, "Time [s]"

# def first_available(data: np.lib.npyio.NpzFile, names: Iterable[str]) -> str | None:
#     """Return the first array name present in an NPZ file."""
#     for name in names:
#         if name in data.files:
#             return name
#     return None


# def dm_for_pol(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray | None, str]:
#     """Prefer the delayed command actually applied to the phase for POL checks."""
#     name = first_available(data, ("dm_cmd_applied", "dm_cmd", "dm_cmd_new"))
#     if name is None:
#         return None, ""
#     return np.asarray(data[name], dtype=float), name


# def truth_atm_total_name(data: np.lib.npyio.NpzFile) -> str | None:
#     return first_available(data, ("truth_modal_atm_total", "truth_modal_atm"))


# def truth_residual_total_name(data: np.lib.npyio.NpzFile) -> str | None:
#     return first_available(data, ("truth_modal_residual_total", "truth_modal_residual"))


# def truth_dm_name(data: np.lib.npyio.NpzFile) -> str | None:
#     return first_available(data, ("truth_modal_dm_applied", "truth_modal_dm"))


# def plot_overview_panels(data: np.lib.npyio.NpzFile, t: np.ndarray, out_dir: Path) -> None:
#     """Less misleading than putting all RMS channels on one y-axis."""
#     series = []
#     if "slopes_err_vec" in data.files:
#         series.append(("WFS slope residual RMS", nan_rms(data["slopes_err_vec"], axis=1)))
#     dm_name = first_available(data, ("dm_cmd_applied", "dm_cmd", "dm_cmd_new"))
#     if dm_name is not None:
#         series.append((f"DM command RMS ({dm_name})", nan_rms(data[dm_name], axis=1)))
#     if "xhat" in data.files:
#         series.append(("Reconstructed modal estimate RMS", nan_rms(data["xhat"], axis=1)))
#     if "dm_phi_rms" in data.files:
#         series.append(("DM phase RMS", np.asarray(data["dm_phi_rms"], dtype=float)))

#     if not series:
#         return

#     fig, axes = plt.subplots(len(series), 1, figsize=(10, 2.2 * len(series)), sharex=True)
#     if len(series) == 1:
#         axes = [axes]
#     for ax, (name, y) in zip(axes, series):
#         ax.plot(t[: len(y)], y)
#         ax.set_ylabel("RMS")
#         ax.set_title(name)
#         ax.grid(True, alpha=0.3)
#     axes[-1].set_xlabel("Time [s]")
#     fig.suptitle("AO telemetry RMS overview", y=1.01)
#     savefig(out_dir, "telemetry_overview_panels.png")


# def plot_rms_timeseries(data: np.lib.npyio.NpzFile, t: np.ndarray, out_dir: Path) -> None:
#     plt.figure(figsize=(10, 5))

#     if "slopes_err_vec" in data.files:
#         plt.plot(t, nan_rms(data["slopes_err_vec"], axis=1), label="WFS slope residual RMS")
#     dm_name = first_available(data, ("dm_cmd_applied", "dm_cmd", "dm_cmd_new"))
#     if dm_name is not None:
#         plt.plot(t, nan_rms(data[dm_name], axis=1), label=f"DM command RMS ({dm_name})")
#     if "xhat" in data.files:
#         plt.plot(t, nan_rms(data["xhat"], axis=1), label="Reconstructed modal estimate RMS")
#     if "dm_phi_rms" in data.files:
#         plt.plot(t, np.asarray(data["dm_phi_rms"]), label="DM phase RMS")

#     plt.xlabel("Time [s]")
#     plt.ylabel("RMS [native units]")
#     plt.title("AO telemetry RMS versus time")
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     savefig(out_dir, "telemetry_rms_timeseries.png")


# def plot_wfe(data: np.lib.npyio.NpzFile, t: np.ndarray, out_dir: Path) -> None:
#     if "wfe_rms" not in data.files:
#         return

#     wfe = np.asarray(data["wfe_rms"])

#     plt.figure(figsize=(10, 5))
#     if wfe.ndim == 2 and wfe.shape[1] >= 2:
#         ratio = wfe[:, 1] / np.maximum(wfe[:, 0], 1e-12)
#         plt.plot(t, wfe[:, 0], label="Uncorrected phase RMS")
#         plt.plot(t, wfe[:, 1], label="Corrected residual phase RMS")
#         plt.plot(t, ratio, label="Corrected / uncorrected")
#     else:
#         plt.plot(t, np.ravel(wfe), label="WFE RMS")

#     plt.xlabel("Time [s]")
#     plt.ylabel("RMS [rad or internal phase units]")
#     plt.title("Wavefront-error proxy versus time")
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     savefig(out_dir, "telemetry_wfe_rms.png")

#     if wfe.ndim == 2 and wfe.shape[1] >= 2:
#         plt.figure(figsize=(10, 4))
#         plt.plot(t, ratio)
#         plt.xlabel("Time [s]")
#         plt.ylabel("Corrected / uncorrected RMS")
#         plt.title("Closed-loop residual ratio")
#         plt.grid(True, alpha=0.3)
#         savefig(out_dir, "telemetry_wfe_ratio.png")


# def plot_tt(data: np.lib.npyio.NpzFile, t: np.ndarray, out_dir: Path) -> None:
#     has_cmd = "tt_cmd_yx" in data.files
#     has_meas = "tt_meas_yx" in data.files
#     if not (has_cmd or has_meas):
#         return

#     plt.figure(figsize=(10, 5))

#     if has_cmd:
#         tt = np.asarray(data["tt_cmd_yx"])
#         if tt.ndim == 2 and tt.shape[1] >= 2:
#             plt.plot(t, tt[:, 0], label="TT command y")
#             plt.plot(t, tt[:, 1], label="TT command x")

#     if has_meas:
#         tm = np.asarray(data["tt_meas_yx"])
#         if tm.ndim == 2 and tm.shape[1] >= 2:
#             plt.plot(t, tm[:, 0], ".", label="TT measurement y")
#             plt.plot(t, tm[:, 1], ".", label="TT measurement x")
#             valid = np.isfinite(tm).all(axis=1)
#             print(f"\nTT measurement valid samples: {valid.sum()} / {len(valid)}")

#     plt.xlabel("Time [s]")
#     plt.ylabel("TT [native units]")
#     plt.title("Tip/tilt telemetry")
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     savefig(out_dir, "telemetry_tt.png")


# def plot_modal_spectra(data: np.lib.npyio.NpzFile, out_dir: Path, max_modes: int | None = None) -> None:
#     plt.figure(figsize=(10, 5))

#     if "dm_cmd" in data.files:
#         dm = np.asarray(data["dm_cmd"])
#         if dm.ndim == 2:
#             y = nan_rms(dm, axis=0)
#             if max_modes is not None:
#                 y = y[:max_modes]
#             plt.plot(np.arange(len(y)), y, label="DM command modal RMS")

#     if "xhat" in data.files:
#         xhat = np.asarray(data["xhat"])
#         if xhat.ndim == 2:
#             y = nan_rms(xhat, axis=0)
#             if max_modes is not None:
#                 y = y[:max_modes]
#             plt.plot(np.arange(len(y)), y, label="Reconstructed mode RMS")

#     plt.xlabel("Mode index")
#     plt.ylabel("RMS [native units]")
#     plt.title("Modal RMS spectrum")
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     savefig(out_dir, "telemetry_modal_rms.png")


# def plot_slope_statistics(data: np.lib.npyio.NpzFile, out_dir: Path) -> None:
#     if "slopes_err_vec" not in data.files:
#         return

#     slopes = np.asarray(data["slopes_err_vec"])
#     if slopes.ndim != 2:
#         return

#     slope_rms_by_channel = nan_rms(slopes, axis=0)

#     plt.figure(figsize=(10, 5))
#     plt.plot(np.arange(slopes.shape[1]), slope_rms_by_channel)
#     plt.xlabel("Packed slope index")
#     plt.ylabel("RMS [native slope units]")
#     plt.title("WFS slope residual RMS by packed slope index")
#     plt.grid(True, alpha=0.3)
#     savefig(out_dir, "telemetry_slope_rms_by_channel.png")

#     plt.figure(figsize=(8, 5))
#     finite = np.ravel(slopes[np.isfinite(slopes)])
#     plt.hist(finite, bins=100)
#     plt.xlabel("Slope residual [native units]")
#     plt.ylabel("Count")
#     plt.title("Distribution of WFS slope residuals")
#     plt.grid(True, alpha=0.3)
#     savefig(out_dir, "telemetry_slope_histogram.png")


# def align_for_delay(a: np.ndarray, b: np.ndarray, t: np.ndarray, delay_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """Return a[k], b[k-delay], t[k] if delay_samples > 0."""
#     n = min(len(a), len(b), len(t))
#     a = a[:n]
#     b = b[:n]
#     t = t[:n]
#     d = int(delay_samples)
#     if d > 0:
#         return a[d:], b[:-d], t[d:]
#     if d < 0:
#         return a[:d], b[-d:], t[:d]
#     return a, b, t


# def plot_pseudo_open_loop(data: np.lib.npyio.NpzFile, t: np.ndarray, out_dir: Path, delay_samples: int = 0) -> None:
#     if "xhat" not in data.files:
#         return
#     dm, dm_name = dm_for_pol(data)
#     if dm is None:
#         return

#     xhat = np.asarray(data["xhat"], dtype=float)
#     x, d, tt = align_for_delay(xhat, dm, t, delay_samples)

#     pol_plus = x + d
#     pol_minus = x - d

#     rms_x = nan_rms(x, axis=1)
#     rms_d = nan_rms(d, axis=1)
#     rms_plus = nan_rms(pol_plus, axis=1)
#     rms_minus = nan_rms(pol_minus, axis=1)

#     plt.figure(figsize=(10, 5))
#     plt.plot(tt, rms_x, label="Residual modal estimate xhat RMS")
#     plt.plot(tt, rms_d, label=f"DM command RMS ({dm_name})")
#     plt.plot(tt, rms_plus, label=f"POL candidate: xhat + {dm_name}")
#     plt.plot(tt, rms_minus, label=f"POL candidate: xhat - {dm_name}")
#     plt.xlabel("Time [s]")
#     plt.ylabel("Modal RMS [native units]")
#     plt.title(f"Pseudo-open-loop modal candidates, delay={delay_samples} sample(s)")
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     savefig(out_dir, f"telemetry_pseudo_open_loop_rms_delay{delay_samples}.png")

#     plt.figure(figsize=(10, 5))
#     plt.plot(np.arange(pol_plus.shape[1]), nan_rms(pol_plus, axis=0), label=f"xhat + {dm_name}")
#     plt.plot(np.arange(pol_minus.shape[1]), nan_rms(pol_minus, axis=0), label=f"xhat - {dm_name}")
#     plt.xlabel("Mode index")
#     plt.ylabel("RMS [native units]")
#     plt.title(f"Pseudo-open-loop modal RMS spectrum, delay={delay_samples} sample(s)")
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     savefig(out_dir, f"telemetry_pseudo_open_loop_modal_spectrum_delay{delay_samples}.png")

#     if len(dm) > 1:
#         ddm = np.diff(dm, axis=0)
#         x_mid = xhat[1 : 1 + len(ddm)]
#         corr = nan_corr(ddm, x_mid, axis=0)

#         plt.figure(figsize=(10, 5))
#         plt.plot(np.arange(len(corr)), corr)
#         plt.axhline(0, linewidth=1)
#         plt.xlabel("Mode index")
#         plt.ylabel("Correlation")
#         plt.title(f"Correlation between {dm_name} increment and xhat")
#         plt.grid(True, alpha=0.3)
#         savefig(out_dir, "telemetry_dm_increment_xhat_correlation.png")


# def plot_layer_truth_summary(data: np.lib.npyio.NpzFile, t: np.ndarray, out_dir: Path, meta: dict | None = None) -> None:
#     if "truth_layer_modal_atm" not in data.files:
#         return
#     layers = np.asarray(data["truth_layer_modal_atm"], dtype=float)  # (T, L, M)
#     if layers.ndim != 3:
#         return
#     n_time, n_layer, _ = layers.shape
#     tt = t[:n_time]
#     layer_rms_t = nan_rms(layers, axis=2)  # (T, L)
#     layer_rms = nan_rms(layers, axis=(0, 2))

#     names = None
#     if meta and isinstance(meta.get("layer_name"), list) and len(meta["layer_name"]) == n_layer:
#         names = [str(x) for x in meta["layer_name"]]
#     else:
#         names = [f"L{i}" for i in range(n_layer)]

#     alt = None
#     if meta and isinstance(meta.get("layer_altitude_m"), list) and len(meta["layer_altitude_m"]) == n_layer:
#         alt = np.asarray(meta["layer_altitude_m"], dtype=float)
#         labels = [f"{names[i]} ({alt[i]/1000:.1f} km)" for i in range(n_layer)]
#     else:
#         labels = names

#     plt.figure(figsize=(10, 5))
#     for i in range(n_layer):
#         plt.plot(tt, layer_rms_t[:, i], label=labels[i])
#     plt.xlabel("Time [s]")
#     plt.ylabel("Modal RMS [native units]")
#     plt.title("Layer-wise atmospheric truth modal RMS")
#     plt.grid(True, alpha=0.3)
#     if n_layer <= 12:
#         plt.legend(fontsize=8)
#     savefig(out_dir, "telemetry_truth_layer_modal_rms_timeseries.png")

#     plt.figure(figsize=(10, 5))
#     x = np.arange(n_layer)
#     plt.bar(x, layer_rms)
#     plt.xticks(x, labels, rotation=45, ha="right")
#     plt.ylabel("Modal RMS [native units]")
#     plt.title("Layer-wise atmospheric truth modal RMS, time-averaged")
#     plt.grid(True, axis="y", alpha=0.3)
#     savefig(out_dir, "telemetry_truth_layer_modal_rms_by_layer.png")


# def plot_truth_modal_comparison(data: np.lib.npyio.NpzFile, t: np.ndarray, out_dir: Path, delay_samples: int = 0) -> None:
#     """Compare pseudo-open-loop candidates to optional simulator-truth modal channels."""
#     if "xhat" not in data.files:
#         return
#     dm, dm_name = dm_for_pol(data)
#     if dm is None:
#         return

#     atm_name = truth_atm_total_name(data)
#     res_name = truth_residual_total_name(data)
#     if atm_name is None and res_name is None:
#         return

#     xhat = np.asarray(data["xhat"], dtype=float)
#     x, d, tt = align_for_delay(xhat, dm, t, delay_samples)
#     pol_plus = x + d
#     pol_minus = x - d

#     if atm_name is not None:
#         truth = np.asarray(data[atm_name], dtype=float)
#         truth = truth[-len(pol_plus) :]
#         n_modes = min(truth.shape[1], pol_plus.shape[1])
#         truth = truth[:, :n_modes]
#         pp = pol_plus[:, :n_modes]
#         pm = pol_minus[:, :n_modes]

#         plt.figure(figsize=(10, 5))
#         plt.plot(tt[-len(truth):], nan_rms(truth, axis=1), label=f"Truth total atmosphere ({atm_name})")
#         plt.plot(tt[-len(truth):], nan_rms(pp, axis=1), label=f"POL xhat + {dm_name}")
#         plt.plot(tt[-len(truth):], nan_rms(pm, axis=1), label=f"POL xhat - {dm_name}")
#         plt.xlabel("Time [s]")
#         plt.ylabel("Modal RMS [native units]")
#         plt.title(f"Pseudo-open-loop vs truth total modal atmosphere, delay={delay_samples}")
#         plt.grid(True, alpha=0.3)
#         plt.legend()
#         savefig(out_dir, f"telemetry_pol_vs_truth_atm_delay{delay_samples}.png")

#         corr_plus = nan_corr(pp, truth, axis=0)
#         corr_minus = nan_corr(pm, truth, axis=0)
#         plt.figure(figsize=(10, 5))
#         plt.plot(np.arange(n_modes), corr_plus, label=f"corr(truth, xhat + {dm_name})")
#         plt.plot(np.arange(n_modes), corr_minus, label=f"corr(truth, xhat - {dm_name})")
#         plt.axhline(0, linewidth=1)
#         plt.xlabel("Mode index")
#         plt.ylabel("Correlation")
#         plt.title("Pseudo-open-loop sign check against truth total modal atmosphere")
#         plt.grid(True, alpha=0.3)
#         plt.legend()
#         savefig(out_dir, f"telemetry_pol_truth_correlation_delay{delay_samples}.png")

#         print("\nPseudo-open-loop truth-total-atmosphere correlation summary:")
#         print(f"  median corr(truth, xhat + {dm_name}) = {np.nanmedian(corr_plus):+.3f}")
#         print(f"  median corr(truth, xhat - {dm_name}) = {np.nanmedian(corr_minus):+.3f}")

#         if "truth_layer_modal_atm" in data.files:
#             layers = np.asarray(data["truth_layer_modal_atm"], dtype=float)
#             if layers.ndim == 3:
#                 layers = layers[-len(pm):, :, :n_modes]
#                 corr_layers = []
#                 for li in range(layers.shape[1]):
#                     corr_layers.append(np.nanmedian(nan_corr(pm, layers[:, li, :], axis=0)))
#                 plt.figure(figsize=(10, 5))
#                 plt.bar(np.arange(len(corr_layers)), corr_layers)
#                 plt.axhline(0, linewidth=1)
#                 plt.xlabel("Layer index")
#                 plt.ylabel("Median modal correlation")
#                 plt.title(f"Correlation of xhat - {dm_name} with each truth layer, delay={delay_samples}")
#                 plt.grid(True, axis="y", alpha=0.3)
#                 savefig(out_dir, f"telemetry_pol_layer_correlation_delay{delay_samples}.png")
#                 print("  layer median corr(truth_layer, xhat - command):", [f"{c:+.3f}" for c in corr_layers])

#     if res_name is not None:
#         truth_res = np.asarray(data[res_name], dtype=float)
#         n = min(len(truth_res), len(xhat), len(t))
#         truth_res = truth_res[:n]
#         xr = xhat[:n, : truth_res.shape[1]]
#         corr_res = nan_corr(xr, truth_res, axis=0)

#         plt.figure(figsize=(10, 5))
#         plt.plot(t[:n], nan_rms(truth_res, axis=1), label=f"Truth total modal residual ({res_name})")
#         plt.plot(t[:n], nan_rms(xr, axis=1), label="xhat residual estimate")
#         plt.xlabel("Time [s]")
#         plt.ylabel("Modal RMS [native units]")
#         plt.title("Residual modal estimate versus simulator total truth")
#         plt.grid(True, alpha=0.3)
#         plt.legend()
#         savefig(out_dir, "telemetry_xhat_vs_truth_residual_rms.png")

#         plt.figure(figsize=(10, 5))
#         plt.plot(np.arange(len(corr_res)), corr_res)
#         plt.axhline(0, linewidth=1)
#         plt.xlabel("Mode index")
#         plt.ylabel("Correlation")
#         plt.title("Correlation between xhat and truth total modal residual")
#         plt.grid(True, alpha=0.3)
#         savefig(out_dir, "telemetry_xhat_truth_residual_correlation.png")

#         print(f"  median corr(truth total residual, xhat) = {np.nanmedian(corr_res):+.3f}")

# def main() -> None:
#     parser = argparse.ArgumentParser()
#     parser.add_argument("telemetry_npz", type=Path, help="Path to telemetry.npz")
#     parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for plots")
#     parser.add_argument("--dt", type=float, default=None, help="Override timestep in seconds")
#     parser.add_argument("--show", action="store_true", help="Display plots interactively after saving")
#     parser.add_argument(
#         "--delay-samples",
#         type=int,
#         default=0,
#         help="Delay to use for pseudo-open-loop candidates: xhat[k] with dm_cmd[k-delay]",
#     )
#     parser.add_argument(
#         "--also-delay-one",
#         action="store_true",
#         help="Also make pseudo-open-loop plots with delay_samples=1",
#     )
#     parser.add_argument(
#         "--max-modes",
#         type=int,
#         default=None,
#         help="Optionally truncate modal spectra to the first N modes",
#     )
#     args = parser.parse_args()

#     telemetry_path = args.telemetry_npz.expanduser().resolve()
#     if not telemetry_path.exists():
#         raise FileNotFoundError(telemetry_path)

#     out_dir = args.out_dir.resolve() if args.out_dir else telemetry_path.parent / "telemetry_quicklook"
#     data = np.load(telemetry_path)
#     meta = load_json_if_exists(telemetry_path.parent / "telemetry_summary.json")
#     dt = args.dt if args.dt is not None else load_dt(telemetry_path)

#     print(f"Telemetry file: {telemetry_path}")
#     print(f"Output directory: {out_dir}")
#     print(f"Using dt = {dt:.6g} s")

#     print_summary(data, meta=meta)
#     t, _ = get_time_axis(data, dt)

#     plot_overview_panels(data, t, out_dir)
#     plot_rms_timeseries(data, t, out_dir)
#     plot_wfe(data, t, out_dir)
#     plot_tt(data, t, out_dir)
#     plot_modal_spectra(data, out_dir, max_modes=args.max_modes)
#     plot_slope_statistics(data, out_dir)
#     plot_layer_truth_summary(data, t, out_dir, meta)
#     plot_pseudo_open_loop(data, t, out_dir, delay_samples=args.delay_samples)
#     plot_truth_modal_comparison(data, t, out_dir, delay_samples=args.delay_samples)
#     if args.also_delay_one and args.delay_samples != 1:
#         plot_pseudo_open_loop(data, t, out_dir, delay_samples=1)
#         plot_truth_modal_comparison(data, t, out_dir, delay_samples=1)

#     if args.show:
#         plt.show()
#     else:
#         plt.close("all")


# if __name__ == "__main__":
#     main()

# # #!/usr/bin/env python3
# # """
# # Quick-look plots for AOsim headless telemetry outputs.

# # This script loads a ``telemetry.npz`` file written by the headless
# # PSF-reconstruction experiment pipeline and produces diagnostic plots for:

# # - basic channel/shape/finite-value checks,
# # - AO telemetry RMS time series,
# # - WFE proxy time series,
# # - tip/tilt telemetry,
# # - modal spectra,
# # - pseudo-open-loop modal candidates,
# # - optional simulator-truth modal comparisons, if truth modal channels exist.

# # The pseudo-open-loop candidates are diagnostic only until the command sign and
# # latency convention are validated against truth telemetry.

# # Example
# # -------
# # From the AOsim repository root:

# #     python experiments/ultimate_wfi_psf_reco/plot_telemetry_npz.py \
# #         experiments/ultimate_wfi_psf_reco/outputs/telemetry_smoke/longrun_YYYYMMDD_HHMMSS/telemetry.npz \
# #         --show
# # """

# # from __future__ import annotations

# # import argparse
# # import json
# # from pathlib import Path
# # from typing import Iterable

# # import matplotlib.pyplot as plt
# # import numpy as np


# # def nan_rms(a: np.ndarray, axis=None) -> np.ndarray:
# #     """RMS ignoring NaNs."""
# #     return np.sqrt(np.nanmean(np.asarray(a, dtype=np.float64) ** 2, axis=axis))


# # def finite_fraction(a: np.ndarray) -> float:
# #     """Fraction of finite entries in an array."""
# #     a = np.asarray(a)
# #     if a.size == 0:
# #         return np.nan
# #     return float(np.isfinite(a).sum() / a.size)


# # def nan_corr(a: np.ndarray, b: np.ndarray, axis=None) -> np.ndarray:
# #     """Pearson-like correlation ignoring NaNs along a given axis."""
# #     a = np.asarray(a, dtype=np.float64)
# #     b = np.asarray(b, dtype=np.float64)
# #     mask = np.isfinite(a) & np.isfinite(b)
# #     aa = np.where(mask, a, np.nan)
# #     bb = np.where(mask, b, np.nan)
# #     aa = aa - np.nanmean(aa, axis=axis, keepdims=True)
# #     bb = bb - np.nanmean(bb, axis=axis, keepdims=True)
# #     num = np.nansum(aa * bb, axis=axis)
# #     den = np.sqrt(np.nansum(aa * aa, axis=axis) * np.nansum(bb * bb, axis=axis))
# #     return num / np.maximum(den, 1e-30)


# # def load_json_if_exists(path: Path) -> dict:
# #     if path.exists():
# #         try:
# #             with path.open("r") as f:
# #                 return json.load(f)
# #         except Exception:
# #             return {}
# #     return {}


# # def load_dt(telemetry_path: Path, default_dt: float = 0.002) -> float:
# #     """Read dt_s from telemetry_summary.json or headless_summary.json if available."""
# #     for name in ("telemetry_summary.json", "headless_summary.json"):
# #         d = load_json_if_exists(telemetry_path.parent / name)
# #         if "dt_s" in d:
# #             return float(d["dt_s"])
# #         if "dt" in d:
# #             return float(d["dt"])
# #     return default_dt


# # def savefig(out_dir: Path, name: str) -> None:
# #     out_dir.mkdir(parents=True, exist_ok=True)
# #     path = out_dir / name
# #     plt.tight_layout()
# #     plt.savefig(path, dpi=160)
# #     print(f"saved: {path}")


# # def print_summary(data: np.lib.npyio.NpzFile, meta: dict | None = None) -> None:
# #     print("\n=== Telemetry arrays ===")
# #     for k in data.files:
# #         arr = data[k]
# #         msg = (
# #             f"{k:22s} shape={str(arr.shape):18s} "
# #             f"dtype={str(arr.dtype):10s} finite={finite_fraction(arr):.3f}"
# #         )
# #         if np.issubdtype(arr.dtype, np.number):
# #             with np.errstate(all="ignore"):
# #                 msg += (
# #                     f" min={np.nanmin(arr): .4e}"
# #                     f" max={np.nanmax(arr): .4e}"
# #                     f" rms={nan_rms(arr): .4e}"
# #                 )
# #         print(msg)

# #     if meta:
# #         print("\n=== Telemetry metadata ===")
# #         for k in (
# #             "dt_s",
# #             "n_frames_requested",
# #             "discard_first_frames",
# #             "stride",
# #             "n_samples",
# #             "psf_tt_mode",
# #             "n_wfs",
# #             "n_slopes",
# #             "n_modes",
# #             "tt_enabled",
# #             "gain",
# #             "leak",
# #             "dm_delay_frames",
# #             "truth_modal_projection",
# #         ):
# #             if k in meta:
# #                 print(f"{k:28s}: {meta[k]}")


# # def get_time_axis(data: np.lib.npyio.NpzFile, dt: float) -> tuple[np.ndarray, str]:
# #     if "frame_index" in data.files:
# #         frames = np.asarray(data["frame_index"])
# #         return frames * dt, "Time [s]"
# #     n = len(data[data.files[0]])
# #     return np.arange(n) * dt, "Time [s]"


# # def plot_overview_panels(data: np.lib.npyio.NpzFile, t: np.ndarray, out_dir: Path) -> None:
# #     """Less misleading than putting all RMS channels on one y-axis."""
# #     series = []
# #     if "slopes_err_vec" in data.files:
# #         series.append(("WFS slope residual RMS", nan_rms(data["slopes_err_vec"], axis=1)))
# #     if "dm_cmd" in data.files:
# #         series.append(("DM/modal command RMS", nan_rms(data["dm_cmd"], axis=1)))
# #     if "xhat" in data.files:
# #         series.append(("Reconstructed modal estimate RMS", nan_rms(data["xhat"], axis=1)))
# #     if "dm_phi_rms" in data.files:
# #         series.append(("DM phase RMS", np.asarray(data["dm_phi_rms"], dtype=float)))

# #     if not series:
# #         return

# #     fig, axes = plt.subplots(len(series), 1, figsize=(10, 2.2 * len(series)), sharex=True)
# #     if len(series) == 1:
# #         axes = [axes]
# #     for ax, (name, y) in zip(axes, series):
# #         ax.plot(t[: len(y)], y)
# #         ax.set_ylabel("RMS")
# #         ax.set_title(name)
# #         ax.grid(True, alpha=0.3)
# #     axes[-1].set_xlabel("Time [s]")
# #     fig.suptitle("AO telemetry RMS overview", y=1.01)
# #     savefig(out_dir, "telemetry_overview_panels.png")


# # def plot_rms_timeseries(data: np.lib.npyio.NpzFile, t: np.ndarray, out_dir: Path) -> None:
# #     plt.figure(figsize=(10, 5))

# #     if "slopes_err_vec" in data.files:
# #         plt.plot(t, nan_rms(data["slopes_err_vec"], axis=1), label="WFS slope residual RMS")
# #     if "dm_cmd" in data.files:
# #         plt.plot(t, nan_rms(data["dm_cmd"], axis=1), label="DM/modal command RMS")
# #     if "xhat" in data.files:
# #         plt.plot(t, nan_rms(data["xhat"], axis=1), label="Reconstructed modal estimate RMS")
# #     if "dm_phi_rms" in data.files:
# #         plt.plot(t, np.asarray(data["dm_phi_rms"]), label="DM phase RMS")

# #     plt.xlabel("Time [s]")
# #     plt.ylabel("RMS [native units]")
# #     plt.title("AO telemetry RMS versus time")
# #     plt.grid(True, alpha=0.3)
# #     plt.legend()
# #     savefig(out_dir, "telemetry_rms_timeseries.png")


# # def plot_wfe(data: np.lib.npyio.NpzFile, t: np.ndarray, out_dir: Path) -> None:
# #     if "wfe_rms" not in data.files:
# #         return

# #     wfe = np.asarray(data["wfe_rms"])

# #     plt.figure(figsize=(10, 5))
# #     if wfe.ndim == 2 and wfe.shape[1] >= 2:
# #         ratio = wfe[:, 1] / np.maximum(wfe[:, 0], 1e-12)
# #         plt.plot(t, wfe[:, 0], label="Uncorrected phase RMS")
# #         plt.plot(t, wfe[:, 1], label="Corrected residual phase RMS")
# #         plt.plot(t, ratio, label="Corrected / uncorrected")
# #     else:
# #         plt.plot(t, np.ravel(wfe), label="WFE RMS")

# #     plt.xlabel("Time [s]")
# #     plt.ylabel("RMS [rad or internal phase units]")
# #     plt.title("Wavefront-error proxy versus time")
# #     plt.grid(True, alpha=0.3)
# #     plt.legend()
# #     savefig(out_dir, "telemetry_wfe_rms.png")

# #     if wfe.ndim == 2 and wfe.shape[1] >= 2:
# #         plt.figure(figsize=(10, 4))
# #         plt.plot(t, ratio)
# #         plt.xlabel("Time [s]")
# #         plt.ylabel("Corrected / uncorrected RMS")
# #         plt.title("Closed-loop residual ratio")
# #         plt.grid(True, alpha=0.3)
# #         savefig(out_dir, "telemetry_wfe_ratio.png")


# # def plot_tt(data: np.lib.npyio.NpzFile, t: np.ndarray, out_dir: Path) -> None:
# #     has_cmd = "tt_cmd_yx" in data.files
# #     has_meas = "tt_meas_yx" in data.files
# #     if not (has_cmd or has_meas):
# #         return

# #     plt.figure(figsize=(10, 5))

# #     if has_cmd:
# #         tt = np.asarray(data["tt_cmd_yx"])
# #         if tt.ndim == 2 and tt.shape[1] >= 2:
# #             plt.plot(t, tt[:, 0], label="TT command y")
# #             plt.plot(t, tt[:, 1], label="TT command x")

# #     if has_meas:
# #         tm = np.asarray(data["tt_meas_yx"])
# #         if tm.ndim == 2 and tm.shape[1] >= 2:
# #             plt.plot(t, tm[:, 0], ".", label="TT measurement y")
# #             plt.plot(t, tm[:, 1], ".", label="TT measurement x")
# #             valid = np.isfinite(tm).all(axis=1)
# #             print(f"\nTT measurement valid samples: {valid.sum()} / {len(valid)}")

# #     plt.xlabel("Time [s]")
# #     plt.ylabel("TT [native units]")
# #     plt.title("Tip/tilt telemetry")
# #     plt.grid(True, alpha=0.3)
# #     plt.legend()
# #     savefig(out_dir, "telemetry_tt.png")


# # def plot_modal_spectra(data: np.lib.npyio.NpzFile, out_dir: Path, max_modes: int | None = None) -> None:
# #     plt.figure(figsize=(10, 5))

# #     if "dm_cmd" in data.files:
# #         dm = np.asarray(data["dm_cmd"])
# #         if dm.ndim == 2:
# #             y = nan_rms(dm, axis=0)
# #             if max_modes is not None:
# #                 y = y[:max_modes]
# #             plt.plot(np.arange(len(y)), y, label="DM command modal RMS")

# #     if "xhat" in data.files:
# #         xhat = np.asarray(data["xhat"])
# #         if xhat.ndim == 2:
# #             y = nan_rms(xhat, axis=0)
# #             if max_modes is not None:
# #                 y = y[:max_modes]
# #             plt.plot(np.arange(len(y)), y, label="Reconstructed mode RMS")

# #     plt.xlabel("Mode index")
# #     plt.ylabel("RMS [native units]")
# #     plt.title("Modal RMS spectrum")
# #     plt.grid(True, alpha=0.3)
# #     plt.legend()
# #     savefig(out_dir, "telemetry_modal_rms.png")


# # def plot_slope_statistics(data: np.lib.npyio.NpzFile, out_dir: Path) -> None:
# #     if "slopes_err_vec" not in data.files:
# #         return

# #     slopes = np.asarray(data["slopes_err_vec"])
# #     if slopes.ndim != 2:
# #         return

# #     slope_rms_by_channel = nan_rms(slopes, axis=0)

# #     plt.figure(figsize=(10, 5))
# #     plt.plot(np.arange(slopes.shape[1]), slope_rms_by_channel)
# #     plt.xlabel("Packed slope index")
# #     plt.ylabel("RMS [native slope units]")
# #     plt.title("WFS slope residual RMS by packed slope index")
# #     plt.grid(True, alpha=0.3)
# #     savefig(out_dir, "telemetry_slope_rms_by_channel.png")

# #     plt.figure(figsize=(8, 5))
# #     finite = np.ravel(slopes[np.isfinite(slopes)])
# #     plt.hist(finite, bins=100)
# #     plt.xlabel("Slope residual [native units]")
# #     plt.ylabel("Count")
# #     plt.title("Distribution of WFS slope residuals")
# #     plt.grid(True, alpha=0.3)
# #     savefig(out_dir, "telemetry_slope_histogram.png")


# # def align_for_delay(a: np.ndarray, b: np.ndarray, t: np.ndarray, delay_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
# #     """Return a[k], b[k-delay], t[k] if delay_samples > 0."""
# #     n = min(len(a), len(b), len(t))
# #     a = a[:n]
# #     b = b[:n]
# #     t = t[:n]
# #     d = int(delay_samples)
# #     if d > 0:
# #         return a[d:], b[:-d], t[d:]
# #     if d < 0:
# #         return a[:d], b[-d:], t[:d]
# #     return a, b, t


# # def plot_pseudo_open_loop(data: np.lib.npyio.NpzFile, t: np.ndarray, out_dir: Path, delay_samples: int = 0) -> None:
# #     if "xhat" not in data.files or "dm_cmd" not in data.files:
# #         return

# #     xhat = np.asarray(data["xhat"], dtype=float)
# #     dm = np.asarray(data["dm_cmd"], dtype=float)
# #     x, d, tt = align_for_delay(xhat, dm, t, delay_samples)

# #     pol_plus = x + d
# #     pol_minus = x - d

# #     rms_x = nan_rms(x, axis=1)
# #     rms_d = nan_rms(d, axis=1)
# #     rms_plus = nan_rms(pol_plus, axis=1)
# #     rms_minus = nan_rms(pol_minus, axis=1)

# #     plt.figure(figsize=(10, 5))
# #     plt.plot(tt, rms_x, label="Residual modal estimate xhat RMS")
# #     plt.plot(tt, rms_d, label="DM command RMS")
# #     plt.plot(tt, rms_plus, label="POL candidate: xhat + dm_cmd")
# #     plt.plot(tt, rms_minus, label="POL candidate: xhat - dm_cmd")
# #     plt.xlabel("Time [s]")
# #     plt.ylabel("Modal RMS [native units]")
# #     plt.title(f"Pseudo-open-loop modal candidates, delay={delay_samples} sample(s)")
# #     plt.grid(True, alpha=0.3)
# #     plt.legend()
# #     savefig(out_dir, f"telemetry_pseudo_open_loop_rms_delay{delay_samples}.png")

# #     plt.figure(figsize=(10, 5))
# #     plt.plot(np.arange(pol_plus.shape[1]), nan_rms(pol_plus, axis=0), label="xhat + dm_cmd")
# #     plt.plot(np.arange(pol_minus.shape[1]), nan_rms(pol_minus, axis=0), label="xhat - dm_cmd")
# #     plt.xlabel("Mode index")
# #     plt.ylabel("RMS [native units]")
# #     plt.title(f"Pseudo-open-loop modal RMS spectrum, delay={delay_samples} sample(s)")
# #     plt.grid(True, alpha=0.3)
# #     plt.legend()
# #     savefig(out_dir, f"telemetry_pseudo_open_loop_modal_spectrum_delay{delay_samples}.png")

# #     if len(dm) > 1:
# #         ddm = np.diff(dm, axis=0)
# #         x_mid = xhat[1 : 1 + len(ddm)]
# #         corr = nan_corr(ddm, x_mid, axis=0)

# #         plt.figure(figsize=(10, 5))
# #         plt.plot(np.arange(len(corr)), corr)
# #         plt.axhline(0, linewidth=1)
# #         plt.xlabel("Mode index")
# #         plt.ylabel("Correlation")
# #         plt.title("Correlation between DM command increment and xhat")
# #         plt.grid(True, alpha=0.3)
# #         savefig(out_dir, "telemetry_dm_increment_xhat_correlation.png")


# # def plot_truth_modal_comparison(data: np.lib.npyio.NpzFile, t: np.ndarray, out_dir: Path, delay_samples: int = 0) -> None:
# #     """Compare pseudo-open-loop candidates to optional simulator-truth modal channels."""
# #     if "xhat" not in data.files or "dm_cmd" not in data.files:
# #         return
# #     if "truth_modal_atm" not in data.files and "truth_modal_residual" not in data.files:
# #         return

# #     xhat = np.asarray(data["xhat"], dtype=float)
# #     dm = np.asarray(data["dm_cmd"], dtype=float)
# #     x, d, tt = align_for_delay(xhat, dm, t, delay_samples)
# #     pol_plus = x + d
# #     pol_minus = x - d

# #     if "truth_modal_atm" in data.files:
# #         truth = np.asarray(data["truth_modal_atm"], dtype=float)
# #         truth = truth[-len(pol_plus) :]
# #         n_modes = min(truth.shape[1], pol_plus.shape[1])
# #         truth = truth[:, :n_modes]
# #         pp = pol_plus[:, :n_modes]
# #         pm = pol_minus[:, :n_modes]

# #         plt.figure(figsize=(10, 5))
# #         plt.plot(tt[-len(truth):], nan_rms(truth, axis=1), label="Truth modal atmosphere")
# #         plt.plot(tt[-len(truth):], nan_rms(pp, axis=1), label="POL xhat + dm_cmd")
# #         plt.plot(tt[-len(truth):], nan_rms(pm, axis=1), label="POL xhat - dm_cmd")
# #         plt.xlabel("Time [s]")
# #         plt.ylabel("Modal RMS [native units]")
# #         plt.title(f"Pseudo-open-loop vs truth modal atmosphere, delay={delay_samples}")
# #         plt.grid(True, alpha=0.3)
# #         plt.legend()
# #         savefig(out_dir, f"telemetry_pol_vs_truth_atm_delay{delay_samples}.png")

# #         corr_plus = nan_corr(pp, truth, axis=0)
# #         corr_minus = nan_corr(pm, truth, axis=0)
# #         plt.figure(figsize=(10, 5))
# #         plt.plot(np.arange(n_modes), corr_plus, label="corr(truth, xhat + dm_cmd)")
# #         plt.plot(np.arange(n_modes), corr_minus, label="corr(truth, xhat - dm_cmd)")
# #         plt.axhline(0, linewidth=1)
# #         plt.xlabel("Mode index")
# #         plt.ylabel("Correlation")
# #         plt.title("Pseudo-open-loop sign check against truth modal atmosphere")
# #         plt.grid(True, alpha=0.3)
# #         plt.legend()
# #         savefig(out_dir, f"telemetry_pol_truth_correlation_delay{delay_samples}.png")

# #         print("\nPseudo-open-loop truth correlation summary:")
# #         print(f"  median corr(truth, xhat + dm_cmd) = {np.nanmedian(corr_plus):+.3f}")
# #         print(f"  median corr(truth, xhat - dm_cmd) = {np.nanmedian(corr_minus):+.3f}")

# #     if "truth_modal_residual" in data.files:
# #         truth_res = np.asarray(data["truth_modal_residual"], dtype=float)
# #         n = min(len(truth_res), len(xhat), len(t))
# #         truth_res = truth_res[:n]
# #         xr = xhat[:n, : truth_res.shape[1]]
# #         corr_res = nan_corr(xr, truth_res, axis=0)

# #         plt.figure(figsize=(10, 5))
# #         plt.plot(t[:n], nan_rms(truth_res, axis=1), label="Truth modal residual")
# #         plt.plot(t[:n], nan_rms(xr, axis=1), label="xhat residual estimate")
# #         plt.xlabel("Time [s]")
# #         plt.ylabel("Modal RMS [native units]")
# #         plt.title("Residual modal estimate versus simulator truth")
# #         plt.grid(True, alpha=0.3)
# #         plt.legend()
# #         savefig(out_dir, "telemetry_xhat_vs_truth_residual_rms.png")

# #         plt.figure(figsize=(10, 5))
# #         plt.plot(np.arange(len(corr_res)), corr_res)
# #         plt.axhline(0, linewidth=1)
# #         plt.xlabel("Mode index")
# #         plt.ylabel("Correlation")
# #         plt.title("Correlation between xhat and truth modal residual")
# #         plt.grid(True, alpha=0.3)
# #         savefig(out_dir, "telemetry_xhat_truth_residual_correlation.png")

# #         print(f"  median corr(truth residual, xhat) = {np.nanmedian(corr_res):+.3f}")


# # def main() -> None:
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("telemetry_npz", type=Path, help="Path to telemetry.npz")
# #     parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for plots")
# #     parser.add_argument("--dt", type=float, default=None, help="Override timestep in seconds")
# #     parser.add_argument("--show", action="store_true", help="Display plots interactively after saving")
# #     parser.add_argument(
# #         "--delay-samples",
# #         type=int,
# #         default=0,
# #         help="Delay to use for pseudo-open-loop candidates: xhat[k] with dm_cmd[k-delay]",
# #     )
# #     parser.add_argument(
# #         "--also-delay-one",
# #         action="store_true",
# #         help="Also make pseudo-open-loop plots with delay_samples=1",
# #     )
# #     parser.add_argument(
# #         "--max-modes",
# #         type=int,
# #         default=None,
# #         help="Optionally truncate modal spectra to the first N modes",
# #     )
# #     args = parser.parse_args()

# #     telemetry_path = args.telemetry_npz.expanduser().resolve()
# #     if not telemetry_path.exists():
# #         raise FileNotFoundError(telemetry_path)

# #     out_dir = args.out_dir.resolve() if args.out_dir else telemetry_path.parent / "telemetry_quicklook"
# #     data = np.load(telemetry_path)
# #     meta = load_json_if_exists(telemetry_path.parent / "telemetry_summary.json")
# #     dt = args.dt if args.dt is not None else load_dt(telemetry_path)

# #     print(f"Telemetry file: {telemetry_path}")
# #     print(f"Output directory: {out_dir}")
# #     print(f"Using dt = {dt:.6g} s")

# #     print_summary(data, meta=meta)
# #     t, _ = get_time_axis(data, dt)

# #     plot_overview_panels(data, t, out_dir)
# #     plot_rms_timeseries(data, t, out_dir)
# #     plot_wfe(data, t, out_dir)
# #     plot_tt(data, t, out_dir)
# #     plot_modal_spectra(data, out_dir, max_modes=args.max_modes)
# #     plot_slope_statistics(data, out_dir)
# #     plot_pseudo_open_loop(data, t, out_dir, delay_samples=args.delay_samples)
# #     plot_truth_modal_comparison(data, t, out_dir, delay_samples=args.delay_samples)
# #     if args.also_delay_one and args.delay_samples != 1:
# #         plot_pseudo_open_loop(data, t, out_dir, delay_samples=1)
# #         plot_truth_modal_comparison(data, t, out_dir, delay_samples=1)

# #     if args.show:
# #         plt.show()
# #     else:
# #         plt.close("all")


# # if __name__ == "__main__":
# #     main()