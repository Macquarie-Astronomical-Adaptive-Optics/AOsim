#!/usr/bin/env python3
"""
Validate GLAO ground-layer telemetry and regularization-aware covariance completion.

This bridges the toy PSF-R demonstrations and the AOsim GLAO telemetry outputs.
For GLAO, pseudo-open-loop telemetry is compared to the ground-conjugated/common
component, not the total atmosphere.

Typical use:

RUN_DIR=$(ls -td experiments/ultimate_wfi_psf_reco/outputs/telemetry_truth_smoke/longrun_* | head -1)

python experiments/ultimate_wfi_psf_reco/validate_glao_ground_layer_psfr.py \
    "$RUN_DIR" \
    --ground-altitude-max-m 500 \
    --gain-floor 0.12 \
    --delay-candidates -1 0 1 \
    --outdir "$RUN_DIR/glao_ground_psfr_validation" \
    --verbose
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)


def first_present(npz: np.lib.npyio.NpzFile, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in npz.files:
            return name
    return None


def covariance(ts: np.ndarray) -> np.ndarray:
    x = np.asarray(ts, dtype=np.float64)
    x = x - np.nanmean(x, axis=0, keepdims=True)
    n = x.shape[0]
    if n < 2:
        raise ValueError("Need at least two samples for covariance")
    C = (x.T @ x) / float(n - 1)
    return 0.5 * (C + C.T)


def covariance_in_basis(ts: np.ndarray, V: np.ndarray) -> np.ndarray:
    z = np.asarray(ts, dtype=np.float64) @ np.asarray(V, dtype=np.float64)
    return covariance(z)


def rel_fro_error(C_true: np.ndarray, C_est: np.ndarray) -> float:
    denom = np.linalg.norm(C_true)
    if denom <= 0:
        return float("nan")
    return float(np.linalg.norm(C_true - C_est) / denom)


def rel_diag_error(C_true: np.ndarray, C_est: np.ndarray) -> float:
    a = np.diag(C_true)
    b = np.diag(C_est)
    denom = np.linalg.norm(a)
    if denom <= 0:
        return float("nan")
    return float(np.linalg.norm(a - b) / denom)


def block_error(C_true: np.ndarray, C_est: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> float:
    if rows.size == 0 or cols.size == 0:
        return float("nan")
    return rel_fro_error(C_true[np.ix_(rows, cols)], C_est[np.ix_(rows, cols)])


def block_diag_error(C_true: np.ndarray, C_est: np.ndarray, idx: np.ndarray) -> float:
    if idx.size == 0:
        return float("nan")
    return rel_diag_error(C_true[np.ix_(idx, idx)], C_est[np.ix_(idx, idx)])


def nan_corr_by_mode(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(a.shape[0], b.shape[0])
    m = min(a.shape[1], b.shape[1])
    a = a[:n, :m]
    b = b[:n, :m]

    mask = np.isfinite(a) & np.isfinite(b)
    aa = np.where(mask, a, np.nan)
    bb = np.where(mask, b, np.nan)
    aa = aa - np.nanmean(aa, axis=0, keepdims=True)
    bb = bb - np.nanmean(bb, axis=0, keepdims=True)

    num = np.nansum(aa * bb, axis=0)
    den = np.sqrt(np.nansum(aa * aa, axis=0) * np.nansum(bb * bb, axis=0))
    return num / np.maximum(den, 1e-30)


def align_xhat_dm_truth(
    xhat: np.ndarray,
    dm: np.ndarray,
    truth_arrays: dict[str, np.ndarray],
    delay: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Align xhat[k] with dm[k-delay] and truth[k]."""
    n = min(len(xhat), len(dm), *(len(v) for v in truth_arrays.values()))
    xhat = xhat[:n]
    dm = dm[:n]
    truth_arrays = {k: v[:n] for k, v in truth_arrays.items()}

    d = int(delay)
    if d > 0:
        return xhat[d:], dm[:-d], {k: v[d:] for k, v in truth_arrays.items()}
    if d < 0:
        return xhat[:d], dm[-d:], {k: v[:d] for k, v in truth_arrays.items()}
    return xhat, dm, truth_arrays


def collapse_layers(
    truth_layer_modal_atm: np.ndarray,
    telemetry_summary: dict[str, Any],
    ground_altitude_max_m: float,
    ground_layer_indices: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    layers = np.asarray(truth_layer_modal_atm, dtype=np.float64)
    if layers.ndim != 3:
        raise ValueError(f"truth_layer_modal_atm must have shape (time, layer, modes), got {layers.shape}")
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
        ground = np.asarray([int(x.strip()) for x in ground_layer_indices.split(",") if x.strip() != ""], dtype=int)
        ground = ground[(ground >= 0) & (ground < n_layer)]
    elif np.all(np.isfinite(alt_m)):
        ground = np.where(alt_m <= float(ground_altitude_max_m))[0].astype(int)
    else:
        ground = np.asarray([0], dtype=int)

    all_idx = np.arange(n_layer, dtype=int)
    ground_set = set(ground.tolist())
    free = np.asarray([i for i in all_idx if i not in ground_set], dtype=int)

    ground_ts = np.nansum(layers[:, ground, :], axis=1) if ground.size else np.zeros((layers.shape[0], layers.shape[2]))
    free_ts = np.nansum(layers[:, free, :], axis=1) if free.size else np.zeros((layers.shape[0], layers.shape[2]))

    labels = []
    for i in range(n_layer):
        if np.isfinite(alt_m[i]):
            labels.append(f"{names[i]} ({alt_m[i]/1000:.2f} km)")
        else:
            labels.append(names[i])

    return ground_ts, free_ts, ground, free, labels


def compute_control_basis(control: np.lib.npyio.NpzFile, gain_floor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    T_name = first_present(control, ("T_mode_to_mode", "T_mode_to_mode_pre_waffle"))
    if T_name is None:
        raise KeyError("control_bundle.npz does not contain T_mode_to_mode or T_mode_to_mode_pre_waffle")
    T = np.asarray(control[T_name], dtype=np.float64)
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError(f"{T_name} must be square, got {T.shape}")
    _, s, Vt = np.linalg.svd(T, full_matrices=True)
    V = Vt.T
    trusted = s >= float(gain_floor)
    return V, s, trusted, T_name


def make_completion_covariances(
    C_tel: np.ndarray,
    C_prior: np.ndarray,
    trusted: np.ndarray,
    residual_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n = C_prior.shape[0]
    trusted_idx = np.where(trusted)[0]

    if residual_weight is not None:
        w = np.asarray(residual_weight, dtype=np.float64)
        C_prior_eff = (w[:, None] * C_prior) * w[None, :]
    else:
        C_prior_eff = np.array(C_prior, copy=True)

    C_tel_only = np.zeros((n, n), dtype=np.float64)
    C_regaware = np.array(C_prior_eff, copy=True)

    if trusted_idx.size:
        C_tel_only[np.ix_(trusted_idx, trusted_idx)] = C_tel[np.ix_(trusted_idx, trusted_idx)]
        C_regaware[np.ix_(trusted_idx, trusted_idx)] = C_tel[np.ix_(trusted_idx, trusted_idx)]

    return 0.5 * (C_tel_only + C_tel_only.T), 0.5 * (C_regaware + C_regaware.T)


def summarize_cov_errors(C_true: np.ndarray, C_tel: np.ndarray, C_reg: np.ndarray, trusted: np.ndarray) -> dict[str, Any]:
    trusted_idx = np.where(trusted)[0]
    missing_idx = np.where(~trusted)[0]

    def pack(C_est: np.ndarray) -> dict[str, float]:
        return {
            "full_fro": rel_fro_error(C_true, C_est),
            "full_diag": rel_diag_error(C_true, C_est),
            "trusted_trusted_fro": block_error(C_true, C_est, trusted_idx, trusted_idx),
            "missing_missing_fro": block_error(C_true, C_est, missing_idx, missing_idx),
            "trusted_missing_fro": block_error(C_true, C_est, trusted_idx, missing_idx),
            "missing_missing_diag": block_diag_error(C_true, C_est, missing_idx),
        }

    out = {"telemetry_only": pack(C_tel), "regularization_aware": pack(C_reg)}
    for key in list(out["telemetry_only"].keys()):
        tel = out["telemetry_only"][key]
        reg = out["regularization_aware"][key]
        out[f"gain_{key}"] = float(tel / reg) if np.isfinite(tel) and np.isfinite(reg) and reg > 0 else float("nan")
    return out


def plot_singular_values(s: np.ndarray, trusted: np.ndarray, gain_floor: float, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.semilogy(np.arange(len(s)), s, lw=2)
    ax.axhline(gain_floor, ls="--", label=f"gain floor = {gain_floor:g}")
    ax.set_xlabel("Singular direction index")
    ax.set_ylabel("Singular value of T")
    ax.set_title(f"Control operator spectrum: {int(np.sum(trusted))}/{len(s)} trusted")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "glao_control_singular_values.png")
    plt.close(fig)


def plot_pol_correlations(pol_results: list[dict[str, Any]], outdir: Path) -> None:
    labels = [f"{r['label']}, d={r['delay']}" for r in pol_results]
    ground = [r["median_corr_ground"] for r in pol_results]
    free = [r["median_corr_free"] for r in pol_results]
    total = [r["median_corr_total"] for r in pol_results]

    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(10, 1.4 * len(labels)), 5), dpi=150)
    ax.bar(x - width, ground, width, label="Ground truth")
    ax.bar(x, free, width, label="Free truth")
    ax.bar(x + width, total, width, label="Total truth")
    ax.axhline(0, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Median per-mode correlation")
    ax.set_title("POL candidate correlations: GLAO should correlate with ground")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "glao_pol_ground_free_correlations.png")
    plt.close(fig)


def plot_variance_in_basis(
    C_true: np.ndarray,
    C_tel: np.ndarray,
    C_reg: np.ndarray,
    trusted: np.ndarray,
    outdir: Path,
    title: str,
    filename: str,
) -> None:
    diag_true = np.maximum(np.diag(C_true), 1e-30)
    diag_tel = np.maximum(np.diag(C_tel), 1e-30)
    diag_reg = np.maximum(np.diag(C_reg), 1e-30)
    x = np.arange(len(diag_true))

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.semilogy(x, diag_true, label="truth", lw=2)
    ax.semilogy(x, diag_tel, label="telemetry-only", lw=1.6, ls="--")
    ax.semilogy(x, diag_reg, label="regularization-aware", lw=1.6, ls="-.")
    n_trusted = int(np.sum(trusted))
    ax.axvline(n_trusted - 0.5, color="0.5", ls=":", label="trusted/missing boundary")
    ax.set_xlabel("Control-basis singular direction")
    ax.set_ylabel("Variance [native units$^2$]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / filename)
    plt.close(fig)


def plot_error_bars(error_summary: dict[str, Any], outdir: Path, stem: str, title: str) -> None:
    keys = ["full_fro", "trusted_trusted_fro", "missing_missing_fro", "trusted_missing_fro", "missing_missing_diag"]
    labels = ["full", "trusted-trusted", "missing-missing", "trusted-missing", "missing diag"]
    tel = [100.0 * error_summary["telemetry_only"][k] for k in keys]
    reg = [100.0 * error_summary["regularization_aware"][k] for k in keys]
    x = np.arange(len(keys))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    ax.bar(x - width / 2, tel, width, label="telemetry-only")
    ax.bar(x + width / 2, reg, width, label="regularization-aware")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Relative Frobenius/diagonal error [%]")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}_subspace_error_bars.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate GLAO ground-layer PSF-R telemetry/control-bundle logic.")
    p.add_argument("run_dir", type=Path, help="Run directory containing telemetry.npz and control_bundle.npz")
    p.add_argument("--telemetry", type=Path, default=None, help="Override telemetry.npz path")
    p.add_argument("--control-bundle", type=Path, default=None, help="Override control_bundle.npz path")
    p.add_argument("--outdir", type=Path, default=None, help="Output directory")
    p.add_argument("--ground-altitude-max-m", type=float, default=500.0)
    p.add_argument("--ground-layer-indices", type=str, default=None)
    p.add_argument("--gain-floor", type=float, default=0.12)
    p.add_argument("--delay-candidates", type=int, nargs="+", default=[-1, 0, 1])
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    telemetry_path = args.telemetry.expanduser().resolve() if args.telemetry else run_dir / "telemetry.npz"
    control_path = args.control_bundle.expanduser().resolve() if args.control_bundle else run_dir / "control_bundle.npz"
    outdir = args.outdir.expanduser().resolve() if args.outdir else run_dir / "glao_ground_psfr_validation"
    outdir.mkdir(parents=True, exist_ok=True)

    telemetry_summary = load_json(run_dir / "telemetry_summary.json")
    control_summary = load_json(run_dir / "control_bundle_summary.json")

    tel = np.load(telemetry_path)
    control = np.load(control_path)

    for name in ["xhat", "truth_layer_modal_atm"]:
        if name not in tel.files:
            raise KeyError(f"telemetry.npz is missing required channel {name!r}")

    dm_name = first_present(tel, ("dm_cmd_applied", "dm_cmd", "dm_cmd_new"))
    if dm_name is None:
        raise KeyError("No DM command channel found; expected dm_cmd_applied, dm_cmd, or dm_cmd_new")

    xhat = np.asarray(tel["xhat"], dtype=np.float64)
    dm = np.asarray(tel[dm_name], dtype=np.float64)
    truth_total_name = first_present(tel, ("truth_modal_atm_total", "truth_modal_atm"))
    truth_dm_name = first_present(tel, ("truth_modal_dm_applied", "truth_modal_dm"))

    ground, free, ground_idx, free_idx, layer_labels = collapse_layers(
        tel["truth_layer_modal_atm"],
        telemetry_summary,
        ground_altitude_max_m=args.ground_altitude_max_m,
        ground_layer_indices=args.ground_layer_indices,
    )
    total = np.asarray(tel[truth_total_name], dtype=np.float64) if truth_total_name else ground + free
    truth_dm = np.asarray(tel[truth_dm_name], dtype=np.float64) if truth_dm_name else dm

    n_modes = min(xhat.shape[1], dm.shape[1], ground.shape[1], free.shape[1], total.shape[1], truth_dm.shape[1])
    xhat = xhat[:, :n_modes]
    dm = dm[:, :n_modes]
    ground = ground[:, :n_modes]
    free = free[:, :n_modes]
    total = total[:, :n_modes]
    truth_dm = truth_dm[:, :n_modes]

    V, s, trusted, T_name = compute_control_basis(control, args.gain_floor)
    V = V[:n_modes, :n_modes]
    s = s[:n_modes]
    trusted = trusted[:n_modes]
    missing = ~trusted

    plot_singular_values(s, trusted, args.gain_floor, outdir)

    pol_results = []
    truth_for_align = {"ground": ground, "free": free, "total": total}
    for delay in args.delay_candidates:
        for sign, label in [(+1.0, f"xhat + {dm_name}"), (-1.0, f"xhat - {dm_name}")]:
            xa, da, truth_a = align_xhat_dm_truth(xhat, dm, truth_for_align, delay)
            pol = xa + sign * da
            pol_results.append(
                {
                    "label": label,
                    "sign": sign,
                    "delay": int(delay),
                    "dm_name": dm_name,
                    "n_samples": int(pol.shape[0]),
                    "n_modes": int(pol.shape[1]),
                    "median_corr_ground": float(np.nanmedian(nan_corr_by_mode(pol, truth_a["ground"]))),
                    "median_corr_free": float(np.nanmedian(nan_corr_by_mode(pol, truth_a["free"]))),
                    "median_corr_total": float(np.nanmedian(nan_corr_by_mode(pol, truth_a["total"]))),
                    "mean_corr_ground": float(np.nanmean(nan_corr_by_mode(pol, truth_a["ground"]))),
                    "mean_corr_free": float(np.nanmean(nan_corr_by_mode(pol, truth_a["free"]))),
                    "mean_corr_total": float(np.nanmean(nan_corr_by_mode(pol, truth_a["total"]))),
                }
            )

    best_pol = max(pol_results, key=lambda r: r["median_corr_ground"])
    plot_pol_correlations(pol_results, outdir)

    xa, da, truth_a = align_xhat_dm_truth(
        xhat,
        dm,
        {"ground": ground, "free": free, "total": total, "truth_dm": truth_dm},
        best_pol["delay"],
    )
    pol_best = xa + best_pol["sign"] * da
    ground_a = truth_a["ground"]
    free_a = truth_a["free"]
    total_a = truth_a["total"]
    truth_dm_a = truth_a["truth_dm"]

    # From previous truth bookkeeping: atmosphere ~= residual + DM, so residual ~= atmosphere - DM.
    ground_res_true = ground_a - truth_dm_a

    # Ground-atmosphere covariance completion using POL.  Oracle prior for first validation.
    C_ground_true_b = covariance_in_basis(ground_a, V)
    C_pol_b = covariance_in_basis(pol_best, V)
    C_ground_prior_b = np.array(C_ground_true_b, copy=True)
    C_ground_tel_b, C_ground_reg_b = make_completion_covariances(C_pol_b, C_ground_prior_b, trusted)
    ground_atm_errors = summarize_cov_errors(C_ground_true_b, C_ground_tel_b, C_ground_reg_b, trusted)

    plot_variance_in_basis(
        C_ground_true_b,
        C_ground_tel_b,
        C_ground_reg_b,
        trusted,
        outdir,
        title="Ground atmospheric covariance in control basis",
        filename="ground_atmosphere_variance_control_basis.png",
    )
    plot_error_bars(ground_atm_errors, outdir, "ground_atmosphere", "Ground atmospheric covariance completion errors")

    # Ground-residual covariance completion. Trusted block from xhat; missing block from prior propagated by (1-s).
    C_ground_res_true_b = covariance_in_basis(ground_res_true, V)
    C_xhat_b = covariance_in_basis(xa, V)
    residual_weight = 1.0 - np.clip(s, 0.0, 1.0)
    C_ground_res_tel_b, C_ground_res_reg_b = make_completion_covariances(
        C_xhat_b,
        C_ground_prior_b,
        trusted,
        residual_weight=residual_weight,
    )
    ground_res_errors = summarize_cov_errors(C_ground_res_true_b, C_ground_res_tel_b, C_ground_res_reg_b, trusted)

    plot_variance_in_basis(
        C_ground_res_true_b,
        C_ground_res_tel_b,
        C_ground_res_reg_b,
        trusted,
        outdir,
        title="Ground residual covariance in control basis",
        filename="ground_residual_variance_control_basis.png",
    )
    plot_error_bars(ground_res_errors, outdir, "ground_residual", "Ground residual covariance completion errors")

    total_from_layers = ground + free
    n_common = min(total_from_layers.shape[0], total.shape[0])
    rel_layers_to_total = float(
        np.linalg.norm(total_from_layers[:n_common] - total[:n_common]) / max(np.linalg.norm(total[:n_common]), 1e-30)
    )

    payload = {
        "run_dir": str(run_dir),
        "telemetry_npz": str(telemetry_path),
        "control_bundle_npz": str(control_path),
        "outdir": str(outdir),
        "dm_channel_used": dm_name,
        "truth_total_channel": truth_total_name,
        "truth_dm_channel": truth_dm_name,
        "ground_altitude_max_m": args.ground_altitude_max_m,
        "ground_layer_indices": ground_idx.tolist(),
        "free_layer_indices": free_idx.tolist(),
        "layer_labels": layer_labels,
        "n_samples_input": int(xhat.shape[0]),
        "n_modes": int(n_modes),
        "control_operator": {
            "T_name": T_name,
            "gain_floor": float(args.gain_floor),
            "n_trusted": int(np.sum(trusted)),
            "n_missing": int(np.sum(missing)),
            "s_min": float(np.min(s)),
            "s_median": float(np.median(s)),
            "s_max": float(np.max(s)),
        },
        "layer_truth_bookkeeping": {
            "rel_error_sum_ground_plus_free_vs_truth_total": rel_layers_to_total,
        },
        "pol_ground_free_results": pol_results,
        "best_pol_by_ground_correlation": best_pol,
        "ground_atmosphere_covariance_completion": ground_atm_errors,
        "ground_residual_covariance_completion": ground_res_errors,
        "notes": {
            "oracle_prior": "C_ground_prior is set equal to simulator truth ground covariance. This is a plumbing/isolation test, not an operational PSF-R prior.",
            "glao_interpretation": "POL is evaluated against the collapsed ground component, not the total atmosphere.",
            "residual_estimator": "Trusted residual covariance is estimated from xhat; missing residual covariance is supplied by (1-S) C_ground_prior (1-S)^T in the control-basis singular directions.",
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
            "truth_modal_projection_note": telemetry_summary.get("truth_modal_projection_note"),
        },
        "control_summary_subset": {
            "reconstructor": control_summary.get("reconstructor", {}),
            "runner": control_summary.get("runner", {}),
        },
    }

    summary_path = outdir / "glao_ground_psfr_validation_summary.json"
    save_json(payload, summary_path)

    np.savez_compressed(
        outdir / "glao_ground_psfr_validation_arrays.npz",
        singular_values=s,
        trusted=trusted,
        ground_layer_indices=ground_idx,
        free_layer_indices=free_idx,
        C_ground_true_b=C_ground_true_b,
        C_ground_tel_b=C_ground_tel_b,
        C_ground_reg_b=C_ground_reg_b,
        C_ground_res_true_b=C_ground_res_true_b,
        C_ground_res_tel_b=C_ground_res_tel_b,
        C_ground_res_reg_b=C_ground_res_reg_b,
    )

    print("\n=== GLAO ground-layer validation summary ===")
    print(f"Run dir: {run_dir}")
    print(f"Output:  {outdir}")
    print(f"Ground layers: {ground_idx.tolist()}")
    print(f"Free layers:   {free_idx.tolist()}")
    print(f"Control trusted/missing: {int(np.sum(trusted))}/{int(np.sum(missing))}")
    print(
        "Best POL by ground correlation: "
        f"{best_pol['label']}, delay={best_pol['delay']}, "
        f"corr_ground={best_pol['median_corr_ground']:+.3f}, "
        f"corr_free={best_pol['median_corr_free']:+.3f}, "
        f"corr_total={best_pol['median_corr_total']:+.3f}"
    )
    print("\nGround atmospheric covariance error, full/missing-missing:")
    print(
        f"  telemetry-only: {100*ground_atm_errors['telemetry_only']['full_fro']:.2f}% / "
        f"{100*ground_atm_errors['telemetry_only']['missing_missing_fro']:.2f}%"
    )
    print(
        f"  reg-aware:      {100*ground_atm_errors['regularization_aware']['full_fro']:.2f}% / "
        f"{100*ground_atm_errors['regularization_aware']['missing_missing_fro']:.2f}%"
    )
    print("\nGround residual covariance error, full/missing-missing:")
    print(
        f"  telemetry-only: {100*ground_res_errors['telemetry_only']['full_fro']:.2f}% / "
        f"{100*ground_res_errors['telemetry_only']['missing_missing_fro']:.2f}%"
    )
    print(
        f"  reg-aware:      {100*ground_res_errors['regularization_aware']['full_fro']:.2f}% / "
        f"{100*ground_res_errors['regularization_aware']['missing_missing_fro']:.2f}%"
    )
    print(f"\nWrote: {summary_path}")

    if args.verbose:
        print("\nKey plots:")
        for p in sorted(outdir.glob("*.png")):
            print(f"  {p}")


if __name__ == "__main__":
    main()
# #!/usr/bin/env python3
# """
# Validate AOsim PSF-R telemetry/control-bundle outputs and run a first modal
# regularization-aware covariance-completion diagnostic.

# This is intentionally a validation/development script, not the final PSF-R
# algorithm.  It answers four questions for a headless AOsim run directory:

# 1. Do telemetry.npz and control_bundle.npz have compatible dimensions?
# 2. What is the simulator sign convention for atm/residual/DM truth?
# 3. Which pseudo-open-loop candidate xhat +/- dm_cmd_applied, and which small
#    delay, best matches the simulator atmospheric truth?
# 4. Does the real saved control operator T_mode_to_mode expose damped/weakly
#    controlled modes, and does a regularization-aware covariance splice improve
#    the residual modal covariance relative to a telemetry-only splice?

# Important: the default regularization-aware diagnostic uses an ORACLE prior from
# truth_modal_atm_total.  That is intentionally validation-only.  It tests whether
# our control-bundle plumbing and covariance propagation are sensible before we
# replace the prior with an operational atmospheric model.

# Typical use from the AOsim repository root:

#     RUN_DIR=$(ls -td experiments/ultimate_wfi_psf_reco/outputs/telemetry_truth_smoke/longrun_* | head -1)

#     python experiments/ultimate_wfi_psf_reco/validate_control_bundle_and_modal_psfr.py \
#         "$RUN_DIR" \
#         --gain-floor 0.12 \
#         --delay-candidates -1 0 1 \
#         --outdir "$RUN_DIR/modal_psfr_validation"

# Outputs:
#     modal_psfr_validation_summary.json
#     control_singular_values.png
#     mode_variance_comparison.png
#     residual_covariance_error_bars.png
# """

# from __future__ import annotations

# import argparse
# import json
# from pathlib import Path
# from typing import Any, Iterable

# import numpy as np
# import matplotlib.pyplot as plt


# # -----------------------------------------------------------------------------
# # Basic utilities
# # -----------------------------------------------------------------------------


# def load_json(path: Path) -> dict[str, Any]:
#     if not path.exists():
#         return {}
#     try:
#         with path.open("r", encoding="utf-8") as f:
#             return json.load(f)
#     except Exception:
#         return {}


# def finite_fraction(a: np.ndarray) -> float:
#     a = np.asarray(a)
#     if a.size == 0:
#         return float("nan")
#     return float(np.isfinite(a).sum() / a.size)


# def rms(a: np.ndarray, axis=None) -> np.ndarray:
#     return np.sqrt(np.nanmean(np.asarray(a, dtype=np.float64) ** 2, axis=axis))


# def demean_time(a: np.ndarray) -> np.ndarray:
#     a = np.asarray(a, dtype=np.float64)
#     return a - np.nanmean(a, axis=0, keepdims=True)


# def covariance_time(a: np.ndarray) -> np.ndarray:
#     """Time covariance for array shape (n_time, n_modes)."""
#     a = demean_time(a)
#     valid_rows = np.isfinite(a).all(axis=1)
#     a = a[valid_rows]
#     if len(a) < 2:
#         raise ValueError("Need at least two finite rows for covariance")
#     return (a.T @ a) / float(len(a) - 1)


# def rel_fro_error(truth: np.ndarray, estimate: np.ndarray) -> float:
#     return float(np.linalg.norm(truth - estimate) / max(np.linalg.norm(truth), 1e-30))


# def modal_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
#     """Per-mode Pearson correlation for arrays (T, M), ignoring non-finite rows."""
#     n = min(len(a), len(b))
#     m = min(a.shape[1], b.shape[1])
#     aa = np.asarray(a[:n, :m], dtype=np.float64)
#     bb = np.asarray(b[:n, :m], dtype=np.float64)
#     valid = np.isfinite(aa) & np.isfinite(bb)
#     aa = np.where(valid, aa, np.nan)
#     bb = np.where(valid, bb, np.nan)
#     aa = aa - np.nanmean(aa, axis=0, keepdims=True)
#     bb = bb - np.nanmean(bb, axis=0, keepdims=True)
#     num = np.nansum(aa * bb, axis=0)
#     den = np.sqrt(np.nansum(aa * aa, axis=0) * np.nansum(bb * bb, axis=0))
#     return num / np.maximum(den, 1e-30)


# def align_for_delay(x: np.ndarray, d: np.ndarray, truth: np.ndarray | None, delay: int):
#     """Align xhat[k] with dm[k-delay].  Positive delay means use earlier DM."""
#     n = min(len(x), len(d))
#     if truth is not None:
#         n = min(n, len(truth))
#     x = x[:n]
#     d = d[:n]
#     t = truth[:n] if truth is not None else None

#     if delay > 0:
#         x2 = x[delay:]
#         d2 = d[:-delay]
#         t2 = t[delay:] if t is not None else None
#     elif delay < 0:
#         x2 = x[:delay]
#         d2 = d[-delay:]
#         t2 = t[:delay] if t is not None else None
#     else:
#         x2 = x
#         d2 = d
#         t2 = t
#     return x2, d2, t2


# def first_present(npz: np.lib.npyio.NpzFile, names: Iterable[str]) -> str | None:
#     for name in names:
#         if name in npz.files:
#             return name
#     return None


# def savefig(path: Path) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     plt.tight_layout()
#     plt.savefig(path, dpi=160)
#     plt.close()
#     print(f"saved: {path}")


# def project_to_psd(C: np.ndarray, floor: float = 0.0) -> np.ndarray:
#     C = 0.5 * (C + C.T)
#     w, V = np.linalg.eigh(C)
#     w = np.maximum(w, floor)
#     return 0.5 * ((V * w) @ V.T + ((V * w) @ V.T).T)


# # -----------------------------------------------------------------------------
# # Loading
# # -----------------------------------------------------------------------------


# def resolve_inputs(run_dir_or_file: Path):
#     p = run_dir_or_file.expanduser().resolve()
#     if p.is_dir():
#         return p, p / "telemetry.npz", p / "control_bundle.npz"
#     if p.name == "telemetry.npz":
#         return p.parent, p, p.parent / "control_bundle.npz"
#     raise ValueError(f"Expected run directory or telemetry.npz, got {p}")


# def print_npz_summary(label: str, data: np.lib.npyio.NpzFile) -> None:
#     print(f"\n=== {label} ===")
#     for k in data.files:
#         arr = data[k]
#         line = f"{k:34s} shape={str(arr.shape):20s} dtype={str(arr.dtype):12s} finite={finite_fraction(arr):.3f}"
#         if np.issubdtype(arr.dtype, np.number) and arr.size:
#             with np.errstate(all="ignore"):
#                 line += f" rms={float(rms(arr)):.4e}"
#         print(line)


# # -----------------------------------------------------------------------------
# # Diagnostics
# # -----------------------------------------------------------------------------


# def truth_bookkeeping(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
#     out: dict[str, Any] = {"available": False}
#     req = ["truth_modal_atm_total", "truth_modal_residual_total", "truth_modal_dm_applied"]
#     if not all(k in data.files for k in req):
#         out["reason"] = f"missing one of {req}"
#         return out

#     atm = np.asarray(data["truth_modal_atm_total"], dtype=float)
#     res = np.asarray(data["truth_modal_residual_total"], dtype=float)
#     dm = np.asarray(data["truth_modal_dm_applied"], dtype=float)
#     n = min(len(atm), len(res), len(dm))
#     m = min(atm.shape[1], res.shape[1], dm.shape[1])
#     atm, res, dm = atm[:n, :m], res[:n, :m], dm[:n, :m]

#     e_plus = rms(atm - (res + dm)) / max(float(rms(atm)), 1e-30)
#     e_minus = rms(atm - (res - dm)) / max(float(rms(atm)), 1e-30)
#     out.update(
#         {
#             "available": True,
#             "n_samples": int(n),
#             "n_modes": int(m),
#             "rel_rms_error_atm_minus_res_plus_dm": float(e_plus),
#             "rel_rms_error_atm_minus_res_minus_dm": float(e_minus),
#             "preferred_truth_identity": "atm ≈ residual + dm" if e_plus < e_minus else "atm ≈ residual - dm",
#         }
#     )
#     return out


# def choose_pol_candidate(data: np.lib.npyio.NpzFile, delay_candidates: list[int]) -> dict[str, Any]:
#     out: dict[str, Any] = {"available": False}
#     if "xhat" not in data.files or "truth_modal_atm_total" not in data.files:
#         out["reason"] = "need xhat and truth_modal_atm_total"
#         return out
#     dm_name = first_present(data, ("dm_cmd_applied", "dm_cmd", "dm_cmd_new"))
#     if dm_name is None:
#         out["reason"] = "need one of dm_cmd_applied, dm_cmd, dm_cmd_new"
#         return out

#     xhat = np.asarray(data["xhat"], dtype=float)
#     dm = np.asarray(data[dm_name], dtype=float)
#     truth = np.asarray(data["truth_modal_atm_total"], dtype=float)

#     candidates = []
#     for delay in delay_candidates:
#         x, d, tr = align_for_delay(xhat, dm, truth, delay)
#         if tr is None or len(x) < 2:
#             continue
#         m = min(x.shape[1], d.shape[1], tr.shape[1])
#         x, d, tr = x[:, :m], d[:, :m], tr[:, :m]
#         for sign, label in [(+1.0, "xhat + dm"), (-1.0, "xhat - dm")]:
#             pol = x + sign * d
#             corr = modal_corr(pol, tr)
#             rel = rms(pol - tr) / max(float(rms(tr)), 1e-30)
#             candidates.append(
#                 {
#                     "label": label,
#                     "sign": float(sign),
#                     "delay": int(delay),
#                     "dm_name": dm_name,
#                     "median_modal_corr": float(np.nanmedian(corr)),
#                     "mean_modal_corr": float(np.nanmean(corr)),
#                     "rel_rms_error_vs_truth_atm": float(rel),
#                     "n_samples": int(len(pol)),
#                     "n_modes": int(m),
#                 }
#             )

#     if not candidates:
#         out["reason"] = "no candidates evaluated"
#         return out
#     # Prefer high median correlation; break ties with low RMS error.
#     best = sorted(candidates, key=lambda c: (-c["median_modal_corr"], c["rel_rms_error_vs_truth_atm"]))[0]
#     out.update({"available": True, "best": best, "all_candidates": candidates})
#     return out


# def get_T_matrix(control: np.lib.npyio.NpzFile) -> tuple[np.ndarray | None, str | None]:
#     for name in ("T_mode_to_mode", "T_mode_to_mode_pre_waffle"):
#         if name in control.files:
#             T = np.asarray(control[name], dtype=float)
#             if T.ndim == 2 and T.shape[0] == T.shape[1]:
#                 return T, name
#     return None, None


# def control_operator_diagnostics(control: np.lib.npyio.NpzFile, outdir: Path, gain_floor: float) -> dict[str, Any]:
#     out: dict[str, Any] = {"available": False}
#     T, name = get_T_matrix(control)
#     if T is None:
#         out["reason"] = "no square T_mode_to_mode found"
#         return out

#     U, s, Vt = np.linalg.svd(T, full_matrices=True)
#     trusted = s >= gain_floor
#     out.update(
#         {
#             "available": True,
#             "T_name": name,
#             "T_shape": list(T.shape),
#             "gain_floor": float(gain_floor),
#             "n_trusted_singular_directions": int(np.sum(trusted)),
#             "n_modes": int(len(s)),
#             "s_min": float(np.min(s)),
#             "s_median": float(np.median(s)),
#             "s_max": float(np.max(s)),
#         }
#     )

#     plt.figure(figsize=(9, 5))
#     plt.semilogy(np.arange(len(s)), s, marker=".")
#     plt.axhline(gain_floor, linestyle="--", linewidth=1.0, label=f"gain floor = {gain_floor:g}")
#     plt.xlabel("Singular direction index")
#     plt.ylabel("Singular value of T_mode_to_mode")
#     plt.title("Saved control operator singular spectrum")
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     savefig(outdir / "control_singular_values.png")
#     return out


# # -----------------------------------------------------------------------------
# # First modal covariance diagnostic
# # -----------------------------------------------------------------------------


# def build_pol(data: np.lib.npyio.NpzFile, best: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
#     xhat = np.asarray(data["xhat"], dtype=float)
#     dm = np.asarray(data[best["dm_name"]], dtype=float)
#     truth = np.asarray(data["truth_modal_atm_total"], dtype=float)
#     x, d, tr = align_for_delay(xhat, dm, truth, int(best["delay"]))
#     m = min(x.shape[1], d.shape[1], tr.shape[1])
#     pol = x[:, :m] + float(best["sign"]) * d[:, :m]
#     tr = tr[:, :m]
#     return pol, tr


# def modal_covariance_diagnostic(
#     data: np.lib.npyio.NpzFile,
#     control: np.lib.npyio.NpzFile,
#     pol_choice: dict[str, Any],
#     outdir: Path,
#     gain_floor: float,
#     use_truth_prior: bool = True,
# ) -> dict[str, Any]:
#     out: dict[str, Any] = {"available": False}
#     T, T_name = get_T_matrix(control)
#     if T is None:
#         out["reason"] = "no T matrix"
#         return out
#     if "truth_modal_residual_total" not in data.files or "truth_modal_atm_total" not in data.files:
#         out["reason"] = "need truth_modal_residual_total and truth_modal_atm_total"
#         return out
#     if not pol_choice.get("available", False):
#         out["reason"] = "no POL choice available"
#         return out

#     pol, truth_atm_aligned = build_pol(data, pol_choice["best"])
#     truth_res = np.asarray(data["truth_modal_residual_total"], dtype=float)
#     # Align residual truth using same delay convention used for POL.
#     delay = int(pol_choice["best"]["delay"])
#     if delay > 0:
#         truth_res = truth_res[delay:]
#     elif delay < 0:
#         truth_res = truth_res[:delay]

#     n = min(len(pol), len(truth_res), len(truth_atm_aligned))
#     m = min(pol.shape[1], truth_res.shape[1], truth_atm_aligned.shape[1], T.shape[0])
#     pol = pol[:n, :m]
#     truth_atm_aligned = truth_atm_aligned[:n, :m]
#     truth_res = truth_res[:n, :m]
#     T = T[:m, :m]

#     C_res_true = covariance_time(truth_res)
#     C_atm_pol = covariance_time(pol)
#     C_atm_truth = covariance_time(truth_atm_aligned)

#     U, s, Vt = np.linalg.svd(T, full_matrices=True)
#     V = Vt.T
#     trusted = s >= gain_floor

#     # Express atmospheric covariance in incoming singular-vector basis.
#     C_pol_s = V.T @ C_atm_pol @ V
#     C_prior = C_atm_truth if use_truth_prior else np.diag(np.diag(C_atm_pol))
#     C_prior_s = V.T @ C_prior @ V

#     C_atm_tel_s = np.zeros_like(C_pol_s)
#     idx = np.where(trusted)[0]
#     if idx.size:
#         C_atm_tel_s[np.ix_(idx, idx)] = C_pol_s[np.ix_(idx, idx)]

#     C_atm_reg_s = C_prior_s.copy()
#     if idx.size:
#         C_atm_reg_s[np.ix_(idx, idx)] = C_pol_s[np.ix_(idx, idx)]

#     C_atm_tel = project_to_psd(V @ C_atm_tel_s @ V.T)
#     C_atm_reg = project_to_psd(V @ C_atm_reg_s @ V.T)

#     M = np.eye(m) - T
#     C_res_tel = project_to_psd(M @ C_atm_tel @ M.T)
#     C_res_reg = project_to_psd(M @ C_atm_reg @ M.T)

#     err_tel = rel_fro_error(C_res_true, C_res_tel)
#     err_reg = rel_fro_error(C_res_true, C_res_reg)
#     gain = err_tel / max(err_reg, 1e-30)

#     out.update(
#         {
#             "available": True,
#             "T_name": T_name,
#             "n_samples": int(n),
#             "n_modes": int(m),
#             "gain_floor": float(gain_floor),
#             "n_trusted": int(np.sum(trusted)),
#             "prior_type": "oracle_truth_atm_covariance" if use_truth_prior else "diag_POL_covariance",
#             "residual_covariance_error_telemetry_only": float(err_tel),
#             "residual_covariance_error_regaware": float(err_reg),
#             "regaware_gain_over_telemetry": float(gain),
#         }
#     )

#     # Plot per-mode residual variances in original modal basis.
#     diag_true = np.diag(C_res_true)
#     diag_tel = np.diag(C_res_tel)
#     diag_reg = np.diag(C_res_reg)
#     mode = np.arange(m)

#     plt.figure(figsize=(10, 5))
#     plt.semilogy(mode, np.maximum(diag_true, 1e-30), label="truth residual")
#     plt.semilogy(mode, np.maximum(diag_tel, 1e-30), linestyle="--", label="telemetry-only")
#     plt.semilogy(mode, np.maximum(diag_reg, 1e-30), linestyle="-.", label="regularization-aware")
#     plt.xlabel("Modal index")
#     plt.ylabel("Residual variance [native units$^2$]")
#     plt.title("Residual modal variance comparison")
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     savefig(outdir / "mode_variance_comparison.png")

#     plt.figure(figsize=(7, 5))
#     labels = ["telemetry-only", "reg-aware"]
#     vals = [100.0 * err_tel, 100.0 * err_reg]
#     plt.bar(np.arange(2), vals)
#     plt.xticks(np.arange(2), labels)
#     plt.ylabel("Relative Frobenius error [%]")
#     plt.title(f"Residual covariance error, gain={gain:.2f}x")
#     plt.grid(True, axis="y", alpha=0.3)
#     savefig(outdir / "residual_covariance_error_bars.png")

#     np.savez_compressed(
#         outdir / "modal_psfr_covariance_diagnostic.npz",
#         C_res_true=C_res_true,
#         C_res_telemetry=C_res_tel,
#         C_res_regaware=C_res_reg,
#         C_atm_pol=C_atm_pol,
#         C_atm_prior=C_prior,
#         T=T,
#         singular_values=s,
#         trusted=trusted,
#     )
#     print(f"saved: {outdir / 'modal_psfr_covariance_diagnostic.npz'}")
#     return out


# # -----------------------------------------------------------------------------
# # Main
# # -----------------------------------------------------------------------------


# def main() -> None:
#     p = argparse.ArgumentParser(description="Validate AOsim control_bundle + telemetry for modal PSF-R development")
#     p.add_argument("run_dir_or_telemetry_npz", type=Path, help="Run directory or telemetry.npz")
#     p.add_argument("--outdir", type=Path, default=None, help="Output directory. Default: <run_dir>/modal_psfr_validation")
#     p.add_argument("--gain-floor", type=float, default=0.12, help="Singular-value floor defining trusted control directions")
#     p.add_argument("--delay-candidates", type=int, nargs="+", default=[-1, 0, 1], help="POL delay candidates to test")
#     p.add_argument("--no-oracle-prior", action="store_true", help="Use diagonal POL covariance as prior instead of truth atmospheric covariance")
#     p.add_argument("--verbose", action="store_true", help="Print full NPZ summaries")
#     args = p.parse_args()

#     run_dir, telem_path, ctrl_path = resolve_inputs(args.run_dir_or_telemetry_npz)
#     outdir = args.outdir.expanduser().resolve() if args.outdir else run_dir / "modal_psfr_validation"
#     outdir.mkdir(parents=True, exist_ok=True)

#     if not telem_path.exists():
#         raise FileNotFoundError(telem_path)
#     if not ctrl_path.exists():
#         raise FileNotFoundError(ctrl_path)

#     data = np.load(telem_path, allow_pickle=True)
#     control = np.load(ctrl_path, allow_pickle=True)
#     telem_meta = load_json(run_dir / "telemetry_summary.json")
#     ctrl_meta = load_json(run_dir / "control_bundle_summary.json")

#     print(f"Run directory:      {run_dir}")
#     print(f"Telemetry:          {telem_path}")
#     print(f"Control bundle:     {ctrl_path}")
#     print(f"Output directory:   {outdir}")

#     if args.verbose:
#         print_npz_summary("telemetry.npz", data)
#         print_npz_summary("control_bundle.npz", control)
#     else:
#         print("Telemetry channels:", ", ".join(data.files))
#         print("Control keys:      ", ", ".join(control.files))

#     result: dict[str, Any] = {
#         "run_dir": str(run_dir),
#         "telemetry_npz": str(telem_path),
#         "control_bundle_npz": str(ctrl_path),
#         "telemetry_summary": telem_meta,
#         "control_bundle_summary": ctrl_meta,
#     }

#     tb = truth_bookkeeping(data)
#     result["truth_bookkeeping"] = tb
#     print("\nTruth bookkeeping:")
#     if tb.get("available"):
#         print(f"  {tb['preferred_truth_identity']}")
#         print(f"  rel error atm-(res+dm): {tb['rel_rms_error_atm_minus_res_plus_dm']:.4e}")
#         print(f"  rel error atm-(res-dm): {tb['rel_rms_error_atm_minus_res_minus_dm']:.4e}")
#     else:
#         print(f"  unavailable: {tb.get('reason')}")

#     pol = choose_pol_candidate(data, args.delay_candidates)
#     result["pol_sign_delay"] = pol
#     print("\nPOL sign/delay candidates:")
#     if pol.get("available"):
#         for c in pol["all_candidates"]:
#             print(
#                 f"  {c['label']:10s} delay={c['delay']:+d} using {c['dm_name']:14s} "
#                 f"median_corr={c['median_modal_corr']:+.3f} rel_rms={c['rel_rms_error_vs_truth_atm']:.3f}"
#             )
#         b = pol["best"]
#         print(f"  BEST: {b['label']} delay={b['delay']:+d} using {b['dm_name']}")
#     else:
#         print(f"  unavailable: {pol.get('reason')}")

#     cd = control_operator_diagnostics(control, outdir, args.gain_floor)
#     result["control_operator"] = cd
#     print("\nControl operator:")
#     if cd.get("available"):
#         print(f"  {cd['T_name']} shape={cd['T_shape']}")
#         print(f"  trusted singular directions: {cd['n_trusted_singular_directions']} / {cd['n_modes']}")
#         print(f"  singular values min/median/max: {cd['s_min']:.3e} / {cd['s_median']:.3e} / {cd['s_max']:.3e}")
#     else:
#         print(f"  unavailable: {cd.get('reason')}")

#     md = modal_covariance_diagnostic(
#         data,
#         control,
#         pol,
#         outdir,
#         gain_floor=args.gain_floor,
#         use_truth_prior=not args.no_oracle_prior,
#     )
#     result["modal_covariance_diagnostic"] = md
#     print("\nModal covariance diagnostic:")
#     if md.get("available"):
#         print(f"  prior type: {md['prior_type']}")
#         print(f"  trusted modes: {md['n_trusted']} / {md['n_modes']}")
#         print(f"  telemetry-only residual covariance error: {100*md['residual_covariance_error_telemetry_only']:.2f}%")
#         print(f"  reg-aware residual covariance error:      {100*md['residual_covariance_error_regaware']:.2f}%")
#         print(f"  gain: {md['regaware_gain_over_telemetry']:.2f}x")
#     else:
#         print(f"  unavailable: {md.get('reason')}")

#     summary_path = outdir / "modal_psfr_validation_summary.json"
#     with summary_path.open("w", encoding="utf-8") as f:
#         json.dump(result, f, indent=2, allow_nan=True)
#     print(f"\nsaved: {summary_path}")


# if __name__ == "__main__":
#     main()
