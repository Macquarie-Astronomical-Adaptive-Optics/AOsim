#!/usr/bin/env python3
"""
Oracle diagnostic for trusted-subspace calibration in AOsim GLAO PSF-R tests.

This follows the previous GLAO ground-layer validator, but asks a narrower question:

    Is the remaining residual-covariance error mainly due to the trusted-subspace
    telemetry covariance being mis-scaled/mis-calibrated?

It compares residual covariance models in the control-basis singular directions:

  1. telemetry-only
       trusted block = C_xhat, missing block = 0

  2. regularization-aware, uncalibrated
       trusted block = C_xhat, missing block = oracle missing residual prior

  3. regularization-aware + scalar trusted calibration
       trusted block = beta * C_xhat, missing block = oracle missing residual prior

  4. regularization-aware + diagonal trusted calibration
       trusted block = K C_xhat K^T, missing block = oracle missing residual prior

The scalar beta and diagonal K are fitted against simulator truth, so this is NOT an
operational PSF-R algorithm. It is a diagnostic to decide whether the next operational
step should be trusted-subspace transfer calibration.

Typical use:

RUN_DIR=$(ls -td experiments/ultimate_wfi_psf_reco/outputs/telemetry_truth_10s/longrun_* | head -1)

python experiments/ultimate_wfi_psf_reco/validate_trusted_subspace_calibration.py \
    "$RUN_DIR" \
    --ground-altitude-max-m 500 \
    --gain-floor 0.12 \
    --use-gram-ls-truth \
    --gram-rcond 1e-6 \
    --outdir "$RUN_DIR/trusted_subspace_calibration" \
    --verbose
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt


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


def center_ts(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x - np.nanmean(x, axis=0, keepdims=True)


def covariance(ts: np.ndarray) -> np.ndarray:
    x = center_ts(ts)
    n = x.shape[0]
    if n < 2:
        raise ValueError("Need at least two samples for covariance")
    C = (x.T @ x) / float(n - 1)
    return 0.5 * (C + C.T)


def covariance_in_basis(ts: np.ndarray, V: np.ndarray) -> np.ndarray:
    z = np.asarray(ts, dtype=np.float64) @ np.asarray(V, dtype=np.float64)
    return covariance(z)


def rel_fro_error(C_true: np.ndarray, C_est: np.ndarray) -> float:
    den = np.linalg.norm(C_true)
    return float(np.linalg.norm(C_true - C_est) / den) if den > 0 else float("nan")


def rel_diag_error(C_true: np.ndarray, C_est: np.ndarray) -> float:
    a = np.diag(C_true)
    b = np.diag(C_est)
    den = np.linalg.norm(a)
    return float(np.linalg.norm(a - b) / den) if den > 0 else float("nan")


def block_error(C_true: np.ndarray, C_est: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> float:
    if rows.size == 0 or cols.size == 0:
        return float("nan")
    return rel_fro_error(C_true[np.ix_(rows, cols)], C_est[np.ix_(rows, cols)])


def block_diag_error(C_true: np.ndarray, C_est: np.ndarray, idx: np.ndarray) -> float:
    if idx.size == 0:
        return float("nan")
    return rel_diag_error(C_true[np.ix_(idx, idx)], C_est[np.ix_(idx, idx)])


def summarize_errors(C_true: np.ndarray, C_est: np.ndarray, trusted: np.ndarray) -> dict[str, float]:
    trusted_idx = np.where(trusted)[0]
    missing_idx = np.where(~trusted)[0]
    return {
        "full_fro": rel_fro_error(C_true, C_est),
        "full_diag": rel_diag_error(C_true, C_est),
        "trusted_trusted_fro": block_error(C_true, C_est, trusted_idx, trusted_idx),
        "trusted_trusted_diag": block_diag_error(C_true, C_est, trusted_idx),
        "missing_missing_fro": block_error(C_true, C_est, missing_idx, missing_idx),
        "missing_missing_diag": block_diag_error(C_true, C_est, missing_idx),
        "trusted_missing_fro": block_error(C_true, C_est, trusted_idx, missing_idx),
    }


def collapse_layers(
    truth_layer_modal_atm: np.ndarray,
    telemetry_summary: dict[str, Any],
    ground_altitude_max_m: float,
    ground_layer_indices: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    layers = np.asarray(truth_layer_modal_atm, dtype=np.float64)
    if layers.ndim != 3:
        raise ValueError(f"truth_layer_modal_atm must be (time, layer, mode), got {layers.shape}")
    n_layer = layers.shape[1]

    names = telemetry_summary.get("layer_name")
    if not isinstance(names, list) or len(names) != n_layer:
        names = [f"L{i}" for i in range(n_layer)]
    names = [str(x) for x in names]

    alt = telemetry_summary.get("layer_altitude_m")
    alt_m = np.asarray(alt, dtype=float) if isinstance(alt, list) and len(alt) == n_layer else np.full(n_layer, np.nan)

    if ground_layer_indices:
        ground = np.asarray([int(x.strip()) for x in ground_layer_indices.split(",") if x.strip()], dtype=int)
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
        labels.append(f"{names[i]} ({alt_m[i]/1000:.2f} km)" if np.isfinite(alt_m[i]) else names[i])
    return ground_ts, free_ts, ground, free, labels


def gram_ls_transform(c_diag: np.ndarray, G: np.ndarray, G_diag: np.ndarray, rcond: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert diagonal inner-product coefficients to Gram-aware LS coefficients."""
    c_diag = np.asarray(c_diag, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    G_diag = np.asarray(G_diag, dtype=np.float64)
    if G.shape[0] != G.shape[1]:
        raise ValueError(f"basis_gram must be square, got {G.shape}")
    n_modes = min(c_diag.shape[1], G.shape[0], G_diag.shape[0])
    c_diag = c_diag[:, :n_modes]
    G = G[:n_modes, :n_modes]
    G_diag = G_diag[:n_modes]

    # b_i = <B_i, phi> = c_diag_i * <B_i, B_i>
    b = c_diag * G_diag[None, :]

    evals, evecs = np.linalg.eigh(0.5 * (G + G.T))
    emax = float(np.nanmax(evals))
    keep = evals > float(rcond) * emax
    if not np.any(keep):
        raise RuntimeError("Gram matrix inversion kept zero eigenvalues; reduce --gram-rcond")
    Gpinv = (evecs[:, keep] / evals[keep][None, :]) @ evecs[:, keep].T
    c_ls = b @ Gpinv.T
    info = {
        "n_modes": int(n_modes),
        "gram_rcond": float(rcond),
        "gram_eig_min": float(np.nanmin(evals)),
        "gram_eig_max": emax,
        "gram_eig_median": float(np.nanmedian(evals)),
        "gram_n_kept": int(np.sum(keep)),
        "gram_n_rejected": int(len(evals) - np.sum(keep)),
    }
    return c_ls, info


def control_basis(control: np.lib.npyio.NpzFile, gain_floor: float, n_modes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    T_name = first_present(control, ("T_mode_to_mode", "T_mode_to_mode_pre_waffle"))
    if T_name is None:
        raise KeyError("control_bundle missing T_mode_to_mode")
    T = np.asarray(control[T_name], dtype=np.float64)
    T = T[:n_modes, :n_modes]
    _, s, Vt = np.linalg.svd(T, full_matrices=True)
    V = Vt.T
    trusted = s >= float(gain_floor)
    return V, s, trusted, T_name


def build_estimate(n: int, trusted: np.ndarray, trusted_block: np.ndarray | None, missing_block: np.ndarray | None) -> np.ndarray:
    trusted_idx = np.where(trusted)[0]
    missing_idx = np.where(~trusted)[0]
    C = np.zeros((n, n), dtype=np.float64)
    if trusted_block is not None and trusted_idx.size:
        C[np.ix_(trusted_idx, trusted_idx)] = trusted_block
    if missing_block is not None and missing_idx.size:
        C[np.ix_(missing_idx, missing_idx)] = missing_block
    return 0.5 * (C + C.T)


def fit_scalar_beta(C_true_tt: np.ndarray, C_x_tt: np.ndarray) -> float:
    den = float(np.sum(C_x_tt * C_x_tt))
    if den <= 0:
        return float("nan")
    beta = float(np.sum(C_true_tt * C_x_tt) / den)
    return max(beta, 0.0)


def fit_diag_k(C_true_tt: np.ndarray, C_x_tt: np.ndarray, clip: tuple[float, float]) -> np.ndarray:
    v_true = np.maximum(np.diag(C_true_tt), 0.0)
    v_x = np.maximum(np.diag(C_x_tt), 1e-30)
    k = np.sqrt(v_true / v_x)
    k = np.where(np.isfinite(k), k, 1.0)
    if clip[0] > 0 or np.isfinite(clip[1]):
        k = np.clip(k, clip[0], clip[1])
    return k


def plot_variance_models(C_true: np.ndarray, models: dict[str, np.ndarray], trusted: np.ndarray, outdir: Path) -> None:
    x = np.arange(C_true.shape[0])
    fig, ax = plt.subplots(figsize=(13, 5), dpi=150)
    ax.semilogy(x, np.maximum(np.diag(C_true), 1e-30), label="truth", lw=2)
    for name, C in models.items():
        ax.semilogy(x, np.maximum(np.diag(C), 1e-30), label=name, lw=1.5, ls="--" if "uncal" in name else "-.")
    ax.axvline(int(np.sum(trusted)) - 0.5, color="0.5", ls=":", label="trusted/missing boundary")
    ax.set_xlabel("Control-basis singular direction")
    ax.set_ylabel("Residual variance [native units$^2$]")
    ax.set_title("Trusted-subspace residual calibration diagnostic")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "trusted_calibrated_residual_variance_control_basis.png")
    plt.close(fig)


def plot_error_bars(errors: dict[str, dict[str, float]], outdir: Path) -> None:
    keys = ["full_fro", "trusted_trusted_fro", "missing_missing_fro", "trusted_missing_fro", "full_diag"]
    labels = ["full", "trusted-trusted", "missing-missing", "trusted-missing", "full diag"]
    model_names = list(errors.keys())
    x = np.arange(len(keys))
    width = min(0.8 / max(len(model_names), 1), 0.22)
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150)
    for j, name in enumerate(model_names):
        vals = [100.0 * errors[name][k] for k in keys]
        ax.bar(x + (j - (len(model_names)-1)/2) * width, vals, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Relative error [%]")
    ax.set_title("Residual covariance errors after trusted-subspace calibration")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "trusted_calibrated_residual_error_bars.png")
    plt.close(fig)


def plot_calibration_factors(k_diag: np.ndarray, beta: float, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
    ax.plot(np.arange(len(k_diag)), k_diag, lw=1.2, label="diagonal K")
    if np.isfinite(beta):
        ax.axhline(np.sqrt(beta), ls="--", label=f"scalar sqrt(beta)={np.sqrt(beta):.3g}")
    ax.set_xlabel("Trusted singular direction index")
    ax.set_ylabel("Amplitude calibration factor")
    ax.set_title("Oracle trusted-subspace calibration factors")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "trusted_calibration_factors.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Oracle trusted-subspace calibration diagnostic for GLAO PSF-R.")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--telemetry", type=Path, default=None)
    p.add_argument("--control-bundle", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--ground-altitude-max-m", type=float, default=500.0)
    p.add_argument("--ground-layer-indices", type=str, default=None)
    p.add_argument("--gain-floor", type=float, default=0.12)
    p.add_argument("--use-gram-ls-truth", action="store_true")
    p.add_argument("--gram-rcond", type=float, default=1e-6)
    p.add_argument("--diag-k-min", type=float, default=0.0)
    p.add_argument("--diag-k-max", type=float, default=np.inf)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    telemetry_path = args.telemetry.expanduser().resolve() if args.telemetry else run_dir / "telemetry.npz"
    control_path = args.control_bundle.expanduser().resolve() if args.control_bundle else run_dir / "control_bundle.npz"
    outdir = args.outdir.expanduser().resolve() if args.outdir else run_dir / "trusted_subspace_calibration"
    outdir.mkdir(parents=True, exist_ok=True)

    telemetry_summary = load_json(run_dir / "telemetry_summary.json")
    control_summary = load_json(run_dir / "control_bundle_summary.json")
    tel = np.load(telemetry_path)
    control = np.load(control_path)

    for key in ("xhat", "truth_layer_modal_atm", "truth_modal_dm_applied"):
        if key not in tel.files:
            raise KeyError(f"telemetry missing {key}")

    xhat = np.asarray(tel["xhat"], dtype=np.float64)
    ground, free, ground_idx, free_idx, layer_labels = collapse_layers(
        tel["truth_layer_modal_atm"], telemetry_summary, args.ground_altitude_max_m, args.ground_layer_indices
    )
    truth_dm = np.asarray(tel["truth_modal_dm_applied"], dtype=np.float64)

    n_modes = min(xhat.shape[1], ground.shape[1], truth_dm.shape[1])
    xhat = xhat[:, :n_modes]
    ground = ground[:, :n_modes]
    truth_dm = truth_dm[:, :n_modes]

    gram_info = None
    truth_projection = "diagonal"
    if args.use_gram_ls_truth:
        if "basis_gram" not in control.files or "basis_gram_diag" not in control.files:
            raise KeyError("Need basis_gram and basis_gram_diag in control_bundle for --use-gram-ls-truth")
        ground, gram_info = gram_ls_transform(ground, control["basis_gram"], control["basis_gram_diag"], args.gram_rcond)
        truth_dm, gram_info_dm = gram_ls_transform(truth_dm, control["basis_gram"], control["basis_gram_diag"], args.gram_rcond)
        n_modes = min(xhat.shape[1], ground.shape[1], truth_dm.shape[1])
        xhat = xhat[:, :n_modes]
        ground = ground[:, :n_modes]
        truth_dm = truth_dm[:, :n_modes]
        truth_projection = "Gram-LS"

    ground_res_true = ground - truth_dm

    V, s, trusted, T_name = control_basis(control, args.gain_floor, n_modes)
    n_modes = V.shape[0]
    trusted_idx = np.where(trusted)[0]
    missing_idx = np.where(~trusted)[0]

    C_true_b = covariance_in_basis(ground_res_true, V)
    C_xhat_b = covariance_in_basis(xhat, V)
    C_ground_prior_b = covariance_in_basis(ground, V)

    # Missing residual prior in control basis. For hard-projected modes this is effectively C_ground_prior in missing block.
    w = 1.0 - np.clip(s, 0.0, 1.0)
    C_missing_prior_full = (w[:, None] * C_ground_prior_b) * w[None, :]
    C_missing_prior = C_missing_prior_full[np.ix_(missing_idx, missing_idx)] if missing_idx.size else None

    C_x_tt = C_xhat_b[np.ix_(trusted_idx, trusted_idx)] if trusted_idx.size else None
    C_true_tt = C_true_b[np.ix_(trusted_idx, trusted_idx)] if trusted_idx.size else None

    models: dict[str, np.ndarray] = {}
    errors: dict[str, dict[str, float]] = {}

    # 1. telemetry-only
    C_telonly = build_estimate(n_modes, trusted, C_x_tt, None)
    models["telemetry-only"] = C_telonly
    errors["telemetry-only"] = summarize_errors(C_true_b, C_telonly, trusted)

    # 2. reg-aware uncalibrated
    C_reg_uncal = build_estimate(n_modes, trusted, C_x_tt, C_missing_prior)
    models["reg-aware uncal"] = C_reg_uncal
    errors["reg-aware uncal"] = summarize_errors(C_true_b, C_reg_uncal, trusted)

    # 3. scalar trusted calibration
    beta = fit_scalar_beta(C_true_tt, C_x_tt) if trusted_idx.size else float("nan")
    C_scalar_tt = beta * C_x_tt if np.isfinite(beta) else C_x_tt
    C_reg_scalar = build_estimate(n_modes, trusted, C_scalar_tt, C_missing_prior)
    models["reg-aware scalar"] = C_reg_scalar
    errors["reg-aware scalar"] = summarize_errors(C_true_b, C_reg_scalar, trusted)

    # 4. diagonal trusted calibration
    k_diag = fit_diag_k(C_true_tt, C_x_tt, (args.diag_k_min, args.diag_k_max)) if trusted_idx.size else np.array([])
    C_diag_tt = (k_diag[:, None] * C_x_tt) * k_diag[None, :] if trusted_idx.size else C_x_tt
    C_reg_diag = build_estimate(n_modes, trusted, C_diag_tt, C_missing_prior)
    models["reg-aware diag"] = C_reg_diag
    errors["reg-aware diag"] = summarize_errors(C_true_b, C_reg_diag, trusted)

    plot_variance_models(C_true_b, models, trusted, outdir)
    plot_error_bars(errors, outdir)
    plot_calibration_factors(k_diag, beta, outdir)

    # Diagnostics on calibration factors.
    k_stats = {}
    if k_diag.size:
        finite = k_diag[np.isfinite(k_diag)]
        k_stats = {
            "n": int(finite.size),
            "min": float(np.min(finite)),
            "p05": float(np.percentile(finite, 5)),
            "median": float(np.median(finite)),
            "p95": float(np.percentile(finite, 95)),
            "max": float(np.max(finite)),
        }

    payload = {
        "run_dir": str(run_dir),
        "telemetry_npz": str(telemetry_path),
        "control_bundle_npz": str(control_path),
        "outdir": str(outdir),
        "truth_projection": truth_projection,
        "gram_info": gram_info,
        "ground_altitude_max_m": float(args.ground_altitude_max_m),
        "ground_layer_indices": ground_idx.tolist(),
        "free_layer_indices": free_idx.tolist(),
        "layer_labels": layer_labels,
        "control_operator": {
            "T_name": T_name,
            "gain_floor": float(args.gain_floor),
            "n_modes": int(n_modes),
            "n_trusted": int(trusted_idx.size),
            "n_missing": int(missing_idx.size),
            "s_min": float(np.min(s)),
            "s_median": float(np.median(s)),
            "s_max": float(np.max(s)),
        },
        "trusted_calibration": {
            "scalar_beta_variance_factor": float(beta),
            "scalar_amplitude_factor_sqrt_beta": float(np.sqrt(beta)) if np.isfinite(beta) else float("nan"),
            "diag_amplitude_factor_stats": k_stats,
            "diag_k_min_clip": float(args.diag_k_min),
            "diag_k_max_clip": float(args.diag_k_max) if np.isfinite(args.diag_k_max) else "inf",
        },
        "residual_covariance_errors": errors,
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
        "notes": {
            "oracle_warning": "Scalar and diagonal trusted calibrations are fitted against simulator truth. They are diagnostics, not operational PSF-R inputs.",
            "missing_prior": "Missing residual covariance uses oracle ground covariance propagated by (1 - singular_value).",
        },
    }
    summary_path = outdir / "trusted_subspace_calibration_summary.json"
    save_json(payload, summary_path)

    print("\n=== Trusted-subspace calibration diagnostic ===")
    print(f"Run dir: {run_dir}")
    print(f"Output:  {outdir}")
    print(f"Truth projection: {truth_projection}")
    if gram_info:
        print(f"Gram rcond: {gram_info['gram_rcond']}; kept {gram_info['gram_n_kept']}/{gram_info['n_modes']} Gram eigenmodes")
    print(f"Control trusted/missing: {trusted_idx.size}/{missing_idx.size}")
    print(f"Scalar beta variance factor: {beta:.6g}; amplitude factor sqrt(beta): {np.sqrt(beta) if np.isfinite(beta) else np.nan:.6g}")
    if k_stats:
        print(
            "Diagonal amplitude K stats: "
            f"median={k_stats['median']:.4g}, p05={k_stats['p05']:.4g}, p95={k_stats['p95']:.4g}"
        )

    print("\nResidual covariance errors:")
    for name, e in errors.items():
        print(
            f"  {name:18s} "
            f"full={100*e['full_fro']:7.2f}%  "
            f"TT={100*e['trusted_trusted_fro']:7.2f}%  "
            f"MM={100*e['missing_missing_fro']:7.2f}%  "
            f"TM={100*e['trusted_missing_fro']:7.2f}%  "
            f"diag={100*e['full_diag']:7.2f}%"
        )
    print(f"\nWrote: {summary_path}")
    if args.verbose:
        print("\nKey plots:")
        for p in sorted(outdir.glob("*.png")):
            print(f"  {p}")


if __name__ == "__main__":
    main()
