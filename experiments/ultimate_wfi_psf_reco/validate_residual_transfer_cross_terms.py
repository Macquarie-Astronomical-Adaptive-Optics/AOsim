#!/usr/bin/env python3
"""
Validate residual-transfer cross terms for GLAO regularization-aware PSF-R.

This is the next diagnostic after:
  - telemetry/control semantics are validated,
  - Gram-LS truth projection is available,
  - missing-mode covariance completion works,
  - trusted-subspace scalar/diagonal calibration shows that diagonal variances can be improved
    but full covariance remains limited by trusted-missing cross terms.

Main question
-------------
Can the remaining residual-covariance error be reduced by using a residual
transfer operator that preserves trusted-missing cross-covariance from the
ground-layer prior?

The block-diagonal regularization-aware model is roughly:

    C_res ~= P_s C_xhat P_s^T + P_m C_prior P_m^T

which sets trusted-missing covariance to zero.

This script tests models of the form:

    C_res ~= H C_ground_prior H^T

where H is a scalar or diagonal residual-transfer operator in the control-basis
singular-vector coordinates.

The diagonal transfer is an oracle diagnostic by default:
    h_i = sqrt( C_res_true[i,i] / C_ground_prior[i,i] )

This is NOT an operational PSF-R model yet. It tells us whether the missing
full-covariance error is mainly due to neglected trusted-missing cross terms.

Typical use
-----------
RUN_DIR=$(ls -td experiments/ultimate_wfi_psf_reco/outputs/telemetry_truth_10s/longrun_* | head -1)
export RUN_DIR

python experiments/ultimate_wfi_psf_reco/validate_residual_transfer_cross_terms.py \
    "$RUN_DIR" \
    --ground-altitude-max-m 500 \
    --gain-floor 0.12 \
    --use-gram-ls-truth \
    --gram-rcond 1e-6 \
    --outdir "$RUN_DIR/residual_transfer_cross_terms" \
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


def center(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x - np.nanmean(x, axis=0, keepdims=True)


def covariance(x: np.ndarray) -> np.ndarray:
    x = center(x)
    n = x.shape[0]
    if n < 2:
        raise ValueError("Need at least two samples")
    C = (x.T @ x) / float(n - 1)
    return 0.5 * (C + C.T)


def covariance_in_basis(x: np.ndarray, V: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) @ np.asarray(V, dtype=np.float64)
    return covariance(z)


def rel_fro(C_true: np.ndarray, C_est: np.ndarray) -> float:
    den = np.linalg.norm(C_true)
    return float(np.linalg.norm(C_true - C_est) / den) if den > 0 else float("nan")


def rel_diag(C_true: np.ndarray, C_est: np.ndarray) -> float:
    a = np.diag(C_true)
    b = np.diag(C_est)
    den = np.linalg.norm(a)
    return float(np.linalg.norm(a - b) / den) if den > 0 else float("nan")


def block(C: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return C[np.ix_(rows, cols)]


def block_err(C_true: np.ndarray, C_est: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> float:
    if rows.size == 0 or cols.size == 0:
        return float("nan")
    return rel_fro(block(C_true, rows, cols), block(C_est, rows, cols))


def block_diag_err(C_true: np.ndarray, C_est: np.ndarray, idx: np.ndarray) -> float:
    if idx.size == 0:
        return float("nan")
    return rel_diag(block(C_true, idx, idx), block(C_est, idx, idx))


def summarize(C_true: np.ndarray, C_est: np.ndarray, trusted: np.ndarray) -> dict[str, float]:
    s = np.where(trusted)[0]
    m = np.where(~trusted)[0]
    return {
        "full_fro": rel_fro(C_true, C_est),
        "full_diag": rel_diag(C_true, C_est),
        "trusted_trusted_fro": block_err(C_true, C_est, s, s),
        "missing_missing_fro": block_err(C_true, C_est, m, m),
        "trusted_missing_fro": block_err(C_true, C_est, s, m),
        "missing_missing_diag": block_diag_err(C_true, C_est, m),
    }


def fit_scalar_beta(C_true: np.ndarray, C_pred: np.ndarray, idx: np.ndarray | None = None) -> float:
    """Fit C_true ~= beta C_pred in Frobenius least squares."""
    if idx is not None:
        C_true = C_true[np.ix_(idx, idx)]
        C_pred = C_pred[np.ix_(idx, idx)]
    a = C_true.ravel()
    b = C_pred.ravel()
    den = float(np.dot(b, b))
    return float(np.dot(a, b) / den) if den > 0 else float("nan")


def collapse_layers(
    layers: np.ndarray,
    telemetry_summary: dict[str, Any],
    ground_altitude_max_m: float,
    ground_layer_indices: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    layers = np.asarray(layers, dtype=np.float64)
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

    ground = np.nansum(layers[:, ground_idx, :], axis=1) if ground_idx.size else np.zeros((layers.shape[0], layers.shape[2]))
    free = np.nansum(layers[:, free_idx, :], axis=1) if free_idx.size else np.zeros((layers.shape[0], layers.shape[2]))

    labels = []
    for i in range(n_layer):
        if np.isfinite(alt_m[i]):
            labels.append(f"{names[i]} ({alt_m[i]/1000:.2f} km)")
        else:
            labels.append(names[i])
    return ground, free, ground_idx, free_idx, labels


def gram_ls_transform_coeffs(
    coeff_diag: np.ndarray,
    G: np.ndarray,
    rcond: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Convert diagonal inner-product coefficients to approximate Gram-LS coefficients.

    Current diagnostic coefficients are approximately:
        c_diag_i = <B_i, phi> / G_ii

    Let b_i = <B_i, phi> = c_diag_i G_ii.
    Solve:
        G c_ls = b

    via eigendecomposition with cutoff.
    """
    C = np.asarray(coeff_diag, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
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
        raise KeyError("control_bundle.npz has no basis_gram; rerun with control_bundle_include_gram=True")

    G = np.asarray(control["basis_gram"], dtype=np.float64)
    out = {}
    info_total = None
    for key, arr in arrays.items():
        out[key], info = gram_ls_transform_coeffs(arr, G, rcond)
        if info_total is None:
            info_total = info
    info_total = info_total or {}
    info_total["truth_projection"] = "Gram-LS"
    return out, info_total


def control_basis(control: np.lib.npyio.NpzFile, gain_floor: float, n_modes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    T_name = first_present(control, ("T_mode_to_mode", "T_mode_to_mode_pre_waffle"))
    if T_name is None:
        raise KeyError("control_bundle.npz missing T_mode_to_mode")
    T = np.asarray(control[T_name], dtype=np.float64)[:n_modes, :n_modes]
    _, s, Vt = np.linalg.svd(T, full_matrices=True)
    V = Vt.T
    trusted = s >= float(gain_floor)
    return V, s, trusted, T_name


def diag_transfer_from_truth(C_res_true: np.ndarray, C_prior: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    num = np.maximum(np.diag(C_res_true), 0.0)
    den = np.maximum(np.diag(C_prior), floor)
    h = np.sqrt(num / den)
    h[~np.isfinite(h)] = 0.0
    return h


def apply_diag_transfer(C_prior: np.ndarray, h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    C = (h[:, None] * C_prior) * h[None, :]
    return 0.5 * (C + C.T)


def make_block_diag_model(C_xhat: np.ndarray, C_prior: np.ndarray, trusted: np.ndarray, residual_weight: np.ndarray) -> np.ndarray:
    """Existing regularization-aware block-diagonal residual covariance model."""
    n = C_prior.shape[0]
    C = (residual_weight[:, None] * C_prior) * residual_weight[None, :]
    s = np.where(trusted)[0]
    C[np.ix_(s, s)] = C_xhat[np.ix_(s, s)]
    return 0.5 * (C + C.T)


def plot_model_errors(model_errors: dict[str, dict[str, float]], outdir: Path) -> None:
    fields = [
        ("full_fro", "full"),
        ("trusted_trusted_fro", "TT"),
        ("missing_missing_fro", "MM"),
        ("trusted_missing_fro", "TM"),
        ("full_diag", "diag"),
    ]
    model_names = list(model_errors.keys())

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    x = np.arange(len(fields))
    width = 0.8 / max(len(model_names), 1)

    for j, name in enumerate(model_names):
        vals = [100.0 * model_errors[name][field] for field, _ in fields]
        ax.bar(x + (j - (len(model_names)-1)/2) * width, vals, width, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in fields])
    ax.set_ylabel("Relative error [%]")
    ax.set_title("Residual covariance model errors")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "residual_transfer_cross_term_error_bars.png")
    plt.close(fig)


def plot_variance(C_true: np.ndarray, models: dict[str, np.ndarray], trusted: np.ndarray, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.semilogy(np.maximum(np.diag(C_true), 1e-30), label="truth", lw=2)
    for name, C in models.items():
        ax.semilogy(np.maximum(np.diag(C), 1e-30), label=name, lw=1.5)
    ax.axvline(int(np.sum(trusted)) - 0.5, color="0.5", ls=":", label="trusted/missing boundary")
    ax.set_xlabel("Control-basis singular direction")
    ax.set_ylabel("Residual variance [native units$^2$]")
    ax.set_title("Residual variance models in control basis")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "residual_transfer_cross_term_variance.png")
    plt.close(fig)


def plot_transfer_factors(h_models: dict[str, np.ndarray], trusted: np.ndarray, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    for name, h in h_models.items():
        ax.plot(h, label=name, lw=1.4)
    ax.axvline(int(np.sum(trusted)) - 0.5, color="0.5", ls=":", label="trusted/missing boundary")
    ax.set_xlabel("Control-basis singular direction")
    ax.set_ylabel("Amplitude residual transfer h")
    ax.set_title("Residual transfer factors")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "residual_transfer_factors.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate residual-transfer cross terms for GLAO PSF-R.")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--telemetry", type=Path, default=None)
    p.add_argument("--control-bundle", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--ground-altitude-max-m", type=float, default=500.0)
    p.add_argument("--ground-layer-indices", type=str, default=None)
    p.add_argument("--gain-floor", type=float, default=0.12)
    p.add_argument("--use-gram-ls-truth", action="store_true")
    p.add_argument("--gram-rcond", type=float, default=1e-6)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    telemetry_path = args.telemetry.expanduser().resolve() if args.telemetry else run_dir / "telemetry.npz"
    control_path = args.control_bundle.expanduser().resolve() if args.control_bundle else run_dir / "control_bundle.npz"
    outdir = args.outdir.expanduser().resolve() if args.outdir else run_dir / "residual_transfer_cross_terms"
    outdir.mkdir(parents=True, exist_ok=True)

    tel = np.load(telemetry_path)
    control = np.load(control_path)
    telemetry_summary = load_json(run_dir / "telemetry_summary.json")
    control_summary = load_json(run_dir / "control_bundle_summary.json")

    for key in ("truth_layer_modal_atm", "truth_modal_dm_applied", "xhat"):
        if key not in tel.files:
            raise KeyError(f"telemetry.npz missing required {key}")

    ground, free, ground_idx, free_idx, labels = collapse_layers(
        tel["truth_layer_modal_atm"],
        telemetry_summary,
        args.ground_altitude_max_m,
        args.ground_layer_indices,
    )
    truth_dm = np.asarray(tel["truth_modal_dm_applied"], dtype=np.float64)
    xhat = np.asarray(tel["xhat"], dtype=np.float64)

    n_samples = min(len(ground), len(truth_dm), len(xhat))
    n_modes = min(ground.shape[1], truth_dm.shape[1], xhat.shape[1])
    ground = ground[:n_samples, :n_modes]
    truth_dm = truth_dm[:n_samples, :n_modes]
    xhat = xhat[:n_samples, :n_modes]

    arrays, gram_info = maybe_gram_ls(
        {"ground": ground, "truth_dm": truth_dm},
        control,
        use_gram=args.use_gram_ls_truth,
        rcond=args.gram_rcond,
    )
    ground = arrays["ground"]
    truth_dm = arrays["truth_dm"]

    # In prior validation the truth sign was atm ~= residual + dm, so residual ~= ground - dm.
    ground_res_true = ground - truth_dm

    V, svals, trusted, T_name = control_basis(control, args.gain_floor, n_modes)
    missing = ~trusted
    sidx = np.where(trusted)[0]
    midx = np.where(missing)[0]

    C_ground_prior = covariance_in_basis(ground, V)
    C_res_true = covariance_in_basis(ground_res_true, V)
    C_xhat = covariance_in_basis(xhat, V)

    residual_weight = 1.0 - np.clip(svals, 0.0, 1.0)

    # Model 0: telemetry-only block. Trusted xhat, zero missing.
    C_tel = np.zeros_like(C_res_true)
    C_tel[np.ix_(sidx, sidx)] = C_xhat[np.ix_(sidx, sidx)]
    C_tel = 0.5 * (C_tel + C_tel.T)

    # Model 1: previous block-diagonal regularization-aware model.
    C_block = make_block_diag_model(C_xhat, C_ground_prior, trusted, residual_weight)

    # Model 2: scalar residual-transfer using one beta over all covariance.
    beta_all = fit_scalar_beta(C_res_true, C_ground_prior)
    h_scalar_all = np.sqrt(max(beta_all, 0.0)) * np.ones(n_modes)
    C_transfer_scalar_all = apply_diag_transfer(C_ground_prior, h_scalar_all)

    # Model 3: scalar transfer fitted in trusted and missing blocks separately.
    beta_s = fit_scalar_beta(C_res_true, C_ground_prior, sidx)
    beta_m = fit_scalar_beta(C_res_true, C_ground_prior, midx)
    h_scalar_blocks = np.zeros(n_modes)
    h_scalar_blocks[sidx] = np.sqrt(max(beta_s, 0.0))
    h_scalar_blocks[midx] = np.sqrt(max(beta_m, 0.0))
    C_transfer_scalar_blocks = apply_diag_transfer(C_ground_prior, h_scalar_blocks)

    # Model 4: diagonal oracle transfer preserving all prior cross terms.
    h_diag = diag_transfer_from_truth(C_res_true, C_ground_prior)
    C_transfer_diag = apply_diag_transfer(C_ground_prior, h_diag)

    models = {
        "telemetry-only": C_tel,
        "reg-aware block": C_block,
        "transfer scalar all": C_transfer_scalar_all,
        "transfer scalar blocks": C_transfer_scalar_blocks,
        "transfer diag oracle": C_transfer_diag,
    }
    model_errors = {name: summarize(C_res_true, C, trusted) for name, C in models.items()}

    plot_model_errors(model_errors, outdir)
    plot_variance(C_res_true, models, trusted, outdir)
    plot_transfer_factors(
        {
            "scalar all": h_scalar_all,
            "scalar blocks": h_scalar_blocks,
            "diag oracle": h_diag,
        },
        trusted,
        outdir,
    )

    summary = {
        "run_dir": str(run_dir),
        "telemetry_npz": str(telemetry_path),
        "control_bundle_npz": str(control_path),
        "outdir": str(outdir),
        "truth_projection": gram_info.get("truth_projection"),
        "gram_info": gram_info,
        "ground_altitude_max_m": args.ground_altitude_max_m,
        "ground_layer_indices": ground_idx.tolist(),
        "free_layer_indices": free_idx.tolist(),
        "layer_labels": labels,
        "n_samples": int(n_samples),
        "n_modes": int(n_modes),
        "control_operator": {
            "T_name": T_name,
            "gain_floor": float(args.gain_floor),
            "n_trusted": int(np.sum(trusted)),
            "n_missing": int(np.sum(missing)),
            "s_min": float(np.min(svals)),
            "s_median": float(np.median(svals)),
            "s_max": float(np.max(svals)),
        },
        "transfer_factors": {
            "beta_all_variance": float(beta_all),
            "h_all_amplitude": float(np.sqrt(max(beta_all, 0.0))),
            "beta_trusted_variance": float(beta_s),
            "h_trusted_amplitude": float(np.sqrt(max(beta_s, 0.0))),
            "beta_missing_variance": float(beta_m),
            "h_missing_amplitude": float(np.sqrt(max(beta_m, 0.0))),
            "h_diag_stats": {
                "median": float(np.nanmedian(h_diag)),
                "p05": float(np.nanpercentile(h_diag, 5)),
                "p95": float(np.nanpercentile(h_diag, 95)),
                "trusted_median": float(np.nanmedian(h_diag[sidx])) if sidx.size else float("nan"),
                "missing_median": float(np.nanmedian(h_diag[midx])) if midx.size else float("nan"),
            },
        },
        "model_errors": model_errors,
        "notes": {
            "oracle_diagnostic": "Transfer factors are fitted using simulator truth. This is not operational PSF-R.",
            "cross_term_question": "If transfer models reduce TM/full error, remaining problem was cross-covariance/residual transfer rather than missing variance alone.",
            "model_formula": "C_res_est = H C_ground_prior H^T in the control-basis singular-vector coordinates.",
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

    summary_path = outdir / "residual_transfer_cross_terms_summary.json"
    save_json(summary, summary_path)

    print("\n=== Residual-transfer cross-term diagnostic ===")
    print(f"Run dir: {run_dir}")
    print(f"Output:  {outdir}")
    print(f"Truth projection: {gram_info.get('truth_projection')}")
    if args.use_gram_ls_truth:
        print(f"Gram rcond: {gram_info.get('gram_rcond')}; kept {gram_info.get('gram_n_kept')}/{gram_info.get('gram_n_eig')} Gram eigenmodes")
    print(f"Control trusted/missing: {int(np.sum(trusted))}/{int(np.sum(missing))}")
    print("\nTransfer factors:")
    print(f"  scalar all:     h={np.sqrt(max(beta_all, 0.0)):.4g}  beta={beta_all:.4g}")
    print(f"  scalar trusted: h={np.sqrt(max(beta_s, 0.0)):.4g}  beta={beta_s:.4g}")
    print(f"  scalar missing: h={np.sqrt(max(beta_m, 0.0)):.4g}  beta={beta_m:.4g}")
    print(f"  diag oracle:    median={np.nanmedian(h_diag):.4g}, p05={np.nanpercentile(h_diag,5):.4g}, p95={np.nanpercentile(h_diag,95):.4g}")

    print("\nResidual covariance errors:")
    order = ["telemetry-only", "reg-aware block", "transfer scalar all", "transfer scalar blocks", "transfer diag oracle"]
    for name in order:
        e = model_errors[name]
        print(
            f"  {name:22s} "
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
