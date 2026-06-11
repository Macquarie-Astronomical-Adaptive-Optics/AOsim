#!/usr/bin/env python3
"""
Validate telemetry-derived residual transfer factors for GLAO PSF-R.

This is the first operational step after the oracle residual-transfer diagnostic.

Goal
----
Estimate a diagonal residual-transfer operator H from telemetry rather than from
simulator residual truth, then test:

    C_res_est = H_telemetry C_ground_prior H_telemetry^T

This keeps the spatial covariance and trusted/missing cross terms from the
ground-layer prior, while using telemetry to estimate how strongly each modal
direction is corrected.

Important
---------
This is still a validation script because it compares against simulator truth.
But the telemetry-derived H models below do NOT use the true residual covariance.

Models compared
---------------
1. telemetry-only
   Trusted block from C_xhat, zeros elsewhere.

2. reg-aware block
   Trusted block from C_xhat, missing block from prior, trusted/missing set to zero.

3. transfer telemetry/prior
   h_i^2 = Var(xhat_i) / Var(ground_prior_i) for trusted modes.
   h_i = 1 for missing modes.

4. transfer telemetry/POL
   h_i^2 = Var(xhat_i) / Var(POL_i) for trusted modes, where POL = xhat - dm_applied.
   h_i = 1 for missing modes.

5. transfer singular
   simple non-oracle control-only heuristic h_i = 1 - clipped singular value.

6. transfer oracle
   h_i^2 = Var(true_residual_i) / Var(ground_prior_i).
   This is included only as a best-case diagnostic.

Typical use
-----------
RUN_DIR=$(ls -td experiments/ultimate_wfi_psf_reco/outputs/telemetry_truth_10s/longrun_* | head -1)
export RUN_DIR

python experiments/ultimate_wfi_psf_reco/validate_telemetry_residual_transfer.py \
    "$RUN_DIR" \
    --ground-altitude-max-m 500 \
    --gain-floor 0.12 \
    --use-gram-ls-truth \
    --gram-rcond 1e-6 \
    --outdir "$RUN_DIR/telemetry_residual_transfer" \
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


def summarize(C_true: np.ndarray, C_est: np.ndarray, trusted: np.ndarray) -> dict[str, float]:
    s = np.where(trusted)[0]
    m = np.where(~trusted)[0]
    return {
        "full_fro": rel_fro(C_true, C_est),
        "full_diag": rel_diag(C_true, C_est),
        "trusted_trusted_fro": block_err(C_true, C_est, s, s),
        "missing_missing_fro": block_err(C_true, C_est, m, m),
        "trusted_missing_fro": block_err(C_true, C_est, s, m),
    }


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


def make_block_model(
    C_xhat: np.ndarray,
    C_prior: np.ndarray,
    trusted: np.ndarray,
    missing_h: np.ndarray | None = None,
) -> np.ndarray:
    s = np.where(trusted)[0]
    m = np.where(~trusted)[0]
    C = np.zeros_like(C_prior)
    C[np.ix_(s, s)] = C_xhat[np.ix_(s, s)]
    if m.size:
        if missing_h is None:
            C[np.ix_(m, m)] = C_prior[np.ix_(m, m)]
        else:
            C_missing = apply_diag_transfer(C_prior, missing_h)
            C[np.ix_(m, m)] = C_missing[np.ix_(m, m)]
    return 0.5 * (C + C.T)


def fit_scalar_beta(C_true: np.ndarray, C_pred: np.ndarray, idx: np.ndarray | None = None) -> float:
    if idx is not None:
        C_true = C_true[np.ix_(idx, idx)]
        C_pred = C_pred[np.ix_(idx, idx)]
    a = C_true.ravel()
    b = C_pred.ravel()
    den = float(np.dot(b, b))
    return float(np.dot(a, b) / den) if den > 0 else float("nan")


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


def corr(a: np.ndarray, b: np.ndarray, idx: np.ndarray | None = None) -> float:
    a = finite_clip(a)
    b = finite_clip(b)
    if idx is not None:
        a = a[idx]
        b = b[idx]
    a = a - np.mean(a)
    b = b - np.mean(b)
    den = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    return float(np.dot(a, b) / den) if den > 0 else float("nan")


def describe_h(h: np.ndarray, trusted: np.ndarray, oracle: np.ndarray | None = None) -> dict[str, float]:
    s = np.where(trusted)[0]
    m = np.where(~trusted)[0]
    d = {
        "median": float(np.nanmedian(h)),
        "p05": float(np.nanpercentile(h, 5)),
        "p95": float(np.nanpercentile(h, 95)),
        "trusted_median": float(np.nanmedian(h[s])) if s.size else float("nan"),
        "missing_median": float(np.nanmedian(h[m])) if m.size else float("nan"),
    }
    if oracle is not None:
        d["corr_with_oracle_all"] = corr(h, oracle)
        d["corr_with_oracle_trusted"] = corr(h, oracle, s) if s.size else float("nan")
        d["corr_with_oracle_missing"] = corr(h, oracle, m) if m.size else float("nan")
        d["rel_error_vs_oracle_all"] = float(np.linalg.norm(h - oracle) / np.linalg.norm(oracle))
        d["rel_error_vs_oracle_trusted"] = float(np.linalg.norm(h[s] - oracle[s]) / np.linalg.norm(oracle[s])) if s.size and np.linalg.norm(oracle[s]) > 0 else float("nan")
        d["rel_error_vs_oracle_missing"] = float(np.linalg.norm(h[m] - oracle[m]) / np.linalg.norm(oracle[m])) if m.size and np.linalg.norm(oracle[m]) > 0 else float("nan")
    return d


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
    ax.set_title("Telemetry-derived residual-transfer model errors")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "telemetry_transfer_error_bars.png")
    plt.close(fig)


def plot_transfer_factors(h_models: dict[str, np.ndarray], trusted: np.ndarray, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    for name, h in h_models.items():
        ax.plot(h, lw=1.4, label=name)
    ax.axvline(int(np.sum(trusted)) - 0.5, color="0.5", ls=":", label="trusted/missing boundary")
    ax.set_xlabel("Control-basis singular direction")
    ax.set_ylabel("Amplitude residual transfer h")
    ax.set_title("Telemetry-derived residual transfer factors")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "telemetry_transfer_factors.png")
    plt.close(fig)


def plot_variance(C_true: np.ndarray, models: dict[str, np.ndarray], trusted: np.ndarray, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.semilogy(np.maximum(np.diag(C_true), 1e-30), label="truth", lw=2)
    for name, C in models.items():
        ax.semilogy(np.maximum(np.diag(C), 1e-30), label=name, lw=1.4)
    ax.axvline(int(np.sum(trusted)) - 0.5, color="0.5", ls=":", label="trusted/missing boundary")
    ax.set_xlabel("Control-basis singular direction")
    ax.set_ylabel("Residual variance [native units$^2$]")
    ax.set_title("Residual variance from telemetry-derived transfer")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "telemetry_transfer_residual_variance.png")
    plt.close(fig)


def plot_oracle_scatter(h_models: dict[str, np.ndarray], h_oracle: np.ndarray, trusted: np.ndarray, outdir: Path) -> None:
    s = np.where(trusted)[0]
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    for name, h in h_models.items():
        if name == "oracle":
            continue
        ax.scatter(h_oracle[s], h[s], s=8, alpha=0.45, label=name)
    lo = 0.0
    hi = max(1.2, float(np.nanpercentile(h_oracle[s], 99)) * 1.1)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Oracle h, trusted modes")
    ax.set_ylabel("Telemetry/model h")
    ax.set_title("Residual transfer factors vs oracle")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "telemetry_transfer_vs_oracle_scatter.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate telemetry-derived residual transfer factors.")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--telemetry", type=Path, default=None)
    p.add_argument("--control-bundle", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--ground-altitude-max-m", type=float, default=500.0)
    p.add_argument("--ground-layer-indices", type=str, default=None)
    p.add_argument("--gain-floor", type=float, default=0.12)
    p.add_argument("--use-gram-ls-truth", action="store_true")
    p.add_argument("--gram-rcond", type=float, default=1e-6)
    p.add_argument("--pol-delay", type=int, default=0, help="Use dm_cmd_applied[k-pol_delay] in POL = xhat[k] - dm[k-pol_delay]")
    p.add_argument("--h-clip-max", type=float, default=2.0, help="Clip telemetry transfer factors to [0, h_clip_max]")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    telemetry_path = args.telemetry.expanduser().resolve() if args.telemetry else run_dir / "telemetry.npz"
    control_path = args.control_bundle.expanduser().resolve() if args.control_bundle else run_dir / "control_bundle.npz"
    outdir = args.outdir.expanduser().resolve() if args.outdir else run_dir / "telemetry_residual_transfer"
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

    n_samples = min(len(ground), len(truth_dm), len(xhat), len(dm))
    n_modes = min(ground.shape[1], truth_dm.shape[1], xhat.shape[1], dm.shape[1])
    ground = ground[:n_samples, :n_modes]
    truth_dm = truth_dm[:n_samples, :n_modes]
    xhat = xhat[:n_samples, :n_modes]
    dm = dm[:n_samples, :n_modes]

    # Optional POL delay alignment.
    d = int(args.pol_delay)
    if d > 0:
        xhat_pol = xhat[d:]
        dm_pol = dm[:-d]
        ground = ground[d:]
        truth_dm = truth_dm[d:]
        xhat = xhat[d:]
        dm = dm[d:]
    elif d < 0:
        dabs = abs(d)
        xhat_pol = xhat[:-dabs]
        dm_pol = dm[dabs:]
        ground = ground[:-dabs]
        truth_dm = truth_dm[:-dabs]
        xhat = xhat[:-dabs]
        dm = dm[:-dabs]
    else:
        xhat_pol = xhat
        dm_pol = dm
    pol = xhat_pol - dm_pol

    n_samples = min(len(ground), len(truth_dm), len(xhat), len(dm), len(pol))
    ground = ground[:n_samples]
    truth_dm = truth_dm[:n_samples]
    xhat = xhat[:n_samples]
    pol = pol[:n_samples]

    arrays, gram_info = maybe_gram_ls(
        {"ground": ground, "truth_dm": truth_dm},
        control,
        use_gram=args.use_gram_ls_truth,
        rcond=args.gram_rcond,
    )
    ground = arrays["ground"]
    truth_dm = arrays["truth_dm"]

    # In previous validation: atmosphere ~= residual + DM, so residual ~= ground - DM.
    ground_res_true = ground - truth_dm

    V, svals, trusted, T_name = control_basis(control, args.gain_floor, n_modes)
    sidx = np.where(trusted)[0]
    midx = np.where(~trusted)[0]

    C_ground = covariance_in_basis(ground, V)
    C_res_true = covariance_in_basis(ground_res_true, V)
    C_xhat = covariance_in_basis(xhat, V)
    C_pol = covariance_in_basis(pol, V)

    var_ground = np.diag(C_ground)
    var_res_true = np.diag(C_res_true)
    var_xhat = np.diag(C_xhat)
    var_pol = np.diag(C_pol)

    # Oracle diagonal transfer, for comparison only.
    h_oracle = safe_sqrt_ratio(var_res_true, var_ground)

    # Telemetry-derived transfer 1: residual telemetry over model/prior ground variance.
    h_tel_prior = safe_sqrt_ratio(var_xhat, var_ground)
    h_tel_prior[midx] = 1.0
    h_tel_prior = np.clip(h_tel_prior, 0.0, args.h_clip_max)

    # Telemetry-derived transfer 2: residual telemetry over POL telemetry variance.
    h_tel_pol = safe_sqrt_ratio(var_xhat, var_pol)
    h_tel_pol[midx] = 1.0
    h_tel_pol = np.clip(h_tel_pol, 0.0, args.h_clip_max)

    # Control-only simple heuristic.
    h_singular = 1.0 - np.clip(svals, 0.0, 1.0)
    h_singular = np.clip(h_singular, 0.0, 1.0)

    # Slightly less aggressive singular heuristic with a floor, useful for plotting comparison.
    h_singular_floor = np.maximum(h_singular, np.where(trusted, np.nanmedian(h_tel_pol[sidx]) if sidx.size else 0.0, 1.0))
    h_singular_floor[midx] = 1.0
    h_singular_floor = np.clip(h_singular_floor, 0.0, args.h_clip_max)

    # Existing models.
    C_tel_only = np.zeros_like(C_res_true)
    C_tel_only[np.ix_(sidx, sidx)] = C_xhat[np.ix_(sidx, sidx)]
    C_tel_only = 0.5 * (C_tel_only + C_tel_only.T)

    C_block = make_block_model(C_xhat, C_ground, trusted)

    models = {
        "telemetry-only": C_tel_only,
        "reg-aware block": C_block,
        "H telemetry/prior": apply_diag_transfer(C_ground, h_tel_prior),
        "H telemetry/POL": apply_diag_transfer(C_ground, h_tel_pol),
        "H singular floor": apply_diag_transfer(C_ground, h_singular_floor),
        "H oracle diag": apply_diag_transfer(C_ground, h_oracle),
    }

    model_errors = {name: summarize(C_res_true, C, trusted) for name, C in models.items()}

    h_models = {
        "telemetry/prior": h_tel_prior,
        "telemetry/POL": h_tel_pol,
        "singular floor": h_singular_floor,
        "oracle": h_oracle,
    }

    h_stats = {name: describe_h(h, trusted, h_oracle) for name, h in h_models.items()}

    plot_model_errors(model_errors, outdir)
    plot_transfer_factors(h_models, trusted, outdir)
    plot_variance(C_res_true, models, trusted, outdir)
    plot_oracle_scatter(h_models, h_oracle, trusted, outdir)

    summary = {
        "run_dir": str(run_dir),
        "telemetry_npz": str(telemetry_path),
        "control_bundle_npz": str(control_path),
        "outdir": str(outdir),
        "truth_projection": gram_info.get("truth_projection"),
        "gram_info": gram_info,
        "ground_altitude_max_m": float(args.ground_altitude_max_m),
        "ground_layer_indices": ground_idx.tolist(),
        "free_layer_indices": free_idx.tolist(),
        "layer_labels": labels,
        "n_samples": int(n_samples),
        "n_modes": int(n_modes),
        "pol_definition": {
            "formula": "POL = xhat - dm_cmd_applied",
            "pol_delay": int(args.pol_delay),
        },
        "control_operator": {
            "T_name": T_name,
            "gain_floor": float(args.gain_floor),
            "n_trusted": int(np.sum(trusted)),
            "n_missing": int(np.sum(~trusted)),
            "s_min": float(np.min(svals)),
            "s_median": float(np.median(svals)),
            "s_max": float(np.max(svals)),
        },
        "h_clip_max": float(args.h_clip_max),
        "h_stats": h_stats,
        "model_errors": model_errors,
        "notes": {
            "main_test": "Can H estimated from telemetry approach the oracle diagonal transfer model?",
            "operational_model": "C_res_est = H_telemetry C_ground_prior H_telemetry^T.",
            "telemetry_prior_h": "Trusted h_i = sqrt(Var(xhat_i)/Var(ground_prior_i)); missing h_i = 1.",
            "telemetry_pol_h": "Trusted h_i = sqrt(Var(xhat_i)/Var(POL_i)); missing h_i = 1.",
            "oracle_h": "h_i = sqrt(Var(true_residual_i)/Var(ground_prior_i)); diagnostic only.",
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

    summary_path = outdir / "telemetry_residual_transfer_summary.json"
    save_json(summary, summary_path)

    print("\n=== Telemetry-derived residual-transfer diagnostic ===")
    print(f"Run dir: {run_dir}")
    print(f"Output:  {outdir}")
    print(f"Truth projection: {gram_info.get('truth_projection')}")
    if args.use_gram_ls_truth:
        print(f"Gram rcond: {gram_info.get('gram_rcond')}; kept {gram_info.get('gram_n_kept')}/{gram_info.get('gram_n_eig')} Gram eigenmodes")
    print(f"Control trusted/missing: {int(np.sum(trusted))}/{int(np.sum(~trusted))}")
    print(f"POL: xhat - dm_cmd_applied, delay={int(args.pol_delay)}")

    print("\nTransfer-factor stats:")
    for name in ("telemetry/prior", "telemetry/POL", "singular floor", "oracle"):
        st = h_stats[name]
        print(
            f"  {name:16s} "
            f"median={st['median']:.3f}  "
            f"trusted={st['trusted_median']:.3f}  "
            f"missing={st['missing_median']:.3f}  "
            f"corr_oracle_T={st.get('corr_with_oracle_trusted', float('nan')):+.3f}  "
            f"relerr_oracle_T={st.get('rel_error_vs_oracle_trusted', float('nan')):.3f}"
        )

    print("\nResidual covariance errors:")
    order = [
        "telemetry-only",
        "reg-aware block",
        "H telemetry/prior",
        "H telemetry/POL",
        "H singular floor",
        "H oracle diag",
    ]
    for name in order:
        e = model_errors[name]
        print(
            f"  {name:20s} "
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
