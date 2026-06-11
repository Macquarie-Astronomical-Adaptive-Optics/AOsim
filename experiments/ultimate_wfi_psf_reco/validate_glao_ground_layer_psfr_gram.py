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



def diag_projection_to_gram_ls(
    coeff_diag: np.ndarray,
    basis_gram: np.ndarray,
    *,
    gram_rcond: float = 1e-6,
    chunk_rows: int = 2048,
) -> np.ndarray:
    """Convert diagonal inner-product coefficients to Gram-aware LS coefficients.

    The simulator diagnostic truth coefficients are approximately

        c_diag_i = <B_i, phi> / G_ii,

    where G_ij=<B_i,B_j>.  Therefore the right-hand side vector for the
    least-squares normal equations is

        b_i = <B_i, phi> = c_diag_i G_ii,

    and the Gram-aware coefficient vector is

        c_ls = G^+ b.

    This conversion is applied row-wise to a time x modes array, or to any array
    whose last dimension is modes.
    """
    G = np.asarray(basis_gram, dtype=np.float64)
    x = np.asarray(coeff_diag, dtype=np.float64)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"basis_gram must be square, got {G.shape}")
    n_modes = min(G.shape[0], x.shape[-1])
    G = G[:n_modes, :n_modes]
    diag = np.diag(G)
    if np.any(~np.isfinite(diag)) or np.any(diag <= 0):
        raise ValueError("basis_gram diagonal contains non-finite or non-positive entries")

    # SVD pinv is stable enough for the diagnostic stage.  rcond should be
    # tightened/loosened if the LS truth becomes noisy.
    G_pinv = np.linalg.pinv(G, rcond=float(gram_rcond))

    orig_shape = x.shape
    flat = x[..., :n_modes].reshape(-1, n_modes)
    out = np.empty_like(flat, dtype=np.float64)
    chunk = max(1, int(chunk_rows))
    for i0 in range(0, flat.shape[0], chunk):
        i1 = min(flat.shape[0], i0 + chunk)
        rhs = flat[i0:i1] * diag[None, :]
        out[i0:i1] = rhs @ G_pinv.T
    return out.reshape(orig_shape[:-1] + (n_modes,))


def maybe_gram_ls_convert_truth(
    *,
    ground: np.ndarray,
    free: np.ndarray,
    total: np.ndarray,
    truth_dm: np.ndarray,
    control: np.lib.npyio.NpzFile,
    enabled: bool,
    gram_rcond: float,
    gram_chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Optionally convert simulator diagnostic truth channels to Gram-LS coefficients."""
    info: dict[str, Any] = {"enabled": bool(enabled)}
    if not enabled:
        info["basis_gram_available"] = "basis_gram" in control.files
        return ground, free, total, truth_dm, info
    if "basis_gram" not in control.files:
        raise KeyError(
            "--use-gram-ls-truth was requested but control_bundle.npz does not contain basis_gram. "
            "Rerun with save_control_bundle=True and control_bundle_include_gram=True."
        )
    G = np.asarray(control["basis_gram"], dtype=np.float64)
    info.update(
        {
            "basis_gram_available": True,
            "basis_gram_shape": list(G.shape),
            "basis_gram_rcond": float(gram_rcond),
            "basis_gram_condition_estimate": float(np.linalg.cond(G)),
        }
    )
    ground_ls = diag_projection_to_gram_ls(ground, G, gram_rcond=gram_rcond, chunk_rows=gram_chunk_rows)
    free_ls = diag_projection_to_gram_ls(free, G, gram_rcond=gram_rcond, chunk_rows=gram_chunk_rows)
    # Prefer total = ground + free after conversion, because the layer split is
    # linear and this avoids small inconsistencies in separately projected totals.
    total_ls = ground_ls + free_ls
    dm_ls = diag_projection_to_gram_ls(truth_dm, G, gram_rcond=gram_rcond, chunk_rows=gram_chunk_rows)
    return ground_ls, free_ls, total_ls, dm_ls, info


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
    p.add_argument(
        "--use-gram-ls-truth",
        action="store_true",
        help="Convert simulator diagonal-projection truth channels to Gram-aware LS coefficients using control_bundle['basis_gram'].",
    )
    p.add_argument("--gram-rcond", type=float, default=1e-6, help="Pseudo-inverse rcond for Gram LS conversion.")
    p.add_argument("--gram-chunk-rows", type=int, default=2048, help="Rows per chunk for Gram LS truth conversion.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    telemetry_path = args.telemetry.expanduser().resolve() if args.telemetry else run_dir / "telemetry.npz"
    control_path = args.control_bundle.expanduser().resolve() if args.control_bundle else run_dir / "control_bundle.npz"
    outdir = args.outdir.expanduser().resolve() if args.outdir else run_dir / ("glao_ground_psfr_validation_gramls" if args.use_gram_ls_truth else "glao_ground_psfr_validation")
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

    ground, free, total, truth_dm, truth_projection_info = maybe_gram_ls_convert_truth(
        ground=ground,
        free=free,
        total=total,
        truth_dm=truth_dm,
        control=control,
        enabled=bool(args.use_gram_ls_truth),
        gram_rcond=float(args.gram_rcond),
        gram_chunk_rows=int(args.gram_chunk_rows),
    )

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
        "truth_projection": truth_projection_info,
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
    print(f"Truth projection: {'Gram-LS' if args.use_gram_ls_truth else 'diagonal diagnostic'}")
    if args.use_gram_ls_truth:
        print(f"Gram rcond: {args.gram_rcond:g}")
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
