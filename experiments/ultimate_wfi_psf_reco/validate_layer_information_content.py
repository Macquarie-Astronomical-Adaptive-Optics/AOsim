#!/usr/bin/env python3
"""
Empirical layer information-content diagnostic for GLAO PSF-R telemetry.

Question
--------
How much information does the saved telemetry contain about each simulated
atmospheric layer?

This is not a formal analytical Fisher matrix F = H^T N^-1 H. Instead it is an
empirical, data-driven information diagnostic:

    layer truth coefficients a_l(t) ~= linear_model(telemetry_stream(t))

The script fits a regularized linear predictor on a training section of the run
and evaluates it on a held-out test section. The main metric is variance-weighted
test R^2:

    R2 = 1 - sum((truth - prediction)^2) / sum((truth - train_mean)^2)

Plain interpretation:
    R2 ~= 1   telemetry contains strong information about that layer
    R2 ~= 0   telemetry has little useful linear information about that layer
    R2 < 0    prediction is worse than a constant train-set mean

Telemetry streams compared by default:
    - xhat
    - POL-like estimate: xhat - dm_cmd_applied
    - slopes_err_vec  (optionally PCA-compressed because it is high-dimensional)

Typical use
-----------
RUN_DIR=$(ls -td experiments/ultimate_wfi_psf_reco/outputs/telemetry_truth_10s/longrun_* | head -1)
export RUN_DIR

python experiments/ultimate_wfi_psf_reco/validate_layer_information_content.py \
    "$RUN_DIR" \
    --ground-altitude-max-m 500 \
    --target-basis control \
    --use-gram-ls-truth \
    --gram-rcond 1e-6 \
    --max-components 256 \
    --outdir "$RUN_DIR/layer_information_content" \
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
    x = np.asarray(x, dtype=np.float64)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


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
    C = finite_clip(coeff_diag)
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


def control_basis(
    control: np.lib.npyio.NpzFile | None,
    gain_floor: float,
    n_modes: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, str | None]:
    if control is None:
        return None, None, None, None

    T_name = first_present(control, ("T_mode_to_mode", "T_mode_to_mode_pre_waffle"))
    if T_name is None:
        return None, None, None, None

    T = np.asarray(control[T_name], dtype=np.float64)[:n_modes, :n_modes]
    _, s, Vt = np.linalg.svd(T, full_matrices=True)
    V = Vt.T
    trusted = s >= float(gain_floor)
    return V, s, trusted, T_name


def layer_metadata(telemetry_summary: dict[str, Any], n_layer: int) -> tuple[list[str], np.ndarray]:
    names = telemetry_summary.get("layer_name")
    if not isinstance(names, list) or len(names) != n_layer:
        names = [f"L{i}" for i in range(n_layer)]
    names = [str(x) for x in names]

    alt = telemetry_summary.get("layer_altitude_m")
    if isinstance(alt, list) and len(alt) == n_layer:
        alt_m = np.asarray(alt, dtype=float)
    else:
        alt_m = np.full(n_layer, np.nan)

    labels = []
    for i in range(n_layer):
        if np.isfinite(alt_m[i]):
            labels.append(f"L{i} {alt_m[i]/1000:.2f} km")
        else:
            labels.append(f"L{i}")
    return labels, alt_m


def reduce_predictors_pca(
    X_train: np.ndarray,
    X_test: np.ndarray,
    max_components: int,
    standardize: bool = True,
    random_state: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    PCA-compress predictors.

    Uses sklearn randomized PCA if available; otherwise falls back to numpy SVD.
    """
    X_train = finite_clip(X_train)
    X_test = finite_clip(X_test)

    mean = np.mean(X_train, axis=0, keepdims=True)
    Xtr = X_train - mean
    Xte = X_test - mean

    if standardize:
        scale = np.std(Xtr, axis=0, keepdims=True)
        scale = np.where(scale > 0, scale, 1.0)
        Xtr = Xtr / scale
        Xte = Xte / scale

    n_comp = int(min(max_components, Xtr.shape[0] - 1, Xtr.shape[1]))
    if n_comp < 1:
        raise ValueError("Need at least one PCA component")

    method = "numpy_svd"
    explained = float("nan")

    try:
        from sklearn.decomposition import PCA  # type: ignore

        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=random_state)
        Ztr = pca.fit_transform(Xtr)
        Zte = pca.transform(Xte)
        explained = float(np.sum(pca.explained_variance_ratio_))
        method = "sklearn_randomized_pca"
    except Exception:
        # Fallback. For the slope matrix this may be slower but avoids hard dependency.
        U, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
        Vt = Vt[:n_comp]
        S = S[:n_comp]
        Ztr = U[:, :n_comp] * S[None, :]
        Zte = Xte @ Vt.T
        total_var = float(np.sum(np.var(Xtr, axis=0)))
        explained = float(np.sum(S**2 / max(Xtr.shape[0] - 1, 1)) / total_var) if total_var > 0 else float("nan")

    # Whiten component scales for stable ridge regularisation.
    z_mean = np.mean(Ztr, axis=0, keepdims=True)
    z_std = np.std(Ztr, axis=0, keepdims=True)
    z_std = np.where(z_std > 0, z_std, 1.0)
    Ztr = (Ztr - z_mean) / z_std
    Zte = (Zte - z_mean) / z_std

    info = {
        "pca_method": method,
        "n_components": int(n_comp),
        "explained_variance_fraction": explained,
        "standardized_predictors": bool(standardize),
    }
    return Ztr, Zte, info


def ridge_predict(
    Z_train: np.ndarray,
    Z_test: np.ndarray,
    Y_train: np.ndarray,
    ridge_alpha: float,
) -> np.ndarray:
    Z_train = finite_clip(Z_train)
    Z_test = finite_clip(Z_test)
    Y_train = finite_clip(Y_train)

    y_mean = np.mean(Y_train, axis=0, keepdims=True)
    Yc = Y_train - y_mean

    k = Z_train.shape[1]
    A = Z_train.T @ Z_train
    # Scale alpha by mean diagonal so the value is less sensitive to sample count.
    scale = float(np.trace(A) / max(k, 1))
    lam = float(ridge_alpha) * (scale if scale > 0 else 1.0)
    A = A + lam * np.eye(k)
    B = Z_train.T @ Yc

    W = np.linalg.solve(A, B)
    return Z_test @ W + y_mean


def r2_variance_weighted(Y_true: np.ndarray, Y_pred: np.ndarray, Y_train: np.ndarray) -> float:
    Y_true = finite_clip(Y_true)
    Y_pred = finite_clip(Y_pred)
    y_mean = np.mean(finite_clip(Y_train), axis=0, keepdims=True)
    sse = float(np.sum((Y_true - Y_pred) ** 2))
    sst = float(np.sum((Y_true - y_mean) ** 2))
    return float(1.0 - sse / sst) if sst > 0 else float("nan")


def per_mode_r2(Y_true: np.ndarray, Y_pred: np.ndarray, Y_train: np.ndarray) -> np.ndarray:
    y_mean = np.mean(finite_clip(Y_train), axis=0, keepdims=True)
    sse = np.sum((finite_clip(Y_true) - finite_clip(Y_pred)) ** 2, axis=0)
    sst = np.sum((finite_clip(Y_true) - y_mean) ** 2, axis=0)
    out = np.full(Y_true.shape[1], np.nan)
    ok = sst > 0
    out[ok] = 1.0 - sse[ok] / sst[ok]
    return out


def per_mode_corr(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    A = finite_clip(Y_true)
    B = finite_clip(Y_pred)
    A = A - np.mean(A, axis=0, keepdims=True)
    B = B - np.mean(B, axis=0, keepdims=True)
    num = np.sum(A * B, axis=0)
    den = np.sqrt(np.sum(A * A, axis=0) * np.sum(B * B, axis=0))
    out = np.full(A.shape[1], np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def metrics_for_prediction(
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    Y_pred: np.ndarray,
    trusted: np.ndarray | None,
) -> dict[str, float]:
    r2m = per_mode_r2(Y_test, Y_pred, Y_train)
    corr = per_mode_corr(Y_test, Y_pred)

    result = {
        "r2_full": r2_variance_weighted(Y_test, Y_pred, Y_train),
        "r2_mode_median": float(np.nanmedian(r2m)),
        "r2_mode_p25": float(np.nanpercentile(r2m, 25)),
        "r2_mode_p75": float(np.nanpercentile(r2m, 75)),
        "corr_mode_median": float(np.nanmedian(corr)),
        "corr_mode_p25": float(np.nanpercentile(corr, 25)),
        "corr_mode_p75": float(np.nanpercentile(corr, 75)),
    }

    if trusted is not None and len(trusted) == Y_test.shape[1]:
        s = np.where(trusted)[0]
        m = np.where(~trusted)[0]
        if s.size:
            result["r2_trusted"] = r2_variance_weighted(Y_test[:, s], Y_pred[:, s], Y_train[:, s])
            result["corr_trusted_median"] = float(np.nanmedian(corr[s]))
        if m.size:
            result["r2_missing"] = r2_variance_weighted(Y_test[:, m], Y_pred[:, m], Y_train[:, m])
            result["corr_missing_median"] = float(np.nanmedian(corr[m]))
    return result


def plot_layer_bars(
    results: dict[str, list[dict[str, Any]]],
    layer_labels: list[str],
    outdir: Path,
    metric: str,
    fname: str,
    title: str,
) -> None:
    streams = list(results.keys())
    n_layer = len(layer_labels)
    x = np.arange(n_layer)
    width = 0.8 / max(len(streams), 1)

    fig, ax = plt.subplots(figsize=(max(10, 0.8 * n_layer + 4), 5), dpi=150)
    for j, stream in enumerate(streams):
        vals = [float(row["metrics"].get(metric, np.nan)) for row in results[stream]]
        ax.bar(x + (j - (len(streams)-1)/2) * width, vals, width, label=stream)

    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / fname)
    plt.close(fig)


def plot_ground_free_summary(
    results: dict[str, list[dict[str, Any]]],
    ground_idx: np.ndarray,
    free_idx: np.ndarray,
    outdir: Path,
) -> None:
    streams = list(results.keys())
    data = []
    for stream in streams:
        vals = np.asarray([float(row["metrics"].get("r2_full", np.nan)) for row in results[stream]], dtype=float)
        g = float(np.nanmean(vals[ground_idx])) if ground_idx.size else float("nan")
        f = float(np.nanmean(vals[free_idx])) if free_idx.size else float("nan")
        data.append((g, f))

    x = np.arange(len(streams))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.bar(x - width / 2, [d[0] for d in data], width, label="ground layers")
    ax.bar(x + width / 2, [d[1] for d in data], width, label="free layers")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(streams, rotation=20, ha="right")
    ax.set_ylabel("Mean layer R²")
    ax.set_title("Empirical layer information: ground vs free")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "layer_information_ground_free_summary.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Empirical layer information-content diagnostic.")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--telemetry", type=Path, default=None)
    p.add_argument("--control-bundle", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)

    p.add_argument("--ground-altitude-max-m", type=float, default=500.0)
    p.add_argument("--ground-layer-indices", type=str, default=None)

    p.add_argument("--streams", type=str, default="xhat,pol,slopes", help="Comma-separated from: xhat,pol,slopes")
    p.add_argument("--pol-delay", type=int, default=0, help="Use dm_cmd_applied[k-pol_delay] in POL = xhat[k] - dm[k-pol_delay]")
    p.add_argument("--max-components", type=int, default=256)
    p.add_argument("--ridge-alpha", type=float, default=1e-2)
    p.add_argument("--train-fraction", type=float, default=0.7)
    p.add_argument("--standardize-predictors", action="store_true", default=True)
    p.add_argument("--no-standardize-predictors", action="store_false", dest="standardize_predictors")

    p.add_argument("--target-basis", choices=("dm", "control"), default="control")
    p.add_argument("--gain-floor", type=float, default=0.12)

    p.add_argument("--use-gram-ls-truth", action="store_true")
    p.add_argument("--gram-rcond", type=float, default=1e-6)

    p.add_argument("--max-frames", type=int, default=0, help="Optional cap for quick tests. 0 means use all.")
    p.add_argument("--random-state", type=int, default=1)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    telemetry_path = args.telemetry.expanduser().resolve() if args.telemetry else run_dir / "telemetry.npz"
    control_path = args.control_bundle.expanduser().resolve() if args.control_bundle else run_dir / "control_bundle.npz"
    outdir = args.outdir.expanduser().resolve() if args.outdir else run_dir / "layer_information_content"
    outdir.mkdir(parents=True, exist_ok=True)

    tel = np.load(telemetry_path)
    if "truth_layer_modal_atm" not in tel.files:
        raise KeyError("telemetry.npz missing truth_layer_modal_atm")

    control = np.load(control_path) if control_path.exists() else None
    telemetry_summary = load_json(run_dir / "telemetry_summary.json")
    control_summary = load_json(run_dir / "control_bundle_summary.json")

    layers = finite_clip(tel["truth_layer_modal_atm"])
    n_total, n_layer, n_modes = layers.shape

    if args.max_frames and args.max_frames > 0:
        n_total = min(n_total, int(args.max_frames))
        layers = layers[:n_total]

    layer_labels, alt_m = layer_metadata(telemetry_summary, n_layer)

    # Ground/free grouping.
    if args.ground_layer_indices:
        ground_idx = np.asarray([int(p.strip()) for p in args.ground_layer_indices.split(",") if p.strip()], dtype=int)
        ground_idx = ground_idx[(ground_idx >= 0) & (ground_idx < n_layer)]
    elif np.all(np.isfinite(alt_m)):
        ground_idx = np.where(alt_m <= float(args.ground_altitude_max_m))[0].astype(int)
    else:
        ground_idx = np.asarray([0], dtype=int)
    ground_set = set(ground_idx.tolist())
    free_idx = np.asarray([i for i in range(n_layer) if i not in ground_set], dtype=int)

    # Optional Gram-LS truth conversion, layer by layer.
    gram_info: dict[str, Any] = {"truth_projection": "diagonal_inner_product_on_dm_basis"}
    if args.use_gram_ls_truth:
        if control is None or "basis_gram" not in control.files:
            raise KeyError("control_bundle.npz missing basis_gram; rerun with control_bundle_include_gram=True")
        G = np.asarray(control["basis_gram"], dtype=np.float64)[:n_modes, :n_modes]
        converted = np.empty_like(layers, dtype=np.float64)
        for li in range(n_layer):
            converted[:, li, :], info = gram_ls_transform_coeffs(layers[:, li, :], G, args.gram_rcond)
            gram_info = info
        gram_info["truth_projection"] = "Gram-LS"
        layers = converted

    # Control-basis rotation for targets.
    V, svals, trusted, T_name = control_basis(control, args.gain_floor, n_modes)
    if args.target_basis == "control":
        if V is None:
            raise KeyError("--target-basis control requested, but control_bundle has no T_mode_to_mode")
        for li in range(n_layer):
            layers[:, li, :] = layers[:, li, :] @ V
    else:
        trusted = None

    # Prepare telemetry streams.
    requested = [s.strip().lower() for s in args.streams.split(",") if s.strip()]
    streams: dict[str, np.ndarray] = {}

    if "xhat" in requested:
        if "xhat" not in tel.files:
            raise KeyError("Requested xhat stream but telemetry.npz has no xhat")
        streams["xhat"] = finite_clip(tel["xhat"][:n_total, :n_modes])

    if "pol" in requested:
        if "xhat" not in tel.files or "dm_cmd_applied" not in tel.files:
            raise KeyError("Requested pol stream but telemetry.npz needs xhat and dm_cmd_applied")
        xhat = finite_clip(tel["xhat"][:n_total, :n_modes])
        dm = finite_clip(tel["dm_cmd_applied"][:n_total, :n_modes])
        d = int(args.pol_delay)
        if d > 0:
            xhat = xhat[d:]
            dm = dm[:-d]
            layers = layers[d:]
            n_total = layers.shape[0]
        elif d < 0:
            dabs = abs(d)
            xhat = xhat[:-dabs]
            dm = dm[dabs:]
            layers = layers[:-dabs]
            n_total = layers.shape[0]
        streams["pol=xhat-dm"] = xhat - dm

    if "slopes" in requested:
        if "slopes_err_vec" not in tel.files:
            raise KeyError("Requested slopes stream but telemetry.npz has no slopes_err_vec")
        streams["slopes_err_vec"] = finite_clip(tel["slopes_err_vec"][:n_total])

    if not streams:
        raise ValueError("No valid telemetry streams requested")

    # Align all streams to final n_total, in case POL delay modified layers.
    n_total = min([layers.shape[0]] + [x.shape[0] for x in streams.values()])
    layers = layers[:n_total]
    for k in list(streams.keys()):
        streams[k] = streams[k][:n_total]

    n_train = int(round(float(args.train_fraction) * n_total))
    n_train = max(10, min(n_train, n_total - 10))
    train = slice(0, n_train)
    test = slice(n_train, n_total)

    results: dict[str, list[dict[str, Any]]] = {}
    predictor_info: dict[str, Any] = {}

    for stream_name, X in streams.items():
        if args.verbose:
            print(f"Preparing predictor stream {stream_name}: shape={X.shape}")

        Ztr, Zte, info = reduce_predictors_pca(
            X[train],
            X[test],
            max_components=args.max_components,
            standardize=args.standardize_predictors,
            random_state=args.random_state,
        )
        predictor_info[stream_name] = info
        results[stream_name] = []

        for li in range(n_layer):
            Y = layers[:, li, :]
            Ytr = Y[train]
            Yte = Y[test]
            Ypred = ridge_predict(Ztr, Zte, Ytr, ridge_alpha=args.ridge_alpha)
            metrics = metrics_for_prediction(Ytr, Yte, Ypred, trusted)
            results[stream_name].append(
                {
                    "layer_index": int(li),
                    "layer_label": layer_labels[li],
                    "layer_altitude_m": float(alt_m[li]) if np.isfinite(alt_m[li]) else None,
                    "is_ground": bool(li in ground_set),
                    "metrics": metrics,
                }
            )

    plot_layer_bars(
        results,
        layer_labels,
        outdir,
        metric="r2_full",
        fname="layer_information_r2_full.png",
        title="Empirical layer information content: variance-weighted R²",
    )
    plot_layer_bars(
        results,
        layer_labels,
        outdir,
        metric="corr_mode_median",
        fname="layer_information_median_modal_corr.png",
        title="Empirical layer information content: median modal correlation",
    )
    if trusted is not None:
        plot_layer_bars(
            results,
            layer_labels,
            outdir,
            metric="r2_trusted",
            fname="layer_information_r2_trusted.png",
            title="Empirical layer information content: trusted-mode R²",
        )
        plot_layer_bars(
            results,
            layer_labels,
            outdir,
            metric="r2_missing",
            fname="layer_information_r2_missing.png",
            title="Empirical layer information content: missing-mode R²",
        )
    plot_ground_free_summary(results, ground_idx, free_idx, outdir)

    # Compact ground/free means for summary.
    ground_free_summary: dict[str, Any] = {}
    for stream_name, rows in results.items():
        vals = np.asarray([float(r["metrics"].get("r2_full", np.nan)) for r in rows], dtype=float)
        ground_free_summary[stream_name] = {
            "mean_r2_ground_layers": float(np.nanmean(vals[ground_idx])) if ground_idx.size else float("nan"),
            "mean_r2_free_layers": float(np.nanmean(vals[free_idx])) if free_idx.size else float("nan"),
            "max_r2_layer": int(np.nanargmax(vals)),
            "max_r2_value": float(np.nanmax(vals)),
        }

    summary = {
        "run_dir": str(run_dir),
        "telemetry_npz": str(telemetry_path),
        "control_bundle_npz": str(control_path) if control_path.exists() else None,
        "outdir": str(outdir),
        "n_frames": int(n_total),
        "n_train": int(n_train),
        "n_test": int(n_total - n_train),
        "n_layers": int(n_layer),
        "n_modes": int(n_modes),
        "target_basis": args.target_basis,
        "truth_projection": gram_info.get("truth_projection"),
        "gram_info": gram_info,
        "ground_altitude_max_m": float(args.ground_altitude_max_m),
        "ground_layer_indices": ground_idx.tolist(),
        "free_layer_indices": free_idx.tolist(),
        "layer_labels": layer_labels,
        "control_operator": {
            "T_name": T_name,
            "gain_floor": float(args.gain_floor),
            "n_trusted": int(np.sum(trusted)) if trusted is not None else None,
            "n_missing": int(np.sum(~trusted)) if trusted is not None else None,
            "s_min": float(np.min(svals)) if svals is not None else None,
            "s_median": float(np.median(svals)) if svals is not None else None,
            "s_max": float(np.max(svals)) if svals is not None else None,
        },
        "predictor_info": predictor_info,
        "ridge_alpha": float(args.ridge_alpha),
        "results": results,
        "ground_free_summary": ground_free_summary,
        "notes": {
            "metric": "Held-out regularized linear predictability of each layer's modal truth from telemetry streams.",
            "not_formal_fisher": "This is empirical information content, not analytical F = H^T N^-1 H.",
            "interpretation": "High R2 means telemetry stream contains linearly recoverable information about that layer; low R2 means little usable information.",
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

    summary_path = outdir / "layer_information_content_summary.json"
    save_json(summary, summary_path)

    print("\n=== Empirical layer information-content diagnostic ===")
    print(f"Run dir: {run_dir}")
    print(f"Output:  {outdir}")
    print(f"Truth projection: {gram_info.get('truth_projection')}")
    if args.use_gram_ls_truth:
        print(f"Gram rcond: {gram_info.get('gram_rcond')}; kept {gram_info.get('gram_n_kept')}/{gram_info.get('gram_n_eig')} Gram eigenmodes")
    print(f"Target basis: {args.target_basis}")
    if trusted is not None:
        print(f"Control trusted/missing: {int(np.sum(trusted))}/{int(np.sum(~trusted))}")
    print(f"Frames train/test: {n_train}/{n_total-n_train}")
    print("\nLayer R2 by stream:")
    header = "layer".ljust(14) + "".join([name.rjust(18) for name in results.keys()])
    print(header)
    for li, lab in enumerate(layer_labels):
        row = lab.ljust(14)
        for name in results.keys():
            row += f"{results[name][li]['metrics']['r2_full']:18.3f}"
        print(row)

    print("\nGround/free mean R2:")
    for name, gf in ground_free_summary.items():
        print(
            f"  {name:16s} "
            f"ground={gf['mean_r2_ground_layers']:+.3f}  "
            f"free={gf['mean_r2_free_layers']:+.3f}  "
            f"best_layer=L{gf['max_r2_layer']} ({gf['max_r2_value']:+.3f})"
        )

    print(f"\nWrote: {summary_path}")
    if args.verbose:
        print("\nKey plots:")
        for p in sorted(outdir.glob("*.png")):
            print(f"  {p}")


if __name__ == "__main__":
    main()
