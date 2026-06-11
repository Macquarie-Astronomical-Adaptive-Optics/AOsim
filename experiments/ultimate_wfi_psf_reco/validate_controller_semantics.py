#!/usr/bin/env python3
"""
Validate controller telemetry semantics for AOsim GLAO PSF-R runs.

This checks what saved telemetry actually means before we use it for PSF-R:

1. Reconstructor consistency:
       xhat ?= R_raw_slope_to_mode @ slopes_err_vec

2. DM update law:
       dm_cmd_new[k] ?= a * state[k-1] +/- gain * xhat[k or k-1]

3. DM delay convention:
       dm_cmd_applied[k] ?= dm_cmd_new[k - delay]

4. Optional truth bookkeeping:
       truth_atm ?= truth_residual + truth_dm

Typical use:

RUN_DIR=$(ls -td experiments/ultimate_wfi_psf_reco/outputs/telemetry_truth_smoke/longrun_* | head -1)

python experiments/ultimate_wfi_psf_reco/validate_controller_semantics.py \
    "$RUN_DIR" \
    --outdir "$RUN_DIR/controller_semantics_validation" \
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


def as_float(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def finite_mask_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & np.isfinite(b)


def rel_rms_error(target: np.ndarray, estimate: np.ndarray) -> float:
    """Relative RMS error ||target-estimate|| / ||target|| over finite entries."""
    target = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    m = finite_mask_pair(target, estimate)
    if not np.any(m):
        return float("nan")
    den = np.sqrt(np.nanmean(target[m] ** 2))
    if den <= 0:
        return float("nan")
    num = np.sqrt(np.nanmean((target[m] - estimate[m]) ** 2))
    return float(num / den)


def rms(a: np.ndarray, axis=None) -> np.ndarray:
    return np.sqrt(np.nanmean(np.asarray(a, dtype=np.float64) ** 2, axis=axis))


def corr_flat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    m = finite_mask_pair(a, b)
    if np.sum(m) < 2:
        return float("nan")
    aa = a[m] - np.nanmean(a[m])
    bb = b[m] - np.nanmean(b[m])
    den = np.sqrt(np.sum(aa * aa) * np.sum(bb * bb))
    if den <= 0:
        return float("nan")
    return float(np.sum(aa * bb) / den)


def corr_by_mode(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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


def fit_scalar_scale(target: np.ndarray, predictor: np.ndarray) -> float:
    """Least-squares scalar alpha for target ~= alpha * predictor."""
    a = np.asarray(target, dtype=np.float64).ravel()
    b = np.asarray(predictor, dtype=np.float64).ravel()
    m = finite_mask_pair(a, b)
    if not np.any(m):
        return float("nan")
    den = np.dot(b[m], b[m])
    if den <= 0:
        return float("nan")
    return float(np.dot(a[m], b[m]) / den)


def subset_contiguous(arr: np.ndarray, start: int, max_samples: int | None) -> np.ndarray:
    n = arr.shape[0]
    start = max(0, min(int(start), n))
    stop = n if max_samples is None or max_samples <= 0 else min(n, start + int(max_samples))
    return arr[start:stop]


def predict_xhat_from_slopes(slopes: np.ndarray, R: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Compute slopes @ R.T in chunks. R has shape modes x slopes."""
    slopes = np.asarray(slopes, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError(f"R must be 2D, got {R.shape}")
    if R.shape[1] != slopes.shape[1]:
        raise ValueError(f"R has {R.shape[1]} slopes but telemetry has {slopes.shape[1]}")
    n = slopes.shape[0]
    out = np.empty((n, R.shape[0]), dtype=np.float64)
    Rt = R.T
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        out[i0:i1] = slopes[i0:i1] @ Rt
    return out


def evaluate_xhat_reconstruction(xhat, slopes, control, chunk: int):
    candidates = []
    best_arrays = None

    for r_name in ("R_raw_slope_to_mode", "R_weighted_slope_to_mode"):
        if r_name not in control.files:
            continue
        R = np.asarray(control[r_name])
        if R.ndim != 2 or R.shape[1] != slopes.shape[1]:
            continue

        pred0 = predict_xhat_from_slopes(slopes, R, chunk=chunk)
        n_modes = min(xhat.shape[1], pred0.shape[1])
        xx = xhat[:, :n_modes]
        pp0 = pred0[:, :n_modes]

        for sign in (+1.0, -1.0):
            pp = sign * pp0
            alpha = fit_scalar_scale(xx, pp)
            pp_scaled = alpha * pp if np.isfinite(alpha) else pp
            corr_modes = corr_by_mode(xx, pp)
            item = {
                "matrix": r_name,
                "sign": sign,
                "label": f"xhat ~= {sign:+.0f} * {r_name} @ slopes",
                "n_samples": int(xx.shape[0]),
                "n_modes": int(n_modes),
                "rel_rms_error": rel_rms_error(xx, pp),
                "corr_flat": corr_flat(xx, pp),
                "median_modal_corr": float(np.nanmedian(corr_modes)),
                "mean_modal_corr": float(np.nanmean(corr_modes)),
                "best_scalar_alpha_for_xhat_eq_alpha_pred": alpha,
                "rel_rms_error_after_scalar_fit": rel_rms_error(xx, pp_scaled),
            }
            candidates.append(item)

            score = item["rel_rms_error_after_scalar_fit"]
            if best_arrays is None or score < best_arrays["score"]:
                best_arrays = {
                    "score": score,
                    "candidate": item,
                    "target": xx,
                    "pred": pp,
                    "pred_scaled": pp_scaled,
                }

    return candidates, best_arrays


def evaluate_dm_update_law(xhat, channels: dict[str, np.ndarray], gain: float, leak: float):
    """Test dm_cmd_new[k] against a*state[k-1] +/- gain*xhat[k or k-1]."""
    if "dm_cmd_new" not in channels:
        return []

    target = channels["dm_cmd_new"]
    n = min(target.shape[0], xhat.shape[0], *(v.shape[0] for v in channels.values()))
    target = target[:n]
    xhat = xhat[:n]
    channels = {k: v[:n] for k, v in channels.items()}

    out = []
    memory_options = [
        ("1-leak", 1.0 - float(leak)),
        ("1", 1.0),
        ("leak", float(leak)),
        ("0", 0.0),
    ]
    state_names = [k for k in ("dm_cmd", "dm_cmd_new", "dm_cmd_applied") if k in channels]

    for state_name in state_names:
        state = channels[state_name]
        for mem_name, mem_coeff in memory_options:
            for sign in (+1.0, -1.0):
                for xhat_lag in (0, 1):
                    y = target[1:]
                    xterm = xhat[1:] if xhat_lag == 0 else xhat[:-1]
                    pred = mem_coeff * state[:-1] + sign * float(gain) * xterm
                    m = min(y.shape[1], pred.shape[1])
                    y = y[:, :m]
                    pred = pred[:, :m]
                    alpha = fit_scalar_scale(y, pred)
                    pred_scaled = alpha * pred if np.isfinite(alpha) else pred
                    out.append({
                        "target": "dm_cmd_new[k]",
                        "state": f"{state_name}[k-1]",
                        "memory_coeff_label": mem_name,
                        "memory_coeff": float(mem_coeff),
                        "xhat_term": f"xhat[k-{xhat_lag}]",
                        "xhat_sign": float(sign),
                        "gain": float(gain),
                        "equation": (
                            f"dm_cmd_new[k] ~= {mem_coeff:.6g}*{state_name}[k-1] "
                            f"{'+' if sign > 0 else '-'} {gain:.6g}*xhat[k-{xhat_lag}]"
                        ),
                        "n_samples": int(y.shape[0]),
                        "n_modes": int(m),
                        "rel_rms_error": rel_rms_error(y, pred),
                        "corr_flat": corr_flat(y, pred),
                        "best_scalar_alpha_for_target_eq_alpha_pred": alpha,
                        "rel_rms_error_after_scalar_fit": rel_rms_error(y, pred_scaled),
                    })

    out.sort(key=lambda d: (np.inf if not np.isfinite(d["rel_rms_error"]) else d["rel_rms_error"]))
    return out


def evaluate_dm_delay(channels: dict[str, np.ndarray], max_delay: int):
    if "dm_cmd_applied" not in channels:
        return []
    applied = channels["dm_cmd_applied"]
    source_names = [k for k in ("dm_cmd_new", "dm_cmd") if k in channels]
    out = []
    for src_name in source_names:
        src = channels[src_name]
        n = min(applied.shape[0], src.shape[0])
        a = applied[:n]
        s = src[:n]
        for d in range(0, max_delay + 1):
            if d == 0:
                aa = a
                ss = s
            else:
                aa = a[d:]
                ss = s[:-d]
            m = min(aa.shape[1], ss.shape[1])
            aa = aa[:, :m]
            ss = ss[:, :m]
            out.append({
                "equation": f"dm_cmd_applied[k] ~= {src_name}[k-{d}]",
                "source": src_name,
                "delay": int(d),
                "n_samples": int(aa.shape[0]),
                "n_modes": int(m),
                "rel_rms_error": rel_rms_error(aa, ss),
                "corr_flat": corr_flat(aa, ss),
            })
    out.sort(key=lambda d: (np.inf if not np.isfinite(d["rel_rms_error"]) else d["rel_rms_error"]))
    return out


def evaluate_truth_bookkeeping(tel, start: int, max_samples: int | None):
    required = ("truth_modal_atm_total", "truth_modal_residual_total", "truth_modal_dm_applied")
    if not all(k in tel.files for k in required):
        return {"available": False}
    atm = subset_contiguous(as_float(tel["truth_modal_atm_total"]), start, max_samples)
    res = subset_contiguous(as_float(tel["truth_modal_residual_total"]), start, max_samples)
    dm = subset_contiguous(as_float(tel["truth_modal_dm_applied"]), start, max_samples)
    n = min(atm.shape[0], res.shape[0], dm.shape[0])
    m = min(atm.shape[1], res.shape[1], dm.shape[1])
    atm = atm[:n, :m]
    res = res[:n, :m]
    dm = dm[:n, :m]
    e_plus = rel_rms_error(atm, res + dm)
    e_minus = rel_rms_error(atm, res - dm)
    return {
        "available": True,
        "n_samples": int(n),
        "n_modes": int(m),
        "rel_rms_error_atm_minus_res_plus_dm": e_plus,
        "rel_rms_error_atm_minus_res_minus_dm": e_minus,
        "preferred_identity": "atm ~= residual + dm" if e_plus <= e_minus else "atm ~= residual - dm",
    }


def plot_xhat_best(best, outdir: Path):
    target = best["target"]
    pred = best["pred"]
    pred_scaled = best["pred_scaled"]
    cand = best["candidate"]
    t = np.arange(target.shape[0])

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(t, rms(target, axis=1), label="xhat RMS")
    ax.plot(t, rms(pred, axis=1), label="prediction RMS")
    ax.plot(t, rms(pred_scaled, axis=1), label="scaled prediction RMS")
    ax.plot(t, rms(target - pred_scaled, axis=1), label="residual RMS")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("RMS [native units]")
    ax.set_title(f"Best xhat-from-slopes: {cand['matrix']}, sign={cand['sign']:+.0f}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "xhat_from_slopes_best_timeseries.png")
    plt.close(fig)

    corr = corr_by_mode(target, pred_scaled)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(np.arange(len(corr)), corr)
    ax.axhline(0, lw=1)
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Correlation")
    ax.set_title("Per-mode correlation: xhat vs best predicted xhat")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "xhat_from_slopes_best_modal_correlation.png")
    plt.close(fig)


def plot_dm_delay(delay_results, outdir: Path):
    if not delay_results:
        return
    sources = sorted(set(r["source"] for r in delay_results))
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for src in sources:
        rr = sorted([r for r in delay_results if r["source"] == src], key=lambda x: x["delay"])
        ax.plot([r["delay"] for r in rr], [r["rel_rms_error"] for r in rr], marker="o", label=src)
    ax.set_xlabel("Candidate delay [frames]")
    ax.set_ylabel("Relative RMS error")
    ax.set_title("DM delay convention check")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "dm_delay_check.png")
    plt.close(fig)


def plot_dm_update_top(update_results, outdir: Path, n_top: int = 12):
    if not update_results:
        return
    rr = update_results[:n_top]
    labels = [r["equation"].replace("dm_cmd_new[k] ~= ", "") for r in rr]
    vals = [r["rel_rms_error"] for r in rr]
    fig, ax = plt.subplots(figsize=(11, max(5, 0.35 * len(rr))), dpi=150)
    y = np.arange(len(rr))
    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Relative RMS error")
    ax.set_title("Best DM update-law candidates")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "dm_update_law_top_candidates.png")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Validate AOsim controller telemetry semantics.")
    p.add_argument("run_dir", type=Path, help="Run directory containing telemetry.npz and control_bundle.npz")
    p.add_argument("--telemetry", type=Path, default=None)
    p.add_argument("--control-bundle", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--start-sample", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    p.add_argument("--matmul-chunk", type=int, default=256)
    p.add_argument("--max-delay", type=int, default=6)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    telemetry_path = args.telemetry.expanduser().resolve() if args.telemetry else run_dir / "telemetry.npz"
    control_path = args.control_bundle.expanduser().resolve() if args.control_bundle else run_dir / "control_bundle.npz"
    outdir = args.outdir.expanduser().resolve() if args.outdir else run_dir / "controller_semantics_validation"
    outdir.mkdir(parents=True, exist_ok=True)

    telemetry_summary = load_json(run_dir / "telemetry_summary.json")
    control_summary = load_json(run_dir / "control_bundle_summary.json")

    tel = np.load(telemetry_path)
    control = np.load(control_path)

    if "xhat" not in tel.files:
        raise KeyError("telemetry.npz is missing xhat")
    if "slopes_err_vec" not in tel.files:
        raise KeyError("telemetry.npz is missing slopes_err_vec")

    max_samples = None if args.max_samples <= 0 else args.max_samples
    xhat = subset_contiguous(as_float(tel["xhat"]), args.start_sample, max_samples)
    slopes = subset_contiguous(as_float(tel["slopes_err_vec"]), args.start_sample, max_samples)
    n = min(xhat.shape[0], slopes.shape[0])
    xhat = xhat[:n]
    slopes = slopes[:n]

    channel_names = [k for k in ("dm_cmd", "dm_cmd_new", "dm_cmd_applied") if k in tel.files]
    channels = {k: subset_contiguous(as_float(tel[k]), args.start_sample, max_samples)[:n] for k in channel_names}

    gain = float(telemetry_summary.get("gain", 1.0))
    leak = float(telemetry_summary.get("leak", 0.0))
    dm_delay_frames = telemetry_summary.get("dm_delay_frames", None)

    xhat_results, best_xhat = evaluate_xhat_reconstruction(xhat, slopes, control, chunk=args.matmul_chunk)
    if best_xhat is not None:
        plot_xhat_best(best_xhat, outdir)

    update_results = evaluate_dm_update_law(xhat, channels, gain=gain, leak=leak)
    delay_results = evaluate_dm_delay(channels, max_delay=args.max_delay)
    truth_bookkeeping = evaluate_truth_bookkeeping(tel, args.start_sample, max_samples)

    plot_dm_delay(delay_results, outdir)
    plot_dm_update_top(update_results, outdir)

    summary = {
        "run_dir": str(run_dir),
        "telemetry_npz": str(telemetry_path),
        "control_bundle_npz": str(control_path),
        "outdir": str(outdir),
        "sample_range": {
            "start_sample": int(args.start_sample),
            "n_samples_used": int(n),
            "max_samples_arg": int(args.max_samples),
        },
        "telemetry_dimensions": {
            "xhat_shape_used": list(xhat.shape),
            "slopes_err_vec_shape_used": list(slopes.shape),
            "dm_channels_used": {k: list(v.shape) for k, v in channels.items()},
        },
        "controller_metadata": {
            "gain": gain,
            "leak": leak,
            "dm_delay_frames": dm_delay_frames,
        },
        "xhat_from_slopes": {
            "available": bool(xhat_results),
            "candidates": xhat_results,
            "best_by_scaled_rel_rms_error": best_xhat["candidate"] if best_xhat is not None else None,
        },
        "dm_update_law": {
            "available": bool(update_results),
            "best_by_rel_rms_error": update_results[0] if update_results else None,
            "top_candidates": update_results[:20],
        },
        "dm_delay": {
            "available": bool(delay_results),
            "best_by_rel_rms_error": delay_results[0] if delay_results else None,
            "all_candidates": delay_results,
        },
        "truth_bookkeeping": truth_bookkeeping,
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

    summary_path = outdir / "controller_semantics_summary.json"
    save_json(summary, summary_path)

    print("\n=== Controller semantics validation summary ===")
    print(f"Run dir: {run_dir}")
    print(f"Output:  {outdir}")
    print(f"Samples used: {n}")
    print(f"gain={gain:g}, leak={leak:g}, dm_delay_frames={dm_delay_frames}")

    if best_xhat is not None:
        b = best_xhat["candidate"]
        print("\nBest xhat-from-slopes candidate:")
        print(f"  {b['label']}")
        print(f"  rel RMS error = {b['rel_rms_error']:.4g}")
        print(f"  rel RMS error after scalar fit = {b['rel_rms_error_after_scalar_fit']:.4g}")
        print(f"  alpha = {b['best_scalar_alpha_for_xhat_eq_alpha_pred']:.6g}")
        print(f"  median modal corr = {b['median_modal_corr']:+.3f}")
    else:
        print("\nNo usable R_*_slope_to_mode matrix found for xhat check.")

    if update_results:
        b = update_results[0]
        print("\nBest DM update-law candidate:")
        print(f"  {b['equation']}")
        print(f"  rel RMS error = {b['rel_rms_error']:.4g}")
        print(f"  rel RMS error after scalar fit = {b['rel_rms_error_after_scalar_fit']:.4g}")
        print(f"  corr = {b['corr_flat']:+.4f}")

    if delay_results:
        b = delay_results[0]
        print("\nBest DM delay candidate:")
        print(f"  {b['equation']}")
        print(f"  rel RMS error = {b['rel_rms_error']:.4g}")
        print(f"  corr = {b['corr_flat']:+.4f}")

    if truth_bookkeeping.get("available"):
        print("\nTruth bookkeeping:")
        print(f"  {truth_bookkeeping['preferred_identity']}")
        print(f"  err(atm - (res+dm)) = {truth_bookkeeping['rel_rms_error_atm_minus_res_plus_dm']:.4g}")
        print(f"  err(atm - (res-dm)) = {truth_bookkeeping['rel_rms_error_atm_minus_res_minus_dm']:.4g}")

    print(f"\nWrote: {summary_path}")
    if args.verbose:
        print("\nKey plots:")
        for p in sorted(outdir.glob("*.png")):
            print(f"  {p}")


if __name__ == "__main__":
    main()
