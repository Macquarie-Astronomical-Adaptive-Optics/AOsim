#How wrong would WFI science be if we assumed one PSF across the whole field? 
#!/usr/bin/env python3
"""
Compare simple PSF reconstruction baselines against MAAO true PSF metrics.

This is the first, deliberately simple PSF-R analysis script for
ULTIMATE/WFI headless field-map runs.

Input:
    A headless_summary.json from a run that contains offaxis_metrics.

Typical run type:
    write_eval_fits = True
    eval_offaxis_arcsec = [...]
    psf_tt_mode = "none"

What this script does:
    1. Reads true PSF metrics from offaxis_metrics.
    2. Builds simple baseline PSF models from those metrics.
    3. Compares baseline predictions to the MAAO truth at each field point.
    4. Writes CSV, JSON, and simple diagnostic plots.

Current baseline models:
    constant_center
        Use the centre or nearest-to-centre field point as the predicted PSF
        everywhere. This quantifies the error from assuming one constant PSF
        over the WFI field.

    field_mean
        Use the mean PSF metric over all field points as the predicted PSF
        everywhere. This is another simple no-spatial-information baseline.

Important:
    These are not final PSF reconstruction algorithms. They are zero-point
    baselines. They tell us how much field-dependent PSF variation exists
    before introducing TIPTOP/P3, telemetry, or hybrid methods.

Example:
    RUN_DIR=$(ls -td experiments/ultimate_wfi_psf_reco/outputs/wfi_3x3_fieldmap_10s/longrun_* | head -1)

    python3 experiments/ultimate_wfi_psf_reco/compare_psf_baselines.py \
        "$RUN_DIR/headless_summary.json" \
        --mode corrected \
        --baseline constant_center \
        --outdir "$RUN_DIR/psf_baseline_analysis"

Outputs:
    psf_baseline_errors_<mode>_<baseline>.csv
    psf_baseline_errors_<mode>_<baseline>.json
    <metric>_<mode>_<baseline>_absolute_error_map.png
    <metric>_<mode>_<baseline>_fractional_error_map.png
    <metric>_<mode>_<baseline>_error_vs_field_radius.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt


METRIC_KEYS = {
    "gaussian": {
        "label": "Gaussian FWHM",
        "suffix": "",
        "unit": "arcsec",
    },
    "moffat": {
        "label": "Moffat FWHM",
        "suffix": "_moffat",
        "unit": "arcsec",
    },
    "moffat_wings": {
        "label": "Moffat wings FWHM",
        "suffix": "_moffat_wings",
        "unit": "arcsec",
    },
    "contour": {
        "label": "Contour equivalent FWHM",
        "suffix": "_contour",
        "unit": "arcsec",
    },
    "contour_major": {
        "label": "Contour major FWHM",
        "suffix": "_contour_major",
        "unit": "arcsec",
    },
    "contour_minor": {
        "label": "Contour minor FWHM",
        "suffix": "_contour_minor",
        "unit": "arcsec",
    },
}


def _as_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _metric_key(metric_name: str, mode: str) -> str:
    if metric_name not in METRIC_KEYS:
        raise KeyError(f"Unknown metric {metric_name!r}. Valid: {sorted(METRIC_KEYS)}")
    return f"fwhm_arcsec_{mode}{METRIC_KEYS[metric_name]['suffix']}"


def load_offaxis_metrics(summary_json: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with summary_json.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    metrics = summary.get("offaxis_metrics", [])
    if not metrics:
        raise ValueError(
            "No offaxis_metrics found in summary JSON. "
            "Use a field-map run with write_eval_fits=True and eval_offaxis_arcsec set."
        )

    return summary, metrics


def choose_center_index(dx: np.ndarray, dy: np.ndarray) -> int:
    r = np.hypot(dx, dy)
    return int(np.nanargmin(r))


def baseline_value(
    values: np.ndarray,
    baseline: str,
    center_index: int,
) -> float:
    finite = values[np.isfinite(values)]

    if baseline == "constant_center":
        return float(values[center_index])

    if baseline == "field_mean":
        if finite.size == 0:
            return float("nan")
        return float(np.mean(finite))

    if baseline == "field_median":
        if finite.size == 0:
            return float("nan")
        return float(np.median(finite))

    raise ValueError(f"Unknown baseline {baseline!r}")


def build_rows(
    metrics: list[dict[str, Any]],
    mode: str,
    baseline: str,
    metric_names: list[str],
) -> list[dict[str, Any]]:
    dx = np.array([_as_float(m.get("dx_arcsec")) for m in metrics], dtype=float)
    dy = np.array([_as_float(m.get("dy_arcsec")) for m in metrics], dtype=float)
    r_arcsec = np.hypot(dx, dy)
    r_arcmin = r_arcsec / 60.0
    theta_deg = np.degrees(np.arctan2(dy, dx))
    theta_deg = np.where(theta_deg < 0, theta_deg + 360.0, theta_deg)

    center_index = choose_center_index(dx, dy)

    rows: list[dict[str, Any]] = []

    for metric_name in metric_names:
        key = _metric_key(metric_name, mode)
        label = METRIC_KEYS[metric_name]["label"]

        truth_arcsec = np.array([_as_float(m.get(key)) for m in metrics], dtype=float)
        pred_arcsec = baseline_value(truth_arcsec, baseline=baseline, center_index=center_index)

        for i in range(len(metrics)):
            truth = float(truth_arcsec[i])
            pred = float(pred_arcsec)

            abs_err = truth - pred if np.isfinite(truth) and np.isfinite(pred) else float("nan")
            abs_err_mas = abs_err * 1000.0 if np.isfinite(abs_err) else float("nan")
            truth_mas = truth * 1000.0 if np.isfinite(truth) else float("nan")
            pred_mas = pred * 1000.0 if np.isfinite(pred) else float("nan")

            frac_err = abs_err / truth if np.isfinite(abs_err) and np.isfinite(truth) and truth != 0 else float("nan")
            frac_err_percent = frac_err * 100.0 if np.isfinite(frac_err) else float("nan")

            rows.append(
                {
                    "mode": mode,
                    "baseline": baseline,
                    "metric": metric_name,
                    "metric_label": label,
                    "metric_key": key,
                    "point_index": i,
                    "is_center_reference": bool(i == center_index),
                    "dx_arcsec": float(dx[i]),
                    "dy_arcsec": float(dy[i]),
                    "field_radius_arcmin": float(r_arcmin[i]),
                    "field_theta_deg": float(theta_deg[i]),
                    "truth_arcsec": truth,
                    "baseline_arcsec": pred,
                    "truth_mas": truth_mas,
                    "baseline_mas": pred_mas,
                    "absolute_error_arcsec": abs_err,
                    "absolute_error_mas": abs_err_mas,
                    "fractional_error": float(frac_err),
                    "fractional_error_percent": float(frac_err_percent),
                    "absolute_fractional_error_percent": float(abs(frac_err_percent)) if np.isfinite(frac_err_percent) else float("nan"),
                }
            )

    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    metrics = sorted(set(str(r["metric"]) for r in rows))
    for metric in metrics:
        rr = [r for r in rows if r["metric"] == metric and not r["is_center_reference"]]

        abs_err_mas = np.array([_as_float(r["absolute_error_mas"]) for r in rr], dtype=float)
        frac_abs_percent = np.array([_as_float(r["absolute_fractional_error_percent"]) for r in rr], dtype=float)

        abs_err_mas = abs_err_mas[np.isfinite(abs_err_mas)]
        frac_abs_percent = frac_abs_percent[np.isfinite(frac_abs_percent)]

        def stats(v: np.ndarray) -> dict[str, float | int]:
            if v.size == 0:
                return {
                    "n": 0,
                    "mean": float("nan"),
                    "median": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                    "rms": float("nan"),
                }
            return {
                "n": int(v.size),
                "mean": float(np.mean(v)),
                "median": float(np.median(v)),
                "std": float(np.std(v)),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
                "rms": float(np.sqrt(np.mean(v**2))),
            }

        summary[metric] = {
            "absolute_error_mas": stats(abs_err_mas),
            "absolute_fractional_error_percent": stats(frac_abs_percent),
        }

    return summary


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_json(payload: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)


def _rows_for_metric(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    return [r for r in rows if r["metric"] == metric]


def plot_error_map(
    rows: list[dict[str, Any]],
    metric: str,
    value_key: str,
    out_path: Path,
    title: str,
    colorbar_label: str,
    show: bool = False,
) -> None:
    rr = _rows_for_metric(rows, metric)
    if not rr:
        return

    x = np.array([_as_float(r["dx_arcsec"]) / 60.0 for r in rr], dtype=float)
    y = np.array([_as_float(r["dy_arcsec"]) / 60.0 for r in rr], dtype=float)
    v = np.array([_as_float(r[value_key]) for r in rr], dtype=float)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    if not np.any(finite):
        return

    fig, ax = plt.subplots(figsize=(8, 7), dpi=140)

    sc = ax.scatter(
        x[finite],
        y[finite],
        c=v[finite],
        marker="s",
        s=360,
        edgecolors="0.2",
        linewidths=1.0,
    )

    for xi, yi, vi in zip(x[finite], y[finite], v[finite]):
        ax.text(
            xi,
            yi,
            f"{vi:.1f}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    ax.axhline(0, linewidth=0.8, color="0.7")
    ax.axvline(0, linewidth=0.8, color="0.7")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Field X [arcmin]")
    ax.set_ylabel("Field Y [arcmin]")
    ax.set_title(title)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(colorbar_label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_error_vs_radius(
    rows: list[dict[str, Any]],
    metric: str,
    out_path: Path,
    title: str,
    show: bool = False,
) -> None:
    rr = _rows_for_metric(rows, metric)
    if not rr:
        return

    r = np.array([_as_float(x["field_radius_arcmin"]) for x in rr], dtype=float)
    abs_err = np.array([_as_float(x["absolute_error_mas"]) for x in rr], dtype=float)
    frac = np.array([_as_float(x["fractional_error_percent"]) for x in rr], dtype=float)

    finite = np.isfinite(r) & np.isfinite(abs_err) & np.isfinite(frac)
    if not np.any(finite):
        return

    order = np.argsort(r[finite])
    r = r[finite][order]
    abs_err = abs_err[finite][order]
    frac = frac[finite][order]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.plot(r, abs_err, marker="o", linestyle="-", label="Absolute error")
    ax.axhline(0, linewidth=0.8, color="0.7")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Field radius [arcmin]")
    ax.set_ylabel("Truth minus baseline [mas]")
    ax.set_title(title)

    ax2 = ax.twinx()
    ax2.plot(r, frac, marker="s", linestyle="--", label="Fractional error")
    ax2.set_ylabel("Fractional error [%]")

    # Combine legends from both axes.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def make_plots(
    rows: list[dict[str, Any]],
    metrics: list[str],
    mode: str,
    baseline: str,
    outdir: Path,
    show: bool = False,
) -> None:
    for metric in metrics:
        label = METRIC_KEYS[metric]["label"]
        stem = f"{metric}_{mode}_{baseline}"

        plot_error_map(
            rows,
            metric=metric,
            value_key="absolute_error_mas",
            out_path=outdir / f"{stem}_absolute_error_map.png",
            title=f"{label}: {baseline} error map ({mode})",
            colorbar_label="Truth minus baseline [mas]",
            show=show,
        )

        plot_error_map(
            rows,
            metric=metric,
            value_key="fractional_error_percent",
            out_path=outdir / f"{stem}_fractional_error_map.png",
            title=f"{label}: {baseline} fractional error map ({mode})",
            colorbar_label="Fractional error [%]",
            show=show,
        )

        plot_error_vs_radius(
            rows,
            metric=metric,
            out_path=outdir / f"{stem}_error_vs_field_radius.png",
            title=f"{label}: {baseline} error versus field radius ({mode})",
            show=show,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare simple PSF baseline models against MAAO off-axis truth metrics."
    )

    parser.add_argument(
        "summary_json",
        type=Path,
        help="Path to headless_summary.json containing offaxis_metrics.",
    )

    parser.add_argument(
        "--mode",
        choices=["corrected", "uncorrected"],
        default="corrected",
        help="Compare corrected or uncorrected PSF metrics.",
    )

    parser.add_argument(
        "--baseline",
        choices=["constant_center", "field_mean", "field_median"],
        default="constant_center",
        help="Baseline PSF metric model to compare against truth.",
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["gaussian", "moffat", "moffat_wings", "contour"],
        choices=sorted(METRIC_KEYS.keys()),
        help="PSF metrics to compare.",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory. Default is <run_dir>/psf_baseline_analysis.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary_json = args.summary_json.expanduser().resolve()
    run_dir = summary_json.parent

    outdir = args.outdir
    if outdir is None:
        outdir = run_dir / "psf_baseline_analysis"
    outdir = outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    summary, offaxis_metrics = load_offaxis_metrics(summary_json)

    rows = build_rows(
        offaxis_metrics,
        mode=args.mode,
        baseline=args.baseline,
        metric_names=list(args.metrics),
    )

    stats = summarize_rows(rows)

    csv_path = outdir / f"psf_baseline_errors_{args.mode}_{args.baseline}.csv"
    json_path = outdir / f"psf_baseline_errors_{args.mode}_{args.baseline}.json"

    write_csv(rows, csv_path)

    payload = {
        "summary_json": str(summary_json),
        "mode": args.mode,
        "baseline": args.baseline,
        "metrics": list(args.metrics),
        "n_offaxis_points": len(offaxis_metrics),
        "stats_excluding_center_reference": stats,
        "rows": rows,
    }
    write_json(payload, json_path)

    make_plots(
        rows,
        metrics=list(args.metrics),
        mode=args.mode,
        baseline=args.baseline,
        outdir=outdir,
        show=args.show,
    )

    print("Wrote:")
    print(" ", csv_path)
    print(" ", json_path)
    print(" ", outdir)
    print()
    print("Summary, excluding the centre reference point where applicable:")
    for metric, metric_stats in stats.items():
        abs_stats = metric_stats["absolute_error_mas"]
        frac_stats = metric_stats["absolute_fractional_error_percent"]
        print(
            f"  {metric:14s} "
            f"RMS abs error = {abs_stats['rms']:.3g} mas, "
            f"median abs frac error = {frac_stats['median']:.3g} %"
        )


if __name__ == "__main__":
    main()