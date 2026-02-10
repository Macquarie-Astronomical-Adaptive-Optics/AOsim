#!/usr/bin/env python3
"""
Plot "FWHM map" style figures (similar to the provided example) from a summary JSON.

Creates three PNGs (Gaussian, Moffat, Moffat Wings) using the off-axis metrics:
- fwhm_arcsec_{mode}
- fwhm_arcsec_{mode}_moffat
- fwhm_arcsec_{mode}_moffat_wings

Usage:
  python plot_fwhm_maps.py summary.json
  python plot_fwhm_maps.py summary.json --mode uncorrected --outdir plots --show

Notes:
- Axes are in arcmin (dx_arcsec/60, dy_arcsec/60)
- Values are plotted in mas (arcsec * 1000)
- LGS markers at r=10' (stars on axes), NGS at r≈7' (diamonds at ±5',±5')
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D


def _finite(a):
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def load_points(summary_path: Path):
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("offaxis_metrics", [])
    if not metrics:
        raise ValueError("No 'offaxis_metrics' found in the JSON.")

    dx_arcsec = np.array([m["dx_arcsec"] for m in metrics], dtype=float)
    dy_arcsec = np.array([m["dy_arcsec"] for m in metrics], dtype=float)

    # Convert to arcmin for plotting
    x_arcmin = dx_arcsec / 60.0
    y_arcmin = dy_arcsec / 60.0
    return data, x_arcmin, y_arcmin, metrics


def make_single_map(
    x_arcmin,
    y_arcmin,
    fwhm_mas,
    out_path: Path,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str = "RdYlGn_r",
    show: bool = False,
):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=140)

    # Scatter squares
    sc = ax.scatter(
        x_arcmin,
        y_arcmin,
        c=fwhm_mas,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        marker="s",
        s=320,
        edgecolors="0.2",
        linewidths=1.2,
        zorder=3,
    )

    # Value labels (rounded to integer mas)
    for xi, yi, vi in zip(x_arcmin, y_arcmin, fwhm_mas):
        if not np.isfinite(vi):
            continue
        ax.text(
            xi,
            yi,
            f"{int(round(vi))}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="black",
            zorder=4,
        )

    # 14'×14' FoV rectangle centered at (0,0): from -7 to +7 arcmin
    fov_rect = Rectangle(
        (-7, -7), 14, 14,
        fill=False,
        linewidth=2.5,
        edgecolor="0.2",
        zorder=2,
    )
    ax.add_patch(fov_rect)

    # LGS (r=10') stars at cardinal points
    lgs = np.array([(10, 0), (-10, 0), (0, 10), (0, -10)], dtype=float)
    ax.scatter(
        lgs[:, 0], lgs[:, 1],
        marker="*",
        s=420,
        c="#ff9f1c",
        edgecolors="#e36414",
        linewidths=1.2,
        zorder=5,
    )

    # NGS (r≈7') diamonds at (±5,±5) arcmin
    ngs = np.array([(5, 5), (-5, 5), (5, -5), (-5, -5)], dtype=float)
    ax.scatter(
        ngs[:, 0], ngs[:, 1],
        marker="D",
        s=140,
        c="#7b2cbf",
        edgecolors="#3c096c",
        linewidths=1.0,
        zorder=5,
    )

    # Axes styling
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("Field X [arcmin]", fontsize=13)
    ax.set_ylabel("Field Y [arcmin]", fontsize=13)
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.25, zorder=0)
    ax.axhline(0, linewidth=0.8, color="0.7", zorder=1)
    ax.axvline(0, linewidth=0.8, color="0.7", zorder=1)

    # Legend (FoV + guide-star markers)
    legend_handles = [
        Patch(facecolor="none", edgecolor="0.2", linewidth=2.5, label="14'×14' FoV"),
        Line2D([0], [0], marker="*", linestyle="None", markersize=14,
               markerfacecolor="#ff9f1c", markeredgecolor="#e36414", label="LGS (r=10')"),
        Line2D([0], [0], marker="D", linestyle="None", markersize=9,
               markerfacecolor="#7b2cbf", markeredgecolor="#3c096c", label="NGS (r=7')"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("FWHM [mas]", fontsize=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("summary_json", type=Path, help="Path to summary.json")
    p.add_argument("--mode", choices=["corrected", "uncorrected"], default="corrected",
                   help="Use corrected or uncorrected FWHM fields (default: corrected)")
    p.add_argument("--outdir", type=Path, default=Path("."), help="Output directory for PNGs")
    p.add_argument("--prefix", type=str, default="ULTIMATE_Subaru_GLAO",
                   help="Filename prefix for outputs")
    p.add_argument("--common-scale", action="store_true", default=True,
                   help="Use a common color scale across all 3 plots (default: on)")
    p.add_argument("--no-common-scale", dest="common_scale", action="store_false",
                   help="Use independent color scales for each plot")
    p.add_argument("--show", action="store_true", help="Display figures interactively")
    args = p.parse_args()

    data, x_arcmin, y_arcmin, metrics = load_points(args.summary_json)

    # Keys for each PSF model
    key_gauss = f"fwhm_arcsec_{args.mode}"
    key_moff  = f"fwhm_arcsec_{args.mode}_moffat"
    key_wings = f"fwhm_arcsec_{args.mode}_moffat_wings"

    f_gauss_mas = np.array([m.get(key_gauss, np.nan) for m in metrics], dtype=float) * 1000.0
    f_moff_mas  = np.array([m.get(key_moff,  np.nan) for m in metrics], dtype=float) * 1000.0
    f_wing_mas  = np.array([m.get(key_wings, np.nan) for m in metrics], dtype=float) * 1000.0

    # Determine color scale
    if args.common_scale:
        all_vals = np.concatenate([_finite(f_gauss_mas), _finite(f_moff_mas), _finite(f_wing_mas)])
        if all_vals.size == 0:
            raise ValueError("All FWHM values are NaN/inf; cannot set color scale.")
        vmin = float(np.min(all_vals))
        vmax = float(np.max(all_vals))
    else:
        vmin = vmax = math.nan  # will be overridden per-plot

    title_prefix = "ULTIMATE Subaru GLAO — FWHM Map (14'×14' FoV)"
    mode_str = args.mode.capitalize()

    # Gaussian
    if not args.common_scale:
        vals = _finite(f_gauss_mas); vmin, vmax = float(vals.min()), float(vals.max())
    make_single_map(
        x_arcmin, y_arcmin, f_gauss_mas,
        args.outdir / f"{args.prefix}_fwhm_gaussian_{args.mode}.png",
        f"{title_prefix}\nGaussian ({mode_str})",
        vmin, vmax,
        show=args.show,
    )

    # Moffat
    if not args.common_scale:
        vals = _finite(f_moff_mas); vmin, vmax = float(vals.min()), float(vals.max())
    make_single_map(
        x_arcmin, y_arcmin, f_moff_mas,
        args.outdir / f"{args.prefix}_fwhm_moffat_{args.mode}.png",
        f"{title_prefix}\nMoffat ({mode_str})",
        vmin, vmax,
        show=args.show,
    )

    # Moffat wings
    if not args.common_scale:
        vals = _finite(f_wing_mas); vmin, vmax = float(vals.min()), float(vals.max())
    make_single_map(
        x_arcmin, y_arcmin, f_wing_mas,
        args.outdir / f"{args.prefix}_fwhm_moffat_wings_{args.mode}.png",
        f"{title_prefix}\nMoffat Wings ({mode_str})",
        vmin, vmax,
        show=args.show,
    )

    print("Wrote:")
    print(" -", args.outdir / f"{args.prefix}_fwhm_gaussian_{args.mode}.png")
    print(" -", args.outdir / f"{args.prefix}_fwhm_moffat_{args.mode}.png")
    print(" -", args.outdir / f"{args.prefix}_fwhm_moffat_wings_{args.mode}.png")


if __name__ == "__main__":
    main()
