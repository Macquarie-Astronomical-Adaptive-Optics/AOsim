#!/usr/bin/env python3
"""
Create "FWHM vs Field Position" plots (similar to the example figure) from a summary JSON.

It reads `offaxis_metrics` entries, converts each point to:
  r_arcmin = sqrt(dx_arcsec^2 + dy_arcsec^2) / 60

Then plots FWHM [mas] vs r [arcmin] using *all* available points (no need for
specific radii like 0,1,2,4,6,10 arcmin). Optionally includes an on-axis point
if it exists and is finite in the JSON.

Generates three plots per mode:
  - Gaussian
  - Moffat
  - Moffat wings

Usage:
  python plot_fwhm_vs_field.py summary.json
  python plot_fwhm_vs_field.py summary.json --mode uncorrected
  python plot_fwhm_vs_field.py summary.json --outdir plots --prefix run1

Optional:
  --annotate-theta      also annotate each point with theta [deg]
  --per-angle-bins 45   additionally make a multi-line plot grouped by angle bins
                        (useful only if your sampling includes multiple radii per angle bin)

Notes:
- Colors are left to matplotlib defaults (no explicit colors).
- Reference lines (seeing-limited, diffraction limit) are drawn after the main series
  so they get different default colors automatically.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _finite_mask(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return np.isfinite(a)


def _theta_deg(dx_arcsec: float, dy_arcsec: float) -> float:
    th = math.degrees(math.atan2(dy_arcsec, dx_arcsec))
    return th + 360.0 if th < 0 else th


def load_summary(summary_path: Path):
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = data.get("offaxis_metrics", [])
    if not metrics:
        raise ValueError("No 'offaxis_metrics' entries found in the JSON.")
    return data, metrics


def extract_series(data, metrics, mode: str, key_suffix: str):
    """
    key_suffix in {"", "_moffat", "_moffat_wings"} for Gaussian/Moffat/Wings.
    Returns arrays (r_arcmin, theta_deg, fwhm_mas).
    """
    key = f"fwhm_arcsec_{mode}{key_suffix}"

    dx = np.array([m["dx_arcsec"] for m in metrics], dtype=float)
    dy = np.array([m["dy_arcsec"] for m in metrics], dtype=float)

    r_arcmin = np.hypot(dx, dy) / 60.0
    theta = np.array([_theta_deg(float(x), float(y)) for x, y in zip(dx, dy)], dtype=float)

    fwhm_arcsec = np.array([m.get(key, np.nan) for m in metrics], dtype=float)
    fwhm_mas = fwhm_arcsec * 1000.0

    # Optional on-axis point if available and finite
    # Root-level keys are: fwhm_arcsec_{mode}{suffix}
    root_key = f"fwhm_arcsec_{mode}{key_suffix}"
    root_val = data.get(root_key, np.nan)
    if root_val is not None:
        try:
            root_val = float(root_val)
        except Exception:
            root_val = np.nan

        if np.isfinite(root_val):
            r_arcmin = np.concatenate([np.array([0.0]), r_arcmin])
            theta = np.concatenate([np.array([0.0]), theta])
            fwhm_mas = np.concatenate([np.array([root_val * 1000.0]), fwhm_mas])

    # Keep only finite values
    msk = _finite_mask(fwhm_mas) & _finite_mask(r_arcmin)
    return r_arcmin[msk], theta[msk], fwhm_mas[msk]


def plot_single(
    r_arcmin: np.ndarray,
    fwhm_mas: np.ndarray,
    out_path: Path,
    title: str,
    series_label: str,
    seeing_mas: float,
    diffraction_mas: float,
    annotate_theta: np.ndarray | None = None,
    show: bool = False,
):
    # Sort by r for a nice line
    idx = np.argsort(r_arcmin)
    r = r_arcmin[idx]
    y = fwhm_mas[idx]
    th = annotate_theta[idx] if annotate_theta is not None else None

    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=140)

    # Main series (line + markers)
    ax.plot(r, y, marker="o", linestyle="-", linewidth=2.2, markersize=8, label=series_label)

    # Axis limits first (used for reference lines)
    xmin = max(-0.25, float(np.min(r) - 0.25))
    xmax = float(np.max(r) + 0.5)
    ax.set_xlim(xmin, xmax)

    # Reference lines drawn after main series so they get different default colors
    ax.plot([xmin, xmax], [seeing_mas, seeing_mas], linestyle="--", linewidth=1.6,
            label=f"Seeing-limited @ K ({seeing_mas:.0f} mas)")
    ax.plot([xmin, xmax], [diffraction_mas, diffraction_mas], linestyle=":", linewidth=1.8,
            label=f"Diffraction limit ({diffraction_mas:.0f} mas)")

    # Annotate points with FWHM values (and optionally theta)
    for i, (xi, yi) in enumerate(zip(r, y)):
        txt = f"{yi:.1f}"
        if th is not None:
            txt = f"{txt}\nθ={th[i]:.0f}°"
        ax.annotate(
            txt,
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    ax.set_title(title, fontsize=18, fontweight="bold", pad=10)
    ax.set_xlabel("Field of View [arcmin]", fontsize=14)
    ax.set_ylabel("FWHM [mas]", fontsize=14)
    ax.grid(True, alpha=0.3)

    # y-limits: include reference lines comfortably
    ymax = float(np.nanmax([np.max(y), seeing_mas, diffraction_mas])) * 1.08
    ax.set_ylim(0, ymax)

    ax.legend(loc="upper left", frameon=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_per_angle_bins(
    r_arcmin: np.ndarray,
    theta_deg: np.ndarray,
    fwhm_mas: np.ndarray,
    out_path: Path,
    title: str,
    series_label_prefix: str,
    seeing_mas: float,
    diffraction_mas: float,
    angle_step_deg: float,
    show: bool = False,
):
    """
    Multi-line plot: group points by angle bin (rounded to nearest `angle_step_deg`),
    and plot FWHM vs r for each bin.
    """
    # Bin angles to nearest step
    bins = np.round(theta_deg / angle_step_deg) * angle_step_deg
    bins = np.mod(bins, 360.0)

    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=140)

    # Plot each bin (only if it has >=2 points; otherwise it isn't a "curve")
    unique_bins = sorted(set(float(b) for b in bins))
    for b in unique_bins:
        msk = bins == b
        if np.sum(msk) < 2:
            continue
        rr = r_arcmin[msk]
        yy = fwhm_mas[msk]
        idx = np.argsort(rr)
        ax.plot(rr[idx], yy[idx], marker="o", linestyle="-", linewidth=1.8, markersize=6,
                label=f"{series_label_prefix} θ≈{b:.0f}°")

    # If no bins had >=2 points, still write an informative plot (scatter)
    if len(ax.lines) == 0:
        idx = np.argsort(r_arcmin)
        ax.plot(r_arcmin[idx], fwhm_mas[idx], marker="o", linestyle="-", linewidth=2.0, markersize=7,
                label=series_label_prefix)

    xmin = max(-0.25, float(np.min(r_arcmin) - 0.25))
    xmax = float(np.max(r_arcmin) + 0.5)
    ax.set_xlim(xmin, xmax)

    # Reference lines (different default colors by plotting after the main lines)
    ax.plot([xmin, xmax], [seeing_mas, seeing_mas], linestyle="--", linewidth=1.6,
            label=f"Seeing-limited @ K ({seeing_mas:.0f} mas)")
    ax.plot([xmin, xmax], [diffraction_mas, diffraction_mas], linestyle=":", linewidth=1.8,
            label=f"Diffraction limit ({diffraction_mas:.0f} mas)")

    ax.set_title(title, fontsize=18, fontweight="bold", pad=10)
    ax.set_xlabel("Field of View [arcmin]", fontsize=14)
    ax.set_ylabel("FWHM [mas]", fontsize=14)
    ax.grid(True, alpha=0.3)
    ymax = float(np.nanmax([np.max(fwhm_mas), seeing_mas, diffraction_mas])) * 1.08
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper left", frameon=True, ncols=1)
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
    p.add_argument("--label", type=str, default="ULTIMATE GLAO (K-band)",
                   help="Legend label for the main series")
    p.add_argument("--seeing-mas", type=float, default=486.0,
                   help="Seeing-limited reference FWHM at K [mas] (default: 486)")
    p.add_argument("--diffraction-mas", type=float, default=66.0,
                   help="Diffraction limit reference FWHM [mas] (default: 66)")
    p.add_argument("--annotate-theta", action="store_true",
                   help="Annotate each point with theta [deg] as well as FWHM")
    p.add_argument("--per-angle-bins", type=float, default=None,
                   help="Also output a multi-line plot grouped by angle bins of this size (deg), "
                        "e.g. 30 or 45. Only bins with >=2 points become separate curves.")
    p.add_argument("--show", action="store_true", help="Display figures interactively")
    args = p.parse_args()

    data, metrics = load_summary(args.summary_json)

    mode_str = args.mode.capitalize()
    title_base = "ULTIMATE Subaru GLAO — K-band PSF FWHM vs Field Position"

    series = [
        ("gaussian", "", f"{title_base}\nGaussian ({mode_str})"),
        ("moffat", "_moffat", f"{title_base}\nMoffat ({mode_str})"),
        ("moffat_wings", "_moffat_wings", f"{title_base}\nMoffat Wings ({mode_str})"),
    ]

    wrote = []
    for name, suffix, title in series:
        r_arcmin, theta_deg, fwhm_mas = extract_series(data, metrics, args.mode, suffix)

        out_path = args.outdir / f"{args.prefix}_fwhm_vs_fov_{name}_{args.mode}.png"
        plot_single(
            r_arcmin,
            fwhm_mas,
            out_path,
            title=title,
            series_label=args.label,
            seeing_mas=args.seeing_mas,
            diffraction_mas=args.diffraction_mas,
            annotate_theta=(theta_deg if args.annotate_theta else None),
            show=args.show,
        )
        wrote.append(out_path)

        if args.per_angle_bins is not None:
            out_path_bins = args.outdir / f"{args.prefix}_fwhm_vs_fov_{name}_{args.mode}_anglebins{args.per_angle_bins:g}.png"
            plot_per_angle_bins(
                r_arcmin,
                theta_deg,
                fwhm_mas,
                out_path_bins,
                title=title + f"\n(angle bins {args.per_angle_bins:g}°)",
                series_label_prefix=args.label,
                seeing_mas=args.seeing_mas,
                diffraction_mas=args.diffraction_mas,
                angle_step_deg=float(args.per_angle_bins),
                show=args.show,
            )
            wrote.append(out_path_bins)

    print("Wrote:")
    for pth in wrote:
        print(" -", pth)


if __name__ == "__main__":
    main()
