"""
Headless long-exposure PSF field-map  test for ULTIMATE/WFI.

This script is an early reproducibility test for moving AOsim PSF-reconstruction
experiments away from the interactive GUI workflow and toward scripted,
version-controlled, publication-ready runs.

The GUI is useful for development and visual inspection, but PSF reconstruction
research requires repeatable batch experiments: the same configuration, field
points, exposure length, reconstruction settings, and output products should be
runnable from the command line without manual GUI interaction. This script uses
`scripts.core.headless_factory.run_headless_longrun()` to build the AO system,
turbulence model, sensors, reconstructor, and batch runner directly from
`config_ultimate.json`.

The specific experiment here generates a 10 s delivered-PSF test over a 3x3
grid spanning the nominal 14 arcmin x 14 arcmin WFI field. It writes long-exposure
corrected and uncorrected off-axis PSFs, together with a JSON summary containing
FWHM and fitted PSF diagnostics. The run uses `psf_tt_mode="none"` so that the
reported PSFs correspond to delivered image quality, including residual image
motion, rather than a tip/tilt-removed diagnostic PSF.

This is intended as a baseline engineering/science sanity check, not yet a final
validated ULTIMATE performance prediction. Later versions should add richer AO
telemetry products, encircled-energy metrics, field-uniformity diagnostics, and
a clearer separation between observable telemetry and simulator-truth products.
"""

from scripts.core.headless_factory import run_headless_longrun

result = run_headless_longrun(
    "config_ultimate.json",
    {
        "n_frames": 5000,
        "discard_first_s": 1.0,
        "out_dir": "experiments/ultimate_wfi_psf_reco/outputs/wfi_3x3_10s_delivered_safe",
        "timestamped": True,

        # Do not record every on-axis PSF for this field-map test.
        "record_psfs": False,
        "record_psfs_ttremoved": False,
        "save_tt_series": True,

        # Write long-exposure off-axis PSFs.
        "write_eval_fits": True,
        "write_eval_uncorrected_fits": True,
        "eval_accumulate_uncorrected": True,

        "eval_offaxis_arcsec": [
            (-420, -420), (0, -420), (420, -420),
            (-420,    0), (0,    0), (420,    0),
            (-420,  420), (0,  420), (420,  420),
        ],

        # Important: force smaller off-axis FFT batches.
        "chunk_frames": 4,

        "psf_roi": 256,
        "progress_every": 500,
        "psf_tt_mode": "none",
    },
    build_reconstructor_kwargs={
        "chunk_modes": 32,
        "rcond": 8e-2,
    },
)

print("Run directory:", result["out_dir"])
print("Summary JSON:", result["headless_summary_json"])