"""Configuration dtype helpers used by the Qt config editors.

`Config_table` and the sensor tabs use this mapping to convert text input into
python types.

The simulator sometimes shipped this module externally; we include a minimal
version so the repo zips are self-contained.

Unknown keys fall back to `str`. Cast failures leave the value unchanged.
"""

from __future__ import annotations

from typing import Any, Dict


# Key -> python type. Containers are parsed by Config_table via JSON.
CONFIG_DTYPES: Dict[str, type] = {
    # Global / telescope
    "telescope_diameter": float,
    "telescope_center_obscuration": float,
    "grid_size": int,
    "field_padding": int,
    "actuators": int,

    # Wavelengths
    "wfs_lambda": float,
    "science_lambda": float,

    # Runtime
    "frame_rate": float,
    "use_gpu": bool,
    "data_path": str,

    # DM loop
    "loop_gain": float,
    "loop_leak": float,
    "dm_delay_frames": int,

    # TT loop (new)
    "tt_enabled": bool,
    "tt_gain": float,
    "tt_leak": float,
    "tt_rate_hz": float,
    "tt_delay_frames": int,

    # Sensor defaults
    "sub_apertures": int,
    "dx": float,
    "dy": float,
    "gs_range_m": float,
    "lgs_remove_tt": bool,
    "lgs_launch_offset_px": tuple,
    "lgs_thickness_m": float,
    "lenslet_f_m": float,
    "pixel_pitch_m": float,

    # WFS noise / centroiding (new)
    "wfs_photons_per_subap": float,
    "wfs_read_noise_e": float,
    "wfs_background_photons_per_pix": float,
    "wfs_centroid_window_px": int,
    "wfs_centroid_window_mode": str,
}


def enforce_config_types(params: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort in-place caster for a params dict.

    This is used in a couple of tabs to coerce values loaded from JSON.
    The `Config_table` already handles conversion when editing interactively;
    this is mainly for programmatic loads.

    Returns the same dict for convenience.
    """
    for k, dtype in CONFIG_DTYPES.items():
        if k not in params:
            continue
        v = params.get(k)
        if v is None:
            continue
        try:
            if dtype is bool:
                if isinstance(v, str):
                    params[k] = v.strip().lower() in ("true", "1", "yes", "y", "on")
                else:
                    params[k] = bool(v)
            elif dtype is tuple:
                # Allow list/tuple
                if isinstance(v, (list, tuple)):
                    params[k] = tuple(v)
                else:
                    params[k] = tuple(v)  # may raise
            else:
                params[k] = dtype(v)
        except Exception:
            # keep original
            pass
    return params
