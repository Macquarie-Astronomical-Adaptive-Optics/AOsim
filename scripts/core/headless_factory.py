"""Headless construction utilities for AOsim experiments.

This module is intended to let experiment scripts build the same core AO
objects that the GUI uses, without importing QWidget/PySide GUI tabs.

Typical use from the repository root:

    from scripts.core.headless_factory import make_headless_system, run_headless_longrun

    system = make_headless_system("config_ultimate.json", build_reconstructor=False)
    print(system.sensor_keys)

    result = run_headless_longrun(
        "config_ultimate.json",
        {
            "duration_s": 1.0,
            "discard_first_s": 0.1,
            "out_dir": "experiments/ultimate_wfi_psf_reco/outputs",
            "psf_roi": 128,
            "record_psfs": False,
            "eval_offaxis_arcsec": [(0, 0), (300, 300), (-300, -300)],
        },
    )

Notes:
- This still depends on the core simulator dependencies, especially CuPy.
- The default long-run PSF TT mode is set to "fit" because the current
  AOBatchRunner rejects the older GUI default "hybrid".
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import cupy as cp

from scripts.batch import AOBatchRunner
from scripts.phase_screen.infinite_vonkarman import LayeredInfinitePhaseScreen
from scripts.reconstructor import GLAOConjugateIM_CuPy
from scripts.utilities import Pupil_tools, WFSensor_tools, set_params

logger = logging.getLogger(__name__)

ARCSEC2RAD = math.pi / (180.0 * 3600.0)
RAD2ARCSEC = (180.0 * 3600.0) / math.pi

TURBULENCE_FACTORIES: Dict[str, Callable[..., Any]] = {
    "InfVonKarman": LayeredInfinitePhaseScreen,
}


@dataclass
class HeadlessAOSystem:
    """Container for the headless AO objects needed by experiments."""

    params: Dict[str, Any]
    config_path: Optional[Path]
    project_root: Path

    sensors: "OrderedDict[str, Any]"
    sensor_keys: List[str]
    sensor_list: List[Any]

    turbulence_name: str
    turbulence_source: str
    turbulence_profile: Dict[str, Any]
    turbulence_base_kwargs: Dict[str, Any]
    layer_cfgs: List[Dict[str, Any]]

    sim: Any
    pupil: cp.ndarray
    reconstructor: Optional[Any] = None

    @property
    def dt_s(self) -> float:
        return 1.0 / float(self.params.get("frame_rate", 500.0))

    def build_reconstructor(
        self,
        *,
        chunk_modes: int = 64,
        rcond: float = 8e-2,
        conjugation_height_m: float = 0.0,
        sensor_method: str = "southwell",
        **kwargs: Any,
    ) -> Any:
        """Build and attach the default GLAO reconstructor."""
        self.reconstructor = build_gl_aoconjugate_reconstructor(
            params=self.params,
            sim=self.sim,
            sensors=self.sensor_list,
            chunk_modes=chunk_modes,
            rcond=rcond,
            conjugation_height_m=conjugation_height_m,
            sensor_method=sensor_method,
            **kwargs,
        )
        return self.reconstructor

    def make_batch_runner(self, cfg: Optional[Mapping[str, Any]] = None) -> AOBatchRunner:
        """Create an AOBatchRunner using this system."""
        if self.reconstructor is None:
            self.build_reconstructor()
        return build_batch_runner(
            params=self.params,
            sim=self.sim,
            sensors=self.sensor_list,
            reconstructor=self.reconstructor,
            pupil=self.pupil,
            cfg=dict(cfg or {}),
        )


def find_project_root(start: Optional[str | Path] = None) -> Path:
    """Find the AOsim repository root from a file/dir inside the repository."""
    p = Path(start or Path.cwd()).expanduser().resolve()
    if p.is_file():
        p = p.parent

    for cand in (p, *p.parents):
        if (cand / "scripts").is_dir() and (cand / "turbulence").is_dir():
            return cand

    raise FileNotFoundError(
        f"Could not find AOsim project root from {p}. "
        "Expected a directory containing scripts/ and turbulence/."
    )


def load_json_config(config_path: str | Path) -> Tuple[Dict[str, Any], Path, Path]:
    """Load a JSON config and return (params, config_path, project_root)."""
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        params = json.load(f)
    if not isinstance(params, dict):
        raise TypeError(f"Config JSON must contain an object/dict, got {type(params)!r}")

    root = find_project_root(path)
    return params, path, root


def _copy_jsonlike(x: Any) -> Any:
    """Copy a config object without preserving GUI/shared mutable state."""
    return copy.deepcopy(x)


def ensure_default_sensors(params: MutableMapping[str, Any]) -> None:
    """Back-compatible default sensor constellation.

    This mirrors the old GUI fallback in Poke_tab for configs that do not yet
    contain a sensors section. Modern ULTIMATE configs should already define
    params["sensors"].
    """
    sensors = params.get("sensors")
    if isinstance(sensors, dict) and sensors:
        return

    params["sensors"] = {
        "science": {
            "type": "ScienceImager",
            "wavelength": float(params.get("science_lambda", 2.150e-6)),
            "grid_size": int(params.get("grid_size", 512)),
        },
        "LGS_NE": {
            "type": "ShackHartmann",
            "gs_range_m": 90_000,
            "sub_apertures": 40,
            "dx": 2.828 * 60,
            "dy": 2.828 * 60,
            "wfs_lambda": 589e-9,
            "lgs_thickness_m": 10_000.0,
            "lgs_launch_offset_px": [0.0, 0.0],
            "lgs_remove_tt": True,
        },
        "LGS_NW": {
            "type": "ShackHartmann",
            "gs_range_m": 90_000,
            "sub_apertures": 40,
            "dx": -2.828 * 60,
            "dy": 2.828 * 60,
            "wfs_lambda": 589e-9,
            "lgs_thickness_m": 10_000.0,
            "lgs_launch_offset_px": [0.0, 0.0],
            "lgs_remove_tt": True,
        },
        "LGS_SE": {
            "type": "ShackHartmann",
            "gs_range_m": 90_000,
            "sub_apertures": 40,
            "dx": 2.828 * 60,
            "dy": -2.828 * 60,
            "wfs_lambda": 589e-9,
            "lgs_thickness_m": 10_000.0,
            "lgs_launch_offset_px": [0.0, 0.0],
            "lgs_remove_tt": True,
        },
        "LGS_SW": {
            "type": "ShackHartmann",
            "gs_range_m": 90_000,
            "sub_apertures": 40,
            "dx": -2.828 * 60,
            "dy": -2.828 * 60,
            "wfs_lambda": 589e-9,
            "lgs_thickness_m": 10_000.0,
            "lgs_launch_offset_px": [0.0, 0.0],
            "lgs_remove_tt": True,
        },
        "NGS": {
            "type": "ShackHartmann",
            "gs_range_m": "inf",
            "sub_apertures": 2,
            "dx": 0.0,
            "dy": 0.0,
            "wfs_lambda": 1650e-9,
        },
    }


def is_science_sensor(name: str, sensor: Any) -> bool:
    """Return True for the dedicated science imager channel."""
    if str(name).lower() == "science":
        return True
    try:
        if bool(getattr(sensor, "is_science_sensor", False)):
            return True
    except Exception:
        pass
    return sensor.__class__.__name__.lower() == "scienceimager"


def reorder_sensors_science_first(sensors: Mapping[str, Any]) -> "OrderedDict[str, Any]":
    """Return sensors with science channel first, preserving the rest of the order."""
    out: "OrderedDict[str, Any]" = OrderedDict()
    sci_keys = [k for k, s in sensors.items() if is_science_sensor(k, s)]
    if sci_keys:
        k0 = sci_keys[0]
        out[k0] = sensors[k0]
    for k, s in sensors.items():
        if k not in out:
            out[k] = s
    return out


def build_pupil(params: Mapping[str, Any], *, grid_size: Optional[int] = None) -> cp.ndarray:
    """Build the common circular pupil mask used by sensors and batch runner."""
    N = int(grid_size if grid_size is not None else params.get("grid_size", 512))
    obsc = float(params.get("telescope_center_obscuration", 0.0))
    return Pupil_tools.generate_pupil(
        grid_size=N,
        telescope_center_obscuration=obsc,
        dtype=cp.float32,
    )


def build_sensors(params: MutableMapping[str, Any]) -> "OrderedDict[str, Any]":
    """Build sensors directly from params["sensors"], without any GUI classes.

    Config convention: dx/dy are in arcsec in JSON. The sensor classes convert
    to radians internally.
    """
    ensure_default_sensors(params)

    grid_size = int(params.get("grid_size", 512))
    pupil = build_pupil(params, grid_size=grid_size)

    sensors_cfg = params.get("sensors")
    if not isinstance(sensors_cfg, dict) or not sensors_cfg:
        raise ValueError("params['sensors'] is missing or empty")

    sensors: "OrderedDict[str, Any]" = OrderedDict()

    for name, raw_cfg in sensors_cfg.items():
        if not isinstance(raw_cfg, Mapping):
            raise TypeError(f"Sensor config for {name!r} must be a dict, got {type(raw_cfg)!r}")
        cfg = dict(raw_cfg)
        stype = str(cfg.get("type", "ShackHartmann"))

        if stype == "ScienceImager":
            sensors[name] = WFSensor_tools.ScienceImager(
                wavelength=float(cfg.get("wavelength", params.get("science_lambda", 2.150e-6))),
                dx=float(cfg.get("dx", 0.0)),
                dy=float(cfg.get("dy", 0.0)),
                pupil=pupil,
                grid_size=int(cfg.get("grid_size", grid_size)),
                gs_range_m=cfg.get("gs_range_m", float("inf")),
            )
            continue

        if stype == "ShackHartmann":
            sensors[name] = WFSensor_tools.ShackHartmann(
                gs_range_m=cfg.get("gs_range_m", float("inf")),
                n_sub=cfg.get("sub_apertures", cfg.get("n_sub", params.get("sub_apertures", 40))),
                dx=float(cfg.get("dx", 0.0)),
                dy=float(cfg.get("dy", 0.0)),
                wavelength=float(cfg.get("wfs_lambda", cfg.get("wavelength", params.get("wfs_lambda", 500e-9)))),
                pupil=pupil,
                grid_size=int(cfg.get("grid_size", grid_size)),
                lgs_thickness_m=float(cfg.get("lgs_thickness_m", 0.0) or 0.0),
                lgs_launch_offset_px=tuple(cfg.get("lgs_launch_offset_px", (0.0, 0.0))),
                lgs_remove_tt=bool(cfg.get("lgs_remove_tt", True)),
                lgs_dir_bins=int(cfg.get("lgs_dir_bins", 64)),
                lgs_rad_bins=int(cfg.get("lgs_rad_bins", 16)),
                lgs_elong_scale=float(cfg.get("lgs_elong_scale", 2.0)),
                lgs_half_max=int(cfg.get("lgs_half_max", 24)),
            )
            continue

        raise ValueError(f"Unknown sensor type {stype!r} for sensor {name!r}")

    sensors = reorder_sensors_science_first(sensors)

    logger.info("Built %d sensors", len(sensors))
    for name, sensor in sensors.items():
        dx_arcsec = float(getattr(sensor, "dx", 0.0)) * RAD2ARCSEC
        dy_arcsec = float(getattr(sensor, "dy", 0.0)) * RAD2ARCSEC
        logger.info(
            "  %-12s %-16s field=(%8.2f,%8.2f) arcsec range=%s",
            name,
            sensor.__class__.__name__,
            dx_arcsec,
            dy_arcsec,
            getattr(sensor, "gs_range_m", None),
        )

    return sensors


def _resolve_profile_file(file_ref: str | Path, project_root: str | Path) -> Path:
    p = Path(file_ref).expanduser()
    if not p.is_absolute():
        p = Path(project_root).resolve() / p
    return p.resolve()


def _scan_turbulence_folder(project_root: str | Path) -> List[Path]:
    folder = Path(project_root).resolve() / "turbulence"
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".json")


def load_active_turbulence_profile(
    params: MutableMapping[str, Any],
    project_root: str | Path,
) -> Tuple[str, Dict[str, Any], str]:
    """Load the active turbulence profile using the GUI config schema.

    Supported forms:
        params["turbulence"] = {"source": "file", "file": "turbulence/MaunaKea_median.json"}
        params["turbulence"] = {"source": "inline", "name": "Custom", "profile": {...}}

    Legacy params["turbulence_profiles"] + params["turbulence_active"] is also accepted.
    """
    root = Path(project_root).resolve()

    # Legacy migration, copied in spirit from Turbulence_tab but without GUI state.
    if not isinstance(params.get("turbulence"), dict):
        profiles = params.get("turbulence_profiles")
        if isinstance(profiles, dict) and profiles:
            active = params.get("turbulence_active")
            if not isinstance(active, str) or active not in profiles:
                active = next(iter(profiles.keys()))
            prof = profiles.get(active)
            if not isinstance(prof, dict):
                prof = {}
            params["turbulence"] = {"source": "inline", "name": active, "profile": prof}
        else:
            params["turbulence"] = {}

    tcfg = params.get("turbulence")
    if not isinstance(tcfg, MutableMapping):
        tcfg = {}
        params["turbulence"] = tcfg

    source = str(tcfg.get("source") or "").strip().lower()

    if source == "inline" and isinstance(tcfg.get("profile"), dict):
        name = str(tcfg.get("name") or "Inline")
        return name, _copy_jsonlike(tcfg["profile"]), "inline"

    file_ref = tcfg.get("file") or tcfg.get("path")
    if isinstance(file_ref, str) and file_ref.strip():
        path = _resolve_profile_file(file_ref, root)
        if not path.exists():
            raise FileNotFoundError(f"Turbulence profile file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            prof = json.load(f)
        if not isinstance(prof, dict):
            raise TypeError(f"Turbulence profile must be a JSON object/dict, got {type(prof)!r} in {path}")
        name = str(tcfg.get("name") or path.stem)
        return name, prof, "file"

    # Last-resort fallback: first JSON profile in turbulence/.
    files = _scan_turbulence_folder(root)
    if not files:
        raise FileNotFoundError("No active turbulence profile configured and no turbulence/*.json files found")

    path = files[0]
    with path.open("r", encoding="utf-8") as f:
        prof = json.load(f)
    if not isinstance(prof, dict):
        raise TypeError(f"Turbulence profile must be a JSON object/dict, got {type(prof)!r} in {path}")
    return path.stem, prof, "file"


def split_turbulence_profile(profile: Mapping[str, Any]) -> Tuple[Callable[..., Any], Dict[str, Any], List[Dict[str, Any]]]:
    """Return (screen_factory, base_kwargs, layer_cfgs) from a turbulence profile."""
    fn_name = str(profile.get("function", ""))
    if fn_name not in TURBULENCE_FACTORIES:
        raise KeyError(
            f"Unknown turbulence function {fn_name!r}. "
            f"Known functions: {sorted(TURBULENCE_FACTORIES)}"
        )

    layers_raw = profile.get("layers", [])
    if not isinstance(layers_raw, list):
        raise TypeError(f"Turbulence profile 'layers' must be a list, got {type(layers_raw)!r}")

    base_kwargs = {k: _copy_jsonlike(v) for k, v in profile.items() if k not in ("layers", "function")}
    if "N" not in base_kwargs:
        raise KeyError("Turbulence profile is missing required key 'N'")
    if "dx" not in base_kwargs:
        raise KeyError("Turbulence profile is missing required key 'dx'")

    layers = [_normalise_layer_config(layer, i) for i, layer in enumerate(layers_raw)]
    return TURBULENCE_FACTORIES[fn_name], base_kwargs, layers


def _normalise_layer_config(layer: Mapping[str, Any], idx: int) -> Dict[str, Any]:
    """Accept both profile-layer and sim-layer key names."""
    if not isinstance(layer, Mapping):
        raise TypeError(f"Layer {idx} must be a dict, got {type(layer)!r}")

    name = str(layer.get("name") or f"Layer {idx + 1}")
    altitude = float(layer.get("altitude_m", layer.get("h", 0.0)))
    r0 = float(layer.get("r0"))
    if r0 <= 0.0:
        raise ValueError(f"Layer {name!r} has non-positive r0={r0}")

    if "wind" in layer and layer["wind"] is not None:
        wind_raw = layer["wind"]
        if len(wind_raw) != 2:
            raise ValueError(f"Layer {name!r} wind must have two entries [vx, vy]")
        wind = (float(wind_raw[0]), float(wind_raw[1]))
    else:
        wind = (float(layer.get("vx", 0.0)), float(layer.get("vy", 0.0)))

    return {
        "name": name,
        "altitude_m": altitude,
        "r0": r0,
        "L0": float(layer.get("L0", 30.0)),
        "wind": wind,
        "seed_offset": int(layer.get("seed_offset", idx + 1)),
        "active": bool(layer.get("active", True)),
    }


def _max_sensor_angle_rad(sensors: Iterable[Any]) -> float:
    vals: List[float] = []
    for sensor in sensors:
        try:
            vals.append(float(getattr(sensor, "sen_angle")))
        except Exception:
            try:
                dx, dy = getattr(sensor, "field_angle")
                vals.append(float(math.hypot(float(dx), float(dy))))
            except Exception:
                vals.append(0.0)
    return max(vals) if vals else 0.0


def build_simulation(
    params: MutableMapping[str, Any],
    sensors: Mapping[str, Any] | Sequence[Any],
    project_root: str | Path,
    *,
    active_layers_only: bool = True,
) -> Tuple[Any, str, str, Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """Build and populate the layered phase-screen simulator.

    Returns:
        sim, turbulence_name, turbulence_source, profile, base_kwargs, layer_cfgs
    """
    sensor_list = list(sensors.values()) if isinstance(sensors, Mapping) else list(sensors)
    name, profile, source = load_active_turbulence_profile(params, project_root)
    screen_factory, base_kwargs, layer_cfgs = split_turbulence_profile(profile)

    N_turb = int(base_kwargs["N"])
    N_cfg = int(params.get("grid_size", N_turb))
    if N_turb != N_cfg:
        logger.warning(
            "Turbulence N=%d differs from config grid_size=%d. Continuing with turbulence N.",
            N_turb,
            N_cfg,
        )

    sim = screen_factory(**base_kwargs)
    max_theta = _max_sensor_angle_rad(sensor_list)

    added = 0
    for layer in layer_cfgs:
        if active_layers_only and not bool(layer.get("active", True)):
            continue
        sim.add_layer(
            name=str(layer["name"]),
            altitude_m=float(layer["altitude_m"]),
            r0=float(layer["r0"]),
            L0=float(layer.get("L0", 30.0)),
            wind=tuple(layer.get("wind", (0.0, 0.0))),
            seed_offset=int(layer.get("seed_offset", added + 1)),
            max_theta=max_theta,
        )
        added += 1

    if added == 0:
        raise ValueError("No active turbulence layers were added")

    logger.info(
        "Built %s turbulence sim with %d layer(s), N=%s, dx=%s, r0_total=%s",
        name,
        added,
        base_kwargs.get("N"),
        base_kwargs.get("dx"),
        getattr(sim, "r0_total", None),
    )

    return sim, name, source, profile, base_kwargs, layer_cfgs


def build_gl_aoconjugate_reconstructor(
    *,
    params: Mapping[str, Any],
    sim: Any,
    sensors: Sequence[Any] | Mapping[str, Any],
    chunk_modes: int = 64,
    rcond: float = 8e-2,
    conjugation_height_m: float = 0.0,
    sensor_method: str = "southwell",
    slopes_aggregation: str = "stack",
    slope_weight_mode: str = "pupil",
    slope_weight_threshold: float = 0.20,
    slope_weight_power: float = 2.0,
    reg_alpha: float = 1e-5,
    reg_beta: float = 1e-5,
    **kwargs: Any,
) -> Any:
    """Build the default single-conjugate GLAO reconstructor.

    This mirrors SimWorker._build_tomographic_reconstructor but avoids QObject/QTimer.
    """
    sensor_list = list(sensors.values()) if isinstance(sensors, Mapping) else list(sensors)
    wfs_list = [s for s in sensor_list if not bool(getattr(s, "is_science_sensor", False))]
    if not wfs_list:
        raise ValueError("No WFS sensors available for reconstructor")

    dx_world_m = getattr(sim, "dx", None)
    if dx_world_m is None:
        raise AttributeError("Simulation object has no dx attribute needed for reconstructor")

    patch_M = int(getattr(sim, "N", params.get("grid_size", 512)))

    R = GLAOConjugateIM_CuPy(
        sensors=wfs_list,
        conjugation_height_m=float(conjugation_height_m),
        dx_world_m=float(dx_world_m),
        M=patch_M,
        slopes_aggregation=str(slopes_aggregation),
        slope_weight_mode=str(slope_weight_mode),
        slope_weight_threshold=float(slope_weight_threshold),
        slope_weight_power=float(slope_weight_power),
        reg_alpha=float(reg_alpha),
        reg_beta=float(reg_beta),
        params=dict(params),
        **kwargs,
    )

    logger.info("Building GLAO interaction matrix...")
    R.build_interaction_matrix(chunk_modes=int(chunk_modes), sensor_method=str(sensor_method))
    logger.info("Preparing GLAO runtime buffers...")
    R.prepare_runtime()
    logger.info("Factorizing GLAO reconstructor...")
    R.factorize(rcond=float(rcond))
    logger.info("GLAO reconstructor ready")
    return R


def build_batch_runner(
    *,
    params: Mapping[str, Any],
    sim: Any,
    sensors: Sequence[Any] | Mapping[str, Any],
    reconstructor: Any,
    pupil: cp.ndarray,
    cfg: Optional[Mapping[str, Any]] = None,
) -> AOBatchRunner:
    """Build AOBatchRunner with safe headless defaults."""
    cfg = dict(cfg or {})
    sensor_list = list(sensors.values()) if isinstance(sensors, Mapping) else list(sensors)
    dt_s = 1.0 / float(params.get("frame_rate", 500.0))

    psf_tt_mode = str(cfg.get("psf_tt_mode", "fit")).lower().strip()
    if psf_tt_mode == "hybrid":
        logger.warning("psf_tt_mode='hybrid' is not supported by current AOBatchRunner; using 'fit'")
        psf_tt_mode = "fit"

    return AOBatchRunner(
        sim=sim,
        sensors=sensor_list,
        reconstructor=reconstructor,
        pupil=pupil,
        dt_s=dt_s,
        gain=float(cfg.get("gain", params.get("loop_gain", 0.3))),
        leak=float(cfg.get("leak", params.get("loop_leak", 0.0))),
        wfs_method=str(cfg.get("wfs_method", "southwell_fast")),
        strip_tt_lgs=bool(cfg.get("strip_tt_lgs", True)),
        psf_tt_mode=psf_tt_mode,
        tt_wfs_indices=cfg.get("tt_wfs_indices", None),
        tt_enabled=bool(cfg.get("tt_enabled", params.get("tt_enabled", False))),
        tt_gain=float(cfg.get("tt_gain", params.get("tt_gain", 0.25))),
        tt_leak=float(cfg.get("tt_leak", params.get("tt_leak", 0.0))),
        tt_rate_hz=float(cfg.get("tt_rate_hz", params.get("tt_rate_hz", 0.0))),
        tt_delay_frames=int(cfg.get("tt_delay_frames", params.get("tt_delay_frames", 1))),
        dm_delay_frames=int(cfg.get("dm_delay_frames", params.get("dm_delay_frames", 1))),
        params=dict(params),
    )


def make_headless_system(
    config_path: str | Path,
    *,
    build_reconstructor: bool = False,
    active_layers_only: bool = True,
    reconstructor_kwargs: Optional[Mapping[str, Any]] = None,
) -> HeadlessAOSystem:
    """Construct sensors, turbulence simulator, pupil, and optionally reconstructor."""
    params, cfg_path, project_root = load_json_config(config_path)
    params = _copy_jsonlike(params)

    # Several legacy core classes still read the module-level
    # scripts.utilities.params object inside methods such as
    # ShackHartmann.measure().  Populate it here so the headless
    # path behaves like main.py / the GUI path.
    set_params(params)

    sensors = build_sensors(params)
    sim, turb_name, turb_source, profile, base_kwargs, layer_cfgs = build_simulation(
        params,
        sensors,
        project_root,
        active_layers_only=active_layers_only,
    )
    pupil = build_pupil(params, grid_size=int(getattr(sim, "N", params.get("grid_size", 512))))

    system = HeadlessAOSystem(
        params=params,
        config_path=cfg_path,
        project_root=project_root,
        sensors=sensors,
        sensor_keys=list(sensors.keys()),
        sensor_list=list(sensors.values()),
        turbulence_name=turb_name,
        turbulence_source=turb_source,
        turbulence_profile=profile,
        turbulence_base_kwargs=base_kwargs,
        layer_cfgs=layer_cfgs,
        sim=sim,
        pupil=pupil,
        reconstructor=None,
    )

    if build_reconstructor:
        system.build_reconstructor(**dict(reconstructor_kwargs or {}))

    return system


def _json_safe(x: Any) -> Any:
    """Convert common result objects to JSON-safe summaries."""
    if isinstance(x, (str, int, float, bool)) or x is None:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return str(x)
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.generic):
        return _json_safe(x.item())
    if isinstance(x, cp.ndarray):
        return {"array_type": "cupy", "shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, np.ndarray):
        return {"array_type": "numpy", "shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, Mapping):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return str(x)


def run_headless_longrun(
    config_path: str | Path,
    batch_cfg: Optional[Mapping[str, Any]] = None,
    *,
    build_reconstructor_kwargs: Optional[Mapping[str, Any]] = None,
    progress: bool = True,
) -> Dict[str, Any]:
    """Convenience wrapper: build a headless system, run AOBatchRunner, write summary.

    batch_cfg keys are passed through to AOBatchRunner.run where relevant. Useful
    keys include:
        duration_s, n_frames, discard_first_s, out_dir, timestamped,
        record_psfs, record_psfs_ttremoved, save_tt_series, psf_roi,
        chunk_frames, eval_offaxis_arcsec, write_eval_fits,
        write_eval_uncorrected_fits, eval_accumulate_uncorrected,
        save_telemetry, telemetry_stride, telemetry_channels,
        telemetry_max_samples, save_control_bundle,
        control_bundle_include_matrices, progress_every.
    """
    cfg = dict(batch_cfg or {})
    system = make_headless_system(
        config_path,
        build_reconstructor=True,
        reconstructor_kwargs=build_reconstructor_kwargs,
    )

    dt_s = system.dt_s
    n_frames = cfg.get("n_frames", None)
    if n_frames is None:
        n_frames = int(round(float(cfg.get("duration_s", 60.0)) / max(dt_s, 1e-12)))
    n_frames = int(n_frames)

    out_base = Path(str(cfg.get("out_dir", system.project_root / "output" / "batch_runs"))).expanduser()
    if not out_base.is_absolute():
        out_base = (system.project_root / out_base).resolve()
    timestamped = bool(cfg.get("timestamped", True))
    if timestamped:
        run_dir = out_base / f"longrun_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        run_dir = out_base
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = system.make_batch_runner(cfg)

    def progress_cb(done: int, total: int, fps: float, msg: str) -> None:
        if progress:
            print(f"[{done:>8d}/{total:<8d}] {fps:7.1f} FPS  {msg}", flush=True)

    result = runner.run(
        n_frames=n_frames,
        discard_first_s=float(cfg.get("discard_first_s", cfg.get("discard_s", 0.0))),
        record_psfs=bool(cfg.get("record_psfs", True)),
        record_psfs_ttremoved=bool(cfg.get("record_psfs_ttremoved", False)),
        save_tt_series=bool(cfg.get("save_tt_series", True)),
        out_dir=str(run_dir),
        psf_roi=int(cfg.get("psf_roi", 128)),
        chunk_frames=int(cfg.get("chunk_frames", 256)),
        eval_offaxis_arcsec=cfg.get("eval_offaxis_arcsec", None),
        write_eval_fits=bool(cfg.get("write_eval_fits", True)),
        write_eval_uncorrected_fits=bool(cfg.get("write_eval_uncorrected_fits", True)),
        eval_accumulate_uncorrected=bool(cfg.get("eval_accumulate_uncorrected", True)),
        save_telemetry=bool(cfg.get("save_telemetry", False)),
        telemetry_stride=int(cfg.get("telemetry_stride", 1)),
        telemetry_channels=cfg.get("telemetry_channels", None),
        telemetry_max_samples=cfg.get("telemetry_max_samples", None),
        save_control_bundle=bool(cfg.get("save_control_bundle", False)),
        control_bundle_include_matrices=bool(cfg.get("control_bundle_include_matrices", True)),
        control_bundle_include_pupil=bool(cfg.get("control_bundle_include_pupil", True)),
        control_bundle_include_basis=bool(cfg.get("control_bundle_include_basis", False)),
        progress_cb=progress_cb if progress else None,
        should_stop=None,
        progress_every=int(cfg.get("progress_every", max(256, int(cfg.get("chunk_frames", 256))))),
    )

    summary = _json_safe(result)
    summary["out_dir"] = str(run_dir)
    summary["config_path"] = str(system.config_path) if system.config_path is not None else None
    summary["sensor_keys"] = list(system.sensor_keys)
    summary["turbulence_name"] = system.turbulence_name
    summary["turbulence_source"] = system.turbulence_source
    summary["dt_s"] = float(dt_s)
    summary["n_frames_requested"] = int(n_frames)

    with (run_dir / "headless_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Keep the raw result available to Python callers, but add the run path.
    result = dict(result)
    result["out_dir"] = str(run_dir)
    result["headless_summary_json"] = str(run_dir / "headless_summary.json")
    return result


__all__ = [
    "HeadlessAOSystem",
    "find_project_root",
    "load_json_config",
    "ensure_default_sensors",
    "build_pupil",
    "build_sensors",
    "load_active_turbulence_profile",
    "split_turbulence_profile",
    "build_simulation",
    "build_gl_aoconjugate_reconstructor",
    "build_batch_runner",
    "make_headless_system",
    "run_headless_longrun",
]
# """Headless construction utilities for AOsim experiments.

# This module is intended to let experiment scripts build the same core AO
# objects that the GUI uses, without importing QWidget/PySide GUI tabs.

# Typical use from the repository root:

#     from scripts.core.headless_factory import make_headless_system, run_headless_longrun

#     system = make_headless_system("config_ultimate.json", build_reconstructor=False)
#     print(system.sensor_keys)

#     result = run_headless_longrun(
#         "config_ultimate.json",
#         {
#             "duration_s": 1.0,
#             "discard_first_s": 0.1,
#             "out_dir": "experiments/ultimate_wfi_psf_reco/outputs",
#             "psf_roi": 128,
#             "record_psfs": False,
#             "eval_offaxis_arcsec": [(0, 0), (300, 300), (-300, -300)],
#         },
#     )

# Notes:
# - This still depends on the core simulator dependencies, especially CuPy.
# - The default long-run PSF TT mode is set to "fit" because the current
#   AOBatchRunner rejects the older GUI default "hybrid".
# """

# from __future__ import annotations

# import copy
# import datetime as _dt
# import json
# import logging
# import math
# from collections import OrderedDict
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

# import numpy as np
# import cupy as cp

# from scripts.batch import AOBatchRunner
# from scripts.phase_screen.infinite_vonkarman import LayeredInfinitePhaseScreen
# from scripts.reconstructor import GLAOConjugateIM_CuPy
# from scripts.utilities import Pupil_tools, WFSensor_tools

# logger = logging.getLogger(__name__)

# ARCSEC2RAD = math.pi / (180.0 * 3600.0)
# RAD2ARCSEC = (180.0 * 3600.0) / math.pi

# TURBULENCE_FACTORIES: Dict[str, Callable[..., Any]] = {
#     "InfVonKarman": LayeredInfinitePhaseScreen,
# }


# @dataclass
# class HeadlessAOSystem:
#     """Container for the headless AO objects needed by experiments."""

#     params: Dict[str, Any]
#     config_path: Optional[Path]
#     project_root: Path

#     sensors: "OrderedDict[str, Any]"
#     sensor_keys: List[str]
#     sensor_list: List[Any]

#     turbulence_name: str
#     turbulence_source: str
#     turbulence_profile: Dict[str, Any]
#     turbulence_base_kwargs: Dict[str, Any]
#     layer_cfgs: List[Dict[str, Any]]

#     sim: Any
#     pupil: cp.ndarray
#     reconstructor: Optional[Any] = None

#     @property
#     def dt_s(self) -> float:
#         return 1.0 / float(self.params.get("frame_rate", 500.0))

#     def build_reconstructor(
#         self,
#         *,
#         chunk_modes: int = 64,
#         rcond: float = 8e-2,
#         conjugation_height_m: float = 0.0,
#         sensor_method: str = "southwell",
#         **kwargs: Any,
#     ) -> Any:
#         """Build and attach the default GLAO reconstructor."""
#         self.reconstructor = build_gl_aoconjugate_reconstructor(
#             params=self.params,
#             sim=self.sim,
#             sensors=self.sensor_list,
#             chunk_modes=chunk_modes,
#             rcond=rcond,
#             conjugation_height_m=conjugation_height_m,
#             sensor_method=sensor_method,
#             **kwargs,
#         )
#         return self.reconstructor

#     def make_batch_runner(self, cfg: Optional[Mapping[str, Any]] = None) -> AOBatchRunner:
#         """Create an AOBatchRunner using this system."""
#         if self.reconstructor is None:
#             self.build_reconstructor()
#         return build_batch_runner(
#             params=self.params,
#             sim=self.sim,
#             sensors=self.sensor_list,
#             reconstructor=self.reconstructor,
#             pupil=self.pupil,
#             cfg=dict(cfg or {}),
#         )


# def find_project_root(start: Optional[str | Path] = None) -> Path:
#     """Find the AOsim repository root from a file/dir inside the repository."""
#     p = Path(start or Path.cwd()).expanduser().resolve()
#     if p.is_file():
#         p = p.parent

#     for cand in (p, *p.parents):
#         if (cand / "scripts").is_dir() and (cand / "turbulence").is_dir():
#             return cand

#     raise FileNotFoundError(
#         f"Could not find AOsim project root from {p}. "
#         "Expected a directory containing scripts/ and turbulence/."
#     )


# def load_json_config(config_path: str | Path) -> Tuple[Dict[str, Any], Path, Path]:
#     """Load a JSON config and return (params, config_path, project_root)."""
#     path = Path(config_path).expanduser()
#     if not path.is_absolute():
#         path = (Path.cwd() / path).resolve()
#     else:
#         path = path.resolve()

#     if not path.exists():
#         raise FileNotFoundError(f"Config file not found: {path}")

#     with path.open("r", encoding="utf-8") as f:
#         params = json.load(f)
#     if not isinstance(params, dict):
#         raise TypeError(f"Config JSON must contain an object/dict, got {type(params)!r}")

#     root = find_project_root(path)
#     return params, path, root


# def _copy_jsonlike(x: Any) -> Any:
#     """Copy a config object without preserving GUI/shared mutable state."""
#     return copy.deepcopy(x)


# def ensure_default_sensors(params: MutableMapping[str, Any]) -> None:
#     """Back-compatible default sensor constellation.

#     This mirrors the old GUI fallback in Poke_tab for configs that do not yet
#     contain a sensors section. Modern ULTIMATE configs should already define
#     params["sensors"].
#     """
#     sensors = params.get("sensors")
#     if isinstance(sensors, dict) and sensors:
#         return

#     params["sensors"] = {
#         "science": {
#             "type": "ScienceImager",
#             "wavelength": float(params.get("science_lambda", 2.150e-6)),
#             "grid_size": int(params.get("grid_size", 512)),
#         },
#         "LGS_NE": {
#             "type": "ShackHartmann",
#             "gs_range_m": 90_000,
#             "sub_apertures": 40,
#             "dx": 2.828 * 60,
#             "dy": 2.828 * 60,
#             "wfs_lambda": 589e-9,
#             "lgs_thickness_m": 10_000.0,
#             "lgs_launch_offset_px": [0.0, 0.0],
#             "lgs_remove_tt": True,
#         },
#         "LGS_NW": {
#             "type": "ShackHartmann",
#             "gs_range_m": 90_000,
#             "sub_apertures": 40,
#             "dx": -2.828 * 60,
#             "dy": 2.828 * 60,
#             "wfs_lambda": 589e-9,
#             "lgs_thickness_m": 10_000.0,
#             "lgs_launch_offset_px": [0.0, 0.0],
#             "lgs_remove_tt": True,
#         },
#         "LGS_SE": {
#             "type": "ShackHartmann",
#             "gs_range_m": 90_000,
#             "sub_apertures": 40,
#             "dx": 2.828 * 60,
#             "dy": -2.828 * 60,
#             "wfs_lambda": 589e-9,
#             "lgs_thickness_m": 10_000.0,
#             "lgs_launch_offset_px": [0.0, 0.0],
#             "lgs_remove_tt": True,
#         },
#         "LGS_SW": {
#             "type": "ShackHartmann",
#             "gs_range_m": 90_000,
#             "sub_apertures": 40,
#             "dx": -2.828 * 60,
#             "dy": -2.828 * 60,
#             "wfs_lambda": 589e-9,
#             "lgs_thickness_m": 10_000.0,
#             "lgs_launch_offset_px": [0.0, 0.0],
#             "lgs_remove_tt": True,
#         },
#         "NGS": {
#             "type": "ShackHartmann",
#             "gs_range_m": "inf",
#             "sub_apertures": 2,
#             "dx": 0.0,
#             "dy": 0.0,
#             "wfs_lambda": 1650e-9,
#         },
#     }


# def is_science_sensor(name: str, sensor: Any) -> bool:
#     """Return True for the dedicated science imager channel."""
#     if str(name).lower() == "science":
#         return True
#     try:
#         if bool(getattr(sensor, "is_science_sensor", False)):
#             return True
#     except Exception:
#         pass
#     return sensor.__class__.__name__.lower() == "scienceimager"


# def reorder_sensors_science_first(sensors: Mapping[str, Any]) -> "OrderedDict[str, Any]":
#     """Return sensors with science channel first, preserving the rest of the order."""
#     out: "OrderedDict[str, Any]" = OrderedDict()
#     sci_keys = [k for k, s in sensors.items() if is_science_sensor(k, s)]
#     if sci_keys:
#         k0 = sci_keys[0]
#         out[k0] = sensors[k0]
#     for k, s in sensors.items():
#         if k not in out:
#             out[k] = s
#     return out


# def build_pupil(params: Mapping[str, Any], *, grid_size: Optional[int] = None) -> cp.ndarray:
#     """Build the common circular pupil mask used by sensors and batch runner."""
#     N = int(grid_size if grid_size is not None else params.get("grid_size", 512))
#     obsc = float(params.get("telescope_center_obscuration", 0.0))
#     return Pupil_tools.generate_pupil(
#         grid_size=N,
#         telescope_center_obscuration=obsc,
#         dtype=cp.float32,
#     )


# def build_sensors(params: MutableMapping[str, Any]) -> "OrderedDict[str, Any]":
#     """Build sensors directly from params["sensors"], without any GUI classes.

#     Config convention: dx/dy are in arcsec in JSON. The sensor classes convert
#     to radians internally.
#     """
#     ensure_default_sensors(params)

#     grid_size = int(params.get("grid_size", 512))
#     pupil = build_pupil(params, grid_size=grid_size)

#     sensors_cfg = params.get("sensors")
#     if not isinstance(sensors_cfg, dict) or not sensors_cfg:
#         raise ValueError("params['sensors'] is missing or empty")

#     sensors: "OrderedDict[str, Any]" = OrderedDict()

#     for name, raw_cfg in sensors_cfg.items():
#         if not isinstance(raw_cfg, Mapping):
#             raise TypeError(f"Sensor config for {name!r} must be a dict, got {type(raw_cfg)!r}")
#         cfg = dict(raw_cfg)
#         stype = str(cfg.get("type", "ShackHartmann"))

#         if stype == "ScienceImager":
#             sensors[name] = WFSensor_tools.ScienceImager(
#                 wavelength=float(cfg.get("wavelength", params.get("science_lambda", 2.150e-6))),
#                 dx=float(cfg.get("dx", 0.0)),
#                 dy=float(cfg.get("dy", 0.0)),
#                 pupil=pupil,
#                 grid_size=int(cfg.get("grid_size", grid_size)),
#                 gs_range_m=cfg.get("gs_range_m", float("inf")),
#             )
#             continue

#         if stype == "ShackHartmann":
#             sensors[name] = WFSensor_tools.ShackHartmann(
#                 gs_range_m=cfg.get("gs_range_m", float("inf")),
#                 n_sub=cfg.get("sub_apertures", cfg.get("n_sub", params.get("sub_apertures", 40))),
#                 dx=float(cfg.get("dx", 0.0)),
#                 dy=float(cfg.get("dy", 0.0)),
#                 wavelength=float(cfg.get("wfs_lambda", cfg.get("wavelength", params.get("wfs_lambda", 500e-9)))),
#                 pupil=pupil,
#                 grid_size=int(cfg.get("grid_size", grid_size)),
#                 lgs_thickness_m=float(cfg.get("lgs_thickness_m", 0.0) or 0.0),
#                 lgs_launch_offset_px=tuple(cfg.get("lgs_launch_offset_px", (0.0, 0.0))),
#                 lgs_remove_tt=bool(cfg.get("lgs_remove_tt", True)),
#                 lgs_dir_bins=int(cfg.get("lgs_dir_bins", 64)),
#                 lgs_rad_bins=int(cfg.get("lgs_rad_bins", 16)),
#                 lgs_elong_scale=float(cfg.get("lgs_elong_scale", 2.0)),
#                 lgs_half_max=int(cfg.get("lgs_half_max", 24)),
#             )
#             continue

#         raise ValueError(f"Unknown sensor type {stype!r} for sensor {name!r}")

#     sensors = reorder_sensors_science_first(sensors)

#     logger.info("Built %d sensors", len(sensors))
#     for name, sensor in sensors.items():
#         dx_arcsec = float(getattr(sensor, "dx", 0.0)) * RAD2ARCSEC
#         dy_arcsec = float(getattr(sensor, "dy", 0.0)) * RAD2ARCSEC
#         logger.info(
#             "  %-12s %-16s field=(%8.2f,%8.2f) arcsec range=%s",
#             name,
#             sensor.__class__.__name__,
#             dx_arcsec,
#             dy_arcsec,
#             getattr(sensor, "gs_range_m", None),
#         )

#     return sensors


# def _resolve_profile_file(file_ref: str | Path, project_root: str | Path) -> Path:
#     p = Path(file_ref).expanduser()
#     if not p.is_absolute():
#         p = Path(project_root).resolve() / p
#     return p.resolve()


# def _scan_turbulence_folder(project_root: str | Path) -> List[Path]:
#     folder = Path(project_root).resolve() / "turbulence"
#     if not folder.exists():
#         return []
#     return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".json")


# def load_active_turbulence_profile(
#     params: MutableMapping[str, Any],
#     project_root: str | Path,
# ) -> Tuple[str, Dict[str, Any], str]:
#     """Load the active turbulence profile using the GUI config schema.

#     Supported forms:
#         params["turbulence"] = {"source": "file", "file": "turbulence/MaunaKea_median.json"}
#         params["turbulence"] = {"source": "inline", "name": "Custom", "profile": {...}}

#     Legacy params["turbulence_profiles"] + params["turbulence_active"] is also accepted.
#     """
#     root = Path(project_root).resolve()

#     # Legacy migration, copied in spirit from Turbulence_tab but without GUI state.
#     if not isinstance(params.get("turbulence"), dict):
#         profiles = params.get("turbulence_profiles")
#         if isinstance(profiles, dict) and profiles:
#             active = params.get("turbulence_active")
#             if not isinstance(active, str) or active not in profiles:
#                 active = next(iter(profiles.keys()))
#             prof = profiles.get(active)
#             if not isinstance(prof, dict):
#                 prof = {}
#             params["turbulence"] = {"source": "inline", "name": active, "profile": prof}
#         else:
#             params["turbulence"] = {}

#     tcfg = params.get("turbulence")
#     if not isinstance(tcfg, MutableMapping):
#         tcfg = {}
#         params["turbulence"] = tcfg

#     source = str(tcfg.get("source") or "").strip().lower()

#     if source == "inline" and isinstance(tcfg.get("profile"), dict):
#         name = str(tcfg.get("name") or "Inline")
#         return name, _copy_jsonlike(tcfg["profile"]), "inline"

#     file_ref = tcfg.get("file") or tcfg.get("path")
#     if isinstance(file_ref, str) and file_ref.strip():
#         path = _resolve_profile_file(file_ref, root)
#         if not path.exists():
#             raise FileNotFoundError(f"Turbulence profile file not found: {path}")
#         with path.open("r", encoding="utf-8") as f:
#             prof = json.load(f)
#         if not isinstance(prof, dict):
#             raise TypeError(f"Turbulence profile must be a JSON object/dict, got {type(prof)!r} in {path}")
#         name = str(tcfg.get("name") or path.stem)
#         return name, prof, "file"

#     # Last-resort fallback: first JSON profile in turbulence/.
#     files = _scan_turbulence_folder(root)
#     if not files:
#         raise FileNotFoundError("No active turbulence profile configured and no turbulence/*.json files found")

#     path = files[0]
#     with path.open("r", encoding="utf-8") as f:
#         prof = json.load(f)
#     if not isinstance(prof, dict):
#         raise TypeError(f"Turbulence profile must be a JSON object/dict, got {type(prof)!r} in {path}")
#     return path.stem, prof, "file"


# def split_turbulence_profile(profile: Mapping[str, Any]) -> Tuple[Callable[..., Any], Dict[str, Any], List[Dict[str, Any]]]:
#     """Return (screen_factory, base_kwargs, layer_cfgs) from a turbulence profile."""
#     fn_name = str(profile.get("function", ""))
#     if fn_name not in TURBULENCE_FACTORIES:
#         raise KeyError(
#             f"Unknown turbulence function {fn_name!r}. "
#             f"Known functions: {sorted(TURBULENCE_FACTORIES)}"
#         )

#     layers_raw = profile.get("layers", [])
#     if not isinstance(layers_raw, list):
#         raise TypeError(f"Turbulence profile 'layers' must be a list, got {type(layers_raw)!r}")

#     base_kwargs = {k: _copy_jsonlike(v) for k, v in profile.items() if k not in ("layers", "function")}
#     if "N" not in base_kwargs:
#         raise KeyError("Turbulence profile is missing required key 'N'")
#     if "dx" not in base_kwargs:
#         raise KeyError("Turbulence profile is missing required key 'dx'")

#     layers = [_normalise_layer_config(layer, i) for i, layer in enumerate(layers_raw)]
#     return TURBULENCE_FACTORIES[fn_name], base_kwargs, layers


# def _normalise_layer_config(layer: Mapping[str, Any], idx: int) -> Dict[str, Any]:
#     """Accept both profile-layer and sim-layer key names."""
#     if not isinstance(layer, Mapping):
#         raise TypeError(f"Layer {idx} must be a dict, got {type(layer)!r}")

#     name = str(layer.get("name") or f"Layer {idx + 1}")
#     altitude = float(layer.get("altitude_m", layer.get("h", 0.0)))
#     r0 = float(layer.get("r0"))
#     if r0 <= 0.0:
#         raise ValueError(f"Layer {name!r} has non-positive r0={r0}")

#     if "wind" in layer and layer["wind"] is not None:
#         wind_raw = layer["wind"]
#         if len(wind_raw) != 2:
#             raise ValueError(f"Layer {name!r} wind must have two entries [vx, vy]")
#         wind = (float(wind_raw[0]), float(wind_raw[1]))
#     else:
#         wind = (float(layer.get("vx", 0.0)), float(layer.get("vy", 0.0)))

#     return {
#         "name": name,
#         "altitude_m": altitude,
#         "r0": r0,
#         "L0": float(layer.get("L0", 30.0)),
#         "wind": wind,
#         "seed_offset": int(layer.get("seed_offset", idx + 1)),
#         "active": bool(layer.get("active", True)),
#     }


# def _max_sensor_angle_rad(sensors: Iterable[Any]) -> float:
#     vals: List[float] = []
#     for sensor in sensors:
#         try:
#             vals.append(float(getattr(sensor, "sen_angle")))
#         except Exception:
#             try:
#                 dx, dy = getattr(sensor, "field_angle")
#                 vals.append(float(math.hypot(float(dx), float(dy))))
#             except Exception:
#                 vals.append(0.0)
#     return max(vals) if vals else 0.0


# def build_simulation(
#     params: MutableMapping[str, Any],
#     sensors: Mapping[str, Any] | Sequence[Any],
#     project_root: str | Path,
#     *,
#     active_layers_only: bool = True,
# ) -> Tuple[Any, str, str, Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
#     """Build and populate the layered phase-screen simulator.

#     Returns:
#         sim, turbulence_name, turbulence_source, profile, base_kwargs, layer_cfgs
#     """
#     sensor_list = list(sensors.values()) if isinstance(sensors, Mapping) else list(sensors)
#     name, profile, source = load_active_turbulence_profile(params, project_root)
#     screen_factory, base_kwargs, layer_cfgs = split_turbulence_profile(profile)

#     N_turb = int(base_kwargs["N"])
#     N_cfg = int(params.get("grid_size", N_turb))
#     if N_turb != N_cfg:
#         logger.warning(
#             "Turbulence N=%d differs from config grid_size=%d. Continuing with turbulence N.",
#             N_turb,
#             N_cfg,
#         )

#     sim = screen_factory(**base_kwargs)
#     max_theta = _max_sensor_angle_rad(sensor_list)

#     added = 0
#     for layer in layer_cfgs:
#         if active_layers_only and not bool(layer.get("active", True)):
#             continue
#         sim.add_layer(
#             name=str(layer["name"]),
#             altitude_m=float(layer["altitude_m"]),
#             r0=float(layer["r0"]),
#             L0=float(layer.get("L0", 30.0)),
#             wind=tuple(layer.get("wind", (0.0, 0.0))),
#             seed_offset=int(layer.get("seed_offset", added + 1)),
#             max_theta=max_theta,
#         )
#         added += 1

#     if added == 0:
#         raise ValueError("No active turbulence layers were added")

#     logger.info(
#         "Built %s turbulence sim with %d layer(s), N=%s, dx=%s, r0_total=%s",
#         name,
#         added,
#         base_kwargs.get("N"),
#         base_kwargs.get("dx"),
#         getattr(sim, "r0_total", None),
#     )

#     return sim, name, source, profile, base_kwargs, layer_cfgs


# def build_gl_aoconjugate_reconstructor(
#     *,
#     params: Mapping[str, Any],
#     sim: Any,
#     sensors: Sequence[Any] | Mapping[str, Any],
#     chunk_modes: int = 64,
#     rcond: float = 8e-2,
#     conjugation_height_m: float = 0.0,
#     sensor_method: str = "southwell",
#     slopes_aggregation: str = "stack",
#     slope_weight_mode: str = "pupil",
#     slope_weight_threshold: float = 0.20,
#     slope_weight_power: float = 2.0,
#     reg_alpha: float = 1e-5,
#     reg_beta: float = 1e-5,
#     **kwargs: Any,
# ) -> Any:
#     """Build the default single-conjugate GLAO reconstructor.

#     This mirrors SimWorker._build_tomographic_reconstructor but avoids QObject/QTimer.
#     """
#     sensor_list = list(sensors.values()) if isinstance(sensors, Mapping) else list(sensors)
#     wfs_list = [s for s in sensor_list if not bool(getattr(s, "is_science_sensor", False))]
#     if not wfs_list:
#         raise ValueError("No WFS sensors available for reconstructor")

#     dx_world_m = getattr(sim, "dx", None)
#     if dx_world_m is None:
#         raise AttributeError("Simulation object has no dx attribute needed for reconstructor")

#     patch_M = int(getattr(sim, "N", params.get("grid_size", 512)))

#     R = GLAOConjugateIM_CuPy(
#         sensors=wfs_list,
#         conjugation_height_m=float(conjugation_height_m),
#         dx_world_m=float(dx_world_m),
#         M=patch_M,
#         slopes_aggregation=str(slopes_aggregation),
#         slope_weight_mode=str(slope_weight_mode),
#         slope_weight_threshold=float(slope_weight_threshold),
#         slope_weight_power=float(slope_weight_power),
#         reg_alpha=float(reg_alpha),
#         reg_beta=float(reg_beta),
#         params=dict(params),
#         **kwargs,
#     )

#     logger.info("Building GLAO interaction matrix...")
#     R.build_interaction_matrix(chunk_modes=int(chunk_modes), sensor_method=str(sensor_method))
#     logger.info("Preparing GLAO runtime buffers...")
#     R.prepare_runtime()
#     logger.info("Factorizing GLAO reconstructor...")
#     R.factorize(rcond=float(rcond))
#     logger.info("GLAO reconstructor ready")
#     return R


# def build_batch_runner(
#     *,
#     params: Mapping[str, Any],
#     sim: Any,
#     sensors: Sequence[Any] | Mapping[str, Any],
#     reconstructor: Any,
#     pupil: cp.ndarray,
#     cfg: Optional[Mapping[str, Any]] = None,
# ) -> AOBatchRunner:
#     """Build AOBatchRunner with safe headless defaults."""
#     cfg = dict(cfg or {})
#     sensor_list = list(sensors.values()) if isinstance(sensors, Mapping) else list(sensors)
#     dt_s = 1.0 / float(params.get("frame_rate", 500.0))

#     psf_tt_mode = str(cfg.get("psf_tt_mode", "fit")).lower().strip()
#     if psf_tt_mode == "hybrid":
#         logger.warning("psf_tt_mode='hybrid' is not supported by current AOBatchRunner; using 'fit'")
#         psf_tt_mode = "fit"

#     return AOBatchRunner(
#         sim=sim,
#         sensors=sensor_list,
#         reconstructor=reconstructor,
#         pupil=pupil,
#         dt_s=dt_s,
#         gain=float(cfg.get("gain", params.get("loop_gain", 0.3))),
#         leak=float(cfg.get("leak", params.get("loop_leak", 0.0))),
#         wfs_method=str(cfg.get("wfs_method", "southwell_fast")),
#         strip_tt_lgs=bool(cfg.get("strip_tt_lgs", True)),
#         psf_tt_mode=psf_tt_mode,
#         tt_wfs_indices=cfg.get("tt_wfs_indices", None),
#         tt_enabled=bool(cfg.get("tt_enabled", params.get("tt_enabled", False))),
#         tt_gain=float(cfg.get("tt_gain", params.get("tt_gain", 0.25))),
#         tt_leak=float(cfg.get("tt_leak", params.get("tt_leak", 0.0))),
#         tt_rate_hz=float(cfg.get("tt_rate_hz", params.get("tt_rate_hz", 0.0))),
#         tt_delay_frames=int(cfg.get("tt_delay_frames", params.get("tt_delay_frames", 1))),
#         dm_delay_frames=int(cfg.get("dm_delay_frames", params.get("dm_delay_frames", 1))),
#         params=dict(params),
#     )


# def make_headless_system(
#     config_path: str | Path,
#     *,
#     build_reconstructor: bool = False,
#     active_layers_only: bool = True,
#     reconstructor_kwargs: Optional[Mapping[str, Any]] = None,
# ) -> HeadlessAOSystem:
#     """Construct sensors, turbulence simulator, pupil, and optionally reconstructor."""
#     params, cfg_path, project_root = load_json_config(config_path)
#     params = _copy_jsonlike(params)

#     sensors = build_sensors(params)
#     sim, turb_name, turb_source, profile, base_kwargs, layer_cfgs = build_simulation(
#         params,
#         sensors,
#         project_root,
#         active_layers_only=active_layers_only,
#     )
#     pupil = build_pupil(params, grid_size=int(getattr(sim, "N", params.get("grid_size", 512))))

#     system = HeadlessAOSystem(
#         params=params,
#         config_path=cfg_path,
#         project_root=project_root,
#         sensors=sensors,
#         sensor_keys=list(sensors.keys()),
#         sensor_list=list(sensors.values()),
#         turbulence_name=turb_name,
#         turbulence_source=turb_source,
#         turbulence_profile=profile,
#         turbulence_base_kwargs=base_kwargs,
#         layer_cfgs=layer_cfgs,
#         sim=sim,
#         pupil=pupil,
#         reconstructor=None,
#     )

#     if build_reconstructor:
#         system.build_reconstructor(**dict(reconstructor_kwargs or {}))

#     return system


# def _json_safe(x: Any) -> Any:
#     """Convert common result objects to JSON-safe summaries."""
#     if isinstance(x, (str, int, float, bool)) or x is None:
#         if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
#             return str(x)
#         return x
#     if isinstance(x, Path):
#         return str(x)
#     if isinstance(x, np.generic):
#         return _json_safe(x.item())
#     if isinstance(x, cp.ndarray):
#         return {"array_type": "cupy", "shape": list(x.shape), "dtype": str(x.dtype)}
#     if isinstance(x, np.ndarray):
#         return {"array_type": "numpy", "shape": list(x.shape), "dtype": str(x.dtype)}
#     if isinstance(x, Mapping):
#         return {str(k): _json_safe(v) for k, v in x.items()}
#     if isinstance(x, (list, tuple)):
#         return [_json_safe(v) for v in x]
#     return str(x)


# def run_headless_longrun(
#     config_path: str | Path,
#     batch_cfg: Optional[Mapping[str, Any]] = None,
#     *,
#     build_reconstructor_kwargs: Optional[Mapping[str, Any]] = None,
#     progress: bool = True,
# ) -> Dict[str, Any]:
#     """Convenience wrapper: build a headless system, run AOBatchRunner, write summary.

#     batch_cfg keys are passed through to AOBatchRunner.run where relevant. Useful
#     keys include:
#         duration_s, n_frames, discard_first_s, out_dir, timestamped,
#         record_psfs, record_psfs_ttremoved, save_tt_series, psf_roi,
#         chunk_frames, eval_offaxis_arcsec, write_eval_fits,
#         write_eval_uncorrected_fits, eval_accumulate_uncorrected,
#         progress_every.
#     """
#     cfg = dict(batch_cfg or {})
#     system = make_headless_system(
#         config_path,
#         build_reconstructor=True,
#         reconstructor_kwargs=build_reconstructor_kwargs,
#     )

#     dt_s = system.dt_s
#     n_frames = cfg.get("n_frames", None)
#     if n_frames is None:
#         n_frames = int(round(float(cfg.get("duration_s", 60.0)) / max(dt_s, 1e-12)))
#     n_frames = int(n_frames)

#     out_base = Path(str(cfg.get("out_dir", system.project_root / "output" / "batch_runs"))).expanduser()
#     if not out_base.is_absolute():
#         out_base = (system.project_root / out_base).resolve()
#     timestamped = bool(cfg.get("timestamped", True))
#     if timestamped:
#         run_dir = out_base / f"longrun_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
#     else:
#         run_dir = out_base
#     run_dir.mkdir(parents=True, exist_ok=True)

#     runner = system.make_batch_runner(cfg)

#     def progress_cb(done: int, total: int, fps: float, msg: str) -> None:
#         if progress:
#             print(f"[{done:>8d}/{total:<8d}] {fps:7.1f} FPS  {msg}", flush=True)

#     result = runner.run(
#         n_frames=n_frames,
#         discard_first_s=float(cfg.get("discard_first_s", cfg.get("discard_s", 0.0))),
#         record_psfs=bool(cfg.get("record_psfs", True)),
#         record_psfs_ttremoved=bool(cfg.get("record_psfs_ttremoved", False)),
#         save_tt_series=bool(cfg.get("save_tt_series", True)),
#         out_dir=str(run_dir),
#         psf_roi=int(cfg.get("psf_roi", 128)),
#         chunk_frames=int(cfg.get("chunk_frames", 256)),
#         eval_offaxis_arcsec=cfg.get("eval_offaxis_arcsec", None),
#         write_eval_fits=bool(cfg.get("write_eval_fits", True)),
#         write_eval_uncorrected_fits=bool(cfg.get("write_eval_uncorrected_fits", True)),
#         eval_accumulate_uncorrected=bool(cfg.get("eval_accumulate_uncorrected", True)),
#         progress_cb=progress_cb if progress else None,
#         should_stop=None,
#         progress_every=int(cfg.get("progress_every", max(256, int(cfg.get("chunk_frames", 256))))),
#     )

#     summary = _json_safe(result)
#     summary["out_dir"] = str(run_dir)
#     summary["config_path"] = str(system.config_path) if system.config_path is not None else None
#     summary["sensor_keys"] = list(system.sensor_keys)
#     summary["turbulence_name"] = system.turbulence_name
#     summary["turbulence_source"] = system.turbulence_source
#     summary["dt_s"] = float(dt_s)
#     summary["n_frames_requested"] = int(n_frames)

#     with (run_dir / "headless_summary.json").open("w", encoding="utf-8") as f:
#         json.dump(summary, f, indent=2)

#     # Keep the raw result available to Python callers, but add the run path.
#     result = dict(result)
#     result["out_dir"] = str(run_dir)
#     result["headless_summary_json"] = str(run_dir / "headless_summary.json")
#     return result


# __all__ = [
#     "HeadlessAOSystem",
#     "find_project_root",
#     "load_json_config",
#     "ensure_default_sensors",
#     "build_pupil",
#     "build_sensors",
#     "load_active_turbulence_profile",
#     "split_turbulence_profile",
#     "build_simulation",
#     "build_gl_aoconjugate_reconstructor",
#     "build_batch_runner",
#     "make_headless_system",
#     "run_headless_longrun",
# ]