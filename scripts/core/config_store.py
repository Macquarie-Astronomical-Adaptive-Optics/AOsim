from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from PySide6.QtCore import QObject, Signal


def _list_replace_inplace(dst: list, src: list) -> None:
    """Replace list contents in-place while preserving existing nested objects where possible."""
    n_min = min(len(dst), len(src))

    # Update common prefix
    for i in range(n_min):
        dv = dst[i]
        sv = src[i]
        if isinstance(dv, dict) and isinstance(sv, dict):
            _dict_replace_inplace(dv, sv)
        elif isinstance(dv, list) and isinstance(sv, list):
            _list_replace_inplace(dv, sv)
        else:
            dst[i] = sv

    # Trim or extend
    if len(dst) > len(src):
        del dst[len(src):]
    elif len(dst) < len(src):
        dst.extend(src[n_min:])


def _dict_replace_inplace(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    """Replace dict contents in-place.

    - Removes keys not present in src
    - Recursively replaces nested dict/list objects where possible
    """
    # Drop keys not in src
    for k in list(dst.keys()):
        if k not in src:
            del dst[k]

    for k, sv in src.items():
        if k not in dst:
            dst[k] = sv
            continue

        dv = dst[k]
        if isinstance(dv, dict) and isinstance(sv, dict):
            _dict_replace_inplace(dv, sv)
        elif isinstance(dv, list) and isinstance(sv, list):
            _list_replace_inplace(dv, sv)
        else:
            dst[k] = sv


def _migrate_turbulence_schema_inplace(cfg: Dict[str, Any]) -> None:
    """Normalize turbulence config to store only the active profile.

    New schema (active only):
        cfg["turbulence"] = {"source": "inline", "name": <name>, "profile": <dict>}
        or {"source": "file", "name": <name>, "file": <path>}

    Legacy schema:
        cfg["turbulence_profiles"] = {name: profile_dict, ...}
        cfg["turbulence_active"] = name
    """
    if isinstance(cfg.get("turbulence"), dict):
        return

    profiles = cfg.get("turbulence_profiles")
    if not isinstance(profiles, dict) or not profiles:
        return

    active = cfg.get("turbulence_active")
    if not isinstance(active, str) or active not in profiles:
        active = next(iter(profiles.keys()))

    profile = profiles.get(active)
    if not isinstance(profile, dict):
        profile = {}

    cfg["turbulence"] = {"source": "inline", "name": active, "profile": profile}
    cfg.pop("turbulence_profiles", None)
    cfg.pop("turbulence_active", None)


class ConfigStore(QObject):
    """Central, shared configuration store.

    The key behavior is **in-place replacement** on load, so any widgets holding
    references into the config (including nested dicts/lists) stay valid.
    """

    # Emitted for any in-place config mutation (including load).
    changed = Signal()

    # Emitted specifically after a full config.json has been loaded from disk.
    # Carries the loaded path for logging/debugging.
    loaded = Signal(str)

    def __init__(self, data: Dict[str, Any]):
        super().__init__()
        self.data = data

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4)

    def load(self, path: str | Path) -> None:
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise TypeError(f"Config root must be a JSON object/dict, got {type(loaded)}")

        _dict_replace_inplace(self.data, loaded)
        _migrate_turbulence_schema_inplace(self.data)
        self.changed.emit()
        self.loaded.emit(str(p))


__all__ = ["ConfigStore"]
