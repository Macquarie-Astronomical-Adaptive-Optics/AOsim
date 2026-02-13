from __future__ import annotations

from collections.abc import MutableMapping, Iterator
from typing import Any, Dict, Iterable, Mapping, Optional, Set


class OverlayConfig(MutableMapping):
    """A dict-like view that routes some keys to an overlay mapping.

    Used to centralize config while still supporting per-sensor overrides.

    - Keys in `overlay_keys` are read/written from `overlay`.
    - All other keys are read/written from `base`.
    """

    def __init__(
        self,
        base: MutableMapping,
        overlay: MutableMapping,
        *,
        overlay_keys: Iterable[str],
    ):
        self._base = base
        self._overlay = overlay
        self._overlay_keys: Set[str] = set(overlay_keys)

    # --- MutableMapping ---
    def __getitem__(self, key: str) -> Any:
        # Keys explicitly marked as overlay-owned route to overlay first.
        # For all other keys, prefer the base when present; otherwise
        # fall back to overlay (important for overlay-only keys like
        # per-sensor metadata fields).
        if key in self._overlay_keys:
            if key in self._overlay:
                return self._overlay[key]
            if key in self._base:
                return self._base[key]
            raise KeyError(key)

        if key in self._base:
            return self._base[key]
        if key in self._overlay:
            return self._overlay[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self._overlay_keys:
            self._overlay[key] = value
            return

        # If the base already owns the key, update it there.
        # Otherwise, treat it as overlay-owned (sensor-specific extension).
        if key in self._base:
            self._base[key] = value
        else:
            self._overlay[key] = value

    def __delitem__(self, key: str) -> None:
        if key in self._overlay_keys:
            del self._overlay[key]
            return

        if key in self._base:
            del self._base[key]
        elif key in self._overlay:
            del self._overlay[key]
        else:
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        # Union of keys (stable-ish ordering: base first, then overlay-only)
        seen = set()
        for k in self._base.keys():
            seen.add(k)
            yield k
        for k in self._overlay.keys():
            if k not in seen:
                yield k

    def __len__(self) -> int:
        # Could compute union; keep simple.
        return len(set(self._base.keys()) | set(self._overlay.keys()))

    # --- Convenience ---
    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key in self._overlay_keys:
            return (key in self._overlay) or (key in self._base)
        return (key in self._base) or (key in self._overlay)

    @property
    def base(self) -> MutableMapping:
        return self._base

    @property
    def overlay(self) -> MutableMapping:
        return self._overlay


__all__ = ["OverlayConfig"]
