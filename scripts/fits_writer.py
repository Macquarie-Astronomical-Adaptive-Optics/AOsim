
"""
Minimal FITS writer (no external deps).

Supports writing:
- An empty Primary HDU (NAXIS=0) to start a file
- Appending IMAGE extensions for 2D or 3D float32 arrays
- Streaming chunked writes by appending multiple IMAGE extensions

Notes:
- Data written in big-endian IEEE float32 (BITPIX=-32).
- Each HDU header + data are padded to 2880-byte FITS blocks.
"""

import os

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union, BinaryIO

import numpy as np

_BLOCK = 2880


def _card(keyword: str, value: Optional[str] = None, comment: str = "") -> bytes:
    """Create one 80-byte FITS header card."""
    keyword = (keyword or "")[:8].ljust(8)
    if value is None:
        s = f"{keyword}{comment}"
    else:
        s = f"{keyword}= {value}"
        if comment:
            s += f" / {comment}"
    s = s[:80].ljust(80)
    return s.encode("ascii")


def _format_value(v: Any) -> str:
    """Format FITS header value field (starts after '= ')."""
    if isinstance(v, (np.bool_, bool)):
        return ("T" if bool(v) else "F").rjust(20)
    if isinstance(v, (np.integer, int)):
        return f"{int(v):20d}"
    if isinstance(v, (np.floating, float)):
        # Use a compact general format that FITS readers accept.
        # Right-justify into 20 columns.
        return f"{float(v):20.12g}"
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("ascii", errors="ignore")
    if isinstance(v, str):
        # FITS strings must be quoted.
        # Keep within 20 columns when possible, but FITS allows longer; we trim in _card.
        return f"'{v}'".ljust(20)
    # Fallback to string repr
    return f"'{str(v)}'".ljust(20)


def _pad_to_block(nbytes: int) -> int:
    return (-int(nbytes)) % _BLOCK


def _write_header(f: BinaryIO, cards: Sequence[bytes]) -> None:
    header = b"".join(cards + [_card("END")])
    header += b" " * _pad_to_block(len(header))
    f.write(header)


def write_primary_empty(
    f: BinaryIO,
    extra_header: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a minimal empty Primary HDU (NAXIS=0)."""
    cards = [
        _card("SIMPLE", _format_value(True), "conforms to FITS standard"),
        _card("BITPIX", _format_value(8), "8-bit bytes"),
        _card("NAXIS", _format_value(0), "no image data"),
        _card("EXTEND", _format_value(True), "extensions may be present"),
    ]
    if extra_header:
        for k, v in extra_header.items():
            cards.append(_card(str(k)[:8].upper(), _format_value(v)))
    _write_header(f, cards)


def append_image_hdu(
    f: BinaryIO,
    data: np.ndarray,
    extname: Optional[str] = None,
    extver: Optional[int] = None,
    extra_header: Optional[Dict[str, Any]] = None,
) -> None:
    """Append an IMAGE extension HDU with 2D or 3D float32 data."""
    arr = np.asarray(data)
    if arr.ndim not in (2, 3):
        raise ValueError("FITS IMAGE HDU supports only 2D or 3D arrays here")
    # FITS uses big-endian.
    arr = np.asarray(arr, dtype=">f4", order="C")
    # NAXIS ordering: axis1 = last dimension (x/columns)
    naxis = arr.ndim
    shape = arr.shape  # (H,W) or (B,H,W)
    cards = [
        _card("XTENSION", _format_value("IMAGE"), "Image extension"),
        _card("BITPIX", _format_value(-32), "32-bit floating point"),
        _card("NAXIS", _format_value(naxis), ""),
    ]
    if naxis == 2:
        h, w = int(shape[0]), int(shape[1])
        cards += [
            _card("NAXIS1", _format_value(w), ""),
            _card("NAXIS2", _format_value(h), ""),
        ]
    else:
        b, h, w = int(shape[0]), int(shape[1]), int(shape[2])
        cards += [
            _card("NAXIS1", _format_value(w), ""),
            _card("NAXIS2", _format_value(h), ""),
            _card("NAXIS3", _format_value(b), ""),
        ]
    cards += [
        _card("PCOUNT", _format_value(0), ""),
        _card("GCOUNT", _format_value(1), ""),
    ]
    if extname is not None:
        cards.append(_card("EXTNAME", _format_value(str(extname)), ""))
    if extver is not None:
        cards.append(_card("EXTVER", _format_value(int(extver)), ""))
    if extra_header:
        for k, v in extra_header.items():
            cards.append(_card(str(k)[:8].upper(), _format_value(v)))
    _write_header(f, cards)

    # Write data and pad to 2880 bytes
    raw = arr.tobytes(order="C")
    f.write(raw)
    f.write(b"\0" * _pad_to_block(len(raw)))


@dataclass
class PsfFITSRecorder:
    """Stream PSF batches into a FITS file via multiple IMAGE extensions.

    The file layout:
      - Primary HDU: empty
      - Ext 1..K: PSF chunks, each (B, H, W)
      - Optional final ext: LONGEXP, (H, W)
    """
    out_path: str
    psf_shape: Tuple[int, int]
    extra_header: Optional[Dict[str, Any]] = None
    extname_chunks: str = "PSF"
    _f: Optional[BinaryIO] = None
    _extver: int = 0
    _frames_written: int = 0

    def __post_init__(self):
        os_dir = os.path.dirname(self.out_path)
        if os_dir:
            os.makedirs(os_dir, exist_ok=True)
        self._f = open(self.out_path, "wb")
        write_primary_empty(self._f, self.extra_header)

    @property
    def frames_written(self) -> int:
        return int(self._frames_written)

    def write_chunk(self, psf_batch: Any, extra_header: Optional[Dict[str, Any]] = None) -> None:
        """Append one chunk (B,H,W). Accepts CuPy or NumPy arrays."""
        if self._f is None:
            raise RuntimeError("Recorder closed")
        # Convert to numpy on host.
        try:
            import cupy as cp  # optional
            if isinstance(psf_batch, cp.ndarray):
                arr = cp.asnumpy(psf_batch)
            else:
                arr = np.asarray(psf_batch)
        except Exception:
            arr = np.asarray(psf_batch)

        if arr.ndim != 3:
            raise ValueError("psf_batch must be (B,H,W)")
        if (arr.shape[1], arr.shape[2]) != tuple(self.psf_shape):
            raise ValueError(f"psf_batch spatial shape {arr.shape[1:]} != {self.psf_shape}")

        self._extver += 1
        append_image_hdu(
            self._f,
            arr.astype(np.float32, copy=False),
            extname=self.extname_chunks,
            extver=self._extver,
            extra_header=extra_header,
        )
        self._frames_written += int(arr.shape[0])

    def write_longexp(self, psf2d: Any, extname: str = "LONGEXP", extra_header: Optional[Dict[str, Any]] = None) -> None:
        """Append a 2D long-exposure PSF image."""
        if self._f is None:
            raise RuntimeError("Recorder closed")
        try:
            import cupy as cp
            if isinstance(psf2d, cp.ndarray):
                arr = cp.asnumpy(psf2d)
            else:
                arr = np.asarray(psf2d)
        except Exception:
            arr = np.asarray(psf2d)

        if arr.ndim != 2:
            raise ValueError("psf2d must be (H,W)")
        if (arr.shape[0], arr.shape[1]) != tuple(self.psf_shape):
            raise ValueError(f"psf2d spatial shape {arr.shape} != {self.psf_shape}")

        append_image_hdu(
            self._f,
            arr.astype(np.float32, copy=False),
            extname=extname,
            extver=1,
            extra_header=extra_header,
        )

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
