from __future__ import annotations

from pathlib import Path

import numpy as np


LEMMO_RECOGNITION_LENGTHS = frozenset((128, 256, 976, 1024, 4096, 16_368, 5_000_000))
LEMMO_DESCRIPTION_MINIMUM_LENGTH = 16
LEMMO_DESCRIPTION_MAXIMUM_LENGTH = 8_000_000


def load_iq(
    path: str | Path,
    allowed_lengths: frozenset[int] | None = None,
    minimum_length: int | None = None,
    maximum_length: int | None = None,
) -> np.ndarray:
    """Load real ``[L,2]`` or complex ``[L]`` IQ and apply record max-abs normalization."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".csv":
        value = np.genfromtxt(path, delimiter=",", dtype=np.float32)
        if value.ndim == 2 and value.shape[1] > 2:
            raise ValueError("CSV must contain exactly two numeric columns: I,Q")
        if value.ndim == 2 and np.isnan(value[0]).any():
            value = value[1:]
    else:
        raise ValueError("Only .npy and .csv IQ files are supported")

    value = np.asarray(value)
    if np.iscomplexobj(value):
        if value.ndim != 1:
            raise ValueError("Complex IQ must have shape [L]")
        value = np.stack((value.real, value.imag), axis=-1)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(f"IQ must have shape [L,2], got {tuple(value.shape)}")
    length = int(value.shape[0])
    if allowed_lengths is not None and length not in allowed_lengths:
        valid = ", ".join(f"{item:,}" for item in sorted(allowed_lengths))
        raise ValueError(f"Unsupported IQ length {value.shape[0]:,}; expected one of: {valid}")
    if minimum_length is not None and length < int(minimum_length):
        raise ValueError(f"IQ length must be at least {int(minimum_length):,}, got {length:,}")
    if maximum_length is not None and length > int(maximum_length):
        raise ValueError(f"IQ length must not exceed {int(maximum_length):,}, got {length:,}")
    value = np.asarray(value, dtype=np.float32, order="C")
    if not np.isfinite(value).all():
        raise ValueError("IQ contains NaN or Inf")
    scale = float(np.max(np.abs(value)))
    if not np.isfinite(scale) or scale < 1e-12:
        raise ValueError("IQ is empty or has zero amplitude")
    return np.ascontiguousarray(value / scale)


def resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_path.parent.parent / path).resolve()

