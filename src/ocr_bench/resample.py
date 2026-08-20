from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .util import round_half_up


METHODS = (
    "bin_floor",
    "bin_ceil",
    "opencv_bilinear",
    "lanczos2_7tap_16phase",
    "lanczos3_7tap_16phase",
)
TAP_OFFSETS = np.arange(-3, 4, dtype=np.int64)
PHASES = 16
Q_FRACTION_BITS = 14
Q_SCALE = 1 << Q_FRACTION_BITS


@dataclass(frozen=True)
class ResizeResult:
    pixels: np.ndarray
    metadata: dict[str, Any]


def output_length(source_length: int, ratio: float) -> int:
    return max(1, round_half_up(source_length / ratio))


def integer_factor(ratio: float, mode: str) -> int:
    nearest = round(ratio)
    stable_ratio = float(nearest) if math.isclose(ratio, nearest, abs_tol=1e-12) else ratio
    if mode == "floor":
        return max(1, math.floor(stable_ratio))
    if mode == "ceil":
        return max(1, math.ceil(stable_ratio))
    raise ValueError(f"unknown integer-factor mode: {mode}")


def box_bin(image: np.ndarray, factor: int) -> np.ndarray:
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("box_bin expects a 2D uint8 image")
    if factor < 1:
        raise ValueError("binning factor must be positive")
    height, width = image.shape
    if height % factor or width % factor:
        raise ValueError(
            f"image shape {image.shape} is not divisible by binning factor {factor}"
        )
    blocks = image.astype(np.uint32).reshape(
        height // factor, factor, width // factor, factor
    )
    sums = blocks.sum(axis=(1, 3), dtype=np.uint32)
    divisor = factor * factor
    return ((sums + divisor // 2) // divisor).astype(np.uint8)


def _kernel(distance: np.ndarray, scale: float, lobes: int) -> np.ndarray:
    cutoff = min(1.0, 1.0 / scale)
    inside = np.abs(distance) < lobes
    values = np.zeros_like(distance, dtype=np.float64)
    values[inside] = (
        cutoff
        * np.sinc(cutoff * distance[inside])
        * np.sinc(distance[inside] / lobes)
    )
    return values


def coefficient_bank(scale: float, lobes: int) -> np.ndarray:
    if lobes not in {2, 3}:
        raise ValueError("only Lanczos-2 and Lanczos-3 are supported")
    if scale <= 0:
        raise ValueError("scale must be positive")
    bank = np.zeros((PHASES, len(TAP_OFFSETS)), dtype=np.int16)
    for phase in range(PHASES):
        fraction = phase / PHASES
        distance = TAP_OFFSETS.astype(np.float64) - fraction
        values = _kernel(distance, scale, lobes)
        total = float(values.sum())
        if abs(total) < 1e-12:
            raise RuntimeError("Lanczos coefficient phase has zero gain")
        values /= total
        quantized = np.rint(values * Q_SCALE).astype(np.int64)
        correction_index = int(np.argmin(np.abs(distance)))
        quantized[correction_index] += Q_SCALE - int(quantized.sum())
        if quantized.min() < np.iinfo(np.int16).min or quantized.max() > np.iinfo(np.int16).max:
            raise OverflowError("Q2.14 coefficient does not fit signed int16")
        bank[phase] = quantized.astype(np.int16)
    return bank


def _coordinate(destination: int, actual_scale: float) -> tuple[int, int]:
    coordinate = (destination + 0.5) * actual_scale - 0.5
    base = math.floor(coordinate)
    fraction = coordinate - base
    phase = int(math.floor(fraction * PHASES + 0.5))
    if phase == PHASES:
        base += 1
        phase = 0
    return base, phase


def _round_shift_signed(values: np.ndarray, bits: int) -> np.ndarray:
    scale = 1 << bits
    half = scale >> 1
    return np.where(values >= 0, (values + half) // scale, -((-values + half) // scale))


def lanczos_resize(image: np.ndarray, ratio: float, lobes: int) -> ResizeResult:
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("lanczos_resize expects a 2D uint8 image")
    source_height, source_width = image.shape
    destination_width = output_length(source_width, ratio)
    destination_height = output_length(source_height, ratio)
    scale_x = source_width / destination_width
    scale_y = source_height / destination_height
    bank_x = coefficient_bank(scale_x, lobes)
    bank_y = coefficient_bank(scale_y, lobes)

    # The horizontal intermediate remains in Q14. The vertical pass therefore
    # accumulates Q28 values and performs the only pixel-domain rounding.
    horizontal = np.empty((source_height, destination_width), dtype=np.int64)
    source = image.astype(np.int64)
    white_q14 = 255 * Q_SCALE
    for destination_x in range(destination_width):
        base, phase = _coordinate(destination_x, scale_x)
        coefficients = bank_x[phase].astype(np.int64)
        indices = base + TAP_OFFSETS
        samples = np.full((source_height, len(TAP_OFFSETS)), 255, dtype=np.int64)
        valid = (indices >= 0) & (indices < source_width)
        if valid.any():
            samples[:, valid] = source[:, indices[valid]]
        horizontal[:, destination_x] = (samples * coefficients).sum(axis=1, dtype=np.int64)

    vertical = np.empty((destination_height, destination_width), dtype=np.int64)
    for destination_y in range(destination_height):
        base, phase = _coordinate(destination_y, scale_y)
        coefficients = bank_y[phase].astype(np.int64)
        indices = base + TAP_OFFSETS
        samples = np.full(
            (len(TAP_OFFSETS), destination_width), white_q14, dtype=np.int64
        )
        valid = (indices >= 0) & (indices < source_height)
        if valid.any():
            samples[valid, :] = horizontal[indices[valid], :]
        vertical[destination_y, :] = (samples * coefficients[:, None]).sum(
            axis=0, dtype=np.int64
        )

    rounded = _round_shift_signed(vertical, Q_FRACTION_BITS * 2)
    pixels = np.clip(rounded, 0, 255).astype(np.uint8)
    return ResizeResult(
        pixels=pixels,
        metadata={
            "lobes": lobes,
            "taps": 7,
            "phases": PHASES,
            "tap_offsets": TAP_OFFSETS.tolist(),
            "q_format": "Q2.14",
            "q_fraction_bits": Q_FRACTION_BITS,
            "actual_scale_x": scale_x,
            "actual_scale_y": scale_y,
            "coefficient_bank_x": bank_x.astype(int).tolist(),
            "coefficient_bank_y": bank_y.astype(int).tolist(),
        },
    )


def opencv_bilinear(image: np.ndarray, ratio: float) -> ResizeResult:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required for opencv_bilinear; install the project dependencies"
        ) from error
    source_height, source_width = image.shape
    destination_width = output_length(source_width, ratio)
    destination_height = output_length(source_height, ratio)
    pixels = cv2.resize(
        image,
        (destination_width, destination_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return ResizeResult(
        pixels=np.asarray(pixels, dtype=np.uint8),
        metadata={
            "interpolation": "cv2.INTER_LINEAR",
            "neighborhood": "2x2",
            "actual_scale_x": source_width / destination_width,
            "actual_scale_y": source_height / destination_height,
        },
    )


def resize_image(image: np.ndarray, requested_ratio: float, method: str) -> ResizeResult:
    if method == "bin_floor":
        factor = integer_factor(requested_ratio, "floor")
        pixels = box_bin(image, factor)
        return ResizeResult(
            pixels=pixels,
            metadata={
                "integer_bin_factor": factor,
                "actual_scale_x": float(factor),
                "actual_scale_y": float(factor),
                "averaging": "non-overlapping NxN, integer round-to-nearest",
            },
        )
    if method == "bin_ceil":
        factor = integer_factor(requested_ratio, "ceil")
        pixels = box_bin(image, factor)
        return ResizeResult(
            pixels=pixels,
            metadata={
                "integer_bin_factor": factor,
                "actual_scale_x": float(factor),
                "actual_scale_y": float(factor),
                "averaging": "non-overlapping NxN, integer round-to-nearest",
            },
        )
    if method == "opencv_bilinear":
        return opencv_bilinear(image, requested_ratio)
    if method == "lanczos2_7tap_16phase":
        return lanczos_resize(image, requested_ratio, lobes=2)
    if method == "lanczos3_7tap_16phase":
        return lanczos_resize(image, requested_ratio, lobes=3)
    raise ValueError(f"unknown resize method: {method}")
