from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from .util import round_half_up


METHODS = (
    # "bin_floor",
    # "bin_ceil",
    "opencv_bilinear",
    # "lanczos2_7tap_16phase",
    # "lanczos3_7tap_16phase",
)
TAP_OFFSETS = np.arange(-3, 4, dtype=np.int64)
PHASES = 16
Q_FRACTION_BITS = 14
Q_SCALE = 1 << Q_FRACTION_BITS
REFERENCE_TAPS = 129
REFERENCE_PHASES = 1024
REFERENCE_LOBES = 3
REFERENCE_TAP_OFFSETS = np.arange(-64, 65, dtype=np.int64)


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


def _coordinate_for_phases(
    destination: int, actual_scale: float, phases: int
) -> tuple[int, int]:
    coordinate = (destination + 0.5) * actual_scale - 0.5
    base = math.floor(coordinate)
    fraction = coordinate - base
    phase = int(math.floor(fraction * phases + 0.5))
    if phase == phases:
        base += 1
        phase = 0
    return base, phase


def _coordinate(destination: int, actual_scale: float) -> tuple[int, int]:
    return _coordinate_for_phases(destination, actual_scale, PHASES)


def _reflect101_indices(indices: np.ndarray, length: int) -> np.ndarray:
    if length < 1:
        raise ValueError("sampled image dimension must be positive")
    if length == 1:
        return np.zeros_like(indices)
    period = 2 * length - 2
    reflected = np.mod(indices, period)
    return np.where(reflected < length, reflected, period - reflected)


def _round_shift_signed(values: np.ndarray, bits: int) -> np.ndarray:
    scale = 1 << bits
    half = scale >> 1
    return np.where(values >= 0, (values + half) // scale, -((-values + half) // scale))


def lanczos_resize(
    image: np.ndarray,
    ratio: float,
    lobes: int,
    *,
    boundary: str = "white",
) -> ResizeResult:
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("lanczos_resize expects a 2D uint8 image")
    if boundary not in {"white", "reflect101"}:
        raise ValueError(f"unknown Lanczos boundary mode: {boundary}")
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
        if boundary == "reflect101":
            samples = source[:, _reflect101_indices(indices, source_width)]
        else:
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
        if boundary == "reflect101":
            samples = horizontal[_reflect101_indices(indices, source_height), :]
        else:
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
            "boundary": boundary,
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
    cv2.setUseOptimized(False)
    cv2.setNumThreads(1)
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)
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
            "optimized": False,
            "threads": 1,
            "opencl": False,
            "actual_scale_x": source_width / destination_width,
            "actual_scale_y": source_height / destination_height,
        },
    )


def fixed_point_bilinear(image: np.ndarray, ratio: float) -> ResizeResult:
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("fixed_point_bilinear expects a 2D uint8 image")
    source_height, source_width = image.shape
    destination_width = output_length(source_width, ratio)
    destination_height = output_length(source_height, ratio)
    scale_x_num = source_width
    scale_x_den = destination_width
    scale_y_num = source_height
    scale_y_den = destination_height
    pixels = np.empty((destination_height, destination_width), dtype=np.uint8)

    x_terms: list[tuple[int, int, int]] = []
    for destination_x in range(destination_width):
        coordinate_num = (2 * destination_x + 1) * scale_x_num - scale_x_den
        coordinate_den = 2 * scale_x_den
        base = coordinate_num // coordinate_den
        fraction = coordinate_num - base * coordinate_den
        x0 = min(max(base, 0), source_width - 1)
        x1 = min(max(base + 1, 0), source_width - 1)
        x_terms.append((x0, x1, fraction))

    y_terms: list[tuple[int, int, int]] = []
    for destination_y in range(destination_height):
        coordinate_num = (2 * destination_y + 1) * scale_y_num - scale_y_den
        coordinate_den = 2 * scale_y_den
        base = coordinate_num // coordinate_den
        fraction = coordinate_num - base * coordinate_den
        y0 = min(max(base, 0), source_height - 1)
        y1 = min(max(base + 1, 0), source_height - 1)
        y_terms.append((y0, y1, fraction))

    denominator = 2 * scale_x_den * 2 * scale_y_den
    for destination_y, (y0, y1, fy) in enumerate(y_terms):
        wy0 = 2 * scale_y_den - fy
        wy1 = fy
        for destination_x, (x0, x1, fx) in enumerate(x_terms):
            wx0 = 2 * scale_x_den - fx
            wx1 = fx
            value = (
                int(image[y0, x0]) * wx0 * wy0
                + int(image[y0, x1]) * wx1 * wy0
                + int(image[y1, x0]) * wx0 * wy1
                + int(image[y1, x1]) * wx1 * wy1
            )
            pixels[destination_y, destination_x] = min(
                255, max(0, (value + denominator // 2) // denominator)
            )
    return ResizeResult(
        pixels=pixels,
        metadata={
            "interpolation": "fixed_point_bilinear",
            "coordinate_mapping": "half_pixel_centers",
            "boundary": "clamp",
            "actual_scale_x": source_width / destination_width,
            "actual_scale_y": source_height / destination_height,
        },
    )


@lru_cache(maxsize=64)
def reference_coefficient_bank(scale: float) -> np.ndarray:
    """Build a float64 129-slot/1024-phase scale-aware Lanczos-3 bank."""

    if scale <= 0:
        raise ValueError("scale must be positive")
    cutoff = min(1.0, 1.0 / scale)
    support = REFERENCE_LOBES / cutoff
    if support > float(REFERENCE_TAP_OFFSETS[-1]):
        raise ValueError(
            f"Lanczos-{REFERENCE_LOBES} support {support:.6f} exceeds the "
            f"{REFERENCE_TAPS}-tap reference capacity"
        )

    bank = np.zeros((REFERENCE_PHASES, REFERENCE_TAPS), dtype=np.float64)
    offsets = REFERENCE_TAP_OFFSETS.astype(np.float64)
    for phase in range(REFERENCE_PHASES):
        fraction = phase / REFERENCE_PHASES
        distance = offsets - fraction
        scaled_distance = cutoff * distance
        inside = np.abs(scaled_distance) < REFERENCE_LOBES
        values = np.zeros(REFERENCE_TAPS, dtype=np.float64)
        values[inside] = (
            cutoff
            * np.sinc(scaled_distance[inside])
            * np.sinc(scaled_distance[inside] / REFERENCE_LOBES)
        )
        total = float(values.sum())
        if abs(total) < 1e-15:
            raise RuntimeError("reference Lanczos coefficient phase has zero gain")
        bank[phase] = values / total
    bank.setflags(write=False)
    return bank


def reference_lanczos_resize(image: np.ndarray, ratio: float) -> ResizeResult:
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("reference_lanczos_resize expects a 2D uint8 image")
    source_height, source_width = image.shape
    destination_width = output_length(source_width, ratio)
    destination_height = output_length(source_height, ratio)
    scale_x = source_width / destination_width
    scale_y = source_height / destination_height
    bank_x = reference_coefficient_bank(scale_x)
    bank_y = reference_coefficient_bank(scale_y)
    source = image.astype(np.float64)

    horizontal = np.empty((source_height, destination_width), dtype=np.float64)
    for destination_x in range(destination_width):
        base, phase = _coordinate_for_phases(
            destination_x, scale_x, REFERENCE_PHASES
        )
        coefficients = bank_x[phase]
        active = np.flatnonzero(coefficients)
        indices = _reflect101_indices(
            base + REFERENCE_TAP_OFFSETS[active], source_width
        )
        horizontal[:, destination_x] = (
            source[:, indices] * coefficients[active][None, :]
        ).sum(axis=1, dtype=np.float64)

    vertical = np.empty((destination_height, destination_width), dtype=np.float64)
    for destination_y in range(destination_height):
        base, phase = _coordinate_for_phases(
            destination_y, scale_y, REFERENCE_PHASES
        )
        coefficients = bank_y[phase]
        active = np.flatnonzero(coefficients)
        indices = _reflect101_indices(
            base + REFERENCE_TAP_OFFSETS[active], source_height
        )
        vertical[destination_y, :] = (
            horizontal[indices, :] * coefficients[active][:, None]
        ).sum(axis=0, dtype=np.float64)

    pixels = np.clip(np.floor(vertical + 0.5), 0, 255).astype(np.uint8)
    return ResizeResult(
        pixels=pixels,
        metadata={
            "lobes": REFERENCE_LOBES,
            "taps": REFERENCE_TAPS,
            "phases": REFERENCE_PHASES,
            "tap_offsets": [
                int(REFERENCE_TAP_OFFSETS[0]), int(REFERENCE_TAP_OFFSETS[-1])
            ],
            "coefficient_precision": "float64",
            "boundary": "reflect101",
            "color_domain": "encoded_samples",
            "coordinate_mapping": "half_pixel_centers",
            "pixel_rounding": "round_half_up_then_clip_uint8",
            "actual_scale_x": scale_x,
            "actual_scale_y": scale_y,
        },
    )


def resize_image(
    image: np.ndarray,
    requested_ratio: float,
    method: str,
    *,
    boundary: str = "white",
    exact: bool = False,
) -> ResizeResult:
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
        if exact:
            return fixed_point_bilinear(image, requested_ratio)
        return opencv_bilinear(image, requested_ratio)
    if method == "lanczos2_7tap_16phase":
        return lanczos_resize(image, requested_ratio, lobes=2, boundary=boundary)
    if method == "lanczos3_7tap_16phase":
        return lanczos_resize(image, requested_ratio, lobes=3, boundary=boundary)
    raise ValueError(f"unknown resize method: {method}")
