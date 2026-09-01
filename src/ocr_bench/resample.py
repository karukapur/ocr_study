from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from .util import round_half_up


DEFAULT_TAPS = 7
DEFAULT_PHASES = 16
Q_FRACTION_BITS = 14
Q_SCALE = 1 << Q_FRACTION_BITS
OPENCV_LINEAR_COEF_BITS = 11
OPENCV_LINEAR_COEF_SCALE = 1 << OPENCV_LINEAR_COEF_BITS
REFERENCE_TAPS = 129
REFERENCE_PHASES = 1024
REFERENCE_LOBES = 3
REFERENCE_TAP_OFFSETS = np.arange(-64, 65, dtype=np.int64)


@dataclass(frozen=True)
class ResizeResult:
    """A quantized resize result with an optional pre-quantization output."""

    pixels: np.ndarray
    metadata: dict[str, Any]
    floating_pixels: np.ndarray | None = None


def output_length(source_length: int, ratio: float) -> int:
    if ratio <= 0:
        raise ValueError("scale must be positive")
    return max(1, round_half_up(source_length / ratio))


def _height_scale(scale_w: float, scale_h: float | None) -> float:
    return scale_w if scale_h is None else scale_h


def _tap_offsets(taps: int) -> np.ndarray:
    if taps < 1 or taps % 2 == 0:
        raise ValueError("Lanczos taps must be a positive odd integer")
    radius = taps // 2
    return np.arange(-radius, radius + 1, dtype=np.int64)


def _validate_interpolation(interpolation: str) -> None:
    if interpolation not in {"original", "shift"}:
        raise ValueError(f"unknown interpolation mapping: {interpolation}")


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


def coefficient_bank(
    scale: float,
    lobes: int,
    *,
    taps: int = DEFAULT_TAPS,
    phases: int = DEFAULT_PHASES,
) -> np.ndarray:
    if lobes not in {2, 3}:
        raise ValueError("only Lanczos-2 and Lanczos-3 are supported")
    if scale <= 0:
        raise ValueError("scale must be positive")
    if phases < 1:
        raise ValueError("Lanczos phases must be positive")
    tap_offsets = _tap_offsets(taps)
    bank = np.zeros((phases, taps), dtype=np.int16)
    for phase in range(phases):
        fraction = phase / phases
        distance = tap_offsets.astype(np.float64) - fraction
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
    destination: int,
    actual_scale: float,
    phases: int,
    interpolation: str = "shift",
) -> tuple[int, int]:
    _validate_interpolation(interpolation)
    coordinate = (
        destination * actual_scale
        if interpolation == "original"
        else (destination + 0.5) * actual_scale - 0.5
    )
    base = math.floor(coordinate)
    fraction = coordinate - base
    phase = int(math.floor(fraction * phases + 0.5))
    if phase == phases:
        base += 1
        phase = 0
    return base, phase


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
    scale_w: float,
    lobes: int,
    *,
    scale_h: float | None = None,
    taps: int = DEFAULT_TAPS,
    phases: int = DEFAULT_PHASES,
    interpolation: str = "shift",
    boundary: str = "white",
) -> ResizeResult:
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("lanczos_resize expects a 2D uint8 image")
    if boundary not in {"white", "reflect101"}:
        raise ValueError(f"unknown Lanczos boundary mode: {boundary}")
    _validate_interpolation(interpolation)
    source_height, source_width = image.shape
    scale_h = _height_scale(scale_w, scale_h)
    destination_width = output_length(source_width, scale_w)
    destination_height = output_length(source_height, scale_h)
    scale_x = source_width / destination_width
    scale_y = source_height / destination_height
    tap_offsets = _tap_offsets(taps)
    bank_x = coefficient_bank(scale_x, lobes, taps=taps, phases=phases)
    bank_y = coefficient_bank(scale_y, lobes, taps=taps, phases=phases)

    # The horizontal intermediate remains in Q14. The vertical pass therefore
    # accumulates Q28 values and performs the only pixel-domain rounding.
    horizontal = np.empty((source_height, destination_width), dtype=np.int64)
    source = image.astype(np.int64)
    white_q14 = 255 * Q_SCALE
    for destination_x in range(destination_width):
        base, phase = _coordinate_for_phases(
            destination_x, scale_x, phases, interpolation
        )
        coefficients = bank_x[phase].astype(np.int64)
        indices = base + tap_offsets
        if boundary == "reflect101":
            samples = source[:, _reflect101_indices(indices, source_width)]
        else:
            samples = np.full((source_height, taps), 255, dtype=np.int64)
            valid = (indices >= 0) & (indices < source_width)
            if valid.any():
                samples[:, valid] = source[:, indices[valid]]
        horizontal[:, destination_x] = (samples * coefficients).sum(axis=1, dtype=np.int64)

    vertical = np.empty((destination_height, destination_width), dtype=np.int64)
    for destination_y in range(destination_height):
        base, phase = _coordinate_for_phases(
            destination_y, scale_y, phases, interpolation
        )
        coefficients = bank_y[phase].astype(np.int64)
        indices = base + tap_offsets
        if boundary == "reflect101":
            samples = horizontal[_reflect101_indices(indices, source_height), :]
        else:
            samples = np.full(
                (taps, destination_width), white_q14, dtype=np.int64
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
            "taps": taps,
            "phases": phases,
            "tap_offsets": tap_offsets.tolist(),
            "q_format": "Q2.14",
            "q_fraction_bits": Q_FRACTION_BITS,
            "boundary": boundary,
            "coordinate_mapping": interpolation,
            "actual_scale_x": scale_x,
            "actual_scale_y": scale_y,
            "coefficient_bank_x": bank_x.astype(int).tolist(),
            "coefficient_bank_y": bank_y.astype(int).tolist(),
        },
    )


def opencv_bilinear(
    image: np.ndarray, scale_w: float, scale_h: float | None = None
) -> ResizeResult:
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("opencv_bilinear expects a 2D uint8 image")
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
    scale_h = _height_scale(scale_w, scale_h)
    destination_width = output_length(source_width, scale_w)
    destination_height = output_length(source_height, scale_h)
    pixels = cv2.resize(
        image,
        (destination_width, destination_height),
        interpolation=cv2.INTER_LINEAR,
    )
    floating_pixels = cv2.resize(
        image.astype(np.float64),
        (destination_width, destination_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return ResizeResult(
        pixels=np.asarray(pixels, dtype=np.uint8),
        floating_pixels=np.asarray(floating_pixels, dtype=np.float64),
        metadata={
            "interpolation": "cv2.INTER_LINEAR",
            "neighborhood": "2x2",
            "optimized": False,
            "threads": 1,
            "opencl": False,
            "coordinate_mapping": "shift",
            "floating_pixel_source": "cv2.INTER_LINEAR on float64 input",
            "floating_pixel_precision": "float64",
            "actual_scale_x": source_width / destination_width,
            "actual_scale_y": source_height / destination_height,
        },
    )


def floating_point_bilinear(
    image: np.ndarray,
    scale_w: float,
    scale_h: float | None = None,
    *,
    interpolation: str = "shift",
) -> ResizeResult:
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("floating_point_bilinear expects a 2D uint8 image")
    _validate_interpolation(interpolation)
    source_height, source_width = image.shape
    scale_h = _height_scale(scale_w, scale_h)
    destination_width = output_length(source_width, scale_w)
    destination_height = output_length(source_height, scale_h)
    scale_x = source_width / destination_width
    scale_y = source_height / destination_height

    x_offsets, x_fractions = _floating_linear_terms(
        source_width, destination_width, interpolation
    )
    y_offsets, y_fractions = _floating_linear_terms(
        source_height, destination_height, interpolation
    )
    source = image.astype(np.float32)
    left = source[:, np.clip(x_offsets, 0, source_width - 1)]
    right = source[:, np.clip(x_offsets + 1, 0, source_width - 1)]
    horizontal = (
        left * (np.float32(1.0) - x_fractions)[None, :]
        + right * x_fractions[None, :]
    ).astype(np.float32)
    top = horizontal[np.clip(y_offsets, 0, source_height - 1), :]
    bottom = horizontal[np.clip(y_offsets + 1, 0, source_height - 1), :]
    vertical = (
        top * (np.float32(1.0) - y_fractions)[:, None]
        + bottom * y_fractions[:, None]
    ).astype(np.float32)
    pixels = np.clip(np.floor(vertical + np.float32(0.5)), 0, 255).astype(np.uint8)
    return ResizeResult(
        pixels=pixels,
        floating_pixels=vertical,
        metadata={
            "interpolation": "bilinear float32",
            "kernel": "bilinear_2x2_float32",
            "neighborhood": "2x2",
            "coefficient_precision": "float32",
            "accumulator_precision": "float32",
            "coordinate_mapping": interpolation,
            "boundary": "clamp",
            "pixel_rounding": "round_half_up_then_clip_uint8",
            "actual_scale_x": scale_x,
            "actual_scale_y": scale_y,
        },
    )


def fixed_point_bilinear(
    image: np.ndarray,
    scale_w: float,
    scale_h: float | None = None,
    *,
    interpolation: str = "shift",
) -> ResizeResult:
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("fixed_point_bilinear expects a 2D uint8 image")
    _validate_interpolation(interpolation)
    source_height, source_width = image.shape
    scale_h = _height_scale(scale_w, scale_h)
    destination_width = output_length(source_width, scale_w)
    destination_height = output_length(source_height, scale_h)
    if (destination_width, destination_height) == (source_width, source_height):
        pixels = image.copy()
        kernel = "copy"
    else:
        scale_x = source_width / destination_width
        scale_y = source_height / destination_height
        if (
            interpolation == "shift"
            and math.isclose(scale_x, 2.0, rel_tol=0.0, abs_tol=0.0)
            and math.isclose(scale_y, 2.0, rel_tol=0.0, abs_tol=0.0)
            and source_width >= destination_width * 2
            and source_height >= destination_height * 2
        ):
            blocks = image.astype(np.uint32).reshape(
                destination_height, 2, destination_width, 2
            )
            pixels = ((blocks.sum(axis=(1, 3), dtype=np.uint32) + 2) // 4).astype(
                np.uint8
            )
            kernel = "opencv_inter_area_fast_2x"
        else:
            x_offsets, x_coefficients, x_min, x_max = _opencv_linear_terms(
                source_width, destination_width, interpolation=interpolation
            )
            y_offsets, y_coefficients, _, _ = _opencv_linear_terms(
                source_height,
                destination_height,
                interpolation=interpolation,
                clamp_edge_coefficients=False,
            )
            source = image.astype(np.int64)
            right_offsets = np.minimum(x_offsets + 1, source_width - 1)
            horizontal = (
                source[:, x_offsets] * x_coefficients[:, 0]
                + source[:, right_offsets] * x_coefficients[:, 1]
            )
            if x_min:
                horizontal[:, :x_min] = (
                    source[:, x_offsets[:x_min]] * OPENCV_LINEAR_COEF_SCALE
                )
            if x_max < destination_width:
                horizontal[:, x_max:] = (
                    source[:, x_offsets[x_max:]] * OPENCV_LINEAR_COEF_SCALE
                )

            top = horizontal[np.clip(y_offsets, 0, source_height - 1)]
            bottom = horizontal[np.clip(y_offsets + 1, 0, source_height - 1)]
            beta0 = y_coefficients[:, 0, None]
            beta1 = y_coefficients[:, 1, None]
            # Match OpenCV's legacy CV_8U vertical kernel exactly.  It does
            # not perform one ideal Q22 multiply-and-round.  Each horizontal
            # Q11 value first loses four low bits, each vertical product then
            # takes its high 16 bits, and the two terms are rounded from Q2.
            # Keeping this sequence is what makes the clone agree with
            # cv2.INTER_LINEAR rather than merely approximate it to +/-1.
            vertical_top = (beta0 * (top >> 4)) >> 16
            vertical_bottom = (beta1 * (bottom >> 4)) >> 16
            values = (vertical_top + vertical_bottom + 2) >> 2
            pixels = np.clip(values, 0, 255).astype(np.uint8)
            kernel = (
                "opencv_inter_linear_8u_fixed_point"
                if interpolation == "shift"
                else "bilinear_8u_fixed_point"
            )
    return ResizeResult(
        pixels=pixels,
        metadata={
            "interpolation": (
                "cv2.INTER_LINEAR deterministic clone"
                if interpolation == "shift"
                else "bilinear fixed-point"
            ),
            "kernel": kernel,
            "coordinate_mapping": interpolation,
            "boundary": "clamp",
            "coefficient_bits": OPENCV_LINEAR_COEF_BITS,
            "coefficient_scale": OPENCV_LINEAR_COEF_SCALE,
            "vertical_rounding": "opencv_legacy_cv8u",
            "actual_scale_x": source_width / destination_width,
            "actual_scale_y": source_height / destination_height,
        },
    )


def _opencv_linear_terms(
    source_length: int,
    destination_length: int,
    *,
    interpolation: str = "shift",
    clamp_edge_coefficients: bool = True,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    _validate_interpolation(interpolation)
    offsets = np.empty(destination_length, dtype=np.int64)
    coefficients = np.empty((destination_length, 2), dtype=np.int64)
    minimum = 0
    maximum = destination_length
    scale = source_length / destination_length
    for destination in range(destination_length):
        coordinate = np.float32(
            destination * scale
            if interpolation == "original"
            else (destination + 0.5) * scale - 0.5
        )
        source = math.floor(float(coordinate))
        fraction = np.float32(coordinate - source)
        if source < 0:
            minimum = destination + 1
            if clamp_edge_coefficients:
                source = 0
                fraction = np.float32(0.0)
        if source >= source_length - 1:
            maximum = min(maximum, destination)
            if clamp_edge_coefficients:
                source = source_length - 1
                fraction = np.float32(0.0)
        offsets[destination] = source
        coefficients[destination] = (
            _round_float32(
                np.float32(
                    (np.float32(1.0) - fraction) * OPENCV_LINEAR_COEF_SCALE
                )
            ),
            _round_float32(np.float32(fraction * OPENCV_LINEAR_COEF_SCALE)),
        )
    return offsets, coefficients, minimum, maximum


def _floating_linear_terms(
    source_length: int, destination_length: int, interpolation: str
) -> tuple[np.ndarray, np.ndarray]:
    _validate_interpolation(interpolation)
    offsets = np.empty(destination_length, dtype=np.int64)
    fractions = np.empty(destination_length, dtype=np.float32)
    scale = source_length / destination_length
    for destination in range(destination_length):
        coordinate = np.float32(
            destination * scale
            if interpolation == "original"
            else (destination + 0.5) * scale - 0.5
        )
        source = math.floor(float(coordinate))
        offsets[destination] = source
        fractions[destination] = np.float32(coordinate - source)
    return offsets, fractions


def _round_float32(value: np.float32) -> int:
    return int(np.rint(value))


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


def reference_lanczos_resize(
    image: np.ndarray, scale_w: float, scale_h: float | None = None
) -> ResizeResult:
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("reference_lanczos_resize expects a 2D uint8 image")
    source_height, source_width = image.shape
    scale_h = _height_scale(scale_w, scale_h)
    destination_width = output_length(source_width, scale_w)
    destination_height = output_length(source_height, scale_h)
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
            "coordinate_mapping": "shift",
            "pixel_rounding": "round_half_up_then_clip_uint8",
            "actual_scale_x": scale_x,
            "actual_scale_y": scale_y,
        },
    )


def resize_image(
    image: np.ndarray,
    scale_w: float,
    method: str,
    *,
    scale_h: float | None = None,
    taps: int = DEFAULT_TAPS,
    phases: int = DEFAULT_PHASES,
    interpolation: str = "shift",
    boundary: str = "white",
) -> ResizeResult:
    if scale_h is not None and method in {"bin_floor", "bin_ceil"}:
        raise ValueError("box binning does not support a separate height scale")
    if method == "bin_floor":
        factor = integer_factor(scale_w, "floor")
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
        factor = integer_factor(scale_w, "ceil")
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
        if interpolation != "shift":
            raise ValueError("opencv_bilinear only supports shift interpolation")
        return opencv_bilinear(image, scale_w, scale_h)
    if method == "floating_point_bilinear":
        return floating_point_bilinear(
            image, scale_w, scale_h, interpolation=interpolation
        )
    if method == "fixed_point_bilinear":
        return fixed_point_bilinear(
            image, scale_w, scale_h, interpolation=interpolation
        )
    if method == "lanczos2":
        return lanczos_resize(
            image,
            scale_w,
            lobes=2,
            scale_h=scale_h,
            taps=taps,
            phases=phases,
            interpolation=interpolation,
            boundary=boundary,
        )
    if method == "lanczos3":
        return lanczos_resize(
            image,
            scale_w,
            lobes=3,
            scale_h=scale_h,
            taps=taps,
            phases=phases,
            interpolation=interpolation,
            boundary=boundary,
        )
    raise ValueError(f"unknown resize method: {method}")
