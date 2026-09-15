"""Register-driven RGB row streaming; see C55/README.md for model limits."""

import argparse
import json

import numpy as np
from PIL import Image

from .coordinates import COORDINATE_SCALE, axis_coordinates
from .generate_coefficients import load_coefficients, validate_coefficients


REGISTER_NAMES = (
    "Pipeline: Bypass crop", "Crop: Enable crop",
    "Crop: start x", "Crop: start y", "Crop: size x", "Crop: size y",
    "vfilt_tinc", "height_in", "oheight", "vfilt_coefset",
    "hfilt_tinc", "width_in", "owidth", "hfilt_coefset",
    "Resampler: method", "Resampler: interpolation",
)


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate register: {name}")
        result[name] = value
    return result


def _validate_fields(registers):
    if not isinstance(registers, dict):
        raise ValueError("registers must be a JSON object / dictionary")
    missing = set(REGISTER_NAMES) - registers.keys()
    unknown = registers.keys() - set(REGISTER_NAMES)
    if missing or unknown:
        raise ValueError(f"missing registers: {sorted(missing)}; unknown registers: {list(unknown)}")
    for name in REGISTER_NAMES:
        if type(registers[name]) is not int:
            raise ValueError(f"{name} must be an integer (not a boolean)")
    for name in ("Pipeline: Bypass crop", "Crop: Enable crop", "Resampler: interpolation"):
        if registers[name] not in (0, 1):
            raise ValueError(f"{name} must be 0 or 1")
    if registers["Resampler: method"] not in (0, 1, 2):
        raise ValueError("Resampler: method must be 0, 1, or 2")
    for name in ("hfilt_coefset", "vfilt_coefset"):
        if registers[name] != 0:
            raise ValueError(f"{name} must be 0: only the software bilinear bank is supported")


def load_registers(path) -> dict:
    """Read all register fields; frame-dependent validation happens on processing."""
    with open(path, encoding="utf-8") as stream:
        registers = json.load(stream, object_pairs_hook=_unique_object)
    _validate_fields(registers)
    return registers


def _validate_geometry(frame_shape, registers):
    _validate_fields(registers)
    if (
        not isinstance(frame_shape, (tuple, list)) or len(frame_shape) != 3
        or any(type(value) is not int for value in frame_shape)
        or frame_shape[0] <= 0 or frame_shape[1] <= 0 or frame_shape[2] != 3
    ):
        raise ValueError("frame_shape must be positive (height, width, 3) for uint8 RGB")

    height, width = frame_shape[:2]
    start_x = start_y = 0
    if registers["Pipeline: Bypass crop"] == 0 and registers["Crop: Enable crop"] == 1:
        start_x, start_y = registers["Crop: start x"], registers["Crop: start y"]
        width, height = registers["Crop: size x"], registers["Crop: size y"]
        if start_x < 0 or start_y < 0 or width <= 0 or height <= 0:
            raise ValueError("active crop requires nonnegative origins and positive sizes")
        if start_x + width > frame_shape[1] or start_y + height > frame_shape[0]:
            raise ValueError("active crop extends outside the input frame")

    if registers["width_in"] != width or registers["height_in"] != height:
        raise ValueError("width_in and height_in must match the effective scaler input")
    if width % 8 or height % 8:
        raise ValueError("effective scaler input width and height must be divisible by 8")
    for source, output_name, increment_name in (
        (width, "owidth", "hfilt_tinc"),
        (height, "oheight", "vfilt_tinc"),
    ):
        output = registers[output_name]
        if output <= 0:
            raise ValueError(f"{output_name} must be positive")
        if not source <= 8 * output <= 7 * source:
            raise ValueError(f"{output_name}/input must be within [1/8, 7/8]")
        increment = registers[increment_name]
        if not (1 << 20) <= increment <= (1 << 24) - 1:
            raise ValueError(f"{increment_name} must fit downscale unsigned Q4.20")
        expected = (source << 20) // output
        if increment != expected:
            raise ValueError(f"{increment_name} must equal {expected} for the configured dimensions")
    return start_x, start_y, width, height


def _axis_taps(coordinate, base, phase, source_length, method, bank):
    if method == 0:
        indices = [(coordinate + COORDINATE_SCALE // 2) // COORDINATE_SCALE]
        weights = [1]
    elif method == 2:
        base = coordinate // COORDINATE_SCALE
        indices, weights = [base, base + 1], [1, 1]
    else:
        indices = [base + offset for offset in range(-3, 4)]
        weights = bank[phase]
    return [(max(0, min(index, source_length - 1)), int(weight))
            for index, weight in zip(indices, weights) if weight != 0]


def _horizontal(row, registers, bank):
    width, output_width = registers["width_in"], registers["owidth"]
    method = registers["Resampler: method"]
    result = np.empty((output_width, 3), dtype=np.int64)
    for x, (coordinate, base, phase) in enumerate(axis_coordinates(
        output_width, registers["hfilt_tinc"], registers["Resampler: interpolation"]
    )):
        total = np.zeros(3, dtype=np.int64)
        for index, weight in _axis_taps(coordinate, base, phase, width, method, bank):
            total += row[index].astype(np.int64) * weight
        result[x] = total
    return result


def _read_row(rows, index, frame_shape):
    try:
        row = next(rows)
    except StopIteration:
        raise ValueError(f"input frame is short: missing row {index}") from None
    if (not isinstance(row, np.ndarray) or row.dtype != np.uint8
            or row.shape != (frame_shape[1], 3)):
        raise ValueError(f"input row {index} must be uint8 RGB shaped ({frame_shape[1]}, 3)")
    return row


def _resize_rows(rows, registers, frame_shape, geometry, bank):
    start_x, start_y, width, height = geometry
    method = registers["Resampler: method"]
    rows = iter(rows)
    consumed = 0
    buffered = {}
    for coordinate, base, phase in axis_coordinates(
        registers["oheight"], registers["vfilt_tinc"], registers["Resampler: interpolation"]
    ):
        taps = _axis_taps(coordinate, base, phase, height, method, bank)
        needed = {index for index, _ in taps}
        # Required rows advance monotonically. Retain only this output's support,
        # at most seven filtered rows; no full-frame intermediate is allocated.
        for index in list(buffered):
            if index not in needed:
                del buffered[index]
        last_input = start_y + max(needed)
        while consumed <= last_input:
            row = _read_row(rows, consumed, frame_shape)
            cropped_y = consumed - start_y
            if cropped_y in needed:
                buffered[cropped_y] = _horizontal(row[start_x:start_x + width], registers, bank)
            consumed += 1
        total = np.zeros((registers["owidth"], 3), dtype=np.int64)
        for index, weight in taps:
            total += buffered[index] * weight
        if method == 1:
            quotient, remainder = np.divmod(total, 4096)
            total = quotient + ((remainder > 2048) | ((remainder == 2048) & (quotient % 2 != 0)))
        elif method == 2:
            total = (total + 2) // 4
        yield np.clip(total, 0, 255).astype(np.uint8)

    # Cropped-out and trailing rows must still exist and obey the input contract.
    while consumed < frame_shape[0]:
        _read_row(rows, consumed, frame_shape)
        consumed += 1
    sentinel = object()
    if next(rows, sentinel) is not sentinel:
        raise ValueError("input frame has extra rows")


def resize_rows(rows, registers, *, frame_shape, coefficients=None):
    """Latch setup now; return an iterator of independent RGB output rows.

    Input rows may reuse storage. Exhaust the iterator to validate the entire
    frame: outputs remain provisional until trailing input validation succeeds.
    Coefficients are the dictionary returned by load_coefficients(), or None
    to load the locally generated bank. frame_shape describes the full pre-crop frame.
    """
    geometry = _validate_geometry(frame_shape, registers)
    if coefficients is None:
        coefficients = load_coefficients()
    else:
        validate_coefficients(coefficients)
    bank = np.array(coefficients["banks"]["0"], dtype=np.int64)
    return _resize_rows(rows, registers.copy(), tuple(frame_shape), geometry, bank)


def crop_and_resize(image, registers, *, coefficients=None) -> np.ndarray:
    """Collect the row-streaming implementation without changing caller data."""
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3
            or image.shape[0] == 0 or image.shape[1] == 0):
        raise ValueError("input must be a nonempty uint8 RGB array shaped (height, width, 3)")
    rows = resize_rows(image, registers, frame_shape=image.shape, coefficients=coefficients)
    return np.stack(list(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="8-bit RGB image")
    parser.add_argument("--registers", required=True, help="JSON register file")
    parser.add_argument("--output", required=True, help="output PNG path")
    parser.add_argument("--coefficients", help="PC-generated bilinear coefficient JSON (default: C55/bilinear_coefficients.json)")
    args = parser.parse_args()
    try:
        registers = load_registers(args.registers)
        with Image.open(args.input) as source:
            if source.mode != "RGB":
                raise ValueError(f"input image mode must be RGB, received {source.mode}")
            image = np.array(source)
        coefficients = load_coefficients(args.coefficients) if args.coefficients else None
        output = crop_and_resize(image, registers, coefficients=coefficients)
        Image.fromarray(output).save(args.output, format="PNG")
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
