"""Register validation and RGB crop/resample; see docs/README.md for model limits."""

import argparse
import json

import numpy as np
from PIL import Image

from ocr_bench.resample import fixed_point_bilinear


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
            raise ValueError(f"{name} must be 0: coefficient banks are inactive placeholders")


def load_registers(path) -> dict:
    """Read all register fields; frame-dependent validation happens on processing."""
    with open(path, encoding="utf-8") as stream:
        registers = json.load(stream, object_pairs_hook=_unique_object)
    _validate_fields(registers)
    return registers


def _validate_frame(image, registers):
    _validate_fields(registers)
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] == 0
        or image.shape[1] == 0
    ):
        raise ValueError("input must be a nonempty uint8 RGB array shaped (height, width, 3)")

    height, width = image.shape[:2]
    start_x = start_y = 0
    if registers["Pipeline: Bypass crop"] == 0 and registers["Crop: Enable crop"] == 1:
        start_x, start_y = registers["Crop: start x"], registers["Crop: start y"]
        width, height = registers["Crop: size x"], registers["Crop: size y"]
        if start_x < 0 or start_y < 0 or width <= 0 or height <= 0:
            raise ValueError("active crop requires nonnegative origins and positive sizes")
        if start_x + width > image.shape[1] or start_y + height > image.shape[0]:
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


def _sample_indices(destination, source_length, output_length, interpolation, nearest):
    # Exact dimension-based rational coordinate, with no floating-point setup.
    if interpolation == 0:
        numerator = destination * source_length
        denominator = output_length
    else:
        numerator = (2 * destination + 1) * source_length - output_length
        denominator = 2 * output_length
    if nearest:
        base = (2 * numerator + denominator) // (2 * denominator)
    else:
        base = numerator // denominator
    first = max(0, min(base, source_length - 1))
    second = max(0, min(base + 1, source_length - 1))
    return first, second


def crop_and_resize(image, registers) -> np.ndarray:
    """Validate, crop, and resample RGB without modifying the input array."""
    start_x, start_y, width, height = _validate_frame(image, registers)
    cropped = image[start_y:start_y + height, start_x:start_x + width]
    output_width, output_height = registers["owidth"], registers["oheight"]
    method, interpolation = registers["Resampler: method"], registers["Resampler: interpolation"]
    output = np.empty((output_height, output_width, 3), dtype=np.uint8)

    if method == 1:
        mapping = "shift" if interpolation == 1 else "original"
        for channel in range(3):
            output[:, :, channel] = fixed_point_bilinear(
                cropped[:, :, channel], width / output_width, height / output_height,
                interpolation=mapping,
            ).pixels
        return output

    for y in range(output_height):
        top, bottom = _sample_indices(y, height, output_height, interpolation, method == 0)
        for x in range(output_width):
            left, right = _sample_indices(x, width, output_width, interpolation, method == 0)
            if method == 0:
                output[y, x] = cropped[top, left]
            else:
                for channel in range(3):
                    # Convert before addition: four uint8 samples can sum to 1020.
                    total = (
                        int(cropped[top, left, channel])
                        + int(cropped[top, right, channel])
                        + int(cropped[bottom, left, channel])
                        + int(cropped[bottom, right, channel])
                    )
                    output[y, x, channel] = (total + 2) // 4
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="8-bit RGB image")
    parser.add_argument("--registers", required=True, help="JSON register file")
    parser.add_argument("--output", required=True, help="output PNG path")
    args = parser.parse_args()
    try:
        registers = load_registers(args.registers)
        with Image.open(args.input) as source:
            if source.mode != "RGB":
                raise ValueError(f"input image mode must be RGB, received {source.mode}")
            image = np.array(source)
        output = crop_and_resize(image, registers)
        Image.fromarray(output).save(args.output, format="PNG")
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
