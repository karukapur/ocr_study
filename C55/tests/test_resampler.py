import copy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from C55 import resampler
from C55.generate_register_reference import (
    LUT_COLUMNS, REGISTER_COLUMNS, render_markdown,
)
from ocr_bench.resample import fixed_point_bilinear


ROOT = Path(__file__).resolve().parents[2]
C55 = ROOT / "C55"
DOCS = C55 / "docs"


def registers(width=16, height=16, output_width=8, output_height=8, method=1, mapping=1):
    values = resampler.load_registers(C55 / "registers.example.json")
    values.update({
        "Crop: start x": 0, "Crop: start y": 0,
        "Crop: size x": width, "Crop: size y": height,
        "width_in": width, "height_in": height,
        "owidth": output_width, "oheight": output_height,
        "hfilt_tinc": (width << 20) // output_width,
        "vfilt_tinc": (height << 20) // output_height,
        "Resampler: method": method, "Resampler: interpolation": mapping,
    })
    return values


def pattern(kind, height=24, width=32):
    y, x = np.indices((height, width))
    if kind == "constant":
        return np.full((height, width, 3), (17, 129, 255), dtype=np.uint8)
    if kind == "impulse":
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[height // 2, width // 2] = (255, 129, 17)
        image[0, 0] = (61, 128, 253)
        return image
    return np.stack(((x * 7 + y * 3) % 256, (x * 11) % 256, (y * 13) % 256), axis=2).astype(np.uint8)


def rational_reference(image, output_width, output_height, method, mapping):
    """Small scalar oracle using Fraction coordinates and Python sample sums."""
    height, width = image.shape[:2]
    output = np.empty((output_height, output_width, 3), dtype=np.uint8)
    for v in range(output_height):
        for u in range(output_width):
            x, y = Fraction(u * width, output_width), Fraction(v * height, output_height)
            if mapping:
                x += Fraction(width, 2 * output_width) - Fraction(1, 2)
                y += Fraction(height, 2 * output_height) - Fraction(1, 2)
            if method == 0:
                ix, iy = int(x + Fraction(1, 2)), int(y + Fraction(1, 2))
                output[v, u] = image[min(iy, height - 1), min(ix, width - 1)]
            else:
                ix, iy = int(x), int(y)
                for channel in range(3):
                    total = sum(
                        int(image[min(iy + dy, height - 1), min(ix + dx, width - 1), channel])
                        for dy in (0, 1) for dx in (0, 1)
                    )
                    output[v, u, channel] = (total + 2) // 4
    return output


@pytest.mark.parametrize("kind", ["ramp", "constant", "impulse"])
@pytest.mark.parametrize("method", [0, 1, 2])
@pytest.mark.parametrize("mapping", [0, 1])
def test_methods_and_mappings(kind, method, mapping):
    image = pattern(kind)
    before = image.copy()
    values = registers(32, 24, 19, 7, method, mapping)
    values_before = copy.deepcopy(values)
    output = resampler.crop_and_resize(image, values)
    if method == 1:
        expected = np.stack([
            fixed_point_bilinear(image[:, :, c], 32 / 19, 24 / 7,
                                 interpolation="shift" if mapping else "original").pixels
            for c in range(3)
        ], axis=2)
    else:
        expected = rational_reference(image, 19, 7, method, mapping)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(image, before)
    assert values == values_before
    assert output.dtype == np.uint8 and output.shape == (7, 19, 3)
    assert not np.shares_memory(output, image)


@pytest.mark.parametrize("mapping", [0, 1])
def test_bilinear_exact_two_x_uses_existing_function(mapping, monkeypatch):
    image = pattern("ramp", 16, 16)
    calls = []

    def record(*args, **kwargs):
        calls.append((args[1:], kwargs))
        return fixed_point_bilinear(*args, **kwargs)

    monkeypatch.setattr(resampler, "fixed_point_bilinear", record)
    output = resampler.crop_and_resize(image, registers(mapping=mapping))
    assert len(calls) == 3
    assert all(args == (2.0, 2.0) for args, _ in calls)
    expected = np.stack([
        fixed_point_bilinear(image[:, :, c], 2.0, interpolation="shift" if mapping else "original").pixels
        for c in range(3)
    ], axis=2)
    np.testing.assert_array_equal(output, expected)


def test_hand_calculated_nearest_tie_and_average_rounding():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[0, 0], image[0, 1] = (0, 255, 1), (1, 255, 2)
    image[1, 0], image[1, 1] = (2, 255, 3), (3, 255, 4)
    nearest = resampler.crop_and_resize(image, registers(8, 8, 4, 4, 0, 1))
    average = resampler.crop_and_resize(image, registers(8, 8, 4, 4, 2, 1))
    np.testing.assert_array_equal(nearest[0, 0], [3, 255, 4])
    np.testing.assert_array_equal(average[0, 0], [2, 255, 3])


@pytest.mark.parametrize("method", [0, 1, 2])
def test_fractional_mapping_changes_output(method):
    image = pattern("ramp", 16, 16)
    original = resampler.crop_and_resize(image, registers(16, 16, 5, 7, method, 0))
    shifted = resampler.crop_and_resize(image, registers(16, 16, 5, 7, method, 1))
    assert np.any(original != shifted)


def test_neighbor_helper_clamps_edges():
    # Valid slide ratios do not need out-of-frame neighbors; verify the edge
    # policy independently at zero-weight/extended-neighbor coordinates.
    assert resampler._sample_indices(7, 8, 8, 0, False) == (7, 7)
    assert resampler._sample_indices(0, 1, 2, 1, False) == (0, 0)
    assert resampler._sample_indices(1, 1, 2, 1, True) == (0, 0)


@pytest.mark.parametrize("bypass,enable", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_crop_precedence_and_origin(bypass, enable):
    image = pattern("ramp", 24, 32)
    active = bypass == 0 and enable == 1
    values = registers(16 if active else 32, 16 if active else 24, 8, 8, 0, 0)
    values.update({"Pipeline: Bypass crop": bypass, "Crop: Enable crop": enable})
    if active:
        values.update({"Crop: start x": 3, "Crop: start y": 5})
        effective = image[5:21, 3:19]
    else:
        values.update({key: -999 for key in ("Crop: start x", "Crop: start y", "Crop: size x", "Crop: size y")})
        effective = image
    expected = rational_reference(effective, 8, 8, 0, 0)
    np.testing.assert_array_equal(resampler.crop_and_resize(image, values), expected)


@pytest.mark.parametrize("method", [0, 1, 2])
@pytest.mark.parametrize("mapping", [0, 1])
@pytest.mark.parametrize("size", [1, 7])
def test_scale_endpoints_and_minimum_input(method, mapping, size):
    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    output = resampler.crop_and_resize(image, registers(8, 8, size, size, method, mapping))
    np.testing.assert_array_equal(output, np.full((size, size, 3), 255, dtype=np.uint8))


@pytest.mark.parametrize("changes,match", [
    ({"Crop: start x": -1}, "nonnegative"),
    ({"Crop: start y": -1}, "nonnegative"),
    ({"Crop: size x": 0}, "positive sizes"),
    ({"Crop: size y": -1}, "positive sizes"),
    ({"Crop: start x": 1}, "outside"),
    ({"Crop: start y": 1}, "outside"),
    ({"width_in": 8}, "must match"),
    ({"height_in": 8}, "must match"),
    ({"Crop: size x": 15, "width_in": 15}, "divisible"),
    ({"Crop: size y": 15, "height_in": 15}, "divisible"),
    ({"owidth": 0}, "positive"),
    ({"oheight": -1}, "positive"),
    ({"owidth": 1}, "within"),
    ({"oheight": 15}, "within"),
    ({"owidth": 16}, "within"),
    ({"oheight": 17}, "within"),
    ({"hfilt_tinc": 2097153}, "must equal"),
    ({"vfilt_tinc": 2097151}, "must equal"),
    ({"hfilt_tinc": 1 << 24}, "Q4.20"),
    ({"vfilt_tinc": (1 << 20) - 1}, "Q4.20"),
    ({"hfilt_coefset": 1}, "inactive"),
    ({"vfilt_coefset": 16}, "inactive"),
    ({"Resampler: method": 3}, "method"),
    ({"Resampler: interpolation": 2}, "interpolation"),
    ({"Pipeline: Bypass crop": 2}, "Bypass"),
    ({"Crop: Enable crop": -1}, "Enable"),
    ({"owidth": True}, "integer"),
    ({"owidth": 8.0}, "integer"),
    ({"owidth": "8"}, "integer"),
    ({"Crop: start x": None}, "integer"),
])
def test_invalid_configuration(changes, match):
    values = registers()
    values.update(changes)
    with pytest.raises(ValueError, match=match):
        resampler.crop_and_resize(pattern("ramp", 16, 16), values)


@pytest.mark.parametrize("image", [
    None, [], np.zeros((8, 8), dtype=np.uint8), np.zeros((8, 8, 4), dtype=np.uint8),
    np.zeros((8, 8, 3), dtype=np.uint16), np.zeros((0, 8, 3), dtype=np.uint8),
    np.zeros((8, 0, 3), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.float32),
])
def test_invalid_image(image):
    with pytest.raises(ValueError, match="uint8 RGB"):
        resampler.crop_and_resize(image, registers(8, 8, 4, 4))


@pytest.mark.parametrize("contents,match", [
    ('{"owidth": 1, "owidth": 2}', "duplicate"),
    ("[]", "object"), ("{}", "missing"), ("{", "Expecting"),
])
def test_invalid_json(tmp_path, contents, match):
    path = tmp_path / "registers.json"
    path.write_text(contents)
    with pytest.raises(ValueError, match=match):
        resampler.load_registers(path)


def test_unknown_and_missing_fields(tmp_path):
    values = registers()
    values["owdith"] = values.pop("owidth")
    path = tmp_path / "registers.json"
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="unknown registers"):
        resampler.load_registers(path)
    with pytest.raises(ValueError, match="missing registers"):
        resampler.crop_and_resize(pattern("ramp", 16, 16), values)


def run_cli(input_path, config_path, output_path):
    return subprocess.run(
        [sys.executable, "-m", "C55.resampler", "--input", str(input_path),
         "--registers", str(config_path), "--output", str(output_path)],
        cwd=ROOT, capture_output=True, text=True,
    )


def test_cli_example(tmp_path):
    image = pattern("ramp", 488, 656)
    source, output = tmp_path / "input.png", tmp_path / "output.png"
    Image.fromarray(image).save(source)
    result = run_cli(source, C55 / "registers.example.json", output)
    assert result.returncode == 0, result.stderr
    with Image.open(output) as saved:
        assert saved.format == "PNG" and saved.mode == "RGB" and saved.size == (320, 240)
        expected = resampler.crop_and_resize(image, resampler.load_registers(C55 / "registers.example.json"))
        np.testing.assert_array_equal(np.array(saved), expected)


@pytest.mark.parametrize("failure", ["grayscale", "short_frame", "corrupt_file", "invalid_config"])
def test_cli_failures_do_not_write_output(tmp_path, failure):
    source, output, config = tmp_path / "input.png", tmp_path / "output.png", tmp_path / "registers.json"
    values = registers(8, 8, 4, 4)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    if failure == "grayscale":
        image = image[:, :, 0]
    if failure == "short_frame":
        image = image[:7]
    if failure == "invalid_config":
        values["hfilt_tinc"] = 0
    Image.fromarray(image).save(source)
    if failure == "corrupt_file":
        source.write_bytes(b"not an image")
    config.write_text(json.dumps(values))
    result = run_cli(source, config, output)
    assert result.returncode != 0 and "error:" in result.stderr
    assert not output.exists()


def test_generated_references_match_catalogue_and_example():
    from openpyxl import load_workbook

    catalogue = json.loads((DOCS / "register_catalogue.json").read_text())
    example = resampler.load_registers(C55 / "registers.example.json")
    assert {row["name"]: row["example"] for row in catalogue["registers"]} == example
    assert set(example) == set(resampler.REGISTER_NAMES)
    assert (DOCS / "registers.md").read_text() == render_markdown(catalogue)
    workbook = load_workbook(DOCS / "registers.xlsx", read_only=True)
    try:
        for name, rows, columns in (
            ("Registers", catalogue["registers"], REGISTER_COLUMNS),
            ("LUT layout", catalogue["lut"], LUT_COLUMNS),
            ("Notes", catalogue["notes"], (("topic", "Topic"), ("text", "Description"))),
        ):
            actual = list(workbook[name].values)
            assert actual[0] == tuple(label for _, label in columns)
            assert actual[1:] == [tuple(row[key] for key, _ in columns) for row in rows]
    finally:
        workbook.close()
