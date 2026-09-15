import copy
import csv
from fractions import Fraction
import json
import subprocess
import sys
import weakref

import numpy as np
import pytest

from C55 import resampler
from C55.analyze_phase_error import analyze_axis, write_report
from C55.coordinates import axis_coordinates, quantize_coordinate, round_nearest_even
from C55.generate_coefficients import bilinear_bank, load_coefficients, validate_coefficients
from C55.tests.test_resampler import C55, ROOT, pattern, rational_reference, registers


def test_two_point_six_two_five_sequence():
    terms = list(axis_coordinates(8, 2752512, 0))
    assert [(base, phase) for _, base, phase in terms] == [
        (0, 0), (2, 20), (5, 8), (7, 28), (10, 16), (13, 4), (15, 24), (18, 12),
    ]
    assert [Fraction(coordinate, 1 << 21) for coordinate, _, _ in terms] == [
        Fraction(index * 21, 8) for index in range(8)
    ]


@pytest.mark.parametrize("numerator,expected", [(1, 0), (3, 2), (5, 2), (-1, 0), (-3, -2)])
def test_signed_nearest_even(numerator, expected):
    assert round_nearest_even(numerator, 2) == expected


@pytest.mark.parametrize("phase", range(32))
def test_phase_halfway_and_neighbors(phase):
    coordinate = (2 * phase + 1) * (1 << 15)
    expected = round(Fraction(2 * phase + 1, 2))
    assert quantize_coordinate(coordinate) == divmod(expected, 32)
    assert quantize_coordinate(coordinate - 1) == (0, phase)
    assert quantize_coordinate(coordinate + 1) == divmod(phase + 1, 32)


@pytest.mark.parametrize("mapping", [0, 1])
def test_accumulator_keeps_precision_and_half_bit(mapping):
    increment = (16 << 20) // 7  # Odd increment exercises the extra half-bit.
    assert increment % 2 == 1
    initial = increment - (1 << 20) if mapping else 0
    terms = list(axis_coordinates(10000, increment, mapping))
    for index, (coordinate, base, phase) in enumerate(terms):
        assert coordinate == initial + index * 2 * increment
        assert (base, phase) == divmod(round(Fraction(coordinate * 32, 1 << 21)), 32)


@pytest.mark.parametrize("method", [0, 1, 2])
@pytest.mark.parametrize("mapping", [0, 1])
def test_stream_random_independent_axes_and_reused_input(method, mapping, monkeypatch):
    image = np.random.default_rng(23).integers(0, 256, (168, 32, 3), dtype=np.uint8)
    values = registers(32, 168, 11, 64, method, mapping)
    consumed = []
    live_rows = []
    peak = 0
    horizontal = resampler._horizontal

    def record_horizontal(*args):
        nonlocal peak
        result = horizontal(*args)
        live_rows.append(weakref.ref(result))
        peak = max(peak, sum(ref() is not None for ref in live_rows))
        return result

    monkeypatch.setattr(resampler, "_horizontal", record_horizontal)

    def input_rows():
        storage = np.empty((32, 3), dtype=np.uint8)
        for index, row in enumerate(image):
            storage[:] = row
            consumed.append(index)
            yield storage

    output_rows = resampler.resize_rows(input_rows(), values, frame_shape=image.shape)
    first = next(output_rows)
    first_copy = first.copy()
    assert len(consumed) < len(image)
    remaining = list(output_rows)
    assert len(consumed) == len(image)
    assert peak <= 7
    np.testing.assert_array_equal(first, first_copy)
    assert all(not np.shares_memory(first, row) for row in remaining)
    actual = np.stack([first, *remaining])
    np.testing.assert_array_equal(actual, rational_reference(image, 11, 64, method, mapping))
    # A second frame must restart both axes.
    np.testing.assert_array_equal(actual, np.stack(list(
        resampler.resize_rows(iter(image), values, frame_shape=image.shape)
    )))


def test_setup_is_latched_before_consumption():
    image = pattern("ramp", 16, 16)
    values, coefficients = registers(), bilinear_bank()
    expected = rational_reference(image, 8, 8, 1, 1)
    rows = resampler.resize_rows(image, values, frame_shape=image.shape, coefficients=coefficients)
    values["hfilt_tinc"] = 0
    values["Resampler: method"] = 0
    coefficients["banks"]["0"][16][3] = -128
    np.testing.assert_array_equal(np.stack(list(rows)), expected)


@pytest.mark.parametrize("method", [0, 1, 2])
@pytest.mark.parametrize("mapping", [0, 1])
def test_stream_crop_with_trailing_input(method, mapping):
    image = pattern("ramp", 40, 48)
    values = registers(32, 24, 13, 9, method, mapping)
    values.update({"Crop: start x": 5, "Crop: start y": 3})
    expected = rational_reference(image[3:27, 5:37], 13, 9, method, mapping)
    actual = np.stack(list(resampler.resize_rows(image, values, frame_shape=image.shape)))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("failure", ["short", "extra", "width", "dtype", "channels", "none"])
def test_bad_stream_rows(failure):
    image = pattern("ramp", 8, 8)
    rows = list(image)
    if failure == "short":
        rows.pop()
    elif failure == "extra":
        rows.append(image[0])
    elif failure == "width":
        rows[0] = rows[0][:-1]
    elif failure == "dtype":
        rows[0] = rows[0].astype(np.uint16)
    elif failure == "channels":
        rows[0] = rows[0][:, :2]
    else:
        rows[0] = None
    with pytest.raises(ValueError, match="row"):
        list(resampler.resize_rows(rows, registers(8, 8, 4, 4), frame_shape=image.shape))


@pytest.mark.parametrize("failure", ["short", "malformed", "extra"])
def test_errors_after_last_output_are_not_hidden(failure):
    image = pattern("ramp", 24, 16)
    values = registers(8, 8, 4, 4, 1, 0)
    values.update({"Crop: start x": 2, "Crop: start y": 2})
    rows = list(image)
    if failure == "short":
        rows.pop()
    elif failure == "malformed":
        rows[-1] = None
    else:
        rows.append(image[0])
    output = resampler.resize_rows(rows, values, frame_shape=image.shape)
    for _ in range(4):
        assert next(output).shape == (4, 3)
    with pytest.raises(ValueError, match="row"):
        next(output)


@pytest.mark.parametrize("shape", [None, (8, 8), (0, 8, 3), (8, 8, 4), (8.0, 8, 3), (True, 8, 3)])
def test_invalid_frame_shape(shape):
    with pytest.raises(ValueError, match="frame_shape"):
        resampler.resize_rows([], registers(8, 8, 4, 4), frame_shape=shape)


def test_bilinear_final_rounding_ties_even_without_horizontal_rounding():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    # At half-pixel x/y: channel sums 2, 6, 3 give 0.5, 1.5, 0.75.
    image[0, 0] = (2, 6, 3)
    output = resampler.crop_and_resize(image, registers(8, 8, 4, 4, 1, 1))
    np.testing.assert_array_equal(output[0, 0], [0, 2, 1])


def test_pc_generator_and_explicit_cli_bank(tmp_path):
    bank_path = tmp_path / "bank.json"
    result = subprocess.run([sys.executable, "-m", "C55.generate_coefficients", "--output", str(bank_path)],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert bank_path.read_bytes() == (C55 / "bilinear_coefficients.json").read_bytes()
    assert load_coefficients(bank_path) == bilinear_bank()
    image = pattern("ramp", 168, 168)
    source, output = tmp_path / "input.png", tmp_path / "output.png"
    from PIL import Image
    Image.fromarray(image).save(source)
    command = [sys.executable, "-m", "C55.resampler", "--input", str(source),
               "--registers", str(C55 / "registers.example.json"), "--output", str(output),
               "--coefficients", str(bank_path)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with Image.open(output) as saved:
        np.testing.assert_array_equal(np.array(saved), rational_reference(image, 64, 64, 1, 0))
    bad_output = tmp_path / "bad.png"
    command[command.index(str(output))] = str(bad_output)
    bank_path.write_text('{"phases": 32, "phases": 32}')
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0 and "duplicate" in result.stderr
    assert not bad_output.exists()


@pytest.mark.parametrize("failure", ["fields", "phases", "bits", "offsets", "banks", "rows", "taps", "float", "bool", "range", "sum", "weights"])
def test_bad_coefficients(failure):
    bank = bilinear_bank()
    if failure == "fields":
        bank["extra"] = 0
    elif failure == "phases":
        bank["phases"] = 16
    elif failure == "bits":
        bank["fraction_bits"] = 20
    elif failure == "offsets":
        bank["tap_offsets"][3] = False
    elif failure == "banks":
        bank["banks"]["1"] = copy.deepcopy(bank["banks"]["0"])
    elif failure == "rows":
        bank["banks"]["0"].pop()
    elif failure == "taps":
        bank["banks"]["0"][0].pop()
    elif failure == "float":
        bank["banks"]["0"][0][0] = 0.0
    elif failure == "bool":
        bank["banks"]["0"][0][0] = False
    elif failure == "range":
        bank["banks"]["0"][0][0] = 128
    elif failure == "sum":
        bank["banks"]["0"][0][0] = 1
    else:
        bank["banks"]["0"][0] = [0, 0, 1, 63, 0, 0, 0]
    with pytest.raises(ValueError, match="coefficient"):
        validate_coefficients(bank)


@pytest.mark.parametrize("mapping", [0, 1])
def test_drift_report_matches_analytical_error(tmp_path, mapping):
    source, output = 8192, 3001
    increment = (source << 20) // output
    report = analyze_axis(source, output, increment, mapping, tmp_path / "trace.csv")
    step_error = Fraction(increment, 1 << 20) - Fraction(source, output)
    final_error = (output - 1 + Fraction(mapping, 2)) * step_error
    assert report["final_coordinate_drift_pixels"] == float(final_error)
    assert report["max_abs_coordinate_drift_pixels"] == float(abs(final_error))
    assert report["max_abs_phase_quantization_error_pixels"] <= 1 / 64
    assert report["sample_position_mismatches"] > 0
    with (tmp_path / "trace.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == output
    assert int(rows[-1]["coordinate_q21"]) == (increment - (1 << 20) if mapping else 0) + 2 * increment * (output - 1)


def test_example_diagnostics_and_cli(tmp_path):
    values = resampler.load_registers(C55 / "registers.example.json")
    report = write_report(values, tmp_path / "api")
    for axis in ("x", "y"):
        assert report[axis]["max_abs_coordinate_drift_pixels"] == 0
        assert report[axis]["max_abs_phase_quantization_error_pixels"] == 0
        assert report[axis]["phase_index_mismatches"] == 0
    result = subprocess.run([sys.executable, "-m", "C55.analyze_phase_error",
                             "--registers", str(C55 / "registers.example.json"),
                             "--output-dir", str(tmp_path / "cli")],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "cli" / "summary.json").read_text()) == report
