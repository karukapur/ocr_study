"""Write axis CSV traces and a JSON summary of increment drift and phase error."""

import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path

from .coordinates import COORDINATE_SCALE, INCREMENT_SCALE, PHASES, axis_coordinates
from .resampler import _validate_geometry, load_registers


def analyze_axis(source_length, output_length, increment, interpolation, trace_path):
    exact_step = Fraction(source_length, output_length)
    initial = (exact_step - 1) / 2 if interpolation else Fraction(0)
    max_drift = max_phase_error = max_sampling_error = Fraction(0)
    phase_mismatches = position_mismatches = 0
    final_drift = Fraction(0)
    with open(trace_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "output_index", "coordinate_q21", "exact_coordinate_numerator",
            "exact_coordinate_denominator", "base", "phase", "exact_base", "exact_phase",
            "coordinate_drift_pixels", "phase_quantization_error_pixels", "sampling_error_pixels",
        ])
        for index, (coordinate, base, phase) in enumerate(axis_coordinates(
            output_length, increment, interpolation
        )):
            exact = initial + index * exact_step
            programmed = Fraction(coordinate, COORDINATE_SCALE)
            sampled = base + Fraction(phase, PHASES)
            # Independent exact-rational nearest-even reference, including carry.
            exact_base, exact_phase = divmod(round(exact * PHASES), PHASES)
            final_drift = programmed - exact
            phase_error = sampled - programmed
            sampling_error = sampled - exact
            max_drift = max(max_drift, abs(final_drift))
            max_phase_error = max(max_phase_error, abs(phase_error))
            max_sampling_error = max(max_sampling_error, abs(sampling_error))
            phase_mismatches += phase != exact_phase
            position_mismatches += (base, phase) != (exact_base, exact_phase)
            writer.writerow([
                index, coordinate, exact.numerator, exact.denominator, base, phase,
                exact_base, exact_phase, float(final_drift), float(phase_error), float(sampling_error),
            ])
    return {
        "source_length": source_length,
        "output_length": output_length,
        "increment_q20": increment,
        "increment_error_pixels_per_output": float(Fraction(increment, INCREMENT_SCALE) - exact_step),
        "max_abs_coordinate_drift_pixels": float(max_drift),
        "final_coordinate_drift_pixels": float(final_drift),
        "max_abs_phase_quantization_error_pixels": float(max_phase_error),
        "max_abs_sampling_error_pixels": float(max_sampling_error),
        "phase_index_mismatches": phase_mismatches,
        "sample_position_mismatches": position_mismatches,
    }


def write_report(registers, output_dir):
    # No image is consumed: validate geometry against the minimum enclosing
    # frame implied by the configuration, not a claim about any arriving frame.
    active = registers["Pipeline: Bypass crop"] == 0 and registers["Crop: Enable crop"] == 1
    width = registers["width_in"] + (max(0, registers["Crop: start x"]) if active else 0)
    height = registers["height_in"] + (max(0, registers["Crop: start y"]) if active else 0)
    _validate_geometry((height, width, 3), registers)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "mapping": "shift" if registers["Resampler: interpolation"] else "original",
        "phases": PHASES,
        "phase_rounding": "nearest-even",
        "note": "Phase diagnostics describe LUT sampling; nearest/average use unquantized coordinates.",
    }
    for axis, source, output, increment in (
        ("x", "width_in", "owidth", "hfilt_tinc"),
        ("y", "height_in", "oheight", "vfilt_tinc"),
    ):
        report[axis] = analyze_axis(
            registers[source], registers[output], registers[increment],
            registers["Resampler: interpolation"], output_dir / f"{axis}.csv",
        )
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registers", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        write_report(load_registers(args.registers), args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
