from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from ocr_bench.resample import (
    OPENCV_LINEAR_COEF_BITS,
    OPENCV_LINEAR_COEF_SCALE,
    _opencv_linear_terms,
    _round_float32,
)


OUTPUT_DIR = Path(__file__).resolve().parent


def _coordinate(destination: int, scale: float, interpolation: str) -> np.float32:
    return np.float32(
        destination * scale
        if interpolation == "original"
        else (destination + 0.5) * scale - 0.5
    )


def write_phase_lut(output_path: Path) -> None:
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "phase",
                "coefficient_bits",
                "coefficient_scale",
                "fraction",
                "left_coefficient",
                "right_coefficient",
                "coefficient_sum",
                "left_weight",
                "right_weight",
            ]
        )
        for phase in range(OPENCV_LINEAR_COEF_SCALE + 1):
            fraction = np.float32(phase / OPENCV_LINEAR_COEF_SCALE)
            left = _round_float32(
                np.float32((np.float32(1.0) - fraction) * OPENCV_LINEAR_COEF_SCALE)
            )
            right = _round_float32(np.float32(fraction * OPENCV_LINEAR_COEF_SCALE))
            writer.writerow(
                [
                    phase,
                    OPENCV_LINEAR_COEF_BITS,
                    OPENCV_LINEAR_COEF_SCALE,
                    f"{float(fraction):.10f}",
                    left,
                    right,
                    left + right,
                    f"{left / OPENCV_LINEAR_COEF_SCALE:.10f}",
                    f"{right / OPENCV_LINEAR_COEF_SCALE:.10f}",
                ]
            )


def write_length_terms(
    output_path: Path,
    source_length: int,
    destination_length: int,
    interpolation: str,
    clamp_edge_coefficients: bool,
) -> None:
    offsets, coefficients, minimum, maximum = _opencv_linear_terms(
        source_length,
        destination_length,
        interpolation=interpolation,
        clamp_edge_coefficients=clamp_edge_coefficients,
    )
    scale = source_length / destination_length
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "destination",
                "source_length",
                "destination_length",
                "scale",
                "coordinate",
                "source_offset",
                "right_offset",
                "fraction",
                "left_coefficient",
                "right_coefficient",
                "coefficient_sum",
                "left_weight",
                "right_weight",
                "valid_min_destination",
                "valid_max_destination_exclusive",
                "interpolation",
                "clamp_edge_coefficients",
            ]
        )
        for destination in range(destination_length):
            coordinate = _coordinate(destination, scale, interpolation)
            source = math.floor(float(coordinate))
            fraction = np.float32(coordinate - source)
            left = int(coefficients[destination, 0])
            right = int(coefficients[destination, 1])
            writer.writerow(
                [
                    destination,
                    source_length,
                    destination_length,
                    f"{scale:.12g}",
                    f"{float(coordinate):.10f}",
                    int(offsets[destination]),
                    min(int(offsets[destination]) + 1, source_length - 1),
                    f"{float(fraction):.10f}",
                    left,
                    right,
                    left + right,
                    f"{left / OPENCV_LINEAR_COEF_SCALE:.10f}",
                    f"{right / OPENCV_LINEAR_COEF_SCALE:.10f}",
                    minimum,
                    maximum,
                    interpolation,
                    clamp_edge_coefficients,
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed-point bilinear Q11 coefficient CSV files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for generated CSV files.",
    )
    parser.add_argument("--source-length", type=int)
    parser.add_argument("--destination-length", type=int)
    parser.add_argument("--axis", choices=["x", "y"], default="x")
    parser.add_argument(
        "--interpolation",
        choices=["shift", "original"],
        default="shift",
    )
    parser.add_argument(
        "--no-clamp-edge-coefficients",
        action="store_true",
        help="Match the vertical fixed-point path, which leaves edge coefficients unclamped.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    phase_path = args.output_dir / "opencv_linear_q11_coefficients.csv"
    write_phase_lut(phase_path)
    print(f"wrote {phase_path}")

    if (args.source_length is None) ^ (args.destination_length is None):
        raise SystemExit("--source-length and --destination-length must be provided together")

    if args.source_length is not None and args.destination_length is not None:
        if args.source_length < 1 or args.destination_length < 1:
            raise SystemExit("source and destination lengths must be positive")
        clamp_edge_coefficients = not args.no_clamp_edge_coefficients
        clamp_label = "clamp" if clamp_edge_coefficients else "no_clamp"
        terms_path = (
            args.output_dir
            / (
                f"opencv_linear_terms_{args.axis}_{args.source_length}_to_"
                f"{args.destination_length}_{args.interpolation}_{clamp_label}.csv"
            )
        )
        write_length_terms(
            terms_path,
            args.source_length,
            args.destination_length,
            args.interpolation,
            clamp_edge_coefficients,
        )
        print(f"wrote {terms_path}")


if __name__ == "__main__":
    main()
