"""Compare OpenCV bilinear with the float64 implementation from issue #25018."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from compare_bilinear import PATTERN_DIR
from compare_bilinear_floating_psnr import run_comparison
from ocr_bench.resample import ResizeResult, output_length


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results_floating_psnr_github"
DEFAULT_CSV_NAME = "bilinear_floating_psnr_github_metrics.csv"


def getfloorAndCeil(z: np.ndarray | float, maxValue: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the neighboring indices used by the GitHub issue implementation."""
    z1 = np.floor(z).astype(int)
    z1 = np.minimum(maxValue, z1)
    z2 = np.minimum(maxValue, z + 1).astype(int)
    return z1, z2


def weighted_sum(z1: np.ndarray, z2: np.ndarray, weight: np.ndarray | float) -> np.ndarray:
    return (1 - weight) * z1 + weight * z2


def resize(img: np.ndarray, size: tuple[int, int, int] | None = None) -> np.ndarray:
    """Float64 half-pixel bilinear resize posted in OpenCV issue #25018."""
    orig_size = img.shape
    if size is None:
        size = (orig_size[0] // 2, orig_size[1] // 2, orig_size[2])

    img_output = np.zeros(size)
    delta_height = orig_size[0] / size[0]
    delta_width = orig_size[1] / size[1]

    for i in range(size[0]):
        height_i = (i + 0.5) * delta_height - 0.5
        if height_i < 0:
            height_i = 0
        alpha_height = height_i % 1
        height_1, height_2 = getfloorAndCeil(height_i, orig_size[0] - 1)

        arr_j = np.arange(size[1])
        width_j = (arr_j + 0.5) * delta_width - 0.5
        width_j[width_j < 0] = 0
        alpha_width = width_j % 1
        width_1, width_2 = getfloorAndCeil(width_j, orig_size[1] - 1)
        alpha_width_extra_dim = alpha_width[:, np.newaxis]

        pixel_1 = weighted_sum(
            img[height_1, width_1, :],
            img[height_1, width_2, :],
            alpha_width_extra_dim,
        )
        pixel_2 = weighted_sum(
            img[height_2, width_1, :],
            img[height_2, width_2, :],
            alpha_width_extra_dim,
        )
        img_output[i, arr_j, :] = weighted_sum(pixel_1, pixel_2, alpha_height)

    return img_output


def github_floating_point_bilinear(image: np.ndarray, ratio: float) -> ResizeResult:
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("github_floating_point_bilinear expects a 2D uint8 image")
    destination_height = output_length(image.shape[0], ratio)
    destination_width = output_length(image.shape[1], ratio)
    floating = resize(
        image[:, :, np.newaxis],
        (destination_height, destination_width, 1),
    )[:, :, 0]
    pixels = np.clip(np.floor(floating + 0.5), 0, 255).astype(np.uint8)
    return ResizeResult(
        pixels=pixels,
        floating_pixels=floating,
        metadata={
            "interpolation": "bilinear float64 from OpenCV issue #25018",
            "kernel": "bilinear_2x2_float64_github_issue_25018",
            "coordinate_mapping": "shift",
            "boundary": "clamp",
            "coefficient_precision": "float64",
            "accumulator_precision": "float64",
            "pixel_rounding": "round_half_up_then_clip_uint8",
            "actual_scale_x": image.shape[1] / destination_width,
            "actual_scale_y": image.shape[0] / destination_height,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern-dir", type=Path, default=PATTERN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--write-images",
        action="store_true",
        help="also write resized outputs and absolute-difference PNGs",
    )
    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="reuse an existing metrics CSV and only regenerate the summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_comparison(
        pattern_dir=args.pattern_dir,
        output_dir=args.output_dir,
        write_images=args.write_images,
        reports_only=args.reports_only,
        candidate_resize=github_floating_point_bilinear,
        candidate_name="github_floating_point_bilinear",
        csv_name=DEFAULT_CSV_NAME,
        report_title="OpenCV vs GitHub Float64 Bilinear PSNR",
    )


if __name__ == "__main__":
    main()
