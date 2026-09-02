"""Compare quantized OpenCV and GitHub issue #25018 bilinear outputs.

Run from the repository root using the RATIOS configured in this file:
    .venv/bin/python experiments/replicate_bilinear/uint8/compare_bilinear_floating_psnr_github.py

Override RATIOS with one uniform ratio:
    .venv/bin/python experiments/replicate_bilinear/uint8/compare_bilinear_floating_psnr_github.py --ratio 2.0

Override RATIOS with width and height ratios:
    .venv/bin/python experiments/replicate_bilinear/uint8/compare_bilinear_floating_psnr_github.py --ratio 2.0 1.5
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from collections.abc import Callable, Sequence
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_bench.resample import ResizeResult, opencv_bilinear, output_length  # noqa: E402
from ocr_bench.util import pixel_sha256, save_deterministic_png  # noqa: E402


# Used when no --ratio flags are supplied. A float applies to both dimensions;
# a (width_ratio, height_ratio) tuple scales the dimensions independently.
RATIOS = (
    1.133333333333,
    1.4,
    1.666666666667,
    2.0,
    2.333333333333,
    2.6,
    2.625,
    3.0,
    (2.0, 1.5),  # width ratio, height ratio
)
PATTERN_DIR = Path(__file__).resolve().parent.parent / "test_patterns"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results_github"
DEFAULT_CSV_NAME = "bilinear_floating_uint8_psnr_github_metrics.csv"
PATTERN_SUFFIXES = frozenset({".png", ".tif", ".tiff", ".svg"})
ResizeImplementation = Callable[[np.ndarray, float, float | None], ResizeResult]
RatioPair = tuple[float, float]


def _ratio_value_slug(ratio: float) -> str:
    return f"{ratio:.12f}".rstrip("0").rstrip(".").replace(".", "p")


def ratio_slug(ratio_w: float, ratio_h: float | None = None) -> str:
    if ratio_h is None or ratio_h == ratio_w:
        return f"r_{_ratio_value_slug(ratio_w)}"
    return f"rw_{_ratio_value_slug(ratio_w)}_rh_{_ratio_value_slug(ratio_h)}"


def find_patterns(pattern_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in pattern_dir.iterdir()
            if path.is_file() and path.suffix.lower() in PATTERN_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )


def load_rgb(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".svg":
        try:
            import cairosvg
        except ImportError as error:
            raise RuntimeError(
                "CairoSVG is required to read SVG test patterns; install the "
                "project dependencies"
            ) from error
        encoded = cairosvg.svg2png(url=str(path), background_color="#ffffff")
        with Image.open(BytesIO(encoded)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def resize_rgb_planes(
    image: np.ndarray,
    ratio_w: float,
    implementation: ResizeImplementation,
    ratio_h: float | None = None,
) -> ResizeResult:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("resize_rgb_planes expects an RGB uint8 image")

    pixels: np.ndarray | None = None
    floating_pixels: np.ndarray | None = None
    metadata: dict[str, object] | None = None
    for channel in range(3):
        result = implementation(image[:, :, channel], ratio_w, ratio_h)
        if pixels is None:
            pixels = np.empty((*result.pixels.shape, 3), dtype=np.uint8)
            metadata = result.metadata
            if result.floating_pixels is not None:
                floating_pixels = np.empty(
                    (*result.floating_pixels.shape, 3),
                    dtype=result.floating_pixels.dtype,
                )
        pixels[:, :, channel] = result.pixels
        if (result.floating_pixels is None) != (floating_pixels is None):
            raise ValueError("resize implementation returned inconsistent RGB planes")
        if floating_pixels is not None and result.floating_pixels is not None:
            floating_pixels[:, :, channel] = result.floating_pixels

    if pixels is None or metadata is None:
        raise RuntimeError("RGB resize produced no planes")
    return ResizeResult(
        pixels=pixels,
        metadata=metadata,
        floating_pixels=floating_pixels,
    )


def pattern_group(name: str) -> str:
    prefixes = (
        ("constant_", "constant"),
        ("ramp_", "ramp"),
        ("plane_", "plane"),
        ("impulse_", "impulse"),
        (("step_", "line_"), "edge"),
        ("checkerboard_", "checkerboard"),
        ("sine_bandlimited_", "sine_bandlimited"),
        ("sine_stress_", "sine_stress"),
        ("sine_", "sine_legacy"),
        ("zone_plate_bandlimited_", "zone_plate_bandlimited"),
        ("zone_plate_stress_", "zone_plate_stress"),
        ("zone_plate_", "zone_plate_legacy"),
        ("noise_", "noise"),
        ("tile_2x2_", "tile_2x2"),
    )
    for prefix, group in prefixes:
        if name.startswith(prefix):
            return group
    return "other"


def save_difference(path: Path, error: np.ndarray) -> None:
    abs_error = np.abs(error).clip(0, 255).astype(np.uint8)
    save_deterministic_png(path, abs_error)


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


def github_floating_point_bilinear(
    image: np.ndarray, ratio_w: float, ratio_h: float | None = None
) -> ResizeResult:
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("github_floating_point_bilinear expects a 2D uint8 image")
    ratio_h = ratio_w if ratio_h is None else ratio_h
    destination_height = output_length(image.shape[0], ratio_h)
    destination_width = output_length(image.shape[1], ratio_w)
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


def compare_pixels(
    pattern: str,
    source: np.ndarray,
    ratio_w: float,
    ratio_h: float,
    output_dir: Path,
    write_images: bool,
) -> dict[str, object]:
    opencv = resize_rgb_planes(source, ratio_w, opencv_bilinear, ratio_h)
    candidate = resize_rgb_planes(
        source, ratio_w, github_floating_point_bilinear, ratio_h
    )
    opencv_pixels = opencv.pixels
    candidate_pixels = candidate.pixels
    if candidate_pixels.shape != opencv_pixels.shape:
        raise ValueError(
            f"quantized output shape mismatch: OpenCV {opencv_pixels.shape}, "
            f"GitHub implementation {candidate_pixels.shape}"
        )
    error = candidate_pixels.astype(np.int16) - opencv_pixels.astype(np.int16)
    abs_error = np.abs(error)
    ratio_name = ratio_slug(ratio_w, ratio_h)
    ratio_label = str(ratio_w) if ratio_w == ratio_h else f"{ratio_w}x{ratio_h}"

    if write_images:
        save_deterministic_png(
            output_dir / "resized" / "opencv_bilinear" / pattern / f"{ratio_name}.png",
            opencv_pixels,
        )
        save_deterministic_png(
            output_dir
            / "resized"
            / "github_floating_point_bilinear"
            / pattern
            / f"{ratio_name}.png",
            candidate_pixels,
        )
        quantized_error = (
            candidate_pixels.astype(np.int16) - opencv_pixels.astype(np.int16)
        )
        save_difference(
            output_dir / "diff_abs" / pattern / f"{ratio_name}.png",
            quantized_error,
        )

    return {
        "pattern": pattern,
        "group": pattern_group(pattern),
        "ratio": ratio_label,
        "ratio_width": ratio_w,
        "ratio_height": ratio_h,
        "ratio_slug": ratio_name,
        "source_width": source.shape[1],
        "source_height": source.shape[0],
        "output_width": opencv_pixels.shape[1],
        "output_height": opencv_pixels.shape[0],
        "candidate": "github_floating_point_bilinear",
        "candidate_kernel": candidate.metadata["kernel"],
        "coordinate_mapping": candidate.metadata["coordinate_mapping"],
        "value_domain": "uint8_quantized",
        "opencv_dtype": str(opencv_pixels.dtype),
        "candidate_dtype": str(candidate_pixels.dtype),
        "psnr_db": float(cv2.PSNR(opencv_pixels, candidate_pixels, 255.0)),
        "max_abs_error": int(abs_error.max()),
        "different_pixels": int(np.count_nonzero(error)),
        "pixel_count": int(error.size),
        "different_pixel_fraction": float(np.count_nonzero(error) / error.size),
        "opencv_sha256": pixel_sha256(opencv_pixels),
        "candidate_sha256": pixel_sha256(candidate_pixels),
    }


def markdown_table(frame: pd.DataFrame, floatfmt: str = ".6g") -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = [
            format(row[column], floatfmt)
            if isinstance(row[column], float)
            else str(row[column])
            for column in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary(results: pd.DataFrame, output_dir: Path) -> None:
    if "value_domain" not in results or set(results["value_domain"]) != {
        "uint8_quantized"
    }:
        raise ValueError(
            "metrics CSV is not a quantized uint8 comparison; rerun it"
        )
    by_ratio = (
        results.groupby("ratio", as_index=False)
        .agg(
            mean_psnr_db=("psnr_db", "mean"),
            min_psnr_db=("psnr_db", "min"),
            max_abs_error=("max_abs_error", "max"),
            mean_different_pixel_fraction=("different_pixel_fraction", "mean"),
        )
        .sort_values("ratio")
    )
    worst = results.sort_values(
        ["psnr_db", "different_pixel_fraction"], ascending=[True, False]
    ).head(20)
    exact = int((results["different_pixels"] == 0).sum())
    rows = [
        "# OpenCV vs GitHub Float64 Bilinear Quantized uint8 PSNR",
        "",
        f"Patterns: {results['pattern'].nunique()}",
        f"Ratios: {', '.join(str(value) for value in sorted(results['ratio'].unique()))}",
        f"Comparisons: {len(results)}",
        f"Exact quantized image matches: {exact} / {len(results)}",
        f"Overall mean PSNR: {results['psnr_db'].mean():.6g} dB",
        f"Overall minimum PSNR: {results['psnr_db'].min():.6g} dB",
        "Compared values: quantized uint8 outputs",
        "PSNR implementation: OpenCV cv2.PSNR with R=255",
        "Coordinate mapping: half-pixel shift",
        "",
        "## By Ratio",
        "",
        markdown_table(by_ratio),
        "",
        "## Lowest PSNR Cases",
        "",
        markdown_table(
            worst[
                [
                    "pattern",
                    "group",
                    "ratio",
                    "candidate_kernel",
                    "psnr_db",
                    "max_abs_error",
                    "different_pixel_fraction",
                ]
            ]
        ),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(rows), encoding="utf-8")


def parse_ratio_groups(
    groups: Sequence[float | Sequence[float]] | None,
) -> tuple[RatioPair, ...]:
    ratio_specs = RATIOS if groups is None else groups
    ratios: list[RatioPair] = []
    for spec in ratio_specs:
        if isinstance(spec, (int, float)):
            ratio_w = ratio_h = float(spec)
        elif len(spec) == 1:
            ratio_w = ratio_h = float(spec[0])
        elif len(spec) == 2:
            ratio_w, ratio_h = (float(spec[0]), float(spec[1]))
        else:
            raise ValueError("each ratio accepts either WIDTH or WIDTH HEIGHT")
        if not np.isfinite(ratio_w) or not np.isfinite(ratio_h):
            raise ValueError("ratios must be finite")
        if ratio_w <= 0.0 or ratio_h <= 0.0:
            raise ValueError("ratios must be positive")
        ratios.append((ratio_w, ratio_h))
    return tuple(ratios)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--ratio",
        type=float,
        nargs="+",
        action="append",
        metavar="RATIO",
        help=(
            "ratio WIDTH [HEIGHT]; one value applies to both dimensions. "
            "Repeat --ratio to run multiple ratio configurations"
        ),
    )
    args = parser.parse_args(argv)
    try:
        args.ratios = parse_ratio_groups(args.ratio)
    except ValueError as error:
        parser.error(str(error))
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / DEFAULT_CSV_NAME
    if args.reports_only:
        if not csv_path.is_file():
            raise FileNotFoundError(f"{csv_path} does not exist")
        write_summary(pd.read_csv(csv_path), args.output_dir)
        print(f"Wrote summary to {args.output_dir / 'summary.md'}")
        return

    patterns = find_patterns(args.pattern_dir)
    if not patterns:
        raise FileNotFoundError(
            f"no PNG, TIFF, or SVG test patterns found in {args.pattern_dir}"
        )
    total = len(patterns) * len(args.ratios)
    done = 0
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer: csv.DictWriter[str] | None = None
        for image_path in patterns:
            source = load_rgb(image_path)
            for ratio_w, ratio_h in args.ratios:
                done += 1
                ratio_label = (
                    str(ratio_w) if ratio_w == ratio_h else f"{ratio_w}x{ratio_h}"
                )
                print(
                    f"[{done}/{total}] {image_path.name} ratio={ratio_label}",
                    flush=True,
                )
                record = compare_pixels(
                    image_path.stem,
                    source,
                    ratio_w,
                    ratio_h,
                    args.output_dir,
                    args.write_images,
                )
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(record))
                    writer.writeheader()
                writer.writerow(record)
                del record
                gc.collect()
            del source
            gc.collect()

    write_summary(pd.read_csv(csv_path), args.output_dir)
    print(f"Wrote metrics to {csv_path}")
    print(f"Wrote summary to {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
