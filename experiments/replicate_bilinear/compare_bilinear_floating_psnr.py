"""Compare OpenCV bilinear with the project floating-point implementation."""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compare_bilinear import (  # noqa: E402
    PATTERN_DIR,
    RATIOS,
    find_patterns,
    load_rgb,
    pattern_group,
    ratio_slug,
    resize_rgb_planes,
    save_difference,
)
from ocr_bench.resample import (  # noqa: E402
    ResizeResult,
    floating_point_bilinear,
    opencv_bilinear,
)
from ocr_bench.util import pixel_sha256, save_deterministic_png  # noqa: E402


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results_floating_psnr"
DEFAULT_CSV_NAME = "bilinear_floating_psnr_metrics.csv"
ResizeImplementation = Callable[[np.ndarray, float], ResizeResult]


def compare_pixels(
    pattern: str,
    source: np.ndarray,
    ratio: float,
    output_dir: Path,
    write_images: bool,
    *,
    candidate_resize: ResizeImplementation = floating_point_bilinear,
    candidate_name: str = "floating_point_bilinear",
) -> dict[str, object]:
    opencv = resize_rgb_planes(source, ratio, opencv_bilinear)
    candidate = resize_rgb_planes(source, ratio, candidate_resize)
    opencv_pixels = opencv.pixels
    candidate_pixels = candidate.pixels
    opencv_floating = opencv.floating_pixels
    candidate_floating = candidate.floating_pixels
    if opencv_floating is None:
        raise ValueError("opencv_bilinear did not return floating pixels")
    if candidate_floating is None:
        raise ValueError(f"{candidate_name} did not return floating pixels")
    if candidate_pixels.shape != opencv_pixels.shape:
        raise ValueError(
            f"quantized output shape mismatch: OpenCV {opencv_pixels.shape}, "
            f"{candidate_name} {candidate_pixels.shape}"
        )
    if candidate_floating.shape != opencv_floating.shape:
        raise ValueError(
            f"floating output shape mismatch: OpenCV {opencv_floating.shape}, "
            f"{candidate_name} {candidate_floating.shape}"
        )

    opencv_for_psnr = opencv_floating.astype(np.float64, copy=False)
    candidate_for_psnr = candidate_floating.astype(np.float64, copy=False)
    error = candidate_for_psnr - opencv_for_psnr
    abs_error = np.abs(error)
    ratio_name = ratio_slug(ratio)

    if write_images:
        save_deterministic_png(
            output_dir / "resized" / "opencv_bilinear" / pattern / f"{ratio_name}.png",
            opencv_pixels,
        )
        save_deterministic_png(
            output_dir / "resized" / candidate_name / pattern / f"{ratio_name}.png",
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
        "ratio": ratio,
        "ratio_slug": ratio_name,
        "source_width": source.shape[1],
        "source_height": source.shape[0],
        "output_width": opencv_pixels.shape[1],
        "output_height": opencv_pixels.shape[0],
        "candidate": candidate_name,
        "candidate_kernel": candidate.metadata["kernel"],
        "coordinate_mapping": candidate.metadata["coordinate_mapping"],
        "value_domain": "floating_pre_quantization",
        "opencv_floating_dtype": str(opencv_floating.dtype),
        "candidate_floating_dtype": str(candidate_floating.dtype),
        "psnr_db": float(cv2.PSNR(opencv_for_psnr, candidate_for_psnr, 255.0)),
        "max_abs_error": float(abs_error.max()),
        "different_pixels": int(np.count_nonzero(error)),
        "pixel_count": int(error.size),
        "different_pixel_fraction": float(np.count_nonzero(error) / error.size),
        "opencv_sha256": pixel_sha256(opencv_pixels),
        "candidate_sha256": pixel_sha256(candidate_pixels),
    }


def _markdown_table(frame: pd.DataFrame, floatfmt: str = ".6g") -> str:
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


def write_summary(results: pd.DataFrame, output_dir: Path, title: str) -> None:
    if "value_domain" not in results or set(results["value_domain"]) != {
        "floating_pre_quantization"
    }:
        raise ValueError(
            "metrics CSV is not a floating pre-quantization comparison; rerun it"
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
        f"# {title}",
        "",
        f"Patterns: {results['pattern'].nunique()}",
        f"Ratios: {', '.join(str(value) for value in sorted(results['ratio'].unique()))}",
        f"Comparisons: {len(results)}",
        f"Exact image matches: {exact} / {len(results)}",
        f"Overall mean PSNR: {results['psnr_db'].mean():.6g} dB",
        f"Overall minimum PSNR: {results['psnr_db'].min():.6g} dB",
        "Compared values: floating outputs before uint8 quantization",
        "PSNR implementation: OpenCV cv2.PSNR with R=255",
        "Coordinate mapping: half-pixel shift",
        "",
        "## By Ratio",
        "",
        _markdown_table(by_ratio),
        "",
        "## Lowest PSNR Cases",
        "",
        _markdown_table(
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


def run_comparison(
    *,
    pattern_dir: Path,
    output_dir: Path,
    write_images: bool,
    reports_only: bool,
    candidate_resize: ResizeImplementation,
    candidate_name: str,
    csv_name: str,
    report_title: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / csv_name
    if reports_only:
        if not csv_path.is_file():
            raise FileNotFoundError(f"{csv_path} does not exist")
        results = pd.read_csv(csv_path)
        write_summary(results, output_dir, report_title)
        print(f"Wrote summary to {output_dir / 'summary.md'}")
        return

    patterns = find_patterns(pattern_dir)
    if not patterns:
        raise FileNotFoundError(
            f"no PNG, TIFF, or SVG test patterns found in {pattern_dir}"
        )

    total = len(patterns) * len(RATIOS)
    done = 0
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer: csv.DictWriter[str] | None = None
        for image_path in patterns:
            source = load_rgb(image_path)
            pattern = image_path.stem
            for ratio in RATIOS:
                done += 1
                print(f"[{done}/{total}] {image_path.name} ratio={ratio}", flush=True)
                record = compare_pixels(
                    pattern,
                    source,
                    ratio,
                    output_dir,
                    write_images,
                    candidate_resize=candidate_resize,
                    candidate_name=candidate_name,
                )
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(record))
                    writer.writeheader()
                writer.writerow(record)
                del record
                gc.collect()
            del source
            gc.collect()

    results = pd.read_csv(csv_path)
    write_summary(results, output_dir, report_title)
    print(f"Wrote metrics to {csv_path}")
    print(f"Wrote summary to {output_dir / 'summary.md'}")


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
        candidate_resize=lambda image, ratio: floating_point_bilinear(
            image, ratio, interpolation="shift"
        ),
        candidate_name="floating_point_bilinear",
        csv_name=DEFAULT_CSV_NAME,
        report_title="OpenCV vs Floating-Point Bilinear PSNR",
    )


if __name__ == "__main__":
    main()
