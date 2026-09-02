"""Shared runner for quantized uint8 bilinear agreement experiments."""

from __future__ import annotations

import argparse
import csv
import gc
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ocr_bench.resample import ResizeResult, opencv_bilinear
from ocr_bench.util import pixel_sha256, save_deterministic_png

from experiments.replicate_bilinear.compare_bilinear_floating_psnr import (
    find_patterns,
    load_rgb,
    parse_ratio_groups,
    pattern_group,
    ratio_slug,
    resize_rgb_planes,
    save_difference,
)


ResizeImplementation = Callable[[np.ndarray, float, float | None], ResizeResult]
RatioPair = tuple[float, float]


def compare_pixels(
    pattern: str,
    source: np.ndarray,
    ratio_w: float,
    ratio_h: float,
    output_dir: Path,
    write_images: bool,
    *,
    candidate_resize: ResizeImplementation,
    candidate_name: str,
) -> dict[str, object]:
    opencv = resize_rgb_planes(source, ratio_w, opencv_bilinear, ratio_h)
    candidate = resize_rgb_planes(source, ratio_w, candidate_resize, ratio_h)
    opencv_pixels = opencv.pixels
    candidate_pixels = candidate.pixels
    if candidate_pixels.shape != opencv_pixels.shape:
        raise ValueError(
            f"quantized output shape mismatch: OpenCV {opencv_pixels.shape}, "
            f"{candidate_name} {candidate_pixels.shape}"
        )

    error = candidate_pixels.astype(np.int16) - opencv_pixels.astype(np.int16)
    abs_error = np.abs(error)
    different_pixels = int(np.count_nonzero(error))
    ratio_name = ratio_slug(ratio_w, ratio_h)
    ratio_label = str(ratio_w) if ratio_w == ratio_h else f"{ratio_w}x{ratio_h}"

    if write_images:
        save_deterministic_png(
            output_dir / "resized" / "opencv_bilinear" / pattern / f"{ratio_name}.png",
            opencv_pixels,
        )
        save_deterministic_png(
            output_dir / "resized" / candidate_name / pattern / f"{ratio_name}.png",
            candidate_pixels,
        )
        save_difference(
            output_dir / "diff_abs" / pattern / f"{ratio_name}.png",
            error,
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
        "candidate": candidate_name,
        "candidate_kernel": candidate.metadata["kernel"],
        "coordinate_mapping": candidate.metadata["coordinate_mapping"],
        "value_domain": "uint8_quantized",
        "opencv_dtype": str(opencv_pixels.dtype),
        "candidate_dtype": str(candidate_pixels.dtype),
        "psnr_db": float(cv2.PSNR(opencv_pixels, candidate_pixels, 255.0)),
        "max_abs_error": int(abs_error.max()),
        "different_pixels": different_pixels,
        "pixel_count": int(error.size),
        "different_pixel_fraction": float(different_pixels / error.size),
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
        "uint8_quantized"
    }:
        raise ValueError("metrics CSV is not a quantized uint8 comparison; rerun it")
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
        f"Exact quantized image matches: {exact} / {len(results)}",
        f"Overall mean PSNR: {results['psnr_db'].mean():.6g} dB",
        f"Overall minimum PSNR: {results['psnr_db'].min():.6g} dB",
        "Compared values: quantized uint8 outputs",
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
    ratios: Sequence[RatioPair] | None = None,
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

    selected_ratios = parse_ratio_groups(ratios)
    total = len(patterns) * len(selected_ratios)
    done = 0
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer: csv.DictWriter[str] | None = None
        for image_path in patterns:
            source = load_rgb(image_path)
            pattern = image_path.stem
            for ratio_w, ratio_h in selected_ratios:
                done += 1
                ratio_label = (
                    str(ratio_w) if ratio_w == ratio_h else f"{ratio_w}x{ratio_h}"
                )
                print(
                    f"[{done}/{total}] {image_path.name} ratio={ratio_label}",
                    flush=True,
                )
                record = compare_pixels(
                    pattern,
                    source,
                    ratio_w,
                    ratio_h,
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


def parse_args(
    description: str,
    pattern_dir: Path,
    output_dir: Path,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--pattern-dir", type=Path, default=pattern_dir)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
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
