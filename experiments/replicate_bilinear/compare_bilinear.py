from __future__ import annotations

import argparse
import csv
import gc
import math
import sys
from collections.abc import Callable, Sequence
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_bench.resample import (  # noqa: E402
    ResizeResult,
    fixed_point_bilinear,
    opencv_bilinear,
)
from ocr_bench.util import pixel_sha256, save_deterministic_png  # noqa: E402


RATIOS = (
    1.133333333333,
    1.4,
    1.666666666667,
    2.0,
    2.333333333333,
    2.6,
    2.625,
    3.0,
)
PATTERN_DIR = Path(__file__).resolve().parent / "test_patterns"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"
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


def sqnr_db(signal: np.ndarray, error: np.ndarray) -> float:
    signal_power = float(np.mean(signal.astype(np.float64) ** 2))
    noise_power = float(np.mean(error.astype(np.float64) ** 2))
    if noise_power == 0.0:
        return math.inf
    if signal_power == 0.0:
        return -math.inf
    return 10.0 * math.log10(signal_power / noise_power)


def pearson_r(left: np.ndarray, right: np.ndarray) -> float:
    left_f = left.astype(np.float64, copy=False)
    right_f = right.astype(np.float64, copy=False)
    count = float(left_f.size)
    sum_left = float(left_f.sum())
    sum_right = float(right_f.sum())
    sum_left_sq = float((left_f * left_f).sum())
    sum_right_sq = float((right_f * right_f).sum())
    sum_product = float((left_f * right_f).sum())
    numerator = sum_product - (sum_left * sum_right / count)
    left_var = sum_left_sq - (sum_left * sum_left / count)
    right_var = sum_right_sq - (sum_right * sum_right / count)
    denom = math.sqrt(max(left_var, 0.0) * max(right_var, 0.0))
    if denom == 0.0:
        return 1.0 if np.array_equal(left, right) else math.nan
    return float(numerator / denom)


def pattern_group(name: str) -> str:
    if name.startswith("constant_"):
        return "constant"
    if name.startswith("ramp_"):
        return "ramp"
    if name.startswith("plane_"):
        return "plane"
    if name.startswith("impulse_"):
        return "impulse"
    if name.startswith(("step_", "line_")):
        return "edge"
    if name.startswith("checkerboard_"):
        return "checkerboard"
    if name.startswith("sine_bandlimited_"):
        return "sine_bandlimited"
    if name.startswith("sine_stress_"):
        return "sine_stress"
    if name.startswith("sine_"):
        return "sine_legacy"
    if name.startswith("zone_plate_bandlimited_"):
        return "zone_plate_bandlimited"
    if name.startswith("zone_plate_stress_"):
        return "zone_plate_stress"
    if name.startswith("zone_plate_"):
        return "zone_plate_legacy"
    if name.startswith("noise_"):
        return "noise"
    if name.startswith("tile_2x2_"):
        return "tile_2x2"
    return "other"


def save_difference(path: Path, error: np.ndarray) -> None:
    abs_error = np.abs(error).clip(0, 255).astype(np.uint8)
    save_deterministic_png(path, abs_error)


def compare_pixels(
    pattern: str,
    source: np.ndarray,
    ratio_w: float,
    ratio_h: float,
    output_dir: Path,
    write_images: bool,
) -> dict[str, object]:
    opencv = resize_rgb_planes(source, ratio_w, opencv_bilinear, ratio_h)
    fixed = resize_rgb_planes(source, ratio_w, fixed_point_bilinear, ratio_h)
    opencv_pixels = opencv.pixels
    fixed_pixels = fixed.pixels
    error = fixed_pixels.astype(np.int16) - opencv_pixels.astype(np.int16)
    abs_error = np.abs(error)
    ratio_name = ratio_slug(ratio_w, ratio_h)
    ratio_label = str(ratio_w) if ratio_w == ratio_h else f"{ratio_w}x{ratio_h}"

    if write_images:
        save_deterministic_png(
            output_dir / "resized" / "opencv_bilinear" / pattern / f"{ratio_name}.png",
            opencv_pixels,
        )
        save_deterministic_png(
            output_dir / "resized" / "fixed_point_bilinear" / pattern / f"{ratio_name}.png",
            fixed_pixels,
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
        "fixed_kernel": fixed.metadata["kernel"],
        "sqnr_db_opencv_as_signal": sqnr_db(opencv_pixels, error),
        "pearson_r": pearson_r(opencv_pixels, fixed_pixels),
        "mse": float(np.mean(error.astype(np.float64) ** 2)),
        "rmse": float(math.sqrt(float(np.mean(error.astype(np.float64) ** 2)))),
        "mae": float(np.mean(abs_error.astype(np.float64))),
        "max_abs_error": int(abs_error.max()),
        "mean_error": float(np.mean(error.astype(np.float64))),
        "different_pixels": int(np.count_nonzero(error)),
        "pixel_count": int(error.size),
        "different_pixel_fraction": float(np.count_nonzero(error) / error.size),
        "opencv_sha256": pixel_sha256(opencv_pixels),
        "fixed_sha256": pixel_sha256(fixed_pixels),
    }


def finite_sqnr_frame(results: pd.DataFrame) -> pd.DataFrame:
    return results.replace([np.inf, -np.inf], np.nan)


def write_summary(results: pd.DataFrame, output_dir: Path) -> None:
    def markdown_table(frame: pd.DataFrame, floatfmt: str = ".6g") -> str:
        columns = list(frame.columns)
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for _, row in frame.iterrows():
            values = []
            for column in columns:
                value = row[column]
                if isinstance(value, float):
                    values.append(format(value, floatfmt))
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    rows = []
    rows.append("# Bilinear Replication Report")
    rows.append("")
    rows.append(f"Patterns: {results['pattern'].nunique()}")
    rows.append(
        f"Ratios: {', '.join(str(value) for value in sorted(results['ratio'].unique()))}"
    )
    rows.append(f"Comparisons: {len(results)}")
    rows.append("")

    exact = int((results["different_pixels"] == 0).sum())
    rows.append(f"Exact image matches: {exact} / {len(results)}")
    rows.append("")

    finite = finite_sqnr_frame(results)
    by_ratio = (
        finite.groupby("ratio", as_index=False)
        .agg(
            mean_sqnr_db=("sqnr_db_opencv_as_signal", "mean"),
            min_sqnr_db=("sqnr_db_opencv_as_signal", "min"),
            max_rmse=("rmse", "max"),
            max_abs_error=("max_abs_error", "max"),
            mean_different_pixel_fraction=("different_pixel_fraction", "mean"),
        )
        .sort_values("ratio")
    )
    rows.append("## By Ratio")
    rows.append("")
    rows.append(markdown_table(by_ratio))
    rows.append("")

    worst = finite.sort_values(["sqnr_db_opencv_as_signal", "rmse"], ascending=[True, False]).head(20)
    rows.append("## Lowest SQNR Cases")
    rows.append("")
    rows.append(markdown_table(worst[
        [
            "pattern",
            "group",
            "ratio",
            "fixed_kernel",
            "sqnr_db_opencv_as_signal",
            "rmse",
            "max_abs_error",
            "different_pixel_fraction",
        ]
    ]))
    rows.append("")
    (output_dir / "summary.md").write_text("\n".join(rows), encoding="utf-8")


def svg_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_line_plot(
    frame: pd.DataFrame,
    value_column: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    width, height = 1100, 660
    left, right, top, bottom = 90, 220, 45, 85
    plot_width = width - left - right
    plot_height = height - top - bottom
    ratios = sorted(str(value) for value in frame["ratio"].unique())
    ratio_positions = {ratio: index for index, ratio in enumerate(ratios)}
    groups = sorted(str(value) for value in frame["group"].unique())
    values = frame[value_column].replace([np.inf, -np.inf], np.nan)
    if not values.notna().any():
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="700" height="120" '
            'viewBox="0 0 700 120">\n'
            '<rect width="100%" height="100%" fill="white"/>\n'
            f'<text x="350" y="45" text-anchor="middle" font-family="Arial" '
            f'font-size="20">{svg_escape(title)}</text>\n'
            '<text x="350" y="82" text-anchor="middle" font-family="Arial" '
            'font-size="14">No finite values to plot</text>\n'
            "</svg>\n",
            encoding="utf-8",
        )
        return
    ymin = float(np.nanmin(values))
    ymax = float(np.nanmax(values))
    if math.isclose(ymin, ymax):
        ymin -= 1.0
        ymax += 1.0

    def x_pos(ratio: str) -> float:
        if len(ratios) == 1:
            return left + plot_width / 2
        return left + ratio_positions[ratio] / (len(ratios) - 1) * plot_width

    def y_pos(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * plot_height

    palette = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
        "#8c564b",
        "#7f7f7f",
        "#bcbd22",
        "#e377c2",
        "#004d40",
        "#6a3d9a",
    ]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="25" text-anchor="middle" font-family="Arial" font-size="20">{svg_escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222"/>',
    ]
    for ratio in ratios:
        x = x_pos(ratio)
        lines.append(f'<line x1="{x:.2f}" y1="{top + plot_height}" x2="{x:.2f}" y2="{top + plot_height + 6}" stroke="#222"/>')
        lines.append(f'<text x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle" font-family="Arial" font-size="11">{svg_escape(ratio)}</text>')
    for index in range(6):
        value = ymin + (ymax - ymin) * index / 5
        y = y_pos(value)
        lines.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#222"/>')
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#ddd"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="11">{value:.3g}</text>')
    lines.append(f'<text x="{left + plot_width / 2}" y="{height - 20}" text-anchor="middle" font-family="Arial" font-size="14">Downscale ratio</text>')
    lines.append(f'<text x="22" y="{top + plot_height / 2}" transform="rotate(-90 22 {top + plot_height / 2})" text-anchor="middle" font-family="Arial" font-size="14">{svg_escape(ylabel)}</text>')

    for group_index, group in enumerate(groups):
        scoped = frame[frame["group"] == group].sort_values("ratio")
        color = palette[group_index % len(palette)]
        points = []
        for _, row in scoped.iterrows():
            value = row[value_column]
            if pd.isna(value) or math.isinf(float(value)):
                continue
            points.append(f"{x_pos(str(row['ratio'])):.2f},{y_pos(float(value)):.2f}")
        if len(points) >= 2:
            lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        for point in points:
            x, y = point.split(",")
            lines.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
        legend_y = top + 20 + group_index * 20
        legend_x = left + plot_width + 25
        lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 22}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x + 28}" y="{legend_y + 4}" font-family="Arial" font-size="11">{svg_escape(group)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def svg_heatmap(frame: pd.DataFrame, path: Path) -> None:
    pivot = frame.pivot_table(
        index="group",
        columns="ratio_slug",
        values="sqnr_db_opencv_as_signal",
        aggfunc="mean",
    ).replace([np.inf, -np.inf], np.nan)
    if pivot.empty or not np.isfinite(pivot.to_numpy(dtype=np.float64)).any():
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="700" height="120" '
            'viewBox="0 0 700 120">\n'
            '<rect width="100%" height="100%" fill="white"/>\n'
            '<text x="350" y="45" text-anchor="middle" font-family="Arial" '
            'font-size="20">Mean SQNR heatmap by group</text>\n'
            '<text x="350" y="82" text-anchor="middle" font-family="Arial" '
            'font-size="14">No finite values to plot</text>\n'
            "</svg>\n",
            encoding="utf-8",
        )
        return
    cell = 58
    left, top = 185, 55
    width = left + cell * len(pivot.columns) + 40
    height = top + cell * len(pivot.index) + 120
    values = pivot.to_numpy(dtype=np.float64)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))

    def color(value: float) -> str:
        if pd.isna(value):
            return "#f4f4f4"
        t = 0.0 if math.isclose(vmin, vmax) else (float(value) - vmin) / (vmax - vmin)
        red = int(240 - 180 * t)
        green = int(245 - 80 * (1.0 - abs(t - 0.5) * 2.0))
        blue = int(255 - 190 * (1.0 - t))
        return f"#{red:02x}{green:02x}{blue:02x}"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="20">Mean SQNR heatmap by group</text>',
    ]
    for col_index, column in enumerate(pivot.columns):
        x = left + col_index * cell + cell / 2
        lines.append(f'<text x="{x}" y="{top - 10}" transform="rotate(-45 {x} {top - 10})" text-anchor="end" font-family="Arial" font-size="11">{svg_escape(column)}</text>')
    for row_index, row_name in enumerate(pivot.index):
        y = top + row_index * cell
        lines.append(f'<text x="{left - 10}" y="{y + cell / 2 + 4}" text-anchor="end" font-family="Arial" font-size="12">{svg_escape(row_name)}</text>')
        for col_index, column in enumerate(pivot.columns):
            value = pivot.loc[row_name, column]
            x = left + col_index * cell
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color(value)}" stroke="white"/>')
            label = "" if pd.isna(value) else f"{float(value):.1f}"
            lines.append(f'<text x="{x + cell / 2}" y="{y + cell / 2 + 4}" text-anchor="middle" font-family="Arial" font-size="10">{label}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(results: pd.DataFrame, output_dir: Path) -> None:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    finite = finite_sqnr_frame(results)

    by_ratio_group = (
        finite.groupby(["ratio", "group"], as_index=False)
        .agg(mean_sqnr_db=("sqnr_db_opencv_as_signal", "mean"), mean_rmse=("rmse", "mean"))
        .sort_values(["ratio", "group"])
    )

    svg_line_plot(
        by_ratio_group,
        "mean_sqnr_db",
        "Mean SQNR, OpenCV output as signal (dB)",
        "OpenCV vs fixed-point bilinear SQNR",
        plots / "sqnr_by_ratio_and_group.svg",
    )
    svg_line_plot(
        by_ratio_group,
        "mean_rmse",
        "Mean RMSE (pixel values)",
        "OpenCV vs fixed-point bilinear RMSE",
        plots / "rmse_by_ratio_and_group.svg",
    )
    svg_heatmap(finite, plots / "sqnr_heatmap_by_group.svg")


def parse_ratio_groups(groups: Sequence[Sequence[float]] | None) -> tuple[RatioPair, ...]:
    if groups is None:
        return tuple((ratio, ratio) for ratio in RATIOS)
    ratios: list[RatioPair] = []
    for group in groups:
        if len(group) == 1:
            ratio_w = ratio_h = group[0]
        elif len(group) == 2:
            ratio_w, ratio_h = group
        else:
            raise ValueError("each --ratio accepts either WIDTH or WIDTH HEIGHT")
        if ratio_w <= 0 or ratio_h <= 0:
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
        help="also write resized OpenCV/fixed-point PNGs and absolute-difference PNGs",
    )
    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="reuse an existing metrics CSV and only regenerate summary/plots",
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
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "bilinear_comparison_metrics.csv"
    if args.reports_only:
        if not csv_path.is_file():
            raise FileNotFoundError(f"{csv_path} does not exist")
        results = pd.read_csv(csv_path)
        write_summary(results, output_dir)
        plot_results(results, output_dir)
        print(f"Wrote summary to {output_dir / 'summary.md'}")
        print(f"Wrote plots to {output_dir / 'plots'}")
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
            pattern = image_path.stem
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
                    pattern=pattern,
                    source=source,
                    ratio_w=ratio_w,
                    ratio_h=ratio_h,
                    output_dir=output_dir,
                    write_images=args.write_images,
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
    write_summary(results, output_dir)
    plot_results(results, output_dir)
    print(f"Wrote metrics to {csv_path}")
    print(f"Wrote summary to {output_dir / 'summary.md'}")
    print(f"Wrote plots to {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
