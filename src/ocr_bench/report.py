from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "ocr_bench_plot_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .config import BenchmarkConfig, ratio_slug
from .resample import METHODS


METHOD_LABELS = {
    "bin_floor": "Bin floor",
    "bin_ceil": "Bin ceil",
    "opencv_bilinear": "OpenCV bilinear",
    "lanczos2_7tap_16phase": "Lanczos-2 7t/16p",
    "lanczos3_7tap_16phase": "Lanczos-3 7t/16p",
}
POOLED_METHOD_LABELS = {
    "bin_floor": "binning_floor",
    "bin_ceil": "binning_ceil",
    "opencv_bilinear": "bilinear",
    "lanczos2_7tap_16phase": "lanczos2",
    "lanczos3_7tap_16phase": "lanczos3",
}
METHOD_COLORS = {
    "lanczos2_7tap_16phase": "#117BB5",
    "bin_floor": "#E9A400",
    "lanczos3_7tap_16phase": "#0AA47A",
    "opencv_bilinear": "#DB5A00",
    "bin_ceil": "#C777A5",
}
REFERENCE_HEATMAP = LinearSegmentedColormap.from_list(
    "reference_theme",
    ["#F7F7F7", "#E9A400", "#DB5A00", "#C777A5", "#117BB5"],
)
SCOPE_LABELS = {
    "english": "English",
    "traditional_chinese": "Traditional Chinese",
    "combined": "Combined",
}


def _apply_reference_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "text.color": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.grid": True,
            "grid.color": "#B0B0B0",
            "grid.alpha": 0.3,
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
        }
    )


def _style_axis(axis: plt.Axes, grid_axis: str = "both") -> None:
    axis.set_axisbelow(True)
    axis.grid(True, axis=grid_axis, color="#B0B0B0", alpha=0.3, linewidth=0.8)
    for spine in axis.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)


def _style_heatmap_cells(axis: plt.Axes, rows: int, columns: int) -> None:
    # Major ticks label cell centers. Minor ticks sit on half-step cell edges,
    # so only the minor grid should be visible on an image heatmap.
    axis.grid(False, which="major")
    axis.set_xticks(np.arange(-0.5, columns, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    axis.grid(
        True,
        which="minor",
        color="white",
        linewidth=1.0,
        alpha=0.9,
    )
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)


def _prepare_plots(output: Path, force: bool) -> Path:
    plots = output / "plots"
    if plots.exists():
        if not force:
            raise FileExistsError(f"{plots} already exists; use --force to regenerate reports")
        shutil.rmtree(plots)
    plots.mkdir(parents=True, exist_ok=True)
    return plots


def _prepare_montage(output: Path, force: bool) -> Path:
    montage = output / "montage"
    if montage.exists():
        if not force:
            raise FileExistsError(
                f"{montage} already exists; use --force to regenerate reports"
            )
        shutil.rmtree(montage)
    montage.mkdir(parents=True, exist_ok=True)
    return montage


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_cer_curves(aggregate: pd.DataFrame, plots: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
    for axis, scope in zip(axes, SCOPE_LABELS):
        scoped = aggregate[aggregate["scope"] == scope]
        for method in METHODS:
            values = scoped[scoped["method"] == method].sort_values("requested_ratio")
            axis.plot(
                values["requested_ratio"],
                values["cer"],
                marker="o",
                markersize=3,
                linewidth=1.4,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
            )
        axis.set_title(SCOPE_LABELS[scope])
        axis.set_xlabel("Requested scaling ratio")
        _style_axis(axis)
    axes[0].set_ylabel("Character error rate")
    axes[-1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.suptitle("CER versus requested scaling ratio")
    _save_figure(figure, plots / "cer_vs_ratio.png")

    standalone = (
        ("english", "English", "cer_vs_ratio_english.png"),
        (
            "traditional_chinese",
            "Traditional Chinese",
            "cer_vs_ratio_traditional_chinese.png",
        ),
        ("combined", "Combined", "cer_vs_ratio_combined.png"),
    )
    for scope, title, filename in standalone:
        scoped = aggregate[aggregate["scope"] == scope]
        individual_figure, individual_axis = plt.subplots(figsize=(10, 6))
        for method in METHODS:
            values = scoped[scoped["method"] == method].sort_values("requested_ratio")
            individual_axis.plot(
                values["requested_ratio"],
                values["cer"],
                marker="o",
                markersize=5,
                linewidth=2.0,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
            )
        individual_axis.set_title(f"{title} CER versus requested scaling ratio", fontsize=18)
        individual_axis.set_xlabel("Requested scaling ratio")
        individual_axis.set_ylabel("Character error rate")
        individual_axis.set_ylim(bottom=0)
        individual_axis.legend(fontsize=10, loc="best", frameon=True)
        _style_axis(individual_axis)
        _save_figure(individual_figure, plots / filename)


def _plot_cer_heatmaps(aggregate: pd.DataFrame, plots: Path) -> None:
    ratios = sorted(aggregate["requested_ratio"].unique())
    figure, axes = plt.subplots(3, 1, figsize=(17, 8.5), constrained_layout=True)
    image = None
    for axis, scope in zip(axes, SCOPE_LABELS):
        scoped = aggregate[aggregate["scope"] == scope]
        matrix = np.full((len(METHODS), len(ratios)), np.nan)
        for row_index, method in enumerate(METHODS):
            values = scoped[scoped["method"] == method].set_index("requested_ratio")["cer"]
            for column_index, ratio in enumerate(ratios):
                if ratio in values.index:
                    matrix[row_index, column_index] = values.loc[ratio]
        image = axis.imshow(
            matrix, aspect="auto", interpolation="nearest", cmap=REFERENCE_HEATMAP
        )
        axis.set_yticks(range(len(METHODS)), [METHOD_LABELS[method] for method in METHODS])
        axis.set_xticks(range(len(ratios)), [f"{ratio:.3f}" for ratio in ratios], rotation=60)
        axis.set_title(SCOPE_LABELS[scope])
        _style_heatmap_cells(axis, len(METHODS), len(ratios))
    if image is not None:
        figure.colorbar(image, ax=axes, label="CER", shrink=0.85)
    figure.suptitle("CER heatmaps")
    _save_figure(figure, plots / "cer_heatmaps.png")


def _plot_glyph_heights(results: pd.DataFrame, target: int, plots: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for axis, (language, title) in zip(
        axes, (("en", "English"), ("zh_tra", "Traditional Chinese"))
    ):
        scoped = results[results["language"] == language]
        grouped = (
            scoped.groupby(["method", "requested_ratio"], as_index=False)["output_max_glyph_height"]
            .mean()
        )
        for method in METHODS:
            values = grouped[grouped["method"] == method].sort_values("requested_ratio")
            axis.plot(
                values["requested_ratio"],
                values["output_max_glyph_height"],
                marker="o",
                markersize=3,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
            )
        axis.axhline(target, color="black", linestyle="--", linewidth=1, label="16 px target")
        axis.set_title(title)
        axis.set_xlabel("Requested scaling ratio")
        _style_axis(axis)
    axes[0].set_ylabel("Mean measured maximum glyph height (px)")
    axes[-1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.suptitle("Output glyph height versus requested ratio")
    _save_figure(figure, plots / "glyph_height_vs_ratio.png")


def _plot_pattern_summary(results: pd.DataFrame, plots: Path) -> None:
    patterns = list(dict.fromkeys(results["pattern_id"].tolist()))
    matrix = np.full((len(METHODS), len(patterns)), np.nan)
    for method_index, method in enumerate(METHODS):
        for pattern_index, pattern in enumerate(patterns):
            values = results[
                (results["method"] == method) & (results["pattern_id"] == pattern)
            ]["cer"]
            if not values.empty:
                matrix[method_index, pattern_index] = values.mean()
    figure, axis = plt.subplots(figsize=(10, 4.8))
    image = axis.imshow(
        matrix, aspect="auto", interpolation="nearest", cmap=REFERENCE_HEATMAP
    )
    axis.set_yticks(range(len(METHODS)), [METHOD_LABELS[method] for method in METHODS])
    axis.set_xticks(range(len(patterns)), patterns, rotation=30, ha="right")
    axis.set_title("Mean CER by pattern and method")
    _style_heatmap_cells(axis, len(METHODS), len(patterns))
    figure.colorbar(image, ax=axis, label="Mean CER")
    _save_figure(figure, plots / "pattern_method_summary.png")


def _fit_image(image: Image.Image, maximum_width: int, maximum_height: int) -> Image.Image:
    scale = min(maximum_width / image.width, maximum_height / image.height, 1.0)
    if scale >= 1:
        return image.copy()
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.Resampling.NEAREST)


def _create_montages(config: BenchmarkConfig, output: Path, montage: Path) -> None:
    ratio = 2.625
    font = ImageFont.load_default()
    for pattern in config.patterns:
        panels: list[tuple[str, Image.Image]] = []
        source = output / "canonical" / pattern.id / f"{ratio_slug(ratio)}.png"
        if source.is_file():
            panels.append(("Canonical source", Image.open(source).convert("L")))
        for method in METHODS:
            path = output / "resized" / method / pattern.id / f"{ratio_slug(ratio)}.png"
            if path.is_file():
                panels.append((METHOD_LABELS[method], Image.open(path).convert("L")))
        if not panels:
            continue
        fitted = [(label, _fit_image(image, 1100, 220)) for label, image in panels]
        width = max(image.width for _, image in fitted) + 32
        panel_heights = [image.height + 38 for _, image in fitted]
        canvas = Image.new("RGB", (width, sum(panel_heights) + 16), "white")
        draw = ImageDraw.Draw(canvas)
        y = 8
        for (label, image), panel_height in zip(fitted, panel_heights):
            draw.text((12, y), label, fill="black", font=font)
            y += 22
            canvas.paste(image.convert("RGB"), (12, y))
            y += panel_height - 22
        canvas.save(montage / f"montage_2p625_{pattern.id}.png", format="PNG")


def _plot_pooled_cer(results: pd.DataFrame, output: Path) -> list[Path]:
    scopes = (
        ("combined", results, "Overall pooled CER – fixed 16 px target glyph"),
        (
            "english",
            results[results["language"] == "en"],
            "English pooled CER – fixed 16 px target glyph",
        ),
        (
            "traditional_chinese",
            results[results["language"] == "zh_tra"],
            "Traditional Chinese pooled CER – fixed 16 px target glyph",
        ),
    )
    paths: list[Path] = []
    for scope, scoped, title in scopes:
        pooled = (
            scoped.groupby("method", as_index=False)[["edit_distance", "reference_chars"]]
            .sum()
        )
        pooled["cer_percent"] = 100.0 * pooled["edit_distance"] / pooled["reference_chars"]
        pooled = pooled.sort_values("cer_percent", ascending=True)

        figure, axis = plt.subplots(figsize=(12, 7))
        methods = pooled["method"].tolist()
        values = pooled["cer_percent"].to_numpy()
        labels = [POOLED_METHOD_LABELS[method] for method in methods]
        colors = [METHOD_COLORS[method] for method in methods]
        bars = axis.barh(labels, values, color=colors)
        axis.invert_yaxis()
        axis.set_title(title, fontsize=18, pad=10)
        axis.set_xlabel("Character error rate (%)")
        _style_axis(axis, grid_axis="x")
        headroom = max(float(values.max()) * 0.12, 0.35)
        axis.set_xlim(0, float(values.max()) + headroom)
        for bar, value in zip(bars, values):
            axis.text(
                value + headroom * 0.06,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}%",
                va="center",
                ha="left",
                fontsize=12,
                color="black",
            )
        path = output / f"pooled_cer_{scope}.png"
        _save_figure(figure, path)
        paths.append(path)
    return paths


def _micro_average(results: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = (
        results.groupby(group_columns, as_index=False)[["edit_distance", "reference_chars"]]
        .sum()
    )
    grouped["cer"] = grouped["edit_distance"] / grouped["reference_chars"]
    return grouped


def _extreme_row(table: pd.DataFrame, best: bool) -> tuple[pd.Series, int]:
    target = float(table["cer"].min() if best else table["cer"].max())
    tied = table[np.isclose(table["cer"], target)].copy()
    sort_columns = [
        column
        for column in ("pattern_id", "requested_ratio", "method")
        if column in tied.columns
    ]
    if sort_columns:
        tied = tied.sort_values(sort_columns)
    return tied.iloc[0], len(tied)


def _method_name(method: str) -> str:
    return POOLED_METHOD_LABELS.get(method, method)


def _case_text(row: pd.Series) -> str:
    reference = str(row.get("normalized_reference", row.get("reference", "")))
    prediction = str(row.get("normalized_prediction", row.get("prediction", "")))
    return (
        f"pattern={row['pattern_id']}, method={_method_name(str(row['method']))}, "
        f"ratio={float(row['requested_ratio']):.3f}, CER={100 * float(row['cer']):.2f}%, "
        f"reference={reference!r}, prediction={prediction!r}"
    )


def _write_performance_summary(results: pd.DataFrame, output: Path) -> list[Path]:
    scopes = (
        ("english", "English", results[results["language"] == "en"]),
        (
            "traditional_chinese",
            "Traditional Chinese",
            results[results["language"] == "zh_tra"],
        ),
        ("combined", "Combined", results),
    )
    text_lines = [
        "OCR benchmark performance summary",
        "",
        "Aggregate rankings use micro-averaged CER (total edit distance / total reference characters).",
        "Lower CER is better. Individual-case ties show one deterministic example.",
    ]
    structured: list[dict[str, object]] = []

    for scope_id, scope_label, scoped in scopes:
        text_lines.extend(["", scope_label, "-"])
        aggregate_specs = (
            ("overall_method", ["method"], "overall method"),
            ("scaling_ratio", ["requested_ratio"], "scaling ratio"),
            (
                "method_by_ratio",
                ["method", "requested_ratio"],
                "method×ratio condition",
            ),
        )
        for category, group_columns, display_name in aggregate_specs:
            table = _micro_average(scoped, group_columns)
            for rank, best in (("best", True), ("worst", False)):
                row, tie_count = _extreme_row(table, best)
                method = str(row["method"]) if "method" in row.index else ""
                ratio = (
                    float(row["requested_ratio"])
                    if "requested_ratio" in row.index
                    else None
                )
                value_parts = []
                if method:
                    value_parts.append(_method_name(method))
                if ratio is not None:
                    value_parts.append(f"{ratio:.3f}×")
                text_lines.append(
                    f"- {rank.title()} {display_name}: {' at '.join(value_parts)} "
                    f"({100 * float(row['cer']):.2f}% CER"
                    f"{f', {tie_count} tied' if tie_count > 1 else ''})"
                )
                structured.append(
                    {
                        "scope": scope_id,
                        "category": category,
                        "rank": rank,
                        "method": method,
                        "requested_ratio": "" if ratio is None else ratio,
                        "pattern_id": "",
                        "language": "",
                        "psm": "",
                        "cer": float(row["cer"]),
                        "edit_distance": int(row["edit_distance"]),
                        "reference_chars": int(row["reference_chars"]),
                        "reference": "",
                        "prediction": "",
                        "tie_count": tie_count,
                    }
                )

        for rank, best in (("best", True), ("worst", False)):
            row, tie_count = _extreme_row(scoped, best)
            text_lines.append(
                f"- {rank.title()} individual case: {_case_text(row)}"
                f"{f' ({tie_count} cases tied at this CER)' if tie_count > 1 else ''}"
            )
            structured.append(
                {
                    "scope": scope_id,
                    "category": "individual_case",
                    "rank": rank,
                    "method": row["method"],
                    "requested_ratio": float(row["requested_ratio"]),
                    "pattern_id": row["pattern_id"],
                    "language": row["language"],
                    "psm": int(row["psm"]),
                    "cer": float(row["cer"]),
                    "edit_distance": int(row["edit_distance"]),
                    "reference_chars": int(row["reference_chars"]),
                    "reference": row["normalized_reference"],
                    "prediction": row["normalized_prediction"],
                    "tie_count": tie_count,
                }
            )

    text_path = output / "performance_summary.txt"
    text_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    csv_path = output / "performance_extremes.csv"
    pd.DataFrame(structured).to_csv(csv_path, index=False, encoding="utf-8")
    return [text_path, csv_path]


def generate_report(config: BenchmarkConfig, output: Path, force: bool = False) -> list[Path]:
    image_csv = output / "image_results.csv"
    aggregate_csv = output / "aggregate_results.csv"
    if not image_csv.is_file() or not aggregate_csv.is_file():
        raise FileNotFoundError("study CSV files are missing; run the study stage first")
    plots = _prepare_plots(output, force)
    montage = _prepare_montage(output, force)
    _apply_reference_theme()
    results = pd.read_csv(image_csv)
    aggregate = pd.read_csv(aggregate_csv)
    results = results[results["status"] == "ok"].copy()
    if results.empty:
        raise RuntimeError("there are no successful OCR rows to report")
    _plot_cer_curves(aggregate, plots)
    _plot_cer_heatmaps(aggregate, plots)
    _plot_glyph_heights(results, config.target_height_px, plots)
    _plot_pattern_summary(results, plots)
    _create_montages(config, output, montage)
    pooled_paths = _plot_pooled_cer(results, output)
    summary_paths = _write_performance_summary(results, output)
    return (
        sorted(plots.glob("*.png"))
        + sorted(montage.glob("*.png"))
        + pooled_paths
        + summary_paths
    )
