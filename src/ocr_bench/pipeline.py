from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import BenchmarkConfig, ratio_slug
from .dataset import measure_resized_glyph_height
from .fonts import validate_all_fonts
from .metrics import aggregate_rows, character_error_rate, normalize_text
from .ocr import run_tesseract, tesseract_language, validate_tesseract
from .resample import METHODS, resize_image
from .util import read_json, sha256_file, software_environment, write_json


IMAGE_RESULT_FIELDS = (
    "pattern_id",
    "language",
    "layout",
    "psm",
    "requested_ratio",
    "method",
    "effective_ratio_x",
    "effective_ratio_y",
    "integer_bin_factor",
    "source_path",
    "source_sha256",
    "output_path",
    "output_sha256",
    "font_path",
    "font_index",
    "font_size",
    "target_output_glyph_height",
    "target_source_glyph_height",
    "source_max_glyph_height",
    "source_height_error",
    "output_max_glyph_height",
    "glyph_height_warning",
    "reference",
    "prediction",
    "normalized_reference",
    "normalized_prediction",
    "edit_distance",
    "reference_chars",
    "cer",
    "ocr_seconds",
    "status",
    "error",
)

AGGREGATE_FIELDS = (
    "requested_ratio",
    "method",
    "scope",
    "edit_distance",
    "reference_chars",
    "cer",
    "pattern_count",
)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _prepare_study_directories(output: Path, force: bool) -> None:
    owned = [output / "resized", output / "ocr"]
    result_files = [
        output / "image_results.csv",
        output / "aggregate_results.csv",
        output / "run_manifest.json",
    ]
    existing = [path for path in [*owned, *result_files] if path.exists()]
    if existing and not force:
        raise FileExistsError(
            f"study outputs already exist ({existing[0]}); use --force to regenerate them"
        )
    if force:
        for directory in owned:
            if directory.exists():
                shutil.rmtree(directory)
        for path in result_files:
            if path.exists():
                path.unlink()
    for directory in owned:
        directory.mkdir(parents=True, exist_ok=True)


def run_study(config: BenchmarkConfig, output: Path, force: bool = False) -> dict[str, Any]:
    source_manifest_path = output / "source_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(
            f"{source_manifest_path} is missing; run the generate stage first"
        )
    _prepare_study_directories(output, force)
    source_manifest = read_json(source_manifest_path)
    expected_sources = len(config.patterns) * len(config.ratios)
    if len(source_manifest["sources"]) != expected_sources:
        raise RuntimeError(
            f"source manifest contains {len(source_manifest['sources'])} images, expected {expected_sources}"
        )
    if source_manifest.get("config_sha256") != sha256_file(config.source_path):
        raise RuntimeError("benchmark.yaml changed after source generation; regenerate canonical images")

    font_info = validate_all_fonts(config.fonts, config.patterns)
    tess_info = validate_tesseract(config.tesseract, {"eng", "chi_tra"})
    # Import once here to report a prerequisite error before doing any resizing.
    try:
        import cv2  # noqa: F401
    except ImportError as error:
        raise RuntimeError("OpenCV is missing; install the project dependencies") from error

    rows: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    failures = 0
    for source_record in source_manifest["sources"]:
        source_path = output / source_record["image_path"]
        actual_source_hash = sha256_file(source_path)
        if actual_source_hash != source_record["sha256"]:
            raise RuntimeError(f"canonical source hash mismatch: {source_path}")
        source_pixels = np.asarray(Image.open(source_path).convert("L"), dtype=np.uint8)
        requested_ratio = float(source_record["requested_ratio"])
        for method in METHODS:
            resized = resize_image(source_pixels, requested_ratio, method)
            relative_output = (
                Path("resized")
                / method
                / source_record["pattern_id"]
                / f"{ratio_slug(requested_ratio)}.png"
            )
            output_path = output / relative_output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(resized.pixels).save(output_path, format="PNG", optimize=False)
            output_hash = sha256_file(output_path)
            output_glyph_height = measure_resized_glyph_height(
                resized.pixels,
                int(source_record["canvas_width"]),
                int(source_record["canvas_height"]),
                source_record["glyph_boxes"],
                config.ink_threshold,
            )
            height_controlled_method = method not in {"bin_floor", "bin_ceil"}
            glyph_height_warning = bool(
                height_controlled_method
                and abs(output_glyph_height - config.target_height_px) > 1
            )

            ocr_result = run_tesseract(
                output_path,
                language=tesseract_language(source_record["language"]),
                psm=int(source_record["psm"]),
                spec=config.tesseract,
            )
            ocr_base = (
                output
                / "ocr"
                / method
                / source_record["pattern_id"]
                / ratio_slug(requested_ratio)
            )
            ocr_base.parent.mkdir(parents=True, exist_ok=True)
            ocr_base.with_suffix(".txt").write_text(ocr_result["stdout"], encoding="utf-8")
            ocr_base.with_suffix(".stderr.txt").write_text(
                ocr_result["stderr"], encoding="utf-8"
            )

            normalized_reference = normalize_text(source_record["reference"])
            normalized_prediction = normalize_text(ocr_result["stdout"])
            edit_distance: int | str = ""
            reference_chars: int | str = ""
            cer: float | str = ""
            if ocr_result["status"] == "ok":
                edit_distance, reference_chars, cer = character_error_rate(
                    source_record["reference"], ocr_result["stdout"]
                )
            else:
                failures += 1

            metadata = resized.metadata
            row = {
                "pattern_id": source_record["pattern_id"],
                "language": source_record["language"],
                "layout": source_record["layout"],
                "psm": source_record["psm"],
                "requested_ratio": requested_ratio,
                "method": method,
                "effective_ratio_x": metadata["actual_scale_x"],
                "effective_ratio_y": metadata["actual_scale_y"],
                "integer_bin_factor": metadata.get("integer_bin_factor", ""),
                "source_path": source_record["image_path"],
                "source_sha256": actual_source_hash,
                "output_path": relative_output.as_posix(),
                "output_sha256": output_hash,
                "font_path": source_record["font_path"],
                "font_index": source_record["font_index"],
                "font_size": source_record["font_size"],
                "target_output_glyph_height": source_record["target_output_glyph_height"],
                "target_source_glyph_height": source_record["target_source_glyph_height"],
                "source_max_glyph_height": source_record["source_max_glyph_height"],
                "source_height_error": source_record["source_height_error"],
                "output_max_glyph_height": output_glyph_height,
                "glyph_height_warning": glyph_height_warning,
                "reference": source_record["reference"],
                "prediction": ocr_result["stdout"],
                "normalized_reference": normalized_reference,
                "normalized_prediction": normalized_prediction,
                "edit_distance": edit_distance,
                "reference_chars": reference_chars,
                "cer": cer,
                "ocr_seconds": ocr_result["elapsed_seconds"],
                "status": ocr_result["status"],
                "error": ocr_result["error"],
            }
            rows.append(row)
            audit_records.append(
                {
                    "pattern_id": source_record["pattern_id"],
                    "requested_ratio": requested_ratio,
                    "method": method,
                    "source_sha256": actual_source_hash,
                    "output_sha256": output_hash,
                    "resampler": metadata,
                    "ocr_command": ocr_result["command"],
                    "ocr_returncode": ocr_result["returncode"],
                }
            )

    aggregate = aggregate_rows(rows)
    _write_csv(output / "image_results.csv", rows, IMAGE_RESULT_FIELDS)
    _write_csv(output / "aggregate_results.csv", aggregate, AGGREGATE_FIELDS)
    manifest = {
        "schema_version": 1,
        "config_path": str(config.source_path),
        "config_sha256": sha256_file(config.source_path),
        "software": software_environment(),
        "fonts": font_info,
        "tesseract": tess_info,
        "ratios": list(config.ratios),
        "methods": list(METHODS),
        "source_count": len(source_manifest["sources"]),
        "result_count": len(rows),
        "failure_count": failures,
        "expected_result_count": expected_sources * len(METHODS),
        "records": audit_records,
    }
    write_json(output / "run_manifest.json", manifest)
    if len(rows) != expected_sources * len(METHODS):
        raise RuntimeError(f"created {len(rows)} result rows, expected {expected_sources * len(METHODS)}")
    if failures:
        raise RuntimeError(
            f"Tesseract failed for {failures} images; failures are recorded in image_results.csv"
        )
    return manifest
