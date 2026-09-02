from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .util import pixel_sha256, read_json, sha256_file


DETERMINISTIC_FILES = (
    "aggregate_results.csv",
)


def _file_hash(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return sha256_file(path)


def _load_image_pixel_hash(path: Path) -> str:
    pixels = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return pixel_sha256(pixels)


def _compare_manifest_images(left: Path, right: Path, manifest_name: str) -> list[str]:
    left_manifest = left / manifest_name
    right_manifest = right / manifest_name
    if not left_manifest.is_file() or not right_manifest.is_file():
        return []
    left_data = read_json(left_manifest)
    right_data = read_json(right_manifest)
    problems: list[str] = []
    left_sources = {
        record["image_path"]: record
        for record in left_data.get("sources", [])
        if "image_path" in record
    }
    right_sources = {
        record["image_path"]: record
        for record in right_data.get("sources", [])
        if "image_path" in record
    }
    for image_path in sorted(set(left_sources) | set(right_sources)):
        if image_path not in left_sources or image_path not in right_sources:
            problems.append(f"{manifest_name}: missing image record {image_path}")
            continue
        left_hash = _load_image_pixel_hash(left / image_path)
        right_hash = _load_image_pixel_hash(right / image_path)
        if left_hash != right_hash:
            problems.append(f"{image_path}: pixel hash differs")
    return problems


def _compare_run_records(left: Path, right: Path) -> list[str]:
    problems: list[str] = []
    left_manifest = left / "run_manifest.json"
    right_manifest = right / "run_manifest.json"
    if not left_manifest.is_file() or not right_manifest.is_file():
        return problems
    left_data: dict[str, Any] = read_json(left_manifest)
    right_data: dict[str, Any] = read_json(right_manifest)
    left_records = {
        (
            record.get("method"),
            record.get("pattern_id"),
            f"{float(record.get('requested_ratio')):.12f}",
        ): record
        for record in left_data.get("records", [])
        if "output_sha256" in record
    }
    right_records = {
        (
            record.get("method"),
            record.get("pattern_id"),
            f"{float(record.get('requested_ratio')):.12f}",
        ): record
        for record in right_data.get("records", [])
        if "output_sha256" in record
    }
    for key in sorted(set(left_records) | set(right_records)):
        if key not in left_records or key not in right_records:
            problems.append(f"run_manifest.json: missing result record {key}")
            continue
        left_record = left_records[key]
        right_record = right_records[key]
        left_path = left_record.get("output_path")
        right_path = right_record.get("output_path")
        if not isinstance(left_path, str) or not isinstance(right_path, str):
            continue
        left_hash = _load_image_pixel_hash(left / left_path)
        right_hash = _load_image_pixel_hash(right / right_path)
        if left_hash != right_hash:
            problems.append(f"{key}: output pixel hash differs")
    return problems


def _strip_volatile(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_volatile(item)
            for key, item in value.items()
            if key
            not in {
                "git_commit",
                "software",
                "ocr_seconds",
                "elapsed_seconds",
                "executable",
                "tessdata_dir",
                "sha256",
                "source_sha256",
                "output_sha256",
                "pixel_sha256",
                "source_pixel_sha256",
                "output_pixel_sha256",
            }
        }
    if isinstance(value, list):
        return [_strip_volatile(item) for item in value]
    return value


def _compare_json_semantic(left: Path, right: Path, relative: str) -> list[str]:
    left_path = left / relative
    right_path = right / relative
    if not left_path.exists() and not right_path.exists():
        return []
    if not left_path.exists() or not right_path.exists():
        return [f"{relative}: missing from one run"]
    if _strip_volatile(read_json(left_path)) != _strip_volatile(read_json(right_path)):
        return [f"{relative}: semantic content differs"]
    return []


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(
                {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "ocr_seconds",
                        "source_sha256",
                        "output_sha256",
                        "source_pixel_sha256",
                        "output_pixel_sha256",
                        "ocr_stdout_sha256",
                    }
                }
            )
        return rows


def _compare_csv_semantic(left: Path, right: Path, relative: str) -> list[str]:
    left_path = left / relative
    right_path = right / relative
    if not left_path.exists() and not right_path.exists():
        return []
    if not left_path.exists() or not right_path.exists():
        return [f"{relative}: missing from one run"]
    if _read_csv_rows(left_path) != _read_csv_rows(right_path):
        return [f"{relative}: semantic rows differ"]
    return []


def compare_runs(left: Path, right: Path) -> list[str]:
    problems: list[str] = []
    for relative in DETERMINISTIC_FILES:
        left_path = left / relative
        right_path = right / relative
        if not left_path.exists() and not right_path.exists():
            continue
        if not left_path.exists() or not right_path.exists():
            problems.append(f"{relative}: missing from one run")
            continue
        if _file_hash(left_path) != _file_hash(right_path):
            problems.append(f"{relative}: file hash differs")
    problems.extend(_compare_json_semantic(left, right, "run_manifest.json"))
    problems.extend(_compare_json_semantic(left, right, "source_manifest.json"))
    problems.extend(_compare_csv_semantic(left, right, "image_results.csv"))
    problems.extend(_compare_manifest_images(left, right, "source_manifest.json"))
    problems.extend(_compare_run_records(left, right))
    return problems
