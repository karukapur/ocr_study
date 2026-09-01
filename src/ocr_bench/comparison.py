from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .config import BenchmarkConfig
from .ranking import (
    BINNING_METHODS,
    COMBINED_LANGUAGES,
    RankedCondition,
    select_method_ratio_extremes,
)
from .resample import ResizeResult, reference_lanczos_resize, resize_image
from .util import sha256_file, software_environment, write_json


SUPPORTED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})
COMPARISON_DIRECTORY = "comparison_exports"


def _scan_images(root: Path, category: str) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"{category} input is not a directory: {root}")
    images = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        extensions = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"{category} input contains no supported images ({extensions}): {root}"
        )
    return images


def _declared_bit_depth(opened: Image.Image, path: Path) -> int:
    if opened.format == "PNG":
        with path.open("rb") as handle:
            header = handle.read(25)
        if len(header) >= 25 and header[:8] == b"\x89PNG\r\n\x1a\n":
            return int(header[24])
    if opened.format == "TIFF":
        bits = opened.tag_v2.get(258, 8)
        if isinstance(bits, tuple):
            return max(int(value) for value in bits)
        return int(bits)
    return 8


def _load_image(path: Path) -> tuple[np.ndarray, str, str]:
    with Image.open(path) as opened:
        frame_count = int(getattr(opened, "n_frames", 1))
        if frame_count != 1:
            raise ValueError(
                f"unsupported multi-frame image ({frame_count} frames): {path}"
            )
        original_mode = opened.mode
        bit_depth = _declared_bit_depth(opened, path)
        if bit_depth > 8:
            raise ValueError(
                f"high-bit-depth ({bit_depth}-bit) input is not supported: {path}"
            )
        if "transparency" in opened.info:
            raise ValueError(f"alpha/transparency is not supported: {path}")
        if original_mode in {"RGBA", "LA", "PA", "RGBa", "La"}:
            raise ValueError(
                f"alpha/transparency mode {original_mode!r} is not supported: {path}"
            )
        if original_mode in {"I", "F"} or original_mode.startswith("I;16"):
            raise ValueError(
                f"high-bit-depth/float mode {original_mode!r} is not supported: {path}"
            )
        if original_mode == "CMYK":
            raise ValueError(
                f"CMYK input is not supported; convert it to 8-bit RGB first: {path}"
            )

        oriented = ImageOps.exif_transpose(opened)
        if oriented.mode in {"1", "L"}:
            normalized = oriented.convert("L")
        elif oriented.mode == "P":
            normalized = oriented.convert("RGB")
        elif oriented.mode == "RGB":
            normalized = oriented.copy()
        else:
            raise ValueError(
                f"unsupported image mode {oriented.mode!r}; expected 8-bit grayscale or RGB: {path}"
            )
        pixels = np.asarray(normalized, dtype=np.uint8)
    return pixels, normalized.mode, original_mode


def _resize_channels(
    pixels: np.ndarray,
    resize_channel: Callable[[np.ndarray], ResizeResult],
) -> ResizeResult:
    if pixels.ndim == 2:
        return resize_channel(pixels)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("comparison export expects an L or RGB uint8 image")
    channel_results = [resize_channel(pixels[:, :, index]) for index in range(3)]
    metadata = dict(channel_results[0].metadata)
    floating_pixels = None
    if all(result.floating_pixels is not None for result in channel_results):
        floating_pixels = np.stack(
            [result.floating_pixels for result in channel_results], axis=2
        )
    return ResizeResult(
        pixels=np.stack([result.pixels for result in channel_results], axis=2),
        metadata=metadata,
        floating_pixels=floating_pixels,
    )


def _compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if not key.startswith("coefficient_bank_")
    }


def _tree_hash(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["relative_path"])):
        digest.update(str(record["relative_path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["source_sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _condition_manifest(condition: RankedCondition) -> dict[str, object]:
    return condition.to_dict()


def _save_png(pixels: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path, format="PNG", optimize=False)


def export_comparison(
    config: BenchmarkConfig,
    output: Path,
    natural_input: Path,
    imatest_input: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    image_csv = output / "image_results.csv"
    if not image_csv.is_file():
        raise FileNotFoundError(f"{image_csv} is missing; run the study stage first")

    destination = output / COMPARISON_DIRECTORY
    if destination.exists() and not force:
        raise FileExistsError(
            f"{destination} already exists; use --force to regenerate the comparison export"
        )

    roots = {
        "natural": natural_input.resolve(),
        "imatest": imatest_input.resolve(),
    }
    resolved_destination = destination.resolve()
    for category, root in roots.items():
        if resolved_destination.is_relative_to(root) or root.is_relative_to(
            resolved_destination
        ):
            raise ValueError(
                f"comparison output and {category} input directories cannot overlap: {root}"
            )
    scanned = {category: _scan_images(root, category) for category, root in roots.items()}

    results = pd.read_csv(image_csv)
    conditions = select_method_ratio_extremes(results)
    temporary = Path(tempfile.mkdtemp(prefix=".comparison_exports-", dir=output))
    records: list[dict[str, Any]] = []
    try:
        for category in ("natural", "imatest"):
            root = roots[category]
            for source_path in scanned[category]:
                relative = source_path.relative_to(root)
                pixels, normalized_mode, original_mode = _load_image(source_path)
                source_height, source_width = pixels.shape[:2]
                source_record: dict[str, Any] = {
                    "category": category,
                    "source_path": str(source_path),
                    "relative_path": relative.as_posix(),
                    "source_sha256": sha256_file(source_path),
                    "original_mode": original_mode,
                    "normalized_mode": normalized_mode,
                    "source_width": source_width,
                    "source_height": source_height,
                    "outputs": [],
                }

                for rank in ("best", "worst"):
                    condition = conditions[rank]
                    ratio = condition.requested_ratio
                    reference = _resize_channels(
                        pixels, lambda channel: reference_lanczos_resize(channel, ratio)
                    )
                    candidate = _resize_channels(
                        pixels,
                        lambda channel: resize_image(
                            channel,
                            ratio,
                            condition.method,
                            taps=config.resampling.lanczos_taps,
                            phases=config.resampling.lanczos_phases,
                            interpolation=config.resampling.interpolation,
                            boundary="reflect101",
                        ),
                    )
                    if reference.pixels.shape != candidate.pixels.shape:
                        raise RuntimeError(
                            f"reference/candidate shape mismatch for {source_path} at {rank}: "
                            f"{reference.pixels.shape} != {candidate.pixels.shape}"
                        )

                    output_name = f"{relative.name}.png"
                    for role, resized in (("reference", reference), ("candidate", candidate)):
                        relative_output = (
                            Path(category)
                            / rank
                            / role
                            / relative.parent
                            / output_name
                        )
                        temporary_path = temporary / relative_output
                        _save_png(resized.pixels, temporary_path)
                        output_height, output_width = resized.pixels.shape[:2]
                        source_record["outputs"].append(
                            {
                                "rank": rank,
                                "role": role,
                                "method": (
                                    "lanczos3_129tap_1024phase_reference"
                                    if role == "reference"
                                    else condition.method
                                ),
                                "requested_ratio": ratio,
                                "path": relative_output.as_posix(),
                                "sha256": sha256_file(temporary_path),
                                "width": output_width,
                                "height": output_height,
                                "resampler": _compact_metadata(resized.metadata),
                            }
                        )
                records.append(source_record)

        inputs = {}
        for category, root in roots.items():
            category_records = [
                record for record in records if record["category"] == category
            ]
            inputs[category] = {
                "root": str(root),
                "file_count": len(category_records),
                "tree_sha256": _tree_hash(category_records),
            }
        manifest = {
            "schema_version": 1,
            "benchmark_config": str(config.source_path),
            "benchmark_config_sha256": sha256_file(config.source_path),
            "image_results_path": str(image_csv),
            "image_results_sha256": sha256_file(image_csv),
            "software": software_environment(),
            "selection": {
                "scope": "combined",
                "languages": sorted(COMBINED_LANGUAGES),
                "metric": "micro_averaged_character_error_rate",
                "excluded_methods": sorted(BINNING_METHODS),
                "tie_break": ["lowest_requested_ratio", "method_id"],
                "conditions": {
                    rank: _condition_manifest(condition)
                    for rank, condition in conditions.items()
                },
            },
            "reference_filter": {
                "name": "Lanczos-3",
                "implementation": "separable_polyphase",
                "taps": 129,
                "tap_offsets": [-64, 64],
                "phases": 1024,
                "coefficient_precision": "float64",
                "kernel": "cutoff*sinc(cutoff*d)*sinc(cutoff*d/3)",
                "support": "abs(cutoff*d) < 3",
                "phase_normalization": "unit_dc_gain",
                "boundary": "reflect101",
                "color_domain": "encoded_8bit_samples",
                "coordinate_mapping": "shift",
                "intermediate_clipping": False,
                "pixel_rounding": "round_half_up_then_clip_uint8",
            },
            "inputs": inputs,
            "records": records,
        }
        write_json(temporary / "manifest.json", manifest)

        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
        return manifest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
