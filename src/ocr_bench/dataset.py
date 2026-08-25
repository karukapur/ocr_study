from __future__ import annotations

import math
import shutil
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import BenchmarkConfig, FontSpec, Pattern, ratio_slug
from .deterministic_text import (
    DeterministicFont,
    render_text as deterministic_render_text,
    text_layout_bounds,
)
from .fonts import validate_all_fonts
from .util import pixel_sha256, save_deterministic_png, sha256_file, write_json


def load_font(spec: FontSpec, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(spec.path), size=size, index=spec.index)


def glyph_ink_height(font: ImageFont.FreeTypeFont, character: str, threshold: int) -> int:
    if character.isspace():
        return 0
    mask = font.getmask(character, mode="L")
    width, height = mask.size
    if width == 0 or height == 0:
        return 0
    pixels = np.asarray(mask, dtype=np.uint8).reshape(height, width)
    visible = pixels > (255 - threshold)
    rows = np.flatnonzero(visible.any(axis=1))
    return 0 if rows.size == 0 else int(rows[-1] - rows[0] + 1)


def max_glyph_ink_height(
    font: ImageFont.FreeTypeFont,
    text: str,
    threshold: int,
    ignore_punctuation: bool = True,
) -> int:
    unique = sorted(
        {
            character
            for character in text
            if not character.isspace()
            and not (ignore_punctuation and unicodedata.category(character).startswith("P"))
        }
    )
    heights = [glyph_ink_height(font, character, threshold) for character in unique]
    return max(heights, default=0)


def calibrate_font_size(
    spec: FontSpec,
    text: str,
    target_source_height: float,
    threshold: int,
    ignore_punctuation: bool = True,
    maximum_size: int = 512,
) -> tuple[int, int]:
    best: tuple[float, int, int] | None = None
    past_target_count = 0
    for size in range(1, maximum_size + 1):
        font = load_font(spec, size)
        height = max_glyph_ink_height(font, text, threshold, ignore_punctuation)
        candidate = (abs(height - target_source_height), size, height)
        if height > 0 and (best is None or candidate[:2] < best[:2]):
            best = candidate
        if height > target_source_height + 4:
            past_target_count += 1
            if past_target_count >= 4 and best is not None:
                break
        else:
            past_target_count = 0
    if best is None:
        raise RuntimeError(f"could not render any visible glyphs from {spec.path}")
    _, size, height = best
    return size, height


def calibrate_deterministic_font_size(
    spec: FontSpec,
    text: str,
    target_source_height: float,
    threshold: int,
    ignore_punctuation: bool = True,
    maximum_size: int = 512,
) -> tuple[int, int]:
    best: tuple[float, int, int] | None = None
    unique = sorted(
        {
            character
            for character in text
            if not character.isspace()
            and not (ignore_punctuation and unicodedata.category(character).startswith("P"))
        }
    )
    if not unique:
        raise RuntimeError(f"could not render any visible glyphs from {spec.path}")
    unit_font = DeterministicFont.load(spec, 1024)
    unit_heights: list[float] = []
    for character in unique:
        contours = unit_font.contours(character)
        if not contours:
            continue
        _, bottom, _, top, _ = text_layout_bounds(unit_font, [character], 0)
        unit_heights.append((top - bottom) / (1024 * 4))
    max_unit_height = max(unit_heights, default=0.0)
    if max_unit_height <= 0:
        raise RuntimeError(f"could not render any visible glyphs from {spec.path}")
    estimate = max(1, min(maximum_size, int(round(target_source_height / max_unit_height))))
    for size in range(max(1, estimate - 12), min(maximum_size, estimate + 12) + 1):
        height = max(1, int(round(size * max_unit_height)))
        candidate = (abs(height - target_source_height), size, height)
        if height > 0 and (best is None or candidate[:2] < best[:2]):
            best = candidate
    if best is None:
        raise RuntimeError(f"could not render any visible glyphs from {spec.path}")
    _, size, height = best
    return size, height


def _ceil_multiple(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def _portable_path(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def _portable_font_info(font_info: dict[str, Any], base: Path) -> dict[str, Any]:
    portable: dict[str, Any] = {}
    for language, info in font_info.items():
        if isinstance(info, dict):
            item = dict(info)
            if "path" in item:
                item["path"] = _portable_path(Path(str(item["path"])), base)
            portable[language] = item
        else:
            portable[language] = info
    return portable


def render_pattern(
    pattern: Pattern,
    spec: FontSpec,
    ratio: float,
    target_height: int,
    threshold: int,
    ignore_punctuation: bool,
    padding_target: int,
    line_spacing_target: int,
    canvas_multiple: int,
) -> tuple[Image.Image, dict[str, Any]]:
    target_source_height = target_height * ratio
    font_size, source_max_height = calibrate_font_size(
        spec, pattern.text, target_source_height, threshold, ignore_punctuation
    )
    font = load_font(spec, font_size)
    ascent, descent = font.getmetrics()
    spacing = max(0, int(math.floor(line_spacing_target * ratio + 0.5)))
    line_advance = ascent + descent + spacing

    probe = ImageDraw.Draw(Image.new("L", (1, 1), 255))
    lines = pattern.text.split("\n")
    line_baselines = [index * line_advance for index in range(len(lines))]
    line_boxes: list[tuple[float, float, float, float]] = []
    glyph_boxes_unshifted: list[dict[str, Any]] = []
    for line_index, (line, baseline) in enumerate(zip(lines, line_baselines)):
        if line:
            line_boxes.append(probe.textbbox((0, baseline), line, font=font, anchor="ls"))
        for char_index, character in enumerate(line):
            if character.isspace():
                continue
            prefix_width = float(probe.textlength(line[:char_index], font=font))
            box = probe.textbbox((prefix_width, baseline), character, font=font, anchor="ls")
            glyph_boxes_unshifted.append(
                {
                    "character": character,
                    "line": line_index,
                    "index": char_index,
                    "box": [float(value) for value in box],
                }
            )
    if not line_boxes:
        raise ValueError(f"pattern {pattern.id!r} contains no renderable lines")

    left = min(box[0] for box in line_boxes)
    top = min(box[1] for box in line_boxes)
    right = max(box[2] for box in line_boxes)
    bottom = max(box[3] for box in line_boxes)
    padding = max(1, int(math.floor(padding_target * ratio + 0.5)))
    content_width = max(1, int(math.ceil(right - left)))
    content_height = max(1, int(math.ceil(bottom - top)))
    unrounded_width = content_width + 2 * padding
    unrounded_height = content_height + 2 * padding
    side = _ceil_multiple(max(unrounded_width, unrounded_height), canvas_multiple)
    width = side
    height = side
    extra_x = (side - unrounded_width) // 2
    extra_y = (side - unrounded_height) // 2
    translate_x = padding + extra_x - left
    translate_y = padding + extra_y - top

    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    for line, baseline in zip(lines, line_baselines):
        draw.text(
            (translate_x, translate_y + baseline),
            line,
            font=font,
            fill=0,
            anchor="ls",
        )

    glyph_boxes = []
    for item in glyph_boxes_unshifted:
        x0, y0, x1, y1 = item["box"]
        glyph_boxes.append(
            {
                **{key: value for key, value in item.items() if key != "box"},
                "box": [
                    x0 + translate_x,
                    y0 + translate_y,
                    x1 + translate_x,
                    y1 + translate_y,
                ],
            }
        )

    metadata = {
        "pattern_id": pattern.id,
        "language": pattern.language,
        "layout": pattern.layout,
        "psm": pattern.psm,
        "reference": pattern.text,
        "requested_ratio": ratio,
        "target_output_glyph_height": target_height,
        "target_source_glyph_height": target_source_height,
        "font_size": font_size,
        "source_max_glyph_height": source_max_height,
        "source_height_error": source_max_height - target_source_height,
        "ink_threshold": threshold,
        "ignore_punctuation_in_glyph_height": ignore_punctuation,
        "padding_source_px": padding,
        "line_spacing_source_px": spacing,
        "canvas_width": width,
        "canvas_height": height,
        "glyph_boxes": glyph_boxes,
    }
    return image, metadata


def render_pattern_deterministic(
    pattern: Pattern,
    spec: FontSpec,
    ratio: float,
    target_height: int,
    threshold: int,
    ignore_punctuation: bool,
    padding_target: int,
    line_spacing_target: int,
    canvas_multiple: int,
) -> tuple[Image.Image, dict[str, Any], np.ndarray]:
    target_source_height = target_height * ratio
    font_size, source_max_height = calibrate_deterministic_font_size(
        spec, pattern.text, target_source_height, threshold, ignore_punctuation
    )
    padding = max(1, int(math.floor(padding_target * ratio + 0.5)))
    spacing = max(0, int(math.floor(line_spacing_target * ratio + 0.5)))
    font = DeterministicFont.load(spec, font_size)
    pixels, render_metadata = deterministic_render_text(
        font,
        pattern.text,
        padding_px=padding,
        line_spacing_px=spacing,
        canvas_multiple=canvas_multiple,
    )
    metadata = {
        "pattern_id": pattern.id,
        "language": pattern.language,
        "layout": pattern.layout,
        "psm": pattern.psm,
        "reference": pattern.text,
        "requested_ratio": ratio,
        "target_output_glyph_height": target_height,
        "target_source_glyph_height": target_source_height,
        "font_size": font_size,
        "source_max_glyph_height": source_max_height,
        "source_height_error": source_max_height - target_source_height,
        "ink_threshold": threshold,
        "ignore_punctuation_in_glyph_height": ignore_punctuation,
        "padding_source_px": padding,
        "line_spacing_source_px": spacing,
        "canvas_width": int(pixels.shape[1]),
        "canvas_height": int(pixels.shape[0]),
        "glyph_boxes": render_metadata["glyph_boxes"],
        "renderer": {
            "name": "ocr_bench_deterministic_text",
            "font_engine": "fontTools glyf outlines",
            "hinting": "disabled",
            "supersample": render_metadata["supersample"],
        },
    }
    return Image.fromarray(pixels, mode="L"), metadata, pixels


def measure_resized_glyph_height(
    image: np.ndarray,
    source_width: int,
    source_height: int,
    glyph_boxes: list[dict[str, Any]],
    threshold: int,
) -> int:
    output_height, output_width = image.shape
    scale_x = output_width / source_width
    scale_y = output_height / source_height
    maximum = 0
    for item in glyph_boxes:
        x0, y0, x1, y1 = item["box"]
        left = max(0, int(math.floor(x0 * scale_x)) - 2)
        top = max(0, int(math.floor(y0 * scale_y)) - 2)
        right = min(output_width, int(math.ceil(x1 * scale_x)) + 2)
        bottom = min(output_height, int(math.ceil(y1 * scale_y)) + 2)
        if left >= right or top >= bottom:
            continue
        crop = image[top:bottom, left:right]
        rows = np.flatnonzero((crop < threshold).any(axis=1))
        if rows.size:
            maximum = max(maximum, int(rows[-1] - rows[0] + 1))
    return maximum


def generate_sources(
    config: BenchmarkConfig,
    output: Path,
    force: bool = False,
    *,
    exact: bool = False,
    deterministic_renderer: bool = False,
) -> dict[str, Any]:
    canonical = output / "canonical"
    if canonical.exists():
        if not force:
            raise FileExistsError(
                f"{canonical} already exists; use --force to regenerate the source stage"
            )
        shutil.rmtree(canonical)
    canonical.mkdir(parents=True, exist_ok=True)

    font_info = validate_all_fonts(config.fonts, config.patterns)
    records: list[dict[str, Any]] = []
    for pattern in config.patterns:
        spec = config.fonts[pattern.language]
        pattern_dir = canonical / pattern.id
        pattern_dir.mkdir(parents=True, exist_ok=True)
        for ratio in config.ratios:
            if deterministic_renderer:
                image, metadata, pixels = render_pattern_deterministic(
                    pattern=pattern,
                    spec=spec,
                    ratio=ratio,
                    target_height=config.target_height_px,
                    threshold=config.ink_threshold,
                    ignore_punctuation=config.ignore_punctuation_in_glyph_height,
                    padding_target=config.padding_target_px,
                    line_spacing_target=config.line_spacing_target_px,
                    canvas_multiple=config.canvas_multiple,
                )
            else:
                image, metadata = render_pattern(
                    pattern=pattern,
                    spec=spec,
                    ratio=ratio,
                    target_height=config.target_height_px,
                    threshold=config.ink_threshold,
                    ignore_punctuation=config.ignore_punctuation_in_glyph_height,
                    padding_target=config.padding_target_px,
                    line_spacing_target=config.line_spacing_target_px,
                    canvas_multiple=config.canvas_multiple,
                )
                pixels = np.asarray(image, dtype=np.uint8)
            relative_path = Path("canonical") / pattern.id / f"{ratio_slug(ratio)}.png"
            image_path = output / relative_path
            if exact:
                save_deterministic_png(image_path, pixels)
            else:
                image.save(image_path, format="PNG", optimize=False)
            record = {
                **metadata,
                "font_path": (
                    _portable_path(spec.path, config.source_path.parent)
                    if exact
                    else str(spec.path)
                ),
                "font_index": spec.index,
                "image_path": relative_path.as_posix(),
                "sha256": sha256_file(image_path),
            }
            if exact:
                record["pixel_sha256"] = pixel_sha256(pixels)
            records.append(record)

    manifest = {
        "schema_version": 1,
        "config_path": (
            _portable_path(config.source_path, config.source_path.parent)
            if exact
            else str(config.source_path)
        ),
        "config_sha256": sha256_file(config.source_path),
        "ratios": list(config.ratios),
        "fonts": _portable_font_info(font_info, config.source_path.parent) if exact else font_info,
        "expected_source_count": len(config.patterns) * len(config.ratios),
        "sources": records,
    }
    if exact:
        manifest["exact_mode"] = True
        manifest["deterministic_renderer"] = deterministic_renderer
    write_json(output / "source_manifest.json", manifest)
    return manifest
