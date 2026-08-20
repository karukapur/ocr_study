from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FontSpec:
    path: Path
    index: int
    expected_family: str


@dataclass(frozen=True)
class Pattern:
    id: str
    language: str
    layout: str
    psm: int
    text: str


@dataclass(frozen=True)
class TesseractSpec:
    executable: str
    tessdata_dir: Path | None
    oem: int
    timeout_seconds: int


@dataclass(frozen=True)
class BenchmarkConfig:
    source_path: Path
    raw: dict[str, Any]
    target_height_px: int
    ink_threshold: int
    ignore_punctuation_in_glyph_height: bool
    padding_target_px: int
    line_spacing_target_px: int
    canvas_multiple: int
    ratios: tuple[float, ...]
    fonts: dict[str, FontSpec]
    tesseract: TesseractSpec
    patterns: tuple[Pattern, ...]


def build_ratios(start: float, stop: float, count: int, extra: list[float]) -> tuple[float, ...]:
    if count < 2:
        raise ValueError("ratio_grid.count must be at least 2")
    if not (0 < start <= stop):
        raise ValueError("ratio_grid must satisfy 0 < start <= stop")
    step = (stop - start) / (count - 1)
    values = [start + i * step for i in range(count)] + list(extra)
    # String rounding prevents insignificant binary differences from creating
    # accidental duplicates while preserving the requested values.
    unique = {round(float(value), 12) for value in values}
    return tuple(sorted(unique))


def load_config(path: str | Path) -> BenchmarkConfig:
    source_path = Path(path).resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")

    ratio = raw["ratio_grid"]
    ratios = build_ratios(
        float(ratio["start"]),
        float(ratio["stop"]),
        int(ratio["count"]),
        [float(value) for value in ratio.get("extra", [])],
    )

    base = source_path.parent
    fonts: dict[str, FontSpec] = {}
    for language, value in raw["fonts"].items():
        font_path = Path(value["path"])
        if not font_path.is_absolute():
            font_path = (base / font_path).resolve()
        fonts[language] = FontSpec(
            path=font_path,
            index=int(value.get("index", 0)),
            expected_family=str(value["expected_family"]),
        )

    patterns = tuple(
        Pattern(
            id=str(item["id"]),
            language=str(item["language"]),
            layout=str(item["layout"]),
            psm=int(item["psm"]),
            text=str(item["text"]),
        )
        for item in raw["patterns"]
    )
    _validate_patterns(patterns, fonts)

    tess = raw["tesseract"]
    tessdata_dir_value = tess.get("tessdata_dir")
    tessdata_dir = None
    if tessdata_dir_value:
        tessdata_dir = Path(tessdata_dir_value)
        if not tessdata_dir.is_absolute():
            tessdata_dir = (base / tessdata_dir).resolve()
    return BenchmarkConfig(
        source_path=source_path,
        raw=raw,
        target_height_px=int(raw["target_height_px"]),
        ink_threshold=int(raw["ink_threshold"]),
        ignore_punctuation_in_glyph_height=bool(
            raw.get("ignore_punctuation_in_glyph_height", True)
        ),
        padding_target_px=int(raw["padding_target_px"]),
        line_spacing_target_px=int(raw["line_spacing_target_px"]),
        canvas_multiple=int(raw["canvas_multiple"]),
        ratios=ratios,
        fonts=fonts,
        tesseract=TesseractSpec(
            executable=str(tess.get("executable", "tesseract")),
            tessdata_dir=tessdata_dir,
            oem=int(tess.get("oem", 1)),
            timeout_seconds=int(tess.get("timeout_seconds", 60)),
        ),
        patterns=patterns,
    )


def _validate_patterns(patterns: tuple[Pattern, ...], fonts: dict[str, FontSpec]) -> None:
    ids = [pattern.id for pattern in patterns]
    if len(ids) != len(set(ids)):
        raise ValueError("pattern ids must be unique")
    expected_psm = {"single_char": 10, "single_line": 7, "multiline": 6}
    for pattern in patterns:
        if pattern.language not in fonts:
            raise ValueError(f"pattern {pattern.id!r} has no configured font")
        if pattern.layout not in expected_psm:
            raise ValueError(f"pattern {pattern.id!r} has unsupported layout {pattern.layout!r}")
        if pattern.psm != expected_psm[pattern.layout]:
            raise ValueError(
                f"pattern {pattern.id!r}: PSM {pattern.psm} does not match "
                f"{pattern.layout} (expected {expected_psm[pattern.layout]})"
            )
        if not pattern.text.strip():
            raise ValueError(f"pattern {pattern.id!r} is empty")


def ratio_slug(ratio: float) -> str:
    return f"r_{ratio:.6f}".replace(".", "p")
