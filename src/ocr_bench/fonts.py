from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from fontTools.ttLib import TTCollection, TTFont

from .config import FontSpec, Pattern
from .util import sha256_file


class FontValidationError(RuntimeError):
    pass


def _open_face(path: Path, index: int):
    if path.suffix.lower() in {".ttc", ".otc"}:
        collection = TTCollection(str(path), lazy=False)
        if not 0 <= index < len(collection.fonts):
            collection.close()
            raise FontValidationError(
                f"font collection {path} has {len(collection.fonts)} faces; index {index} is invalid"
            )
        return collection, collection.fonts[index]
    if index != 0:
        raise FontValidationError(f"font {path} is not a collection, so its index must be 0")
    font = TTFont(str(path), lazy=False)
    return font, font


def _family_names(font: TTFont) -> set[str]:
    result: set[str] = set()
    for record in font["name"].names:
        if record.nameID in {1, 16}:
            try:
                result.add(record.toUnicode().strip())
            except UnicodeDecodeError:
                continue
    return {name for name in result if name}


def _codepoints(font: TTFont) -> set[int]:
    result: set[int] = set()
    for table in font["cmap"].tables:
        if table.isUnicode():
            result.update(table.cmap)
    return result


def validate_font(spec: FontSpec, characters: Iterable[str]) -> dict[str, object]:
    if not spec.path.is_file():
        raise FontValidationError(
            f"required font is missing: {spec.path}\n"
            "Place the licensed file at that path or update benchmark.yaml. "
            "No fallback font will be used."
        )
    owner, face = _open_face(spec.path, spec.index)
    try:
        families = _family_names(face)
        wanted = spec.expected_family.casefold()
        if not any(wanted == family.casefold() for family in families):
            raise FontValidationError(
                f"{spec.path} face {spec.index} has families {sorted(families)!r}, "
                f"expected {spec.expected_family!r}"
            )
        cmap = _codepoints(face)
        required = {ord(char) for char in characters if not char.isspace()}
        missing = sorted(required - cmap)
        if missing:
            preview = " ".join(f"U+{codepoint:04X}" for codepoint in missing[:20])
            raise FontValidationError(
                f"{spec.path} face {spec.index} lacks {len(missing)} required glyphs: {preview}"
            )
        return {
            **{key: str(value) if isinstance(value, Path) else value for key, value in asdict(spec).items()},
            "families": sorted(families),
            "sha256": sha256_file(spec.path),
            "required_codepoints": len(required),
        }
    finally:
        owner.close()


def validate_all_fonts(fonts: dict[str, FontSpec], patterns: tuple[Pattern, ...]) -> dict[str, object]:
    result: dict[str, object] = {}
    for language, spec in fonts.items():
        chars = "".join(pattern.text for pattern in patterns if pattern.language == language)
        result[language] = validate_font(spec, chars)
    return result
