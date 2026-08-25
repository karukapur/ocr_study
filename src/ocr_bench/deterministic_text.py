from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cmp_to_key
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTCollection, TTFont

from .config import FontSpec


SUPERSAMPLE = 4


Point = tuple[int, int]
Contour = list[Point]


def _round_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -((-numerator) // denominator)


def _ceil_half_fraction(numerator: int, denominator: int) -> int:
    # ceil(numerator / denominator - 0.5), where denominator is positive.
    return _ceil_div(2 * numerator - denominator, 2 * denominator)


def _compare_rationals(left: tuple[int, int], right: tuple[int, int]) -> int:
    lhs = left[0] * right[1]
    rhs = right[0] * left[1]
    return (lhs > rhs) - (lhs < rhs)


def _open_ttfont(path: str, index: int) -> TTFont:
    font_path = Path(path)
    if font_path.suffix.lower() in {".ttc", ".otc"}:
        collection = TTCollection(str(font_path), lazy=False)
        if not 0 <= index < len(collection.fonts):
            collection.close()
            raise ValueError(f"font collection {font_path} has no face {index}")
        font = collection.fonts[index]
        # Detach enough state for deterministic read-only use after closing the
        # collection object would be risky, so keep the TTFont object cached.
        return font
    if index != 0:
        raise ValueError(f"{font_path} is not a collection, so index must be 0")
    return TTFont(str(font_path), lazy=False)


@lru_cache(maxsize=16)
def _cached_font(path: str, index: int) -> TTFont:
    font = _open_ttfont(path, index)
    if "glyf" not in font:
        raise ValueError("deterministic renderer only supports TrueType glyf fonts")
    return font


class _OutlinePen(BasePen):
    def __init__(self, glyph_set: Any, units_per_em: int, size: int):
        super().__init__(glyph_set)
        self.units_per_em = units_per_em
        self.size = size
        self.contours: list[Contour] = []
        self._current: Contour | None = None

    def _scale(self, point: tuple[float, float]) -> Point:
        x, y = point
        denominator = self.units_per_em
        scale = self.size * SUPERSAMPLE
        return (
            _round_div(int(round(x)) * scale, denominator),
            _round_div(int(round(y)) * scale, denominator),
        )

    def _moveTo(self, point: tuple[float, float]) -> None:
        if self._current:
            self.contours.append(self._current)
        self._current = [self._scale(point)]

    def _lineTo(self, point: tuple[float, float]) -> None:
        if self._current is None:
            self._moveTo(point)
        else:
            self._current.append(self._scale(point))

    def _qCurveToOne(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float],
    ) -> None:
        if self._current is None:
            self._moveTo(point2)
            return
        p0 = self._current[-1]
        p1 = self._scale(point1)
        p2 = self._scale(point2)
        maximum = max(
            abs(p1[0] - p0[0]),
            abs(p1[1] - p0[1]),
            abs(p2[0] - p1[0]),
            abs(p2[1] - p1[1]),
        )
        segments = max(1, min(96, _ceil_div(maximum, 3 * SUPERSAMPLE)))
        denom = segments * segments
        for index in range(1, segments + 1):
            u = segments - index
            t = index
            x = _round_div(u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0], denom)
            y = _round_div(u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1], denom)
            self._current.append((x, y))

    def _curveToOne(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float],
        point3: tuple[float, float],
    ) -> None:
        if self._current is None:
            self._moveTo(point3)
            return
        p0 = self._current[-1]
        p1 = self._scale(point1)
        p2 = self._scale(point2)
        p3 = self._scale(point3)
        maximum = max(
            abs(p1[0] - p0[0]),
            abs(p1[1] - p0[1]),
            abs(p2[0] - p1[0]),
            abs(p2[1] - p1[1]),
            abs(p3[0] - p2[0]),
            abs(p3[1] - p2[1]),
        )
        segments = max(1, min(128, _ceil_div(maximum, 3 * SUPERSAMPLE)))
        denom = segments * segments * segments
        for index in range(1, segments + 1):
            u = segments - index
            t = index
            x = _round_div(
                u * u * u * p0[0]
                + 3 * u * u * t * p1[0]
                + 3 * u * t * t * p2[0]
                + t * t * t * p3[0],
                denom,
            )
            y = _round_div(
                u * u * u * p0[1]
                + 3 * u * u * t * p1[1]
                + 3 * u * t * t * p2[1]
                + t * t * t * p3[1],
                denom,
            )
            self._current.append((x, y))

    def _closePath(self) -> None:
        if self._current:
            if self._current[0] != self._current[-1]:
                self._current.append(self._current[0])
            self.contours.append(self._current)
        self._current = None

    def _endPath(self) -> None:
        self._closePath()

    def finish(self) -> list[Contour]:
        if self._current:
            self._closePath()
        return [contour for contour in self.contours if len(contour) >= 3]


@dataclass(frozen=True)
class DeterministicFont:
    path: Path
    index: int
    size: int
    units_per_em: int
    ascent_sp: int
    descent_sp: int
    cmap: dict[int, str]
    advances: dict[str, int]
    glyph_set: Any
    ttfont: TTFont

    @classmethod
    def load(cls, spec: FontSpec, size: int) -> "DeterministicFont":
        ttfont = _cached_font(str(spec.path), spec.index)
        units_per_em = int(ttfont["head"].unitsPerEm)
        scale = size * SUPERSAMPLE
        ascent = _round_div(int(ttfont["hhea"].ascent) * scale, units_per_em)
        descent = _round_div(abs(int(ttfont["hhea"].descent)) * scale, units_per_em)
        cmap = ttfont.getBestCmap()
        hmtx = ttfont["hmtx"].metrics
        advances = {
            glyph_name: _round_div(int(metrics[0]) * scale, units_per_em)
            for glyph_name, metrics in hmtx.items()
        }
        return cls(
            path=spec.path,
            index=spec.index,
            size=size,
            units_per_em=units_per_em,
            ascent_sp=ascent,
            descent_sp=descent,
            cmap=dict(cmap),
            advances=advances,
            glyph_set=ttfont.getGlyphSet(),
            ttfont=ttfont,
        )

    def glyph_name(self, character: str) -> str:
        codepoint = ord(character)
        try:
            return self.cmap[codepoint]
        except KeyError as error:
            raise ValueError(f"font {self.path} has no glyph for U+{codepoint:04X}") from error

    def advance_sp(self, character: str) -> int:
        if character.isspace():
            glyph_name = self.cmap.get(ord(character), "space")
        else:
            glyph_name = self.glyph_name(character)
        return self.advances.get(glyph_name, self.advances.get(".notdef", 0))

    def contours(self, character: str) -> list[Contour]:
        if character.isspace():
            return []
        glyph_name = self.glyph_name(character)
        pen = _OutlinePen(self.glyph_set, self.units_per_em, self.size)
        self.glyph_set[glyph_name].draw(pen)
        return pen.finish()


def _contour_bounds(contours: list[Contour]) -> tuple[int, int, int, int] | None:
    points = [point for contour in contours for point in contour]
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _fill_contours(mask: np.ndarray, contours: list[Contour], dx: int, baseline_y: int) -> None:
    if not contours:
        return
    height, width = mask.shape
    transformed: list[Contour] = [
        [(x + dx, baseline_y - y) for x, y in contour] for contour in contours
    ]
    min_y = max(0, min(y for contour in transformed for _, y in contour))
    max_y = min(height - 1, max(y for contour in transformed for _, y in contour))
    for row in range(min_y, max_y + 1):
        y2 = 2 * row + 1
        intersections: list[tuple[int, int]] = []
        for contour in transformed:
            for (x0, y0), (x1, y1) in zip(contour, contour[1:]):
                if y0 == y1:
                    continue
                lower = min(y0, y1)
                upper = max(y0, y1)
                if not (2 * lower <= y2 < 2 * upper):
                    continue
                numerator = 2 * x0 * (y1 - y0) + (y2 - 2 * y0) * (x1 - x0)
                denominator = 2 * (y1 - y0)
                if denominator < 0:
                    numerator = -numerator
                    denominator = -denominator
                intersections.append((numerator, denominator))
        if len(intersections) < 2:
            continue
        intersections.sort(key=cmp_to_key(_compare_rationals))
        for left, right in zip(intersections[0::2], intersections[1::2]):
            left_x = max(0, _ceil_half_fraction(left[0], left[1]))
            right_x = min(width, _ceil_half_fraction(right[0], right[1]))
            if left_x < right_x:
                mask[row, left_x:right_x] = 0


def _downsample(mask: np.ndarray) -> np.ndarray:
    height_sp, width_sp = mask.shape
    height = _ceil_div(height_sp, SUPERSAMPLE)
    width = _ceil_div(width_sp, SUPERSAMPLE)
    padded = np.full((height * SUPERSAMPLE, width * SUPERSAMPLE), 255, dtype=np.uint8)
    padded[:height_sp, :width_sp] = mask
    blocks = padded.astype(np.uint32).reshape(
        height, SUPERSAMPLE, width, SUPERSAMPLE
    )
    sums = blocks.sum(axis=(1, 3), dtype=np.uint32)
    divisor = SUPERSAMPLE * SUPERSAMPLE
    return ((sums + divisor // 2) // divisor).astype(np.uint8)


def rasterize_glyph(font: DeterministicFont, character: str) -> np.ndarray:
    contours = font.contours(character)
    bounds = _contour_bounds(contours)
    if bounds is None:
        return np.full((1, 1), 255, dtype=np.uint8)
    left, bottom, right, top = bounds
    margin = SUPERSAMPLE * 2
    width = max(1, right - left + 2 * margin)
    height = max(1, top - bottom + 2 * margin)
    mask = np.full((height, width), 255, dtype=np.uint8)
    _fill_contours(mask, contours, margin - left, margin + top)
    return _downsample(mask)


def glyph_ink_height(font: DeterministicFont, character: str, threshold: int) -> int:
    image = rasterize_glyph(font, character)
    rows = np.flatnonzero((image < threshold).any(axis=1))
    return 0 if rows.size == 0 else int(rows[-1] - rows[0] + 1)


def text_layout_bounds(
    font: DeterministicFont, lines: list[str], line_advance_sp: int
) -> tuple[int, int, int, int, list[dict[str, Any]]]:
    glyph_boxes: list[dict[str, Any]] = []
    left: int | None = None
    right: int | None = None
    top: int | None = None
    bottom: int | None = None
    for line_index, line in enumerate(lines):
        cursor = 0
        baseline = line_index * line_advance_sp
        for char_index, character in enumerate(line):
            contours = font.contours(character)
            bounds = _contour_bounds(contours)
            if bounds is not None:
                x0, y0, x1, y1 = bounds
                box = (cursor + x0, baseline + y0, cursor + x1, baseline + y1)
                left = box[0] if left is None else min(left, box[0])
                bottom = box[1] if bottom is None else min(bottom, box[1])
                right = box[2] if right is None else max(right, box[2])
                top = box[3] if top is None else max(top, box[3])
                glyph_boxes.append(
                    {
                        "character": character,
                        "line": line_index,
                        "index": char_index,
                        "box_sp": list(box),
                    }
                )
            cursor += font.advance_sp(character)
    if left is None or right is None or top is None or bottom is None:
        raise ValueError("text contains no renderable glyphs")
    return left, bottom, right, top, glyph_boxes


def render_text(
    font: DeterministicFont,
    text: str,
    padding_px: int,
    line_spacing_px: int,
    canvas_multiple: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    lines = text.split("\n")
    spacing_sp = line_spacing_px * SUPERSAMPLE
    line_advance_sp = font.ascent_sp + font.descent_sp + spacing_sp
    left, bottom, right, top, glyph_boxes_sp = text_layout_bounds(
        font, lines, line_advance_sp
    )
    padding_sp = padding_px * SUPERSAMPLE
    content_width_sp = max(1, right - left)
    content_height_sp = max(1, top - bottom)
    unrounded_width_px = _ceil_div(content_width_sp, SUPERSAMPLE) + 2 * padding_px
    unrounded_height_px = _ceil_div(content_height_sp, SUPERSAMPLE) + 2 * padding_px
    side_px = int(math.ceil(max(unrounded_width_px, unrounded_height_px) / canvas_multiple) * canvas_multiple)
    side_sp = side_px * SUPERSAMPLE
    extra_x_sp = (side_sp - content_width_sp - 2 * padding_sp) // 2
    extra_y_sp = (side_sp - content_height_sp - 2 * padding_sp) // 2
    translate_x = padding_sp + extra_x_sp - left
    translate_y = padding_sp + extra_y_sp - bottom
    baseline_origin_y_down = side_sp - translate_y

    mask = np.full((side_sp, side_sp), 255, dtype=np.uint8)
    for line_index, line in enumerate(lines):
        cursor = 0
        baseline_y = baseline_origin_y_down - line_index * line_advance_sp
        for character in line:
            contours = font.contours(character)
            _fill_contours(mask, contours, translate_x + cursor, baseline_y)
            cursor += font.advance_sp(character)

    glyph_boxes = []
    for item in glyph_boxes_sp:
        x0, y0, x1, y1 = item["box_sp"]
        glyph_boxes.append(
            {
                "character": item["character"],
                "line": item["line"],
                "index": item["index"],
                "box": [
                    (x0 + translate_x) / SUPERSAMPLE,
                    (side_sp - (y1 + translate_y)) / SUPERSAMPLE,
                    (x1 + translate_x) / SUPERSAMPLE,
                    (side_sp - (y0 + translate_y)) / SUPERSAMPLE,
                ],
            }
        )
    return _downsample(mask), {
        "line_advance_sp": line_advance_sp,
        "supersample": SUPERSAMPLE,
        "glyph_boxes": glyph_boxes,
    }
