from pathlib import Path

import pytest

from ocr_bench.config import FontSpec, Pattern
from ocr_bench.dataset import calibrate_font_size, render_pattern
from ocr_bench.fonts import FontValidationError, validate_font


def _test_font() -> Path:
    candidates = (
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    pytest.skip("no known test TrueType font is available")


def test_missing_font_fails_without_fallback(tmp_path: Path) -> None:
    spec = FontSpec(tmp_path / "missing.ttf", 0, "Missing")
    with pytest.raises(FontValidationError, match="No fallback font"):
        validate_font(spec, "ABC")


def test_calibration_and_canvas_geometry() -> None:
    spec = FontSpec(_test_font(), 0, "Arial")
    size, height = calibrate_font_size(spec, "Ag", 16 * 1.625, 250)
    assert size > 0
    assert abs(height - 16 * 1.625) <= 2

    pattern = Pattern("test", "en", "single_line", 7, "Ag Test")
    image, metadata = render_pattern(
        pattern,
        spec,
        ratio=1.625,
        target_height=16,
        threshold=250,
        ignore_punctuation=True,
        padding_target=8,
        line_spacing_target=8,
        canvas_multiple=6,
    )
    assert image.width % 6 == 0
    assert image.height % 6 == 0
    assert image.width == image.height
    assert metadata["source_max_glyph_height"] == height
    assert metadata["glyph_boxes"]
