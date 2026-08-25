import pytest

from ocr_bench.ocr import _parse_tesseract_version, tesseract_language


def test_dataset_languages_map_to_tesseract_model_codes() -> None:
    assert tesseract_language("en") == "eng"
    assert tesseract_language("zh_tra") == "chi_tra"


def test_unknown_language_is_rejected() -> None:
    with pytest.raises(ValueError, match="no Tesseract language mapping"):
        tesseract_language("unknown")


def test_tesseract_version_parser_accepts_551() -> None:
    assert _parse_tesseract_version("tesseract 5.5.1\n leptonica-1.85.0") == "5.5.1"
