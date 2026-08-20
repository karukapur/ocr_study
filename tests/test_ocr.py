import pytest

from ocr_bench.ocr import tesseract_language


def test_dataset_languages_map_to_tesseract_model_codes() -> None:
    assert tesseract_language("en") == "eng"
    assert tesseract_language("zh_tra") == "chi_tra"


def test_unknown_language_is_rejected() -> None:
    with pytest.raises(ValueError, match="no Tesseract language mapping"):
        tesseract_language("unknown")
