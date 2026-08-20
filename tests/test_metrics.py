import pytest

from ocr_bench.metrics import (
    aggregate_rows,
    character_error_rate,
    levenshtein_distance,
    normalize_text,
)


def test_recognition_normalization_preserves_case_and_punctuation() -> None:
    assert normalize_text("  A,\r\n  B!  ") == "A, B!"
    assert normalize_text("A") != normalize_text("a")


def test_levenshtein_and_cer() -> None:
    assert levenshtein_distance("kitten", "sitting") == 3
    distance, length, cer = character_error_rate("ABC", "ADC")
    assert (distance, length) == (1, 3)
    assert cer == pytest.approx(1 / 3)


def test_combined_cer_is_micro_average() -> None:
    rows = [
        {
            "status": "ok",
            "requested_ratio": 1.0,
            "method": "m",
            "language": "en",
            "edit_distance": 1,
            "reference_chars": 10,
        },
        {
            "status": "ok",
            "requested_ratio": 1.0,
            "method": "m",
            "language": "zh_tra",
            "edit_distance": 2,
            "reference_chars": 2,
        },
    ]
    aggregate = aggregate_rows(rows)
    combined = next(row for row in aggregate if row["scope"] == "combined")
    assert combined["cer"] == pytest.approx(3 / 12)
    assert combined["pattern_count"] == 2

