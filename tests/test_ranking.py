from pathlib import Path

import pandas as pd

from ocr_bench.ranking import select_method_ratio_extremes


ROOT = Path(__file__).resolve().parents[1]


def test_non_binning_ranking_excludes_bins_and_breaks_ties_by_lowest_ratio() -> None:
    rows = pd.DataFrame(
        [
            {"method": "bin_floor", "requested_ratio": 1.0, "edit_distance": 0, "reference_chars": 10, "status": "ok"},
            {"method": "bin_ceil", "requested_ratio": 3.0, "edit_distance": 10, "reference_chars": 10, "status": "ok"},
            {"method": "lanczos3", "requested_ratio": 2.0, "edit_distance": 1, "reference_chars": 10, "status": "ok"},
            {"method": "lanczos3", "requested_ratio": 1.8, "edit_distance": 1, "reference_chars": 10, "status": "ok"},
            {"method": "opencv_bilinear", "requested_ratio": 2.6, "edit_distance": 9, "reference_chars": 10, "status": "ok"},
            {"method": "opencv_bilinear", "requested_ratio": 2.333333333333, "edit_distance": 9, "reference_chars": 10, "status": "ok"},
        ]
    )

    selected = select_method_ratio_extremes(rows)

    assert selected["best"].method == "lanczos3"
    assert selected["best"].requested_ratio == 1.8
    assert selected["best"].tie_count == 2
    assert selected["worst"].method == "opencv_bilinear"
    assert selected["worst"].requested_ratio == 2.333333333333
    assert selected["worst"].tie_count == 2


def test_checked_in_run_selects_expected_combined_conditions() -> None:
    results = pd.read_csv(ROOT / "artifacts/run-001/image_results.csv")
    selected = select_method_ratio_extremes(results)
    assert selected["best"].method == "lanczos3_7tap_16phase"
    assert selected["best"].requested_ratio == 1.8
    assert selected["worst"].method == "opencv_bilinear"
    assert selected["worst"].requested_ratio == 2.333333333333
