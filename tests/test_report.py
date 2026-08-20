from pathlib import Path

import pandas as pd

from ocr_bench.report import _write_performance_summary


ROOT = Path(__file__).resolve().parents[1]


def test_performance_summary_includes_non_binning_extremes(tmp_path: Path) -> None:
    results = pd.read_csv(ROOT / "artifacts/run-001/image_results.csv")
    results = results[results["status"] == "ok"].copy()

    paths = _write_performance_summary(results, tmp_path)

    assert paths == [
        tmp_path / "performance_summary.txt",
        tmp_path / "performance_extremes.csv",
    ]
    summary = paths[0].read_text(encoding="utf-8")
    assert "Best non-binning method×ratio condition: lanczos3 at 1.800×" in summary
    assert "Worst non-binning method×ratio condition: bilinear at 2.333×" in summary
    structured = pd.read_csv(paths[1])
    combined = structured[
        (structured["scope"] == "combined")
        & (structured["category"] == "non_binning_method_by_ratio")
    ]
    assert combined["rank"].tolist() == ["best", "worst"]
