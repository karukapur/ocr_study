from pathlib import Path

from ocr_bench.config import build_ratios, load_config, ratio_slug


ROOT = Path(__file__).resolve().parents[1]


def test_ratio_grid_has_sixteen_base_points_plus_extra() -> None:
    ratios = build_ratios(1.0, 3.0, 16, [2.625])
    assert len(ratios) == 17
    assert ratios[0] == 1.0
    assert ratios[-1] == 3.0
    assert 2.625 in ratios


def test_checked_in_config_has_expected_layout_psms() -> None:
    config = load_config(ROOT / "benchmark.yaml")
    assert len(config.patterns) == 29
    assert len(config.ratios) == 17
    assert all(
        pattern.psm == (10 if pattern.layout == "single_char" else 7)
        for pattern in config.patterns
    )
    assert [pattern.text for pattern in config.patterns[:2]] == ["A", "8"]
    assert [
        pattern.text for pattern in config.patterns if pattern.layout == "single_char"
    ] == ["A", "8", "做", "剪"]


def test_ratio_slug_is_stable() -> None:
    assert ratio_slug(2.625) == "r_2p625000"
