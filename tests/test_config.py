from pathlib import Path

import pytest
import yaml

from ocr_bench.config import build_ratios, load_config, ratio_slug


ROOT = Path(__file__).resolve().parents[1]


def _write_config(tmp_path: Path, resampling) -> Path:
    raw = yaml.safe_load((ROOT / "benchmark.yaml").read_text(encoding="utf-8"))
    if resampling is None:
        raw.pop("resampling", None)
    else:
        raw["resampling"] = resampling
    path = tmp_path / "benchmark.yaml"
    path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return path


def test_ratio_grid_has_sixteen_base_points_plus_extra() -> None:
    ratios = build_ratios(1.0, 3.0, 16, [2.625])
    assert len(ratios) == 17
    assert ratios[0] == 1.0
    assert ratios[-1] == 3.0
    assert 2.625 in ratios


def test_checked_in_config_has_expected_layout_psms() -> None:
    config = load_config(ROOT / "benchmark.yaml")
    assert len(config.patterns) == 13
    assert len(config.ratios) == 17
    assert all(
        pattern.psm == (10 if pattern.layout == "single_char" else 7)
        for pattern in config.patterns
    )
    assert [pattern.text for pattern in config.patterns[:2]] == ["A", "做"]
    assert [
        pattern.text for pattern in config.patterns if pattern.layout == "single_char"
    ] == ["A", "做", "剪"]
    assert config.resampling.methods == (
        "bin_floor",
        "bin_ceil",
        "opencv_bilinear",
        "floating_point_bilinear",
        "fixed_point_bilinear",
        "lanczos2",
        "lanczos3",
    )
    assert config.resampling.lanczos_taps == 7
    assert config.resampling.lanczos_phases == 16
    assert config.resampling.interpolation == "shift"


def test_example_config_is_valid_and_shows_all_methods() -> None:
    config = load_config(ROOT / "benchmark.yaml.example")
    assert config.resampling.methods == (
        "bin_floor",
        "bin_ceil",
        "opencv_bilinear",
        "floating_point_bilinear",
        "fixed_point_bilinear",
        "lanczos2",
        "lanczos3",
    )
    assert {pattern.layout for pattern in config.patterns} == {
        "single_char",
        "single_line",
        "multiline",
    }


def test_resampling_defaults_preserve_opencv_only(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, None))
    assert config.resampling.methods == ("opencv_bilinear",)
    assert config.resampling.lanczos_taps == 7
    assert config.resampling.lanczos_phases == 16
    assert config.resampling.interpolation == "shift"


def test_bilinear_accepts_a_scalar(tmp_path: Path) -> None:
    config = load_config(
        _write_config(
            tmp_path,
            {
                "binning": [],
                "bilinear": "floating",
                "lanczos": {"variants": [], "taps": 5, "phases": 8},
                "interpolation": "original",
            },
        )
    )
    assert config.resampling.methods == ("floating_point_bilinear",)
    assert config.resampling.lanczos_taps == 5
    assert config.resampling.lanczos_phases == 8
    assert config.resampling.interpolation == "original"


@pytest.mark.parametrize(
    ("resampling", "message"),
    [
        (
            {
                "binning": [],
                "bilinear": [],
                "lanczos": {"variants": []},
            },
            "select at least one method",
        ),
        (
            {
                "binning": ["floor", "floor"],
                "bilinear": [],
                "lanczos": {"variants": []},
            },
            "must not contain duplicates",
        ),
        (
            {
                "binning": [],
                "bilinear": ["floating"],
                "lanczos": {"variants": [], "taps": 6},
            },
            "positive odd integer",
        ),
        (
            {
                "binning": [],
                "bilinear": ["floating"],
                "lanczos": {"variants": [], "phases": 0},
            },
            "phases must be positive",
        ),
        (
            {
                "binning": [],
                "bilinear": ["opencv"],
                "lanczos": {"variants": []},
                "interpolation": "original",
            },
            "cannot include 'opencv'",
        ),
    ],
)
def test_invalid_resampling_config_is_rejected(
    tmp_path: Path, resampling: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(_write_config(tmp_path, resampling))


def test_ratio_slug_is_stable() -> None:
    assert ratio_slug(2.625) == "r_2p625000"
