from pathlib import Path

import pytest

from ocr_bench.cli import _parser, main


def test_deterministic_resize_option_is_removed() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["study", "--deterministic-resize"])


def test_baseline_run_requires_exact(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "study",
            "--config",
            "benchmark.yaml",
            "--output",
            str(tmp_path / "run"),
            "--baseline-run",
            str(tmp_path / "baseline"),
        ]
    )
    assert exit_code == 2
    assert "--baseline-run requires --exact" in capsys.readouterr().err
