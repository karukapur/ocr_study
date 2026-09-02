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


def test_generate_copies_used_config_into_run_directory(
    tmp_path: Path, monkeypatch
) -> None:
    source = Path("benchmark.yaml").resolve()
    output = tmp_path / "run"
    monkeypatch.setattr(
        "ocr_bench.cli.generate_sources",
        lambda *args, **kwargs: {"sources": []},
    )

    exit_code = main(
        ["generate", "--config", str(source), "--output", str(output)]
    )

    assert exit_code == 0
    assert (output / "benchmark_config.yaml").read_bytes() == source.read_bytes()
