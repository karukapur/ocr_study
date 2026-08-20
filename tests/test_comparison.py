from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ocr_bench.comparison import export_comparison
from ocr_bench.config import load_config
from ocr_bench.util import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _write_ranking_csv(output: Path) -> None:
    rows = [
        ("bin_floor", 1.0, 0, 10),
        ("bin_ceil", 3.0, 10, 10),
        ("lanczos3_7tap_16phase", 1.8, 1, 10),
        ("lanczos3_7tap_16phase", 2.0, 1, 10),
        ("opencv_bilinear", 2.333333333333, 9, 10),
        ("opencv_bilinear", 2.6, 9, 10),
    ]
    pd.DataFrame(
        [
            {
                "method": method,
                "requested_ratio": ratio,
                "edit_distance": distance,
                "reference_chars": length,
                "status": "ok",
            }
            for method, ratio, distance, length in rows
        ]
    ).to_csv(output / "image_results.csv", index=False)


def _save_inputs(natural: Path, imatest: Path) -> None:
    natural.mkdir()
    imatest.mkdir()
    grayscale = np.arange(12 * 14, dtype=np.uint8).reshape(12, 14)
    rgb = np.stack([grayscale, 255 - grayscale, grayscale // 2], axis=2)
    formats = (
        ("nested/a.png", grayscale),
        ("same.jpg", rgb),
        ("same.jpeg", rgb),
        ("sample.tif", rgb),
        ("sample.tiff", grayscale),
        ("sample.bmp", rgb),
    )
    for relative, pixels in formats:
        path = natural / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(path)

    palette = Image.fromarray(grayscale).convert("P")
    palette.save(imatest / "palette.png")


def test_export_comparison_creates_matched_trees_and_manifest(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    _write_ranking_csv(output)
    natural = tmp_path / "natural"
    imatest = tmp_path / "imatest"
    _save_inputs(natural, imatest)

    manifest = export_comparison(
        load_config(ROOT / "benchmark.yaml"), output, natural, imatest
    )

    destination = output / "comparison_exports"
    assert manifest["inputs"]["natural"]["file_count"] == 6
    assert manifest["inputs"]["imatest"]["file_count"] == 1
    assert manifest["selection"]["conditions"]["best"]["requested_ratio"] == 1.8
    assert manifest["selection"]["conditions"]["worst"]["requested_ratio"] == 2.333333333333
    assert (destination / "natural/best/reference/same.jpg.png").is_file()
    assert (destination / "natural/best/reference/same.jpeg.png").is_file()

    loaded_manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    for record in loaded_manifest["records"]:
        assert len(record["outputs"]) == 4
        for rank in ("best", "worst"):
            pair = [item for item in record["outputs"] if item["rank"] == rank]
            assert {(item["role"]) for item in pair} == {"reference", "candidate"}
            assert len({(item["width"], item["height"]) for item in pair}) == 1
            for item in pair:
                path = destination / item["path"]
                assert path.is_file()
                assert sha256_file(path) == item["sha256"]

    with pytest.raises(FileExistsError, match="--force"):
        export_comparison(load_config(ROOT / "benchmark.yaml"), output, natural, imatest)
    export_comparison(
        load_config(ROOT / "benchmark.yaml"), output, natural, imatest, force=True
    )


@pytest.mark.parametrize("unsupported", ["rgba", "uint16", "multiframe"])
def test_export_rejects_unsupported_images_without_partial_output(
    tmp_path: Path, unsupported: str
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    _write_ranking_csv(output)
    natural = tmp_path / "natural"
    imatest = tmp_path / "imatest"
    natural.mkdir()
    imatest.mkdir()
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(imatest / "valid.png")

    path = natural / "invalid.png"
    if unsupported == "rgba":
        Image.fromarray(np.zeros((8, 8, 4), dtype=np.uint8), mode="RGBA").save(path)
    elif unsupported == "uint16":
        Image.fromarray(np.zeros((8, 8), dtype=np.uint16)).save(path)
    else:
        path = natural / "invalid.tiff"
        frames = [Image.new("L", (8, 8), value) for value in (0, 255)]
        frames[0].save(path, save_all=True, append_images=frames[1:])

    with pytest.raises(ValueError):
        export_comparison(load_config(ROOT / "benchmark.yaml"), output, natural, imatest)
    assert not (output / "comparison_exports").exists()


def test_export_rejects_input_inside_stage_owned_output(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    _write_ranking_csv(output)
    natural = output / "comparison_exports" / "user-input"
    imatest = tmp_path / "imatest"
    natural.mkdir(parents=True)
    imatest.mkdir()
    Image.new("L", (8, 8)).save(natural / "natural.png")
    Image.new("L", (8, 8)).save(imatest / "imatest.png")

    with pytest.raises(ValueError, match="cannot overlap"):
        export_comparison(
            load_config(ROOT / "benchmark.yaml"),
            output,
            natural,
            imatest,
            force=True,
        )
    assert (natural / "natural.png").is_file()
