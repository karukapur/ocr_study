from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image


EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[1] / "experiments" / "replicate_bilinear"
)
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from compare_bilinear import (  # noqa: E402
    find_patterns,
    load_rgb,
    plot_results,
    resize_rgb_planes,
)
from ocr_bench.resample import opencv_bilinear  # noqa: E402


def test_find_and_load_png_tiff_and_svg_as_rgb(tmp_path: Path) -> None:
    pixels = np.array(
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]],
        dtype=np.uint8,
    )
    Image.fromarray(pixels, mode="RGB").save(tmp_path / "pattern.PNG")
    Image.fromarray(pixels, mode="RGB").save(tmp_path / "pattern_tiff.TIFF")
    (tmp_path / "pattern_svg.SVG").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2">'
        '<rect width="1" height="2" fill="#ff0000"/>'
        "</svg>",
        encoding="utf-8",
    )
    (tmp_path / "ignored.txt").write_text("not an image", encoding="utf-8")

    patterns = find_patterns(tmp_path)

    assert [path.name for path in patterns] == [
        "pattern.PNG",
        "pattern_svg.SVG",
        "pattern_tiff.TIFF",
    ]
    for path in patterns:
        loaded = load_rgb(path)
        assert loaded.shape == (2, 2, 3)
        assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(load_rgb(tmp_path / "pattern.PNG"), pixels)
    svg = load_rgb(tmp_path / "pattern_svg.SVG")
    np.testing.assert_array_equal(svg[:, 0], np.array([[255, 0, 0]] * 2))
    np.testing.assert_array_equal(svg[:, 1], np.array([[255, 255, 255]] * 2))


def test_resize_rgb_planes_matches_independent_channel_resizes() -> None:
    image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)

    combined = resize_rgb_planes(image, 2.0, opencv_bilinear)

    expected = np.stack(
        [opencv_bilinear(image[:, :, channel], 2.0).pixels for channel in range(3)],
        axis=2,
    )
    expected_floating = np.stack(
        [
            opencv_bilinear(image[:, :, channel], 2.0).floating_pixels
            for channel in range(3)
        ],
        axis=2,
    )
    np.testing.assert_array_equal(combined.pixels, expected)
    np.testing.assert_array_equal(combined.floating_pixels, expected_floating)


def test_plot_results_handles_all_exact_comparisons(tmp_path: Path) -> None:
    results = pd.DataFrame(
        [
            {
                "ratio": 2.0,
                "ratio_slug": "r_2",
                "group": "other",
                "sqnr_db_opencv_as_signal": np.inf,
                "rmse": 0.0,
            }
        ]
    )

    plot_results(results, tmp_path)

    assert (tmp_path / "plots" / "sqnr_by_ratio_and_group.svg").is_file()
    assert (tmp_path / "plots" / "rmse_by_ratio_and_group.svg").is_file()
    assert (tmp_path / "plots" / "sqnr_heatmap_by_group.svg").is_file()
