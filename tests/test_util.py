from io import BytesIO
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from ocr_bench.util import (
    deterministic_png_bytes,
    git_commit_hash,
    normalize_lf_text,
    pixel_sha256,
)


def test_pixel_hash_is_independent_of_png_encoding(tmp_path: Path) -> None:
    image = np.array([[0, 128], [200, 255]], dtype=np.uint8)
    default_png = tmp_path / "default.png"
    optimized_png = tmp_path / "optimized.png"
    Image.fromarray(image).save(default_png, format="PNG", optimize=False)
    Image.fromarray(image).save(optimized_png, format="PNG", optimize=True)

    default_pixels = np.asarray(Image.open(default_png).convert("L"), dtype=np.uint8)
    optimized_pixels = np.asarray(Image.open(optimized_png).convert("L"), dtype=np.uint8)
    assert pixel_sha256(default_pixels) == pixel_sha256(optimized_pixels)


def test_deterministic_png_bytes_are_stable() -> None:
    image = np.array([[0, 128], [200, 255]], dtype=np.uint8)
    assert deterministic_png_bytes(image) == deterministic_png_bytes(image.copy())


def test_rgb_pixel_hash_and_deterministic_png_are_stable() -> None:
    image = np.array(
        [[[0, 64, 128], [255, 200, 100]], [[12, 34, 56], [78, 90, 123]]],
        dtype=np.uint8,
    )
    encoded = deterministic_png_bytes(image)

    assert pixel_sha256(image) == pixel_sha256(image.copy())
    assert encoded == deterministic_png_bytes(image.copy())
    with Image.open(BytesIO(encoded)) as decoded:
        assert decoded.mode == "RGB"
        np.testing.assert_array_equal(np.asarray(decoded), image)


def test_text_normalization_uses_utf8_and_lf() -> None:
    assert normalize_lf_text("a\r\nb\rc") == "a\nb\nc"
    assert normalize_lf_text("測試\r\n".encode("utf-8")) == "測試\n"


def test_git_commit_hash_returns_head(monkeypatch) -> None:
    commit = "a" * 40
    monkeypatch.setattr(
        "ocr_bench.util.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, commit + "\n", ""),
    )

    assert git_commit_hash() == commit


def test_git_commit_hash_returns_none_when_git_is_unavailable(monkeypatch) -> None:
    def unavailable(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr("ocr_bench.util.subprocess.run", unavailable)

    assert git_commit_hash() is None
