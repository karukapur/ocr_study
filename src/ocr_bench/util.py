from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import binascii
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_lf_text(value: bytes | str) -> str:
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value
    return text.replace("\r\n", "\n").replace("\r", "\n")


def pixel_sha256(image: np.ndarray) -> str:
    if image.dtype != np.uint8 or (
        image.ndim != 2 and not (image.ndim == 3 and image.shape[2] == 3)
    ):
        raise ValueError("pixel_sha256 expects an 8-bit grayscale or RGB image")
    height, width = image.shape[:2]
    digest = hashlib.sha256()
    digest.update(b"L8\0" if image.ndim == 2 else b"RGB8\0")
    digest.update(struct.pack(">II", width, height))
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(kind)
    crc = binascii.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def deterministic_png_bytes(image: np.ndarray) -> bytes:
    if image.dtype != np.uint8 or (
        image.ndim != 2 and not (image.ndim == 3 and image.shape[2] == 3)
    ):
        raise ValueError(
            "deterministic_png_bytes expects an 8-bit grayscale or RGB image"
        )
    height, width = image.shape[:2]
    rows = bytearray()
    contiguous = np.ascontiguousarray(image)
    for row in contiguous:
        rows.append(0)
        rows.extend(row.tobytes())
    color_type = 0 if image.ndim == 2 else 2
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", header),
            _png_chunk(b"IDAT", zlib.compress(bytes(rows), level=0)),
            _png_chunk(b"IEND", b""),
        ]
    )


def save_deterministic_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(deterministic_png_bytes(image))
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def software_environment() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    distributions = {
        "PIL": "Pillow",
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "yaml": "PyYAML",
        "fontTools": "fonttools",
        "cv2": "opencv-python-headless",
    }
    for module_name, distribution in distributions.items():
        try:
            versions[module_name] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[module_name] = "not installed"
    return versions
