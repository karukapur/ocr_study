from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
