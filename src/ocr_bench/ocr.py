from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import TesseractSpec
from .util import sha256_file


class TesseractValidationError(RuntimeError):
    pass


LANGUAGE_CODES = {
    "en": "eng",
    "zh_tra": "chi_tra",
}


def tesseract_language(dataset_language: str) -> str:
    try:
        return LANGUAGE_CODES[dataset_language]
    except KeyError as error:
        raise ValueError(f"no Tesseract language mapping for {dataset_language!r}") from error


def validate_tesseract(spec: TesseractSpec, required_languages: set[str]) -> dict[str, Any]:
    executable = shutil.which(spec.executable)
    if executable is None:
        raise TesseractValidationError(
            f"Tesseract executable {spec.executable!r} was not found on PATH"
        )
    version_process = subprocess.run(
        [executable, "--version"], capture_output=True, text=True, errors="replace", check=False
    )
    if version_process.returncode != 0:
        raise TesseractValidationError(version_process.stderr.strip() or "tesseract --version failed")
    language_command = [executable]
    if spec.tessdata_dir is not None:
        if not spec.tessdata_dir.is_dir():
            raise TesseractValidationError(
                f"configured tessdata directory is missing: {spec.tessdata_dir}"
            )
        language_command.extend(["--tessdata-dir", str(spec.tessdata_dir)])
    language_command.append("--list-langs")
    language_process = subprocess.run(
        language_command, capture_output=True, text=True, errors="replace", check=False
    )
    if language_process.returncode != 0:
        raise TesseractValidationError(language_process.stderr.strip() or "tesseract --list-langs failed")
    lines = [line.strip() for line in language_process.stdout.splitlines() if line.strip()]
    available = set(lines[1:] if lines and lines[0].startswith("List of available") else lines)
    missing = required_languages - available
    if missing:
        raise TesseractValidationError(
            "missing Tesseract language data: "
            + ", ".join(sorted(missing))
            + ". Install Traditional Chinese tessdata (Homebrew: tesseract-lang) "
            "and retry."
        )

    tessdata_match = re.search(r'"([^"]+)"', lines[0]) if lines else None
    tessdata_dir = spec.tessdata_dir or (Path(tessdata_match.group(1)) if tessdata_match else None)
    language_hashes: dict[str, str | None] = {}
    for language in sorted(required_languages):
        traineddata = tessdata_dir / f"{language}.traineddata" if tessdata_dir else None
        language_hashes[language] = (
            sha256_file(traineddata) if traineddata is not None and traineddata.is_file() else None
        )
    return {
        "executable": executable,
        "version": version_process.stdout.splitlines()[0].strip(),
        "available_languages": sorted(available),
        "tessdata_dir": str(tessdata_dir) if tessdata_dir else None,
        "language_sha256": language_hashes,
    }


def run_tesseract(
    image_path: Path,
    language: str,
    psm: int,
    spec: TesseractSpec,
) -> dict[str, Any]:
    command = [
        spec.executable,
        str(image_path),
        "stdout",
    ]
    if spec.tessdata_dir is not None:
        command.extend(["--tessdata-dir", str(spec.tessdata_dir)])
    command.extend([
        "--oem",
        str(spec.oem),
        "--psm",
        str(psm),
        "-l",
        language,
        "-c",
        "preserve_interword_spaces=1",
    ])
    environment = os.environ.copy()
    environment.setdefault("OMP_THREAD_LIMIT", "1")
    if spec.tessdata_dir is not None:
        environment["TESSDATA_PREFIX"] = str(spec.tessdata_dir)
    started = time.perf_counter()
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=spec.timeout_seconds,
            check=False,
            env=environment,
        )
        elapsed = time.perf_counter() - started
        return {
            "command": command,
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "elapsed_seconds": elapsed,
            "status": "ok" if process.returncode == 0 else "error",
            "error": "" if process.returncode == 0 else process.stderr.strip(),
        }
    except subprocess.TimeoutExpired as error:
        return {
            "command": command,
            "returncode": None,
            "stdout": error.stdout or "",
            "stderr": error.stderr or "",
            "elapsed_seconds": time.perf_counter() - started,
            "status": "error",
            "error": f"Tesseract timed out after {spec.timeout_seconds} seconds",
        }
