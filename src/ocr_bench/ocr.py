from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import TesseractSpec
from .util import normalize_lf_text, sha256_bytes, sha256_file


class TesseractValidationError(RuntimeError):
    pass


LANGUAGE_CODES = {
    "en": "eng",
    "zh_tra": "chi_tra",
}
REQUIRED_EXACT_TESSERACT_VERSION = "5.5.1"


def tesseract_language(dataset_language: str) -> str:
    try:
        return LANGUAGE_CODES[dataset_language]
    except KeyError as error:
        raise ValueError(f"no Tesseract language mapping for {dataset_language!r}") from error


def _parse_tesseract_version(stdout: str) -> str:
    first_line = stdout.splitlines()[0].strip() if stdout.splitlines() else ""
    match = re.search(r"tesseract\s+([0-9]+(?:\.[0-9]+)*)", first_line, re.IGNORECASE)
    return match.group(1) if match else first_line


def validate_tesseract(
    spec: TesseractSpec,
    required_languages: set[str],
    *,
    exact: bool = False,
) -> dict[str, Any]:
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
    parsed_version = _parse_tesseract_version(version_process.stdout)
    if exact and parsed_version != REQUIRED_EXACT_TESSERACT_VERSION:
        raise TesseractValidationError(
            f"exact mode requires Tesseract {REQUIRED_EXACT_TESSERACT_VERSION}; "
            f"found {parsed_version or 'unknown'}"
        )
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
    result = {
        "executable": executable,
        "version": version_process.stdout.splitlines()[0].strip(),
        "available_languages": sorted(available),
        "tessdata_dir": str(tessdata_dir) if tessdata_dir else None,
        "language_sha256": language_hashes,
    }
    if exact:
        result["executable_sha256"] = sha256_file(Path(executable))
        result["parsed_version"] = parsed_version
    return result


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
            timeout=spec.timeout_seconds,
            check=False,
            env=environment,
        )
        elapsed = time.perf_counter() - started
        stdout = normalize_lf_text(process.stdout)
        stderr = normalize_lf_text(process.stderr)
        return {
            "command": command,
            "returncode": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_sha256": sha256_bytes(stdout.encode("utf-8")),
            "stderr_sha256": sha256_bytes(stderr.encode("utf-8")),
            "elapsed_seconds": elapsed,
            "status": "ok" if process.returncode == 0 else "error",
            "error": "" if process.returncode == 0 else stderr.strip(),
        }
    except subprocess.TimeoutExpired as error:
        stdout = normalize_lf_text(error.stdout or b"")
        stderr = normalize_lf_text(error.stderr or b"")
        return {
            "command": command,
            "returncode": None,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_sha256": sha256_bytes(stdout.encode("utf-8")),
            "stderr_sha256": sha256_bytes(stderr.encode("utf-8")),
            "elapsed_seconds": time.perf_counter() - started,
            "status": "error",
            "error": f"Tesseract timed out after {spec.timeout_seconds} seconds",
        }
