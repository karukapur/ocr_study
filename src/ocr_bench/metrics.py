from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Any


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    return re.sub(r"\s+", " ", normalized.strip())


def levenshtein_distance(reference: str, prediction: str) -> int:
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for reference_index, reference_char in enumerate(reference, start=1):
        current = [reference_index]
        for prediction_index, prediction_char in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[prediction_index] + 1,
                    previous[prediction_index - 1] + (reference_char != prediction_char),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, prediction: str) -> tuple[int, int, float]:
    normalized_reference = normalize_text(reference)
    normalized_prediction = normalize_text(prediction)
    length = len(normalized_reference)
    distance = levenshtein_distance(normalized_reference, normalized_prediction)
    if length == 0:
        raise ValueError("CER reference is empty after normalization")
    return distance, length, distance / length


def aggregate_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in rows if row.get("status") == "ok"]
    keys = sorted({(float(row["requested_ratio"]), str(row["method"])) for row in valid})
    aggregates: list[dict[str, Any]] = []
    scopes = (
        ("english", {"en"}),
        ("traditional_chinese", {"zh_tra"}),
        ("combined", {"en", "zh_tra"}),
    )
    for ratio, method in keys:
        matching = [
            row
            for row in valid
            if float(row["requested_ratio"]) == ratio and str(row["method"]) == method
        ]
        for scope, languages in scopes:
            scoped = [row for row in matching if row["language"] in languages]
            if not scoped:
                continue
            distance = sum(int(row["edit_distance"]) for row in scoped)
            length = sum(int(row["reference_chars"]) for row in scoped)
            aggregates.append(
                {
                    "requested_ratio": ratio,
                    "method": method,
                    "scope": scope,
                    "edit_distance": distance,
                    "reference_chars": length,
                    "cer": distance / length,
                    "pattern_count": len(scoped),
                }
            )
    return aggregates

