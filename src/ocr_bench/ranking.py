from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd


BINNING_METHODS = frozenset({"bin_floor", "bin_ceil"})
COMBINED_LANGUAGES = frozenset({"en", "zh_tra"})


@dataclass(frozen=True)
class RankedCondition:
    rank: str
    method: str
    requested_ratio: float
    cer: float
    edit_distance: int
    reference_chars: int
    tie_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def select_method_ratio_extremes(
    results: pd.DataFrame,
    *,
    excluded_methods: Iterable[str] = BINNING_METHODS,
) -> dict[str, RankedCondition]:
    """Select deterministic best/worst pooled-CER method/ratio conditions."""

    required = {
        "method",
        "requested_ratio",
        "edit_distance",
        "reference_chars",
    }
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError(f"ranking input is missing columns: {', '.join(missing)}")

    eligible = results.copy()
    if "status" in eligible.columns:
        eligible = eligible[eligible["status"] == "ok"]
    if "language" in eligible.columns:
        eligible = eligible[eligible["language"].isin(COMBINED_LANGUAGES)]
    eligible = eligible[~eligible["method"].isin(set(excluded_methods))]
    if eligible.empty:
        raise ValueError("there are no eligible non-binning OCR results to rank")
    if (eligible["requested_ratio"] <= 0).any():
        raise ValueError("ranking input contains a non-positive requested ratio")

    table = (
        eligible.groupby(["method", "requested_ratio"], as_index=False)[
            ["edit_distance", "reference_chars"]
        ]
        .sum()
    )
    if (table["reference_chars"] <= 0).any():
        raise ValueError("ranking input contains a non-positive reference character count")
    table["cer"] = table["edit_distance"] / table["reference_chars"]

    selected: dict[str, RankedCondition] = {}
    for rank, best in (("best", True), ("worst", False)):
        target = float(table["cer"].min() if best else table["cer"].max())
        tied = table[np.isclose(table["cer"], target)].sort_values(
            ["requested_ratio", "method"], kind="stable"
        )
        row = tied.iloc[0]
        selected[rank] = RankedCondition(
            rank=rank,
            method=str(row["method"]),
            requested_ratio=float(row["requested_ratio"]),
            cer=float(row["cer"]),
            edit_distance=int(row["edit_distance"]),
            reference_chars=int(row["reference_chars"]),
            tie_count=len(tied),
        )
    return selected
