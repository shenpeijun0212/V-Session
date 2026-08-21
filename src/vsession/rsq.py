"""Validated aggregation for externally supplied Reasoning Style Quality ratings.

This module implements the RSQ equations published in the manuscript. It does
not generate ratings, call an LLM judge, or reinterpret binary format audits as
0--10 RSQ judgments.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

DIMENSIONS = (
    "structural_integrity",
    "logical_rigor",
    "calculation_precision",
    "expression_clarity",
)

WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    "equal": dict(zip(DIMENSIONS, (0.25, 0.25, 0.25, 0.25), strict=True)),
    "structure-heavy": dict(zip(DIMENSIONS, (0.40, 0.20, 0.20, 0.20), strict=True)),
    "math-rigor-heavy": dict(zip(DIMENSIONS, (0.15, 0.35, 0.35, 0.15), strict=True)),
    "clarity-heavy": dict(zip(DIMENSIONS, (0.20, 0.20, 0.20, 0.40), strict=True)),
}


class RSQDataError(ValueError):
    """Raised when rating records cannot support an auditable RSQ aggregate."""


def validated_weights(weights: Mapping[str, float] | None = None) -> dict[str, float]:
    """Validate and return one complete non-negative four-dimension weight map."""

    selected: Mapping[str, float] = WEIGHT_PRESETS["equal"] if weights is None else weights
    if set(selected) != set(DIMENSIONS):
        raise RSQDataError(f"weights must contain exactly these dimensions: {DIMENSIONS}")
    result: dict[str, float] = {}
    for dimension in DIMENSIONS:
        value = selected[dimension]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RSQDataError(f"weight for {dimension!r} must be a finite number")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise RSQDataError(f"weight for {dimension!r} must be finite and non-negative")
        result[dimension] = numeric
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RSQDataError("dimension weights must sum to 1")
    return result


def _nonempty_identifier(value: Any, field: str, position: int) -> str:
    if field == "item_id" and isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if not isinstance(value, str) or not value.strip():
        raise RSQDataError(f"record {position}: {field!r} must be a non-empty string")
    if value != value.strip():
        raise RSQDataError(f"record {position}: {field!r} has surrounding whitespace")
    return value


def _validated_score(value: Any, position: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RSQDataError(f"record {position}: score must be a finite number in [0, 10]")
    score = float(value)
    if not math.isfinite(score) or not 0 <= score <= 10:
        raise RSQDataError(f"record {position}: score must be a finite number in [0, 10]")
    return score


def aggregate_rsq(
    records: Iterable[Mapping[str, Any]],
    *,
    weights: Mapping[str, float] | None = None,
    expected_evaluators: int | None = None,
    require_consistent_panel: bool = False,
    require_balanced_items: bool = False,
    include_items: bool = False,
) -> dict[str, Any]:
    """Aggregate long-form external ratings using the paper's RSQ equations.

    The manuscript omits an item index from its displayed equations. For a
    multi-item dataset, this implementation first applies the panel equation to
    each item and then averages items equally, preventing items with more rating
    rows from receiving more weight.
    """

    selected_weights = validated_weights(weights)
    if expected_evaluators is not None and (
        isinstance(expected_evaluators, bool) or expected_evaluators <= 0
    ):
        raise RSQDataError("expected_evaluators must be a positive integer")

    cells: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    styles_to_items: dict[str, set[str]] = defaultdict(set)
    row_count = 0
    for position, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise RSQDataError(f"record {position}: expected an object")
        style = _nonempty_identifier(record.get("style"), "style", position)
        item_id = _nonempty_identifier(record.get("item_id"), "item_id", position)
        evaluator_id = _nonempty_identifier(record.get("evaluator_id"), "evaluator_id", position)
        dimension = record.get("dimension")
        if dimension not in DIMENSIONS:
            raise RSQDataError(
                f"record {position}: dimension must be one of {DIMENSIONS}, got {dimension!r}"
            )
        key = (style, item_id, evaluator_id)
        if dimension in cells[key]:
            raise RSQDataError(
                f"record {position}: duplicate rating for {key!r}, dimension {dimension!r}"
            )
        cells[key][dimension] = _validated_score(record.get("score"), position)
        groups[(style, item_id)].add(evaluator_id)
        styles_to_items[style].add(item_id)
        row_count += 1
    if row_count == 0:
        raise RSQDataError("rating input is empty")

    expected_dimensions = set(DIMENSIONS)
    for key, scores in cells.items():
        if set(scores) != expected_dimensions:
            missing = sorted(expected_dimensions - set(scores))
            raise RSQDataError(f"rating cell {key!r} is missing dimensions: {missing}")

    panel: set[str] | None = None
    for group, evaluator_ids in groups.items():
        if expected_evaluators is not None and len(evaluator_ids) != expected_evaluators:
            raise RSQDataError(
                f"style/item group {group!r} has {len(evaluator_ids)} evaluators; "
                f"expected {expected_evaluators}"
            )
        if require_consistent_panel:
            if panel is None:
                panel = set(evaluator_ids)
            elif evaluator_ids != panel:
                raise RSQDataError(f"style/item group {group!r} uses a different evaluator panel")

    if require_balanced_items:
        first_items: set[str] | None = None
        for style, item_ids in styles_to_items.items():
            if first_items is None:
                first_items = set(item_ids)
            elif item_ids != first_items:
                raise RSQDataError(f"style {style!r} uses a different item set")

    item_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (style, item_id), evaluator_ids in sorted(groups.items()):
        dimension_means = {
            dimension: sum(
                cells[(style, item_id, evaluator)][dimension] for evaluator in evaluator_ids
            )
            / len(evaluator_ids)
            for dimension in DIMENSIONS
        }
        overall = sum(
            selected_weights[dimension] * dimension_means[dimension] for dimension in DIMENSIONS
        )
        item_results[style].append(
            {
                "item_id": item_id,
                "evaluator_count": len(evaluator_ids),
                "dimension_means": dimension_means,
                "overall": overall,
            }
        )

    style_results: list[dict[str, Any]] = []
    for style, items in sorted(item_results.items()):
        dimension_means = {
            dimension: sum(item["dimension_means"][dimension] for item in items) / len(items)
            for dimension in DIMENSIONS
        }
        evaluator_ids = {
            evaluator
            for item_id in styles_to_items[style]
            for evaluator in groups[(style, item_id)]
        }
        result: dict[str, Any] = {
            "style": style,
            "item_count": len(items),
            "evaluator_count": len(evaluator_ids),
            "rating_row_count": sum(item["evaluator_count"] for item in items) * len(DIMENSIONS),
            "dimension_means": dimension_means,
            "overall": sum(
                selected_weights[dimension] * dimension_means[dimension] for dimension in DIMENSIONS
            ),
        }
        if include_items:
            result["items"] = items
        style_results.append(result)

    return {
        "schema_version": 1,
        "dimensions": list(DIMENSIONS),
        "weights": selected_weights,
        "input_rating_rows": row_count,
        "aggregation_convention": (
            "Apply the panel-weighted equation per item, then average items equally. "
            "The item-level extension is an explicit implementation convention because "
            "the manuscript equation omits an item index."
        ),
        "automatic_judging_performed": False,
        "styles": style_results,
    }
