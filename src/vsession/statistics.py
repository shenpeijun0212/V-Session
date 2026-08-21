"""Paired bootstrap confidence intervals and exact McNemar tests."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


class PairedDataError(ValueError):
    """Raised when two result sets cannot be aligned item by item."""


@dataclass(frozen=True)
class PairedComparison:
    n_items: int
    reference_accuracy: float
    candidate_accuracy: float
    difference: float
    difference_direction: str
    confidence_level: float
    ci_lower: float
    ci_upper: float
    bootstrap_resamples: int
    bootstrap_seed: int
    both_correct: int
    reference_only_correct: int
    candidate_only_correct: int
    both_incorrect: int
    mcnemar_exact_p: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _index(
    records: Iterable[Mapping[str, Any]],
    label: str,
    item_id_key: str,
    correct_key: str,
) -> tuple[dict[str, bool], list[str], dict[str, Mapping[str, Any]]]:
    indexed: dict[str, bool] = {}
    order: list[str] = []
    source_records: dict[str, Mapping[str, Any]] = {}
    for position, record in enumerate(records):
        if item_id_key not in record or correct_key not in record:
            raise PairedDataError(
                f"{label} record {position} must contain {item_id_key!r} and {correct_key!r}"
            )
        raw_item_id = record[item_id_key]
        if isinstance(raw_item_id, bool) or not isinstance(raw_item_id, (str, int)):
            raise PairedDataError(
                f"{label} record {position} has an invalid item ID type: "
                f"{type(raw_item_id).__name__}"
            )
        item_id = str(raw_item_id)
        if not item_id.strip():
            raise PairedDataError(f"{label} record {position} has an empty item ID")
        if item_id != item_id.strip():
            raise PairedDataError(f"{label} record {position} has whitespace around its item ID")
        if item_id in indexed:
            raise PairedDataError(f"{label} contains duplicate item ID {item_id!r}")
        correct = record[correct_key]
        if type(correct) is not bool:
            raise PairedDataError(f"{label} item {item_id!r} has non-boolean correctness")
        indexed[item_id] = correct
        source_records[item_id] = record
        order.append(item_id)
    if not indexed:
        raise PairedDataError(f"{label} contains no records")
    return indexed, order, source_records


def align_results(
    reference: Iterable[Mapping[str, Any]],
    candidate: Iterable[Mapping[str, Any]],
    *,
    item_id_key: str = "item_id",
    correct_key: str = "correct",
) -> list[tuple[str, bool, bool]]:
    """Align outcomes by item ID and reject missing or duplicate examples."""

    first, order, first_records = _index(reference, "reference", item_id_key, correct_key)
    second, _, second_records = _index(candidate, "candidate", item_id_key, correct_key)
    if set(first) != set(second):
        missing_candidate = sorted(set(first) - set(second))[:5]
        missing_reference = sorted(set(second) - set(first))[:5]
        raise PairedDataError(
            "item ID sets differ; "
            f"missing from candidate={missing_candidate}, "
            f"missing from reference={missing_reference}"
        )
    for item_id in order:
        for field in ("dataset", "question", "reference"):
            first_record = first_records[item_id]
            second_record = second_records[item_id]
            if (
                field in first_record
                and field in second_record
                and first_record[field] != second_record[field]
            ):
                raise PairedDataError(
                    f"paired item {item_id!r} has different {field!r} values across inputs"
                )
    return [(item_id, first[item_id], second[item_id]) for item_id in order]


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty sequence")
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1 - weight) + values[upper] * weight)


def _mcnemar_p(reference_only: int, candidate_only: int) -> float:
    discordant = reference_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = min(reference_only, candidate_only)
    one_sided = sum(math.comb(discordant, value) for value in range(tail + 1)) / (2**discordant)
    return min(1.0, 2 * one_sided)


def compare_paired(
    reference: Iterable[Mapping[str, Any]],
    candidate: Iterable[Mapping[str, Any]],
    *,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 20260218,
    item_id_key: str = "item_id",
    correct_key: str = "correct",
) -> PairedComparison:
    """Compare candidate minus reference accuracy on exactly paired items."""

    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    aligned = align_results(
        reference,
        candidate,
        item_id_key=item_id_key,
        correct_key=correct_key,
    )
    differences = [
        int(candidate_ok) - int(reference_ok) for _, reference_ok, candidate_ok in aligned
    ]
    n_items = len(aligned)
    rng = random.Random(seed)
    bootstrap = [
        sum(differences[rng.randrange(n_items)] for _ in range(n_items)) / n_items
        for _ in range(n_resamples)
    ]
    bootstrap.sort()
    alpha = (1 - confidence_level) / 2

    both_correct = sum(reference_ok and candidate_ok for _, reference_ok, candidate_ok in aligned)
    reference_only = sum(
        reference_ok and not candidate_ok for _, reference_ok, candidate_ok in aligned
    )
    candidate_only = sum(
        candidate_ok and not reference_ok for _, reference_ok, candidate_ok in aligned
    )
    both_incorrect = n_items - both_correct - reference_only - candidate_only
    return PairedComparison(
        n_items=n_items,
        reference_accuracy=sum(reference_ok for _, reference_ok, _ in aligned) / n_items,
        candidate_accuracy=sum(candidate_ok for _, _, candidate_ok in aligned) / n_items,
        difference=sum(differences) / n_items,
        difference_direction="candidate_minus_reference",
        confidence_level=confidence_level,
        ci_lower=_percentile(bootstrap, alpha),
        ci_upper=_percentile(bootstrap, 1 - alpha),
        bootstrap_resamples=n_resamples,
        bootstrap_seed=seed,
        both_correct=both_correct,
        reference_only_correct=reference_only,
        candidate_only_correct=candidate_only,
        both_incorrect=both_incorrect,
        mcnemar_exact_p=_mcnemar_p(reference_only, candidate_only),
    )
