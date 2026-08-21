from __future__ import annotations

import pytest

from vsession.statistics import PairedDataError, align_results, compare_paired


def test_alignment_uses_item_ids_not_file_order():
    reference = [{"item_id": "a", "correct": True}, {"item_id": "b", "correct": False}]
    candidate = [{"item_id": "b", "correct": True}, {"item_id": "a", "correct": True}]
    assert align_results(reference, candidate) == [("a", True, True), ("b", False, True)]


def test_paired_comparison_and_exact_mcnemar():
    reference = [{"item_id": str(i), "correct": i < 2} for i in range(4)]
    candidate = [{"item_id": str(i), "correct": i != 3} for i in range(4)]
    result = compare_paired(reference, candidate, n_resamples=200, seed=7)
    assert result.reference_accuracy == 0.5
    assert result.candidate_accuracy == 0.75
    assert result.difference == 0.25
    assert result.difference_direction == "candidate_minus_reference"
    assert 0 <= result.mcnemar_exact_p <= 1


def test_known_exact_mcnemar_value():
    reference = [{"item_id": str(i), "correct": False} for i in range(3)]
    candidate = [{"item_id": str(i), "correct": True} for i in range(3)]
    result = compare_paired(reference, candidate, n_resamples=10, seed=1)
    assert result.mcnemar_exact_p == 0.25


def test_pairing_rejects_missing_and_duplicate_ids():
    reference = [{"item_id": "a", "correct": True}]
    with pytest.raises(PairedDataError, match="sets differ"):
        align_results(reference, [{"item_id": "b", "correct": True}])
    with pytest.raises(PairedDataError, match="duplicate"):
        align_results(
            reference,
            [{"item_id": "a", "correct": True}, {"item_id": "a", "correct": False}],
        )


def test_pairing_rejects_invalid_ids_and_mismatched_items():
    with pytest.raises(PairedDataError, match="invalid item ID type"):
        align_results(
            [{"item_id": None, "correct": True}],
            [{"item_id": None, "correct": True}],
        )
    with pytest.raises(PairedDataError, match="different 'question'"):
        align_results(
            [{"item_id": "a", "question": "1+1?", "correct": True}],
            [{"item_id": "a", "question": "2+2?", "correct": True}],
        )
