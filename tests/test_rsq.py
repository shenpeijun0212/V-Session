from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from vsession.rsq import DIMENSIONS, WEIGHT_PRESETS, RSQDataError, aggregate_rsq

ROOT = Path(__file__).resolve().parents[1]


def _panel_rows(style, item_id, evaluator_scores):
    return [
        {
            "style": style,
            "item_id": item_id,
            "evaluator_id": evaluator,
            "dimension": dimension,
            "score": score,
        }
        for evaluator, scores in evaluator_scores.items()
        for dimension, score in zip(DIMENSIONS, scores, strict=True)
    ]


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ("equal", 7.70),
        ("structure-heavy", 7.64),
        ("math-rigor-heavy", 7.74),
        ("clarity-heavy", 7.72),
    ],
)
def test_paper_weight_presets_reproduce_ps_sensitivity_row(preset, expected):
    rows = _panel_rows("ps", "one", {"evaluator-1": (7.4, 7.7, 7.9, 7.8)})
    result = aggregate_rsq(rows, weights=WEIGHT_PRESETS[preset], expected_evaluators=1)
    assert result["styles"][0]["overall"] == pytest.approx(expected)


def test_items_are_weighted_equally_in_multi_item_extension():
    rows = _panel_rows("vsession", "easy", {"a": (0, 0, 0, 0)})
    rows += _panel_rows(
        "vsession",
        "hard",
        {"a": (10, 10, 10, 10), "b": (10, 10, 10, 10)},
    )
    result = aggregate_rsq(rows)
    assert result["styles"][0]["overall"] == 5.0


def test_rsq_rejects_duplicate_missing_and_nonfinite_scores():
    rows = _panel_rows("vsession", "one", {"a": (8, 8, 8, 8)})
    with pytest.raises(RSQDataError, match="duplicate"):
        aggregate_rsq(rows + [rows[0]])
    with pytest.raises(RSQDataError, match="missing dimensions"):
        aggregate_rsq(rows[:-1])
    invalid = [dict(row) for row in rows]
    invalid[0]["score"] = math.nan
    with pytest.raises(RSQDataError, match="finite number"):
        aggregate_rsq(invalid)


def test_paper_panel_requires_eight_consistent_evaluators_and_items():
    evaluators = {f"evaluator-{index}": (8, 8, 8, 8) for index in range(8)}
    rows = _panel_rows("ps", "one", evaluators)
    rows += _panel_rows("vsession", "one", evaluators)
    result = aggregate_rsq(
        rows,
        expected_evaluators=8,
        require_consistent_panel=True,
        require_balanced_items=True,
    )
    assert [style["evaluator_count"] for style in result["styles"]] == [8, 8]


def test_rsq_cli_outputs_input_hash_and_declares_no_automatic_judging(tmp_path):
    source = tmp_path / "ratings.jsonl"
    rows = _panel_rows("vsession", "one", {"human-1": (8, 9, 10, 7)})
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "eval" / "compute_rsq.py"),
            str(source),
            "--expected-evaluators",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["automatic_judging_performed"] is False
    assert len(result["input"]["sha256"]) == 64
