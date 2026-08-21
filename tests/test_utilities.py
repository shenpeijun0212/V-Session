from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from compute_accuracy import compute_accuracy  # noqa: E402


def test_preserved_reference_log_scores():
    math = compute_accuracy(ROOT / "log" / "Qwen2.5-3B_V-Session_MATH500.log")
    gsm = compute_accuracy(ROOT / "log" / "Qwen2.5-3B_V-Session_GSM8K1000.log")
    assert (math["correct"], math["total"]) == (196, 500)
    assert (gsm["correct"], gsm["total"]) == (787, 1000)
    assert gsm["literal_true"] == 784


def test_trace_validator_accepts_a_well_formed_trace(tmp_path):
    source = tmp_path / "traces.jsonl"
    destination = tmp_path / "validated.jsonl"
    record = {
        "answer": "gold work\n#### 5",
        "completion": (
            "Goal: total.\nSolution: add.\nThinking: plan.\n"
            "Reasoning: ≪2 + 3 = 5≫.\nResult: five.\n#### 5"
        ),
        "finish_reason": "stop",
    }
    source.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_traces.py"),
            str(source),
            str(destination),
            "--require-delimiters",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["passed"] == 1
    assert summary["written"] == 1
    assert summary["output_mode"] == "valid_only"
    annotated = json.loads(destination.read_text(encoding="utf-8"))
    assert annotated["trace_validation"]["valid"] is True


def test_trace_validator_filters_invalid_records_and_fails_closed(tmp_path):
    source = tmp_path / "traces.jsonl"
    destination = tmp_path / "validated.jsonl"
    record = {
        "answer": "gold work\n#### 5",
        "completion": (
            "Goal: total.\nSolution: add.\nThinking: plan.\n"
            "Reasoning: 2 + 3 = 5.\nResult: five.\n#### 5"
        ),
        "finish_reason": "unknown",
    }
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_traces.py"),
            str(source),
            str(destination),
            "--require-delimiters",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert completed.returncode == 3
    assert summary["failed"] == 1
    assert summary["written"] == 0
    assert destination.read_text(encoding="utf-8") == ""
