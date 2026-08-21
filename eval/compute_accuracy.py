#!/usr/bin/env python3
"""Compute accuracy from structured predictions or archived human-readable logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "log" / "Qwen2.5-3B_V-Session_MATH500.log"
LOG_RECORD_RE = re.compile(
    r"^\[?idx:(?P<index>\d+)\]?\s+.*?\bacc:(?P<value>True|False|1(?:\.0)?|0(?:\.0)?)\b"
)


def _from_jsonl(path: Path) -> dict[str, Any]:
    outcomes: list[bool] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict) or type(record.get("correct")) is not bool:
                raise ValueError(f"{path}:{line_number}: expected an object with boolean 'correct'")
            raw_item_id = record.get("item_id", line_number)
            if isinstance(raw_item_id, bool) or not isinstance(raw_item_id, (str, int)):
                raise ValueError(f"{path}:{line_number}: item_id must be a string or integer")
            item_id = str(raw_item_id)
            if not item_id.strip() or item_id != item_id.strip():
                raise ValueError(
                    f"{path}:{line_number}: item_id must be non-empty without "
                    "surrounding whitespace"
                )
            if item_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate item_id {item_id!r}")
            seen_ids.add(item_id)
            outcomes.append(record["correct"])
    if not outcomes:
        raise ValueError(f"no prediction records found in {path}")
    correct = sum(outcomes)
    return {
        "source_format": "jsonl",
        "total": len(outcomes),
        "correct": correct,
        "accuracy": correct / len(outcomes),
    }


def _from_legacy_log(path: Path) -> dict[str, Any]:
    outcomes: dict[int, bool] = {}
    literal_true = 0
    numeric_true = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            match = LOG_RECORD_RE.match(line)
            if not match:
                continue
            index = int(match.group("index"))
            if index in outcomes:
                raise ValueError(f"{path}:{line_number}: duplicate idx:{index}")
            raw = match.group("value")
            is_correct = raw in {"True", "1", "1.0"}
            outcomes[index] = is_correct
            literal_true += int(raw == "True")
            numeric_true += int(raw in {"1", "1.0"})
    if not outcomes:
        raise ValueError(f"no indexed accuracy records found in {path}")
    correct = sum(outcomes.values())
    total = len(outcomes)
    return {
        "source_format": "legacy_log",
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "literal_true": literal_true,
        "numeric_true": numeric_true,
        "archived_literal_true_accuracy": literal_true / total,
        "note": (
            "Normalized accuracy treats True and 1.0 as correct. The archived utility counted "
            "only the literal string 'acc:True'; both values are reported for transparency."
        ),
    }


def compute_accuracy(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".jsonl":
        result = _from_jsonl(source)
    else:
        result = _from_legacy_log(source)
    result["path"] = str(source)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()
    result = compute_accuracy(args.path)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"File: {result['path']}")
        print(f"Total samples: {result['total']}")
        print(f"Correct: {result['correct']}")
        print(f"Accuracy: {result['accuracy']:.2%}")
        if result["source_format"] == "legacy_log" and result["numeric_true"]:
            print(
                "Archived literal-True accuracy: "
                f"{result['literal_true']}/{result['total']} "
                f"({result['archived_literal_true_accuracy']:.2%})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
