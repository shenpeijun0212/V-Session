#!/usr/bin/env python3
"""Aggregate externally supplied long-form RSQ ratings without calling a judge model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsession.data import file_sha256  # noqa: E402
from vsession.rsq import WEIGHT_PRESETS, RSQDataError, aggregate_rsq  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RSQDataError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise RSQDataError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--preset", choices=tuple(WEIGHT_PRESETS), default="equal")
    parser.add_argument("--expected-evaluators", type=int)
    parser.add_argument(
        "--paper-panel",
        action="store_true",
        help="require the same eight evaluators and the same item set for every style",
    )
    parser.add_argument("--include-items", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if args.paper_panel and args.expected_evaluators not in {None, 8}:
        parser.error("--paper-panel requires --expected-evaluators 8")
    try:
        result = aggregate_rsq(
            load_jsonl(args.input),
            weights=WEIGHT_PRESETS[args.preset],
            expected_evaluators=8 if args.paper_panel else args.expected_evaluators,
            require_consistent_panel=args.paper_panel,
            require_balanced_items=args.paper_panel,
            include_items=args.include_items,
        )
    except RSQDataError as exc:
        parser.error(str(exc))
    result["input"] = {"path": str(args.input.resolve()), "sha256": file_sha256(args.input)}
    result["weight_preset"] = args.preset
    result["paper_panel_checks"] = args.paper_panel
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
