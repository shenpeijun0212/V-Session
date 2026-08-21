#!/usr/bin/env python3
"""Validate generated GSM8K V-Session traces before supervised fine-tuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsession.evaluation import evaluate_response  # noqa: E402
from vsession.structure import audit_vsession_structure  # noqa: E402


def _required_text(record: dict[str, Any], field: str, path: Path, line_number: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{line_number}: {field!r} must be a non-empty string")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trace-field", default="completion")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--finish-reason-field", default="finish_reason")
    parser.add_argument(
        "--require-delimiters",
        action="store_true",
        help="require at least one ≪...≫ pair; applicability cannot be inferred automatically",
    )
    parser.add_argument(
        "--allow-unknown-finish-reason",
        action="store_true",
        help="allow missing or unrecognized finish reasons; default validation fails closed",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="write annotated invalid records too (audit output; never use directly for training)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="exit successfully after filtering rejected records instead of returning status 3",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"input does not exist: {args.input}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    annotated: list[dict[str, Any]] = []
    with args.input.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{args.input}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{args.input}:{line_number}: expected a JSON object")
            trace = _required_text(record, args.trace_field, args.input, line_number)
            answer = _required_text(record, args.answer_field, args.input, line_number)
            structure = audit_vsession_structure(
                trace,
                dataset="gsm8k",
                require_delimiters=args.require_delimiters,
            )
            score = evaluate_response("gsm8k", trace, answer)
            finish_reason = record.get(args.finish_reason_field)
            if finish_reason is None:
                not_truncated: bool | None = None
            else:
                normalized_finish_reason = str(finish_reason).strip().lower()
                if normalized_finish_reason in {
                    "stop",
                    "eos",
                    "eos_token",
                    "eos_or_stop",
                    "finished",
                }:
                    not_truncated = True
                elif normalized_finish_reason in {
                    "length",
                    "max_tokens",
                    "max_new_tokens",
                    "max_length",
                }:
                    not_truncated = False
                else:
                    not_truncated = None
            strict_answer_match = bool(
                score.extraction_mode == "strict_final_field" and score.correct
            )
            delimiters_used = structure.delimiter_pairs > 0
            valid = (
                structure.stages_present
                and structure.stages_in_order
                and structure.stages_unique
                and structure.stage_bodies_nonempty
                and structure.delimiters_balanced
                and structure.delimiter_style_compliant
                and structure.final_answer_present is True
                and strict_answer_match
                and structure.delimiter_requirement_met
                and (not_truncated is True or args.allow_unknown_finish_reason)
            )
            annotated.append(
                {
                    **record,
                    "trace_validation": {
                        "valid": valid,
                        "strict_answer_match": strict_answer_match,
                        "stages_present": structure.stages_present,
                        "stages_in_order": structure.stages_in_order,
                        "stages_unique": structure.stages_unique,
                        "stage_bodies_nonempty": structure.stage_bodies_nonempty,
                        "delimiters_balanced": structure.delimiters_balanced,
                        "delimiters_used": delimiters_used,
                        "delimiter_style_compliant": structure.delimiter_style_compliant,
                        "delimiter_usage_required": args.require_delimiters,
                        "not_truncated": not_truncated,
                        "unknown_finish_reason_allowed": args.allow_unknown_finish_reason,
                        "parseable_calculations_checked": False,
                        "note": (
                            "The paper requires arithmetic validation, but the exact expression "
                            "grammar was not published; this implementation does not claim "
                            "that check."
                        ),
                    },
                }
            )

    if not annotated:
        raise SystemExit("input contains no trace records")
    passed_records = [record for record in annotated if record["trace_validation"]["valid"]]
    records_to_write = annotated if args.include_invalid else passed_records
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records_to_write:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    passed = len(passed_records)
    failed = len(annotated) - passed
    summary = {
        "total": len(annotated),
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / len(annotated),
        "written": len(records_to_write),
        "output_mode": "annotated_all" if args.include_invalid else "valid_only",
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 3 if failed and not args.allow_partial else 0


if __name__ == "__main__":
    raise SystemExit(main())
