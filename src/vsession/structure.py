"""Audits for the five-stage V-Session response contract."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .evaluation import has_strict_terminal_field

STAGES = ("Goal", "Solution", "Thinking", "Reasoning", "Result")
_STAGE_RE = re.compile(r"(?mi)^\s*(Goal|Solution|Thinking|Reasoning|Result)\s*:")


@dataclass(frozen=True)
class StructureAudit:
    stages_present: bool
    stages_in_order: bool
    stages_unique: bool
    stage_bodies_nonempty: bool
    observed_stages: tuple[str, ...]
    delimiters_balanced: bool
    delimiter_pairs: int
    delimiters_used: bool
    delimiter_style_compliant: bool
    unexpected_delimiters_present: bool
    delimiter_usage_required: bool
    delimiter_requirement_met: bool
    final_answer_present: bool | None
    valid: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _balanced_pairs(text: str, opening: str, closing: str) -> tuple[bool, int]:
    depth = 0
    pairs = 0
    index = 0
    while index < len(text):
        if text.startswith(opening, index):
            depth += 1
            index += len(opening)
            continue
        if text.startswith(closing, index):
            if depth == 0:
                return False, pairs
            depth -= 1
            pairs += 1
            index += len(closing)
            continue
        index += 1
    return depth == 0, pairs


def audit_vsession_structure(
    text: str,
    *,
    dataset: str | None = None,
    delimiter_style: str = "unicode",
    expected_stages: Sequence[str] = STAGES,
    require_delimiters: bool = False,
) -> StructureAudit:
    """Check stage order/uniqueness, calculation delimiters, and final field."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if delimiter_style == "unicode":
        opening, closing = "≪", "≫"
        forbidden = ("<<", ">>")
    elif delimiter_style == "ascii":
        opening, closing = "<<", ">>"
        forbidden = ("≪", "≫")
    elif delimiter_style == "none":
        opening = closing = ""
        forbidden = ("≪", "≫", "<<", ">>")
    else:
        raise ValueError("delimiter_style must be 'unicode', 'ascii', or 'none'")

    expected = tuple(expected_stages)
    if (
        not expected
        or any(stage not in STAGES for stage in expected)
        or len(set(expected)) != len(expected)
    ):
        raise ValueError("expected_stages must contain unique canonical V-Session stage names")
    canonical_subsequence = tuple(stage for stage in STAGES if stage in expected)
    if expected != canonical_subsequence:
        raise ValueError("expected_stages must preserve the canonical V-Session stage order")
    stage_matches = list(_STAGE_RE.finditer(text))
    observed = tuple(match.group(1).title() for match in stage_matches)
    counts = {stage: observed.count(stage) for stage in expected}
    present = all(counts[stage] >= 1 for stage in expected)
    unique = all(counts[stage] == 1 for stage in expected) and all(
        stage in expected for stage in observed
    )
    ordered = observed == expected
    bodies_nonempty = True
    for index, match in enumerate(stage_matches):
        end = stage_matches[index + 1].start() if index + 1 < len(stage_matches) else len(text)
        body_lines = []
        for line in text[match.end() : end].splitlines():
            stripped = line.strip()
            if has_strict_terminal_field("gsm8k", stripped) or has_strict_terminal_field(
                "math500", stripped
            ):
                continue
            body_lines.append(stripped)
        if not any(body_lines):
            bodies_nonempty = False
            break

    if delimiter_style == "none":
        balanced, pairs = True, 0
    else:
        balanced, pairs = _balanced_pairs(text, opening, closing)
    unexpected = any(token in text for token in forbidden)
    style_compliant = not unexpected
    used = pairs > 0
    delimiter_requirement_met = used or not require_delimiters

    last_line = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
    if dataset is None:
        final_present: bool | None = None
    elif dataset == "gsm8k":
        final_present = has_strict_terminal_field("gsm8k", last_line)
    elif dataset == "math500":
        final_present = has_strict_terminal_field("math500", last_line)
    else:
        raise ValueError("dataset must be 'gsm8k', 'math500', or None")

    valid = (
        present
        and ordered
        and unique
        and bodies_nonempty
        and balanced
        and style_compliant
        and delimiter_requirement_met
        and final_present is not False
    )
    return StructureAudit(
        stages_present=present,
        stages_in_order=ordered,
        stages_unique=unique,
        stage_bodies_nonempty=bodies_nonempty,
        observed_stages=observed,
        delimiters_balanced=balanced,
        delimiter_pairs=pairs,
        delimiters_used=used,
        delimiter_style_compliant=style_compliant,
        unexpected_delimiters_present=unexpected,
        delimiter_usage_required=require_delimiters,
        delimiter_requirement_met=delimiter_requirement_met,
        final_answer_present=final_present,
        valid=valid,
    )
