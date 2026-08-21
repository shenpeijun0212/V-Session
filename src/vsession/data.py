"""Strict, deterministic loaders for the benchmark snapshots used by V-Session."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DataError(ValueError):
    """Raised when a benchmark file is missing or malformed."""


@dataclass(frozen=True)
class Example:
    """One normalized benchmark example."""

    item_id: str
    question: str
    answer: str
    index: int
    raw: dict[str, Any]


_FIELDS = {
    "gsm8k": ("question", "answer", ("item_id", "id")),
    "math500": ("problem", "answer", ("unique_id", "item_id", "id")),
}


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_text(record: dict[str, Any], field: str, path: Path, line_number: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DataError(f"{path}:{line_number}: field {field!r} must be a non-empty string")
    return value


def load_examples(
    path: str | Path,
    dataset: str,
    *,
    limit: int | None = None,
    expected_count: int | None = None,
) -> list[Example]:
    """Load JSONL examples and reject bad JSON, missing fields, and duplicate IDs.

    The complete source is validated before ``limit`` is applied. This prevents a
    smoke run from hiding corruption later in the file.
    """

    source = Path(path).expanduser().resolve()
    if dataset not in _FIELDS:
        raise DataError(f"unsupported dataset {dataset!r}; choose from {sorted(_FIELDS)}")
    if limit is not None and (isinstance(limit, bool) or limit <= 0):
        raise DataError("limit must be a positive integer")
    if expected_count is not None and (isinstance(expected_count, bool) or expected_count <= 0):
        raise DataError("expected_count must be a positive integer")
    if not source.is_file():
        hint = ""
        if dataset == "gsm8k":
            hint = "; prepare the canonical test split with scripts/prepare_gsm8k.py"
        raise DataError(f"dataset file does not exist: {source}{hint}")

    question_field, answer_field, id_fields = _FIELDS[dataset]
    examples: list[Example] = []
    seen_ids: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise DataError(f"{source}:{line_number}: expected a JSON object")
            question = _required_text(record, question_field, source, line_number)
            answer = _required_text(record, answer_field, source, line_number)
            raw_id = next(
                (record.get(name) for name in id_fields if record.get(name) is not None), None
            )
            if raw_id is None:
                item_id = f"{dataset}-{len(examples):04d}"
            else:
                if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)):
                    raise DataError(f"{source}:{line_number}: item ID must be a string or integer")
                item_id = str(raw_id)
                if not item_id.strip() or item_id != item_id.strip():
                    raise DataError(
                        f"{source}:{line_number}: item ID must be non-empty without "
                        "surrounding whitespace"
                    )
            if item_id in seen_ids:
                raise DataError(f"{source}:{line_number}: duplicate item ID {item_id!r}")
            seen_ids.add(item_id)
            examples.append(
                Example(
                    item_id=item_id,
                    question=question,
                    answer=answer,
                    index=len(examples),
                    raw=record,
                )
            )

    if not examples:
        raise DataError(f"dataset contains no examples: {source}")
    if expected_count is not None and len(examples) != expected_count:
        raise DataError(
            f"expected {expected_count} {dataset} examples but found {len(examples)} in {source}"
        )
    return examples[:limit] if limit is not None else examples
