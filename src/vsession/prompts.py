"""Load and render immutable prompt templates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

QUESTION_PLACEHOLDER = "{question}"


class PromptError(ValueError):
    """Raised when a prompt cannot be rendered unambiguously."""


@dataclass(frozen=True)
class PromptTemplate:
    path: Path
    text: str
    sha256: str
    render_mode: str

    def render(self, question: str) -> str:
        if not isinstance(question, str) or not question.strip():
            raise PromptError("question must be a non-empty string")
        clean_question = question.strip()
        if self.render_mode == "replace":
            return self.text.replace(QUESTION_PLACEHOLDER, clean_question, 1)
        if self.render_mode == "append":
            # Archived few-shot prompts already contain their final Question/Q cue.
            separator = "" if self.text[-1].isspace() else " "
            return self.text + separator + clean_question
        if self.render_mode == "append_exact":
            return self.text + clean_question
        raise PromptError(f"unsupported render mode: {self.render_mode!r}")


def load_prompt(path: str | Path, render_mode: str = "replace") -> PromptTemplate:
    """Load a UTF-8 template and validate its question-slot contract."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise PromptError(f"prompt file does not exist: {source}")
    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise PromptError(f"prompt is not valid UTF-8: {source}") from exc
    if text.startswith("\ufeff"):
        raise PromptError(f"prompt must not contain a UTF-8 BOM: {source}")
    if not text.strip():
        raise PromptError(f"prompt is empty: {source}")
    if render_mode not in {"replace", "append", "append_exact"}:
        raise PromptError("render_mode must be 'replace', 'append', or 'append_exact'")
    occurrences = text.count(QUESTION_PLACEHOLDER)
    expected = int(render_mode == "replace")
    if occurrences != expected:
        raise PromptError(
            f"{source}: render mode {render_mode!r} requires {expected} "
            f"{QUESTION_PLACEHOLDER!r} placeholder(s), found {occurrences}"
        )
    return PromptTemplate(
        path=source,
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        render_mode=render_mode,
    )
