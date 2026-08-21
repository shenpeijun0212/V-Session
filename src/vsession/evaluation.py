"""Auditable final-answer extraction and equivalence checks.

The revised paper specifies strict final fields for V-Session, but it does not
publish the parser used for the other prompting baselines. This module therefore
records whether a strict field or a documented fallback produced the answer.
It never evaluates model-produced Python code.
"""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any


class EvaluationError(ValueError):
    """Raised for malformed references or unsupported evaluation requests."""


class AnswerExtractionError(EvaluationError):
    """Raised when no defensible final answer can be extracted."""


_NUMBER_PATTERN = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?"
_NUMBER_RE = re.compile(_NUMBER_PATTERN)
_GSM_FINAL_RE = re.compile(rf"(?m)^\s*####\s*({_NUMBER_PATTERN})\s*$")
_MATH_FINAL_RE = re.compile(r"(?mi)^\s*Final\s+Answer\s*:\s*(\S.*)\s*$")
_LEGACY_MATH_FINAL_RE = re.compile(r"(?mi)^\s*####\s*(.+?)(?:\s*<end>)?\s*$")
_ANSWER_IS_RE = re.compile(r"(?mi)^.*\b(?:final\s+)?answer\s+(?:is|=)\s*(\S.*?)\s*$")
_TEXT_WRAPPER_RE = re.compile(r"\\(?:text|mathrm|operatorname)\s*\{([^{}]*)\}")


@dataclass(frozen=True)
class EvaluationResult:
    dataset: str
    extracted_answer: str | None
    reference_answer: str | None
    correct: bool
    extraction_mode: str | None
    strict_correct: bool | None = None
    legacy_correct: bool | None = None
    error: str | None = None
    error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{label} must be a non-empty string")
    return value.strip()


def _canonical_decimal(value: str) -> Decimal:
    if _NUMBER_RE.fullmatch(value) is None:
        raise EvaluationError(f"invalid numeric answer: {value!r}")
    try:
        return Decimal(value.replace(",", ""))
    except InvalidOperation as exc:
        raise EvaluationError(f"invalid numeric answer: {value!r}") from exc


def _extract_gsm_reference(reference: str) -> str:
    reference = _require_text(reference, "GSM8K reference")
    matches = list(_GSM_FINAL_RE.finditer(reference))
    if matches:
        return matches[-1].group(1)
    if _NUMBER_RE.fullmatch(reference):
        return reference
    raise EvaluationError("GSM8K reference is missing a valid '#### <number>' answer")


def _last_nonempty_line(text: str) -> str:
    return next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")


def has_strict_terminal_field(dataset: str, response: str) -> bool:
    """Return whether the last non-empty line follows the dataset's strict protocol."""

    if not isinstance(response, str):
        raise TypeError("response must be a string")
    final_line = _last_nonempty_line(response)
    if dataset == "gsm8k":
        return _GSM_FINAL_RE.fullmatch(final_line) is not None
    if dataset == "math500":
        return _MATH_FINAL_RE.fullmatch(final_line) is not None
    raise EvaluationError("dataset must be 'gsm8k' or 'math500'")


def extract_gsm8k_answer(response: str) -> tuple[str, str]:
    """Return ``(answer, mode)`` using strict field first, then last-number fallback."""

    response = _require_text(response, "GSM8K response")
    strict = _GSM_FINAL_RE.fullmatch(_last_nonempty_line(response))
    if strict:
        return strict.group(1), "strict_final_field"
    numbers = _NUMBER_RE.findall(response)
    if not numbers:
        raise AnswerExtractionError("GSM8K response contains no numeric answer")
    return numbers[-1], "fallback_last_number"


def extract_legacy_last_digit_run(response: str) -> str:
    """Reproduce the archived evaluator's intentionally defective parser."""

    matches = re.findall(r"\d+", _require_text(response, "GSM8K response"))
    if not matches:
        raise AnswerExtractionError("response contains no unsigned digit sequence")
    return matches[-1]


def _consume_braced(text: str, start: int) -> tuple[str, int] | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
        elif text[index] == "}" and (index == 0 or text[index - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    return None


def _boxed_answers(text: str) -> list[str]:
    answers: list[str] = []
    for match in re.finditer(r"\\(?:boxed|fbox)\s*", text):
        start = match.end()
        while start < len(text) and text[start].isspace():
            start += 1
        consumed = _consume_braced(text, start)
        if consumed is not None and consumed[0].strip():
            answers.append(consumed[0].strip())
    return answers


def extract_math500_answer(response: str) -> tuple[str, str]:
    """Extract MATH500 output without guessing an arbitrary parenthesized phrase."""

    response = _require_text(response, "MATH500 response")
    strict = _MATH_FINAL_RE.fullmatch(_last_nonempty_line(response))
    if strict:
        return strict.group(1).strip(), "strict_final_field"
    boxed = _boxed_answers(response)
    if boxed:
        return boxed[-1], "fallback_boxed"
    legacy_matches = list(_LEGACY_MATH_FINAL_RE.finditer(response))
    if legacy_matches:
        return legacy_matches[-1].group(1).strip(), "legacy_hash_field"
    answer_matches = list(_ANSWER_IS_RE.finditer(response))
    if answer_matches:
        return answer_matches[-1].group(1).strip(), "fallback_answer_phrase"
    raise AnswerExtractionError(
        "MATH500 response has no 'Final Answer:', boxed answer, legacy #### field, "
        "or explicit 'answer is' phrase"
    )


def _strip_math_wrappers(value: str) -> str:
    value = value.strip().replace("−", "-").replace("π", r"\pi")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    value = value.replace(r"\!", "").replace(r"\,", "")
    previous = None
    while previous != value:
        previous = value
        value = _TEXT_WRAPPER_RE.sub(lambda match: match.group(1), value)
    wrappers = (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]"))
    changed = True
    while changed:
        changed = False
        value = value.strip().rstrip(".;，")
        for left, right in wrappers:
            if (
                value.startswith(left)
                and value.endswith(right)
                and len(value) > len(left) + len(right)
            ):
                value = value[len(left) : -len(right)].strip()
                changed = True
                break
        box_match = re.fullmatch(r"\\(?:boxed|fbox)\s*\{(.*)\}", value, flags=re.S)
        if box_match:
            value = box_match.group(1).strip()
            changed = True
    return value


def _simple_assignment(value: str) -> tuple[str, str] | None:
    """Return a conservative one-variable assignment without discarding its LHS."""

    match = re.fullmatch(r"([A-Za-z])\s*=\s*(.+)", value, flags=re.S)
    if match is None:
        return None
    return match.group(1), match.group(2).strip()


def _split_top_level(value: str) -> list[str] | None:
    parts: list[str] = []
    start = 0
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in "([{":
            stack.append(char)
        elif char in ")]}":
            if not stack or stack[-1] != pairs[char]:
                return None
            stack.pop()
        elif char == "," and not stack:
            parts.append(value[start:index].strip())
            start = index + 1
    if stack:
        return None
    parts.append(value[start:].strip())
    return parts


def _outer_collection(value: str) -> tuple[str, list[str]] | None:
    value = value.strip()
    candidates = ((r"\{", r"\}", "set", True), ("{", "}", "set", False))
    for left, right, kind, explicit_set in candidates:
        if value.startswith(left) and value.endswith(right):
            inner = value[len(left) : -len(right)]
            parts = _split_top_level(inner)
            if parts is None or any(not part for part in parts):
                return None
            if not explicit_set and len(parts) == 1:
                return None
            return kind, parts
    if len(value) >= 2 and value[0] in "([" and value[-1] in ")]":
        parts = _split_top_level(value[1:-1])
        if parts is None or len(parts) < 2 or any(not part for part in parts):
            return None
        return f"sequence:{value[0]}{value[-1]}", parts
    return None


def _consume_tex_argument(text: str, start: int) -> tuple[str, int] | None:
    """Consume one braced or single-token TeX argument."""

    while start < len(text) and text[start].isspace():
        start += 1
    consumed = _consume_braced(text, start)
    if consumed is not None:
        return consumed
    if start >= len(text) or text[start] in "{}":
        return None
    if text[start] != "\\":
        return text[start], start + 1
    end = start + 1
    if end >= len(text):
        return None
    if text[end].isalpha():
        while end < len(text) and text[end].isalpha():
            end += 1
    else:
        end += 1
    return text[start:end], end


def _replace_latex_command(
    text: str,
    command: str,
    argument_count: int,
    render: Any,
) -> str | None:
    """Replace a whitelisted TeX command using balanced argument parsing."""

    cursor = 0
    output: list[str] = []
    while cursor < len(text):
        position = text.find(command, cursor)
        while position >= 0:
            command_end = position + len(command)
            if command_end == len(text) or not text[command_end].isalpha():
                break
            position = text.find(command, command_end)
        if position < 0:
            output.append(text[cursor:])
            break
        output.append(text[cursor:position])
        end = position + len(command)
        arguments: list[str] = []
        for _ in range(argument_count):
            consumed = _consume_tex_argument(text, end)
            if consumed is None:
                return None
            arguments.append(consumed[0])
            end = consumed[1]
        converted = [_latex_to_safe_expression(argument) for argument in arguments]
        if any(value is None for value in converted):
            return None
        output.append(render(converted))
        cursor = end
    return "".join(output)


def _latex_to_safe_expression(value: str) -> str | None:
    """Convert a small documented LaTeX subset to a Python expression string.

    The returned string is never evaluated. It is parsed with :mod:`ast` and
    converted node-by-node into SymPy objects by ``_safe_symbolic_parse``.
    """

    value = _strip_math_wrappers(value)
    if not value or len(value) > 500:
        return None
    while r"\frac" in value:
        converted = _replace_latex_command(
            value,
            r"\frac",
            2,
            lambda parts: f"(({parts[0]})/({parts[1]}))",
        )
        if converted is None or converted == value:
            return None
        value = converted
    while r"\sqrt" in value:
        converted = _replace_latex_command(
            value,
            r"\sqrt",
            1,
            lambda parts: f"sqrt({parts[0]})",
        )
        if converted is None or converted == value:
            return None
        value = converted
    for source, replacement in {
        r"\cdot": "*",
        r"\times": "*",
        r"\pi": "pi",
    }.items():
        value = value.replace(source, replacement)
    if "\\" in value:
        return None
    value = value.replace("{", "(").replace("}", ")").replace("^", "**")
    value = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", value)
    value = re.sub(r"(?<=\))(?=[A-Za-z\d(])", "*", value)
    if not re.fullmatch(r"[A-Za-z0-9+\-*/().\s]+", value):
        return None
    if "__" in value or "//" in value:
        return None
    return value


def _safe_symbolic_parse(value: str) -> Any | None:
    """Build a SymPy expression from a strict AST whitelist without ``eval``."""

    expression = _latex_to_safe_expression(value)
    if expression is None:
        return None
    try:
        tree = ast.parse(expression, mode="eval")
        import sympy
    except (ImportError, SyntaxError, ValueError, MemoryError, RecursionError):
        return None

    node_count = 0

    def convert(node: ast.AST, depth: int = 0) -> Any:
        nonlocal node_count
        node_count += 1
        if node_count > 200 or depth > 50:
            raise ValueError("symbolic expression is too complex")
        if isinstance(node, ast.Expression):
            return convert(node.body, depth + 1)
        if isinstance(node, ast.Constant):
            if type(node.value) is int:
                if len(str(abs(node.value))) > 100:
                    raise ValueError("integer literal is too large")
                return sympy.Integer(node.value)
            if type(node.value) is float:
                return sympy.Rational(str(node.value))
            raise ValueError("unsupported literal")
        if isinstance(node, ast.Name):
            if node.id == "pi":
                return sympy.pi
            if node.id == "sqrt":
                raise ValueError("sqrt is only valid as a function call")
            return sympy.Symbol(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = convert(node.operand, depth + 1)
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                exponent_node = node.right
                sign = 1
                if isinstance(exponent_node, ast.UnaryOp) and isinstance(
                    exponent_node.op, (ast.UAdd, ast.USub)
                ):
                    sign = -1 if isinstance(exponent_node.op, ast.USub) else 1
                    exponent_node = exponent_node.operand
                if (
                    isinstance(exponent_node, ast.Constant)
                    and type(exponent_node.value) is int
                    and abs(sign * exponent_node.value) > 1_000
                ):
                    raise ValueError("literal exponent is too large")
            left = convert(node.left, depth + 1)
            right = convert(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                if right.is_Integer is True:
                    integer_exponent = int(right)
                    if abs(integer_exponent) > 1_000:
                        raise ValueError("computed exponent is too large")
                    if left.is_Integer is True and integer_exponent > 0:
                        base_digits = len(str(abs(int(left))))
                        if base_digits * integer_exponent > 5_000:
                            raise ValueError("integer power result would be too large")
                return left**right
            raise ValueError("unsupported binary operator")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sqrt"
            and len(node.args) == 1
            and not node.keywords
        ):
            return sympy.sqrt(convert(node.args[0], depth + 1))
        raise ValueError(f"unsupported syntax: {type(node).__name__}")

    try:
        return convert(tree)
    except (ArithmeticError, TypeError, ValueError, MemoryError, RecursionError):
        return None


def _sympy_equivalent(left: str, right: str) -> bool:
    parsed = [_safe_symbolic_parse(left), _safe_symbolic_parse(right)]
    if any(value is None for value in parsed):
        return False
    if parsed[0].free_symbols != parsed[1].free_symbols:
        return False
    try:
        import sympy

        difference = sympy.simplify(parsed[0] - parsed[1])
        return bool(difference == 0)
    except (ArithmeticError, TypeError, ValueError, NotImplementedError):
        return False


def math_equivalent(predicted: str, reference: str) -> bool:
    """Conservative symbolic equivalence with tuple/set structure preserved."""

    left = _strip_math_wrappers(_require_text(predicted, "predicted MATH answer"))
    right = _strip_math_wrappers(_require_text(reference, "reference MATH answer"))
    left_assignment = _simple_assignment(left)
    right_assignment = _simple_assignment(right)
    if (
        left_assignment is not None
        and right_assignment is not None
        and left_assignment[0] != right_assignment[0]
    ):
        return False
    if left_assignment is not None and right_assignment is None:
        parsed_rhs = _safe_symbolic_parse(left_assignment[1])
        if parsed_rhs is None or parsed_rhs.free_symbols:
            return False
    if right_assignment is not None and left_assignment is None:
        parsed_rhs = _safe_symbolic_parse(right_assignment[1])
        if parsed_rhs is None or parsed_rhs.free_symbols:
            return False
    if left_assignment is not None:
        left = left_assignment[1]
    if right_assignment is not None:
        right = right_assignment[1]
    compact_left = re.sub(r"\s+", "", left)
    compact_right = re.sub(r"\s+", "", right)
    if compact_left == compact_right:
        return True

    left_collection = _outer_collection(left)
    right_collection = _outer_collection(right)
    if left_collection is not None or right_collection is not None:
        if left_collection is None or right_collection is None:
            return False
        left_kind, left_parts = left_collection
        right_kind, right_parts = right_collection
        if left_kind != right_kind or len(left_parts) != len(right_parts):
            return False
        if left_kind == "set":
            unmatched = list(right_parts)
            for component in left_parts:
                match_index = next(
                    (
                        index
                        for index, candidate in enumerate(unmatched)
                        if math_equivalent(component, candidate)
                    ),
                    None,
                )
                if match_index is None:
                    return False
                unmatched.pop(match_index)
            return not unmatched
        return all(math_equivalent(a, b) for a, b in zip(left_parts, right_parts, strict=True))

    try:
        return _canonical_decimal(left) == _canonical_decimal(right)
    except EvaluationError:
        pass

    # MATH500 contains some unordered multi-answer references without outer
    # brackets (for example ``1,-2``). Compare those component-wise, but only
    # after the strict thousands-separated numeric parser above has run.
    left_parts = _split_top_level(left)
    right_parts = _split_top_level(right)
    left_is_list = left_parts is not None and len(left_parts) > 1
    right_is_list = right_parts is not None and len(right_parts) > 1
    if left_is_list or right_is_list:
        if not left_is_list or not right_is_list or len(left_parts) != len(right_parts):
            return False
        unmatched = list(right_parts)
        for left_part in left_parts:
            match_index = next(
                (
                    index
                    for index, right_part in enumerate(unmatched)
                    if math_equivalent(left_part, right_part)
                ),
                None,
            )
            if match_index is None:
                return False
            unmatched.pop(match_index)
        return not unmatched
    return _sympy_equivalent(left, right)


def evaluate_response(dataset: str, response: str, reference: str) -> EvaluationResult:
    """Score one response and retain strict/fallback provenance."""

    if dataset == "gsm8k":
        gold = _extract_gsm_reference(reference)
        legacy_correct: bool | None
        try:
            legacy_correct = _canonical_decimal(
                extract_legacy_last_digit_run(response)
            ) == _canonical_decimal(gold)
        except (AnswerExtractionError, EvaluationError):
            legacy_correct = False
        try:
            predicted, mode = extract_gsm8k_answer(response)
            correct = _canonical_decimal(predicted) == _canonical_decimal(gold)
            return EvaluationResult(
                dataset=dataset,
                extracted_answer=predicted,
                reference_answer=gold,
                correct=correct,
                extraction_mode=mode,
                strict_correct=correct if mode == "strict_final_field" else None,
                legacy_correct=legacy_correct,
            )
        except (AnswerExtractionError, EvaluationError) as exc:
            return EvaluationResult(
                dataset=dataset,
                extracted_answer=None,
                reference_answer=gold,
                correct=False,
                extraction_mode=None,
                strict_correct=None,
                legacy_correct=legacy_correct,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    if dataset == "math500":
        gold = _require_text(reference, "MATH500 reference")
        try:
            predicted, mode = extract_math500_answer(response)
            correct = math_equivalent(predicted, gold)
            return EvaluationResult(
                dataset=dataset,
                extracted_answer=predicted,
                reference_answer=gold,
                correct=correct,
                extraction_mode=mode,
                strict_correct=correct if mode == "strict_final_field" else None,
            )
        except (AnswerExtractionError, EvaluationError) as exc:
            return EvaluationResult(
                dataset=dataset,
                extracted_answer=None,
                reference_answer=gold,
                correct=False,
                extraction_mode=None,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    raise EvaluationError("dataset must be 'gsm8k' or 'math500'")
