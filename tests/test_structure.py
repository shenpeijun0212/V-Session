from __future__ import annotations

import pytest

from vsession.structure import audit_vsession_structure

VALID = """Goal: compute the total.
Solution: add the values.
Thinking: identify 2 and 3.
Reasoning: ≪2 + 3 = 5≫.
Result: the total is five.
#### 5
"""


def test_valid_vsession_structure():
    audit = audit_vsession_structure(VALID, dataset="gsm8k")
    assert audit.valid
    assert audit.delimiter_pairs == 1


def test_duplicate_or_out_of_order_stage_is_invalid():
    duplicate = VALID.replace("Solution:", "Goal:")
    assert not audit_vsession_structure(duplicate, dataset="gsm8k").valid
    reordered = VALID.replace(
        "Goal: compute the total.\nSolution: add the values.", "Solution: add.\nGoal: compute."
    )
    assert not audit_vsession_structure(reordered, dataset="gsm8k").valid


def test_unbalanced_delimiter_is_invalid():
    assert not audit_vsession_structure(VALID.replace("≫", ""), dataset="gsm8k").valid


def test_gsm_final_field_uses_the_same_strict_number_grammar_as_scoring():
    exponent = VALID.replace("#### 5", "#### 1e3")
    assert audit_vsession_structure(exponent, dataset="gsm8k").final_answer_present
    malformed = VALID.replace("#### 5", "#### 1,,000")
    assert not audit_vsession_structure(malformed, dataset="gsm8k").final_answer_present


def test_notation_style_and_usage_are_auditable():
    no_symbols = audit_vsession_structure(VALID, dataset="gsm8k", delimiter_style="none")
    assert not no_symbols.valid
    assert no_symbols.unexpected_delimiters_present

    wrong_style = audit_vsession_structure(VALID, dataset="gsm8k", delimiter_style="ascii")
    assert not wrong_style.valid
    assert not wrong_style.delimiter_style_compliant

    without_calculation = VALID.replace("≪2 + 3 = 5≫", "verified directly")
    optional = audit_vsession_structure(without_calculation, dataset="gsm8k")
    required = audit_vsession_structure(
        without_calculation,
        dataset="gsm8k",
        require_delimiters=True,
    )
    assert optional.valid
    assert not optional.delimiters_used
    assert not required.valid


def test_stage_ablation_has_explicit_expected_stages():
    no_goal = """Solution: add.
Thinking: plan.
Reasoning: ≪2+3=5≫.
Result: five.
Final Answer: 5
"""
    audit = audit_vsession_structure(
        no_goal,
        dataset="math500",
        expected_stages=("Solution", "Thinking", "Reasoning", "Result"),
    )
    assert audit.valid


def test_empty_stage_body_and_noncanonical_expected_order_are_rejected():
    assert not audit_vsession_structure(VALID.replace("Goal: compute the total.", "Goal:")).valid
    with pytest.raises(ValueError, match="canonical"):
        audit_vsession_structure(VALID, expected_stages=("Solution", "Goal"))
