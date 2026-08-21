from __future__ import annotations

from vsession.evaluation import (
    evaluate_response,
    extract_gsm8k_answer,
    extract_legacy_last_digit_run,
    extract_math500_answer,
    math_equivalent,
)


def test_gsm_strict_signed_decimal_and_comma():
    answer, mode = extract_gsm8k_answer("work\n#### -1,250.5")
    assert answer == "-1,250.5"
    assert mode == "strict_final_field"


def test_gsm_fallback_and_legacy_defect_are_both_visible():
    answer, mode = extract_gsm8k_answer("The answer is -3.14")
    assert (answer, mode) == ("-3.14", "fallback_last_number")
    assert extract_legacy_last_digit_run("The answer is -3.14") == "14"
    result = evaluate_response("gsm8k", "The answer is -3.14", "solution\n#### -3.14")
    assert result.correct is True
    assert result.legacy_correct is False
    assert result.strict_correct is None


def test_strict_final_field_must_be_the_last_nonempty_line():
    answer, mode = extract_gsm8k_answer("#### 5\nTrailing value 6")
    assert (answer, mode) == ("6", "fallback_last_number")
    answer, mode = extract_math500_answer("Final Answer: 5\nTrailing explanation: answer is 6")
    assert (answer, mode) == ("6", "fallback_answer_phrase")


def test_math_final_answer_precedes_earlier_box():
    response = "An intermediate value is \\boxed{2}.\nFinal Answer: \\frac{1}{2}"
    answer, mode = extract_math500_answer(response)
    assert answer == r"\frac{1}{2}"
    assert mode == "strict_final_field"


def test_math_nested_box_is_extracted():
    answer, mode = extract_math500_answer(r"Thus \boxed{\frac{1+\sqrt{5}}{2}}.")
    assert answer == r"\frac{1+\sqrt{5}}{2}"
    assert mode == "fallback_boxed"


def test_math_equivalence_preserves_structure():
    assert math_equivalent(r"\frac{1}{2}", "0.5")
    assert math_equivalent(r"\frac43", "4/3")
    assert math_equivalent(r"\{1,2\}", r"\{2,1\}")
    assert math_equivalent("x=5", "5")
    assert math_equivalent("x=π", r"\pi")
    assert math_equivalent("(1/2)", "0.5")
    assert math_equivalent("1, -2", "1,-2")
    assert math_equivalent("-2,1", "1,-2")
    assert math_equivalent("7,5,3", "3,5,7")
    assert math_equivalent("58,500", "58500")
    assert not math_equivalent("[0,1)", "(0,1]")
    assert not math_equivalent("(1,2)", "(2,1)")
    assert not math_equivalent("3", "3, 5, 7")
    assert not math_equivalent("1", "1,-2")
    assert not math_equivalent("-21", "-2,1")
    assert not math_equivalent("z=2x+3", "y=2x+3")
    assert not math_equivalent("2x+3", "y=2x+3")
    assert not math_equivalent("-2x", "y=-2x")
    assert not math_equivalent("x", "y=x")


def test_math_symbolic_parser_never_executes_model_text(capsys):
    malicious = "__import__('builtins').print('SHOULD_NOT_RUN')"
    assert not math_equivalent(malicious, "0")
    assert capsys.readouterr().out == ""
    assert not math_equivalent("9**999999999", "1")
    assert not math_equivalent("9**(500+501)", "1")
    assert not math_equivalent("9**(10**9)", "1")


def test_math_no_longer_guesses_arbitrary_parentheses():
    result = evaluate_response("math500", "Work used (1, 2), but I stopped.", "5")
    assert result.correct is False
    assert result.error_type == "AnswerExtractionError"
