from __future__ import annotations

import json

import pytest

from vsession.data import DataError, load_examples
from vsession.prompts import PromptError, load_prompt


def test_replace_prompt_has_one_question_slot(tmp_path):
    path = tmp_path / "prompt.txt"
    path.write_text("Question:\n{question}\nResponse:\n", encoding="utf-8")
    prompt = load_prompt(path, "replace")
    rendered = prompt.render("How many?")
    assert rendered == "Question:\nHow many?\nResponse:\n"
    assert rendered.count("How many?") == 1


def test_append_prompt_does_not_add_a_second_question_cue(tmp_path):
    path = tmp_path / "prompt.txt"
    path.write_text("Example.\nQuestion: ", encoding="utf-8")
    prompt = load_prompt(path, "append")
    assert prompt.render("2 + 2?") == "Example.\nQuestion: 2 + 2?"


def test_append_prompt_inserts_missing_separator(tmp_path):
    path = tmp_path / "prompt.txt"
    path.write_text("Example.\nQuestion:", encoding="utf-8")
    prompt = load_prompt(path, "append")
    exact = load_prompt(path, "append_exact")
    assert prompt.render("2 + 2?").endswith("Question: 2 + 2?")
    assert exact.render("2 + 2?").endswith("Question:2 + 2?")


def test_prompt_contract_rejects_wrong_placeholder_count(tmp_path):
    path = tmp_path / "prompt.txt"
    path.write_text("{question} and {question}", encoding="utf-8")
    with pytest.raises(PromptError, match="found 2"):
        load_prompt(path, "replace")


def test_data_loader_rejects_bad_json_with_line_number(tmp_path):
    path = tmp_path / "math.jsonl"
    path.write_text('{"problem":"ok","answer":"1"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(DataError, match=r":2: invalid JSON"):
        load_examples(path, "math500")


def test_data_loader_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "math.jsonl"
    rows = [
        {"unique_id": "same", "problem": "a", "answer": "1"},
        {"unique_id": "same", "problem": "b", "answer": "2"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(DataError, match="duplicate item ID"):
        load_examples(path, "math500")


def test_data_loader_rejects_non_scalar_item_ids(tmp_path):
    path = tmp_path / "math.jsonl"
    path.write_text(
        json.dumps({"unique_id": ["bad"], "problem": "a", "answer": "1"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DataError, match="string or integer"):
        load_examples(path, "math500")


def test_data_loader_validates_full_file_before_limit(tmp_path):
    path = tmp_path / "gsm.jsonl"
    path.write_text(
        json.dumps({"question": "first", "answer": "#### 1"}) + "\n" + "bad\n",
        encoding="utf-8",
    )
    with pytest.raises(DataError, match=r":2: invalid JSON"):
        load_examples(path, "gsm8k", limit=1)
