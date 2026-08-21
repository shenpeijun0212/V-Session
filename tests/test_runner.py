from __future__ import annotations

import json
from dataclasses import replace

import pytest

from vsession.runner import (
    Generation,
    GenerationConfig,
    RunConfig,
    RunnerError,
    _exclusive_run_lock,
    dry_run,
    run_evaluation,
)


class FakeBackend:
    name = "fake"

    def generate(self, prompts, generation, *, start_index):
        del prompts, generation
        answers = ["#### 3", "#### 7"]
        return [
            Generation(
                text=answers[start_index + offset],
                input_tokens=10,
                output_tokens=2,
                finish_reason="stop",
            )
            for offset in range(len(answers[start_index : start_index + 2]))
        ]

    def metadata(self):
        return {"backend": "fake", "chat_template_applied": False}


class FailingBackend:
    name = "failing"

    def generate(self, prompts, generation, *, start_index):
        del prompts, generation, start_index
        raise RuntimeError("synthetic inference failure")

    def metadata(self):
        return {"backend": "failing", "chat_template_applied": False}


class FallbackBackend:
    name = "fallback"

    def generate(self, prompts, generation, *, start_index):
        del generation
        answers = ["The answer is 3", "#### 7\nTrailing explanation"]
        return [Generation(text=answers[start_index + offset]) for offset in range(len(prompts))]

    def metadata(self):
        return {"backend": "fallback", "chat_template_applied": False}


def _config(tmp_path):
    dataset = tmp_path / "gsm.jsonl"
    rows = [
        {"item_id": "one", "question": "1+2?", "answer": "#### 3"},
        {"item_id": "two", "question": "3+4?", "answer": "#### 7"},
    ]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Solve:\n{question}\n", encoding="utf-8")
    return RunConfig(
        dataset="gsm8k",
        method="direct",
        protocol="paper-zero-shot-test",
        dataset_path=dataset,
        prompt_path=prompt,
        output_dir=tmp_path / "results",
        expected_count=2,
        request_batch_size=2,
        generation=GenerationConfig(max_new_tokens=8),
    )


def test_dry_run_does_not_need_a_model(tmp_path):
    config = _config(tmp_path)
    result = dry_run(config)
    assert result["dry_run"] is True
    assert result["example_count"] == 2
    assert result["expected_source_count"] == 2
    assert result["paper_reported_decoding_requirements"]["greedy"] is True
    assert result["conforms_to_paper_reported_decoding"] is True
    assert (config.output_dir / "dry-run.json").is_file()


def test_fake_backend_end_to_end(tmp_path):
    config = _config(tmp_path)
    summary = run_evaluation(config, FakeBackend())
    assert summary["correct"] == 2
    records = [
        json.loads(line)
        for line in (config.output_dir / "predictions.jsonl").read_text().splitlines()
    ]
    assert [record["item_id"] for record in records] == ["one", "two"]
    assert all(record["correct"] for record in records)


def test_failed_overwrite_preserves_complete_artifacts(tmp_path):
    config = _config(tmp_path)
    run_evaluation(config, FakeBackend())
    paths = [
        config.output_dir / "predictions.jsonl",
        config.output_dir / "summary.json",
        config.output_dir / "metadata.json",
    ]
    before = {path: path.read_bytes() for path in paths}

    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        run_evaluation(replace(config, overwrite=True), FailingBackend())

    assert {path: path.read_bytes() for path in paths} == before
    assert not list(config.output_dir.glob(".*.tmp"))


def test_output_directory_lock_is_exclusive(tmp_path):
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    lock = output_dir / ".run.lock"
    with _exclusive_run_lock(lock):
        with pytest.raises(RunnerError, match="already using"):
            with _exclusive_run_lock(lock):
                pass


def test_paper_vsession_primary_score_requires_terminal_strict_field(tmp_path):
    config = replace(_config(tmp_path), method="vsession")
    summary = run_evaluation(config, FallbackBackend())
    records = [
        json.loads(line)
        for line in (config.output_dir / "predictions.jsonl").read_text().splitlines()
    ]

    assert summary["correct"] == 0
    assert summary["parser_correct"] == 2
    assert summary["fallback_correct"] == 2
    assert all(record["correct"] is False for record in records)
    assert all(record["parser_correct"] is True for record in records)
    assert all(record["fallback_correct"] is True for record in records)
    assert all(record["strict_terminal_field"] is False for record in records)

    strict_config = replace(config, output_dir=tmp_path / "strict-results")
    strict_summary = run_evaluation(strict_config, FakeBackend())
    assert strict_summary["correct"] == 2
    assert strict_summary["strict_correct"] == 2
    assert strict_summary["strict_available"] == 2


def test_legacy_vsession_does_not_apply_modern_structure_contract(tmp_path):
    dataset = tmp_path / "math.jsonl"
    dataset.write_text(
        json.dumps({"unique_id": "one", "problem": "2+3?", "answer": "5"}) + "\n",
        encoding="utf-8",
    )
    prompt = tmp_path / "legacy.txt"
    prompt.write_text("Example.\nQuestion: ", encoding="utf-8")
    config = RunConfig(
        dataset="math500",
        method="vsession",
        protocol="legacy-math500-five-shot",
        dataset_path=dataset,
        prompt_path=prompt,
        output_dir=tmp_path / "legacy-results",
        render_mode="append",
        expected_count=1,
        generation=GenerationConfig(stop=("<end>", "\nQuestion:")),
    )

    class LegacyBackend:
        name = "legacy"

        def generate(self, prompts, generation, *, start_index):
            del prompts, generation, start_index
            return [Generation("Goal: add.\nSolution: calculate.\nResult: 5.\n#### 5")]

        def metadata(self):
            return {"backend": "legacy"}

    summary = run_evaluation(config, LegacyBackend())
    record = json.loads((config.output_dir / "predictions.jsonl").read_text())
    assert summary["correct"] == 1
    assert summary["structure_available"] == 0
    assert record["structure_audit"] is None
