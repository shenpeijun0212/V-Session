"""Thin, auditable benchmark runner with lazy vLLM and Transformers backends."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .data import Example, file_sha256, load_examples
from .evaluation import evaluate_response, has_strict_terminal_field
from .prompts import PromptTemplate, load_prompt
from .structure import audit_vsession_structure


class RunnerError(RuntimeError):
    """Raised when an experiment cannot be run safely."""


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 2048
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int = 42
    stop: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.do_sample and self.temperature <= 0:
            raise ValueError("sampling requires temperature > 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if any(not isinstance(value, str) or not value for value in self.stop):
            raise ValueError("stop strings must be non-empty strings")


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    method: str
    protocol: str
    dataset_path: Path
    prompt_path: Path
    output_dir: Path
    render_mode: str = "replace"
    model_path: str | None = None
    backend: str = "vllm"
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    request_batch_size: int = 16
    trust_remote_code: bool = False
    expected_count: int | None = None
    limit: int | None = None
    overwrite: bool = False
    generation: GenerationConfig = GenerationConfig()

    def __post_init__(self) -> None:
        if self.dataset not in {"gsm8k", "math500"}:
            raise ValueError("dataset must be 'gsm8k' or 'math500'")
        if not self.method.strip() or not self.protocol.strip():
            raise ValueError("method and protocol must be non-empty")
        if self.backend not in {"vllm", "hf"}:
            raise ValueError("backend must be 'vllm' or 'hf'")
        if self.tensor_parallel_size <= 0 or self.request_batch_size <= 0:
            raise ValueError("tensor_parallel_size and request_batch_size must be positive")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")


@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    stop_reason: str | int | None = None


class Backend(Protocol):
    """Minimal interface used by the runner and fake test backends."""

    name: str

    def generate(
        self,
        prompts: Sequence[str],
        generation: GenerationConfig,
        *,
        start_index: int,
    ) -> list[Generation]: ...

    def metadata(self) -> dict[str, Any]: ...


class VLLMBackend:
    """Raw-completion vLLM backend; no chat template is applied."""

    name = "vllm"

    def __init__(self, config: RunConfig):
        if not config.model_path:
            raise RunnerError("model_path is required for inference")
        try:
            import vllm
        except ImportError as exc:
            raise RunnerError(
                "install the vLLM optional dependency before using --backend vllm"
            ) from exc
        self._vllm = vllm
        try:
            self._engine = vllm.LLM(
                model=config.model_path,
                tokenizer=config.model_path,
                dtype=config.dtype,
                tensor_parallel_size=config.tensor_parallel_size,
                gpu_memory_utilization=config.gpu_memory_utilization,
                trust_remote_code=config.trust_remote_code,
                seed=config.generation.seed,
            )
        except Exception as exc:
            raise RunnerError(f"vLLM could not load {config.model_path!r}: {exc}") from exc

    def generate(
        self,
        prompts: Sequence[str],
        generation: GenerationConfig,
        *,
        start_index: int,
    ) -> list[Generation]:
        del start_index  # vLLM receives the explicit experiment seed below.
        params = self._vllm.SamplingParams(
            n=1,
            temperature=generation.temperature if generation.do_sample else 0.0,
            top_p=generation.top_p,
            repetition_penalty=generation.repetition_penalty,
            max_tokens=generation.max_new_tokens,
            seed=generation.seed,
            stop=list(generation.stop) or None,
        )
        try:
            requests = list(self._engine.generate(list(prompts), params, use_tqdm=False))
        except Exception as exc:
            raise RunnerError(f"vLLM generation failed: {exc}") from exc
        if len(requests) != len(prompts):
            raise RunnerError(f"vLLM returned {len(requests)} outputs for {len(prompts)} prompts")
        generations: list[Generation] = []
        for request in requests:
            if len(request.outputs) != 1:
                raise RunnerError("vLLM must return exactly one completion per prompt")
            candidate = request.outputs[0]
            generations.append(
                Generation(
                    text=str(candidate.text),
                    input_tokens=len(request.prompt_token_ids)
                    if request.prompt_token_ids
                    else None,
                    output_tokens=len(candidate.token_ids)
                    if candidate.token_ids is not None
                    else None,
                    finish_reason=str(candidate.finish_reason) if candidate.finish_reason else None,
                    stop_reason=candidate.stop_reason,
                )
            )
        return generations

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "version": str(getattr(self._vllm, "__version__", "unknown")),
            "prompt_mode": "raw_completion",
            "chat_template_applied": False,
        }


class TransformersBackend:
    """Sequential raw-completion fallback for environments without vLLM."""

    name = "hf"

    def __init__(self, config: RunConfig):
        if not config.model_path:
            raise RunnerError("model_path is required for inference")
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise RunnerError(
                "install the hf optional dependencies before using --backend hf"
            ) from exc
        self._torch = torch
        self._transformers = transformers
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }.get(config.dtype)
        if dtype is None:
            raise RunnerError("HF dtype must be bfloat16, float16, float32, or auto")
        try:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                config.model_path,
                trust_remote_code=config.trust_remote_code,
            )
            self._model = transformers.AutoModelForCausalLM.from_pretrained(
                config.model_path,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=config.trust_remote_code,
            ).eval()
        except Exception as exc:
            raise RunnerError(f"Transformers could not load {config.model_path!r}: {exc}") from exc
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    def generate(
        self,
        prompts: Sequence[str],
        generation: GenerationConfig,
        *,
        start_index: int,
    ) -> list[Generation]:
        results: list[Generation] = []
        for offset, prompt in enumerate(prompts):
            seed = generation.seed + start_index + offset
            random.seed(seed)
            self._torch.manual_seed(seed)
            if self._torch.cuda.is_available():
                self._torch.cuda.manual_seed_all(seed)
            encoded = self._tokenizer(prompt, return_tensors="pt", truncation=False)
            input_ids = encoded["input_ids"].to(self._model.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self._model.device)
            kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "do_sample": generation.do_sample,
                "num_beams": 1,
                "max_new_tokens": generation.max_new_tokens,
                "repetition_penalty": generation.repetition_penalty,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }
            if generation.do_sample:
                kwargs.update(temperature=generation.temperature, top_p=generation.top_p)
            encoded_stops: list[tuple[str, list[int]]] = []
            for stop in generation.stop:
                token_ids = self._tokenizer.encode(stop, add_special_tokens=False)
                if token_ids:
                    encoded_stops.append((stop, token_ids))
            if encoded_stops:
                transformers = self._transformers

                class _StopOnTokenSequences(transformers.StoppingCriteria):
                    def __init__(self, sequences: list[tuple[str, list[int]]]):
                        self._sequences = sequences

                    def __call__(self, input_ids, scores, **unused):  # noqa: ANN001
                        del scores, unused
                        row = input_ids[0]
                        return any(
                            len(token_ids) <= row.shape[0]
                            and row[-len(token_ids) :].tolist() == token_ids
                            for _, token_ids in self._sequences
                        )

                kwargs["stopping_criteria"] = transformers.StoppingCriteriaList(
                    [_StopOnTokenSequences(encoded_stops)]
                )
            with self._torch.inference_mode():
                output = self._model.generate(**kwargs)
            generated_ids = output[0, input_ids.shape[1] :]
            output_tokens = int(generated_ids.shape[0])
            raw_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            text, stop_reason = _trim_at_first_stop(raw_text, generation.stop)
            if stop_reason is None:
                generated_list = generated_ids.tolist()
                stop_reason = next(
                    (
                        stop
                        for stop, token_ids in encoded_stops
                        if len(token_ids) <= len(generated_list)
                        and generated_list[-len(token_ids) :] == token_ids
                    ),
                    None,
                )
            results.append(
                Generation(
                    text=text,
                    input_tokens=int(input_ids.shape[1]),
                    output_tokens=output_tokens,
                    finish_reason="stop"
                    if stop_reason is not None
                    else ("length" if output_tokens >= generation.max_new_tokens else "eos"),
                    stop_reason=stop_reason,
                )
            )
        return results

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "transformers_version": str(self._transformers.__version__),
            "torch_version": str(self._torch.__version__),
            "prompt_mode": "raw_completion",
            "chat_template_applied": False,
        }


def _trim_at_first_stop(text: str, stops: Sequence[str]) -> tuple[str, str | None]:
    """Remove the earliest configured stop string and anything after it."""

    matches = [(position, stop) for stop in stops if (position := text.find(stop)) >= 0]
    if not matches:
        return text, None
    position, stop = min(matches, key=lambda item: item[0])
    return text[:position], stop


def _new_temporary_path(target: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    return Path(name)


def _write_json_file(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _commit_staged_files(staged: Sequence[tuple[Path, Path]]) -> None:
    """Atomically replace each target, restoring old files if a commit step fails."""

    backups: dict[Path, Path] = {}
    targets_without_originals = {target for _, target in staged if not target.exists()}
    try:
        for _, target in staged:
            if not target.exists():
                continue
            backup = _new_temporary_path(target)
            backup.unlink()
            try:
                os.link(target, backup)
            except OSError:
                shutil.copy2(target, backup)
            backups[target] = backup
        for temporary, target in staged:
            os.replace(temporary, target)
        if staged:
            _fsync_directory(staged[0][1].parent)
    except Exception:
        for _, target in reversed(staged):
            backup = backups.get(target)
            if backup is not None and backup.exists():
                os.replace(backup, target)
            elif target in targets_without_originals:
                target.unlink(missing_ok=True)
        if staged:
            _fsync_directory(staged[0][1].parent)
        raise
    finally:
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def _json_dump(path: Path, value: Any) -> None:
    temporary = _new_temporary_path(path)
    try:
        _write_json_file(temporary, value)
        _commit_staged_files(((temporary, path),))
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _exclusive_run_lock(path: Path) -> Iterator[None]:
    """Prevent two evaluators from writing the same output directory."""

    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - supported execution target is POSIX
        raise RunnerError("exclusive run locking requires fcntl on this platform") from exc
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunnerError(f"another evaluation is already using {path.parent}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _uses_paper_vsession_scoring(config: RunConfig) -> bool:
    is_vsession_family = (
        config.method == "vsession"
        or config.method.startswith("vsession_")
        or config.method.startswith("no_")
    )
    return config.protocol.startswith("paper-") and is_vsession_family


def _uses_modern_vsession_structure(config: RunConfig) -> bool:
    is_vsession_family = config.method.startswith("vsession") or config.method.startswith("no_")
    return is_vsession_family and not config.protocol.startswith("legacy-")


def _rendered_prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _manifest(config: RunConfig, prompt: PromptTemplate, examples: list[Example]) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "dataset": config.dataset,
        "method": config.method,
        "protocol": config.protocol,
        "dataset_path": str(config.dataset_path),
        "dataset_sha256": file_sha256(config.dataset_path),
        "prompt_path": str(config.prompt_path),
        "prompt_template_sha256": prompt.sha256,
        "render_mode": config.render_mode,
        "model_path": config.model_path,
        "backend": config.backend,
        "dtype": config.dtype,
        "tensor_parallel_size": config.tensor_parallel_size,
        "request_batch_size": config.request_batch_size,
        "example_count": len(examples),
        "expected_source_count": config.expected_count,
        "limit": config.limit,
        "generation": asdict(config.generation),
        "engineering_defaults_disclosure": (
            "The paper does not specify max_new_tokens, seed, dtype, backend, batching, "
            "stop strings, or repetition penalty; configured values are recorded here."
        ),
    }
    if config.protocol.startswith("paper-zero-shot"):
        manifest["paper_reported_decoding_requirements"] = {
            "pass_at_1": True,
            "greedy": True,
            "sampling": False,
            "beam_search": False,
        }
        manifest["conforms_to_paper_reported_decoding"] = not config.generation.do_sample
    return manifest


def prepare_run(config: RunConfig) -> tuple[list[Example], PromptTemplate, list[str]]:
    examples = load_examples(
        config.dataset_path,
        config.dataset,
        limit=config.limit,
        expected_count=config.expected_count,
    )
    prompt = load_prompt(config.prompt_path, config.render_mode)
    rendered = [prompt.render(example.question) for example in examples]
    return examples, prompt, rendered


def dry_run(config: RunConfig) -> dict[str, Any]:
    """Validate data/prompts and write a manifest without importing an ML backend."""

    examples, prompt, rendered = prepare_run(config)
    manifest = _manifest(config, prompt, examples)
    manifest["dry_run"] = True
    manifest["sample"] = [
        {
            "item_id": example.item_id,
            "prompt_sha256": _rendered_prompt_hash(text),
            "prompt_preview": text[:500],
        }
        for example, text in zip(examples[:3], rendered[:3], strict=True)
    ]
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output = config.output_dir / "dry-run.json"
    with _exclusive_run_lock(config.output_dir / ".run.lock"):
        if output.exists() and not config.overwrite:
            raise RunnerError(f"refusing to overwrite {output}; pass --overwrite intentionally")
        _json_dump(output, manifest)
    return manifest


def _make_backend(config: RunConfig) -> Backend:
    if config.backend == "vllm":
        return VLLMBackend(config)
    return TransformersBackend(config)


def run_evaluation(config: RunConfig, backend: Backend | None = None) -> dict[str, Any]:
    """Run inference, write per-item JSONL, and return a compact summary."""

    examples, prompt, rendered = prepare_run(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = config.output_dir / "predictions.jsonl"
    summary_path = config.output_dir / "summary.json"
    metadata_path = config.output_dir / "metadata.json"
    lock_path = config.output_dir / ".run.lock"

    with _exclusive_run_lock(lock_path):
        existing = [
            path for path in (predictions_path, summary_path, metadata_path) if path.exists()
        ]
        if existing and not config.overwrite:
            names = ", ".join(str(path) for path in existing)
            raise RunnerError(f"refusing to overwrite existing run artifacts: {names}")

        active_backend = backend or _make_backend(config)
        paper_vsession_scoring = _uses_paper_vsession_scoring(config)
        metadata = _manifest(config, prompt, examples)
        metadata["runtime"] = active_backend.metadata()
        metadata["scoring_policy"] = (
            "strict_terminal_final_field" if paper_vsession_scoring else "documented_parser"
        )
        metadata["structure_audit_policy"] = (
            "Delimiter style and balance are enforced. Usage is reported separately because "
            "the manuscript says delimiters apply when calculations are applicable but does "
            "not publish an applicability classifier."
        )
        metadata["completed"] = True

        temporary_paths: list[Path] = []
        try:
            metadata_temporary = _new_temporary_path(metadata_path)
            temporary_paths.append(metadata_temporary)
            predictions_temporary = _new_temporary_path(predictions_path)
            temporary_paths.append(predictions_temporary)
            _write_json_file(metadata_temporary, metadata)

            correct_count = 0
            parser_correct_count = 0
            fallback_count = 0
            fallback_available = 0
            strict_count = 0
            strict_available = 0
            legacy_count = 0
            legacy_available = 0
            structure_valid = 0
            structure_available = 0
            delimiter_usage_observed = 0
            delimiter_style_compliant = 0
            extraction_modes: dict[str, int] = {}
            output_characters = 0
            started = time.time()

            with predictions_temporary.open("w", encoding="utf-8") as output:
                for start in range(0, len(examples), config.request_batch_size):
                    batch_examples = examples[start : start + config.request_batch_size]
                    batch_prompts = rendered[start : start + config.request_batch_size]
                    generations = active_backend.generate(
                        batch_prompts,
                        config.generation,
                        start_index=start,
                    )
                    if len(generations) != len(batch_examples):
                        raise RunnerError("backend returned the wrong number of completions")
                    for example, rendered_prompt, generation in zip(
                        batch_examples, batch_prompts, generations, strict=True
                    ):
                        evaluation = evaluate_response(
                            config.dataset, generation.text, example.answer
                        )
                        parser_correct = bool(evaluation.correct)
                        parser_correct_count += int(parser_correct)
                        terminal_final_field = has_strict_terminal_field(
                            config.dataset, generation.text
                        )
                        if paper_vsession_scoring:
                            strict_correct: bool | None = bool(
                                parser_correct
                                and evaluation.extraction_mode == "strict_final_field"
                                and terminal_final_field
                            )
                            primary_correct = strict_correct
                        else:
                            strict_correct = evaluation.strict_correct
                            primary_correct = parser_correct
                        correct_count += int(primary_correct)
                        if strict_correct is not None:
                            strict_available += 1
                            strict_count += int(strict_correct)
                        fallback_correct = (
                            parser_correct
                            if evaluation.extraction_mode not in {None, "strict_final_field"}
                            else None
                        )
                        if fallback_correct is not None:
                            fallback_available += 1
                            fallback_count += int(fallback_correct)
                        if evaluation.legacy_correct is not None:
                            legacy_available += 1
                            legacy_count += int(evaluation.legacy_correct)
                        if evaluation.extraction_mode:
                            extraction_modes[evaluation.extraction_mode] = (
                                extraction_modes.get(evaluation.extraction_mode, 0) + 1
                            )
                        structure = None
                        if _uses_modern_vsession_structure(config):
                            delimiter_style = "ascii" if "ascii" in config.method else "unicode"
                            if "no_symbols" in config.method:
                                delimiter_style = "none"
                            removed = {
                                "no_goal": "Goal",
                                "no_solution": "Solution",
                                "no_thinking": "Thinking",
                                "no_reasoning": "Reasoning",
                                "no_result": "Result",
                            }.get(config.method)
                            expected_stages = tuple(
                                stage
                                for stage in (
                                    "Goal",
                                    "Solution",
                                    "Thinking",
                                    "Reasoning",
                                    "Result",
                                )
                                if stage != removed
                            )
                            structure = audit_vsession_structure(
                                generation.text,
                                dataset=config.dataset,
                                delimiter_style=delimiter_style,
                                expected_stages=expected_stages,
                            )
                            structure_available += 1
                            structure_valid += int(structure.valid)
                            delimiter_usage_observed += int(structure.delimiters_used)
                            delimiter_style_compliant += int(structure.delimiter_style_compliant)
                        output_characters += len(generation.text)
                        evaluation_fields = evaluation.to_dict()
                        evaluation_fields["correct"] = primary_correct
                        evaluation_fields["strict_correct"] = strict_correct
                        record = {
                            "schema_version": 1,
                            "item_id": example.item_id,
                            "index": example.index,
                            "dataset": config.dataset,
                            "method": config.method,
                            "protocol": config.protocol,
                            "question": example.question,
                            "reference": example.answer,
                            "prompt_template_sha256": prompt.sha256,
                            "prompt_sha256": _rendered_prompt_hash(rendered_prompt),
                            "completion": generation.text,
                            **evaluation_fields,
                            "parser_correct": parser_correct,
                            "fallback_correct": fallback_correct,
                            "strict_terminal_field": terminal_final_field,
                            "structure_audit": structure.to_dict() if structure else None,
                            "input_tokens": generation.input_tokens,
                            "output_tokens": generation.output_tokens,
                            "finish_reason": generation.finish_reason,
                            "stop_reason": generation.stop_reason,
                            "seed": config.generation.seed,
                        }
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())

            elapsed = time.time() - started
            total = len(examples)
            summary = {
                "schema_version": 1,
                "dataset": config.dataset,
                "method": config.method,
                "protocol": config.protocol,
                "scoring_policy": metadata["scoring_policy"],
                "total": total,
                "correct": correct_count,
                "accuracy": correct_count / total,
                "parser_correct": parser_correct_count,
                "parser_accuracy": parser_correct_count / total,
                "fallback_correct": fallback_count,
                "fallback_available": fallback_available,
                "strict_correct": strict_count,
                "strict_available": strict_available,
                "legacy_correct": legacy_count,
                "legacy_available": legacy_available,
                "structure_valid": structure_valid,
                "structure_available": structure_available,
                "delimiter_usage_observed": delimiter_usage_observed,
                "delimiter_style_compliant": delimiter_style_compliant,
                "extraction_modes": extraction_modes,
                "mean_output_characters": output_characters / total,
                "elapsed_seconds": elapsed,
            }
            summary_temporary = _new_temporary_path(summary_path)
            temporary_paths.append(summary_temporary)
            _write_json_file(summary_temporary, summary)
            _commit_staged_files(
                (
                    (predictions_temporary, predictions_path),
                    (summary_temporary, summary_path),
                    (metadata_temporary, metadata_path),
                )
            )
            return summary
        finally:
            for temporary in temporary_paths:
                temporary.unlink(missing_ok=True)
