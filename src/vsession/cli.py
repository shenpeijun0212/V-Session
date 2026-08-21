"""Command-line entry point for paper-derived V-Session evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import GenerationConfig, RunConfig, RunnerError, dry_run, run_evaluation

METHODS = (
    "direct",
    "cot",
    "pot",
    "ps",
    "vsession",
    "no_goal",
    "no_solution",
    "no_thinking",
    "no_reasoning",
    "no_result",
    "vsession_no_symbols",
    "vsession_ascii_symbols",
    "vsession_compact",
)
MATH500_ONLY_METHODS = frozenset(
    {
        "no_goal",
        "no_solution",
        "no_thinking",
        "no_reasoning",
        "no_result",
        "vsession_no_symbols",
        "vsession_ascii_symbols",
        "vsession_compact",
    }
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=("gsm8k", "math500"), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--protocol", default="paper-zero-shot-reconstructed")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--prompt-path", type=Path)
    parser.add_argument(
        "--render-mode",
        choices=("replace", "append", "append_exact"),
        default="replace",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--backend", choices=("vllm", "hf"), default="vllm")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--request-batch-size", type=int, default=16)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--stop", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run model inference and evaluation")
    _add_common_arguments(run_parser)
    run_parser.add_argument("--model-path", required=True)
    dry_parser = subparsers.add_parser(
        "dry-run", help="validate and render data/prompts without loading a model"
    )
    _add_common_arguments(dry_parser)
    dry_parser.add_argument("--model-path")
    return parser


def _resolved_config(args: argparse.Namespace) -> RunConfig:
    if args.dataset == "gsm8k" and args.method in MATH500_ONLY_METHODS:
        supported = ", ".join(method for method in METHODS if method not in MATH500_ONLY_METHODS)
        raise ValueError(
            f"method {args.method!r} is only available for MATH500; GSM8K supports: {supported}"
        )
    root = project_root()
    dataset_path = args.dataset_path
    if dataset_path is None:
        dataset_path = (
            root / "data" / ("gsm8k_test.jsonl" if args.dataset == "gsm8k" else "MATH.jsonl")
        )
    prompt_path = args.prompt_path
    if prompt_path is None:
        prompt_path = root / "prompt" / "zero-shot" / args.dataset / f"{args.method}.txt"
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = root / "results" / args.protocol / args.dataset / args.method
    generation = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        stop=tuple(args.stop),
    )
    return RunConfig(
        dataset=args.dataset,
        method=args.method,
        protocol=args.protocol,
        dataset_path=dataset_path.resolve(),
        prompt_path=prompt_path.resolve(),
        output_dir=output_dir.resolve(),
        render_mode=args.render_mode,
        model_path=args.model_path,
        backend=args.backend,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        request_batch_size=args.request_batch_size,
        trust_remote_code=args.trust_remote_code,
        expected_count=args.expected_count,
        limit=args.limit,
        overwrite=args.overwrite,
        generation=generation,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _resolved_config(args)
        result = dry_run(config) if args.command == "dry-run" else run_evaluation(config)
    except (ValueError, RunnerError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
