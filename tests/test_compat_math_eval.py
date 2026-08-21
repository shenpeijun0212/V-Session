from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("compat_math_eval", ROOT / "test" / "math_eval.py")
assert SPEC is not None and SPEC.loader is not None
COMPAT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPAT)


def _args(tmp_path: Path, prompt_mode: str) -> argparse.Namespace:
    return argparse.Namespace(
        prompt_mode=prompt_mode,
        fewshot_prompt_path=tmp_path / "legacy.txt",
        save_dir=tmp_path,
        model_name_or_path="model",
        dataset_path=tmp_path / "math.jsonl",
        backend="vllm",
        dtype="bfloat16",
        tensor_parallel_size=1,
        request_batch_size=2,
        trust_remote_code=False,
        limit=1,
        overwrite=False,
        max_new_tokens=32,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        seed=42,
    )


def test_legacy_five_shot_has_archived_continuation_stops(tmp_path):
    config, _ = COMPAT.build_run_config(
        _args(tmp_path, "legacy-five-shot"), timestamp="20260821-000000"
    )

    assert config.protocol == "legacy-math500-five-shot"
    assert config.render_mode == "append"
    assert config.generation.stop == ("<end>", "\nQuestion:")


def test_paper_zero_shot_does_not_inherit_legacy_stops(tmp_path):
    config, _ = COMPAT.build_run_config(
        _args(tmp_path, "paper-zero-shot"), timestamp="20260821-000000"
    )
    assert config.protocol == "paper-zero-shot-reconstructed"
    assert config.generation.stop == ()
