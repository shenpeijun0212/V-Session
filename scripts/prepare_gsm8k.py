#!/usr/bin/env python3
"""Download and freeze the canonical 1,319-example GSM8K test split as JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/gsm8k_test.jsonl"))
    parser.add_argument("--revision", help="optional immutable Hugging Face dataset revision")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    if metadata_path.exists():
        raise SystemExit(f"refusing to overwrite {metadata_path}")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("install the data extra first: pip install -e '.[data]'") from exc

    dataset = load_dataset("openai/gsm8k", "main", split="test", revision=args.revision)
    if len(dataset) != 1319:
        raise SystemExit(f"expected 1,319 GSM8K test examples, received {len(dataset)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(dataset):
            record = {
                "item_id": f"gsm8k-test-{index:04d}",
                "question": row["question"],
                "answer": row["answer"],
            }
            encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
            handle.write(encoded.decode("utf-8"))
            digest.update(encoded)
    temporary.replace(args.output)
    metadata = {
        "source": "openai/gsm8k",
        "configuration": "main",
        "split": "test",
        "revision": args.revision,
        "count": 1319,
        "sha256": digest.hexdigest(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
