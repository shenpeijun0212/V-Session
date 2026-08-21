# V-Session: Structured Reasoning for Mathematical Problem Solving

This repository contains prompts, evaluation utilities, reference artifacts, and
paper-derived experiment code for **V-Session**, a five-stage reasoning framework:
`Goal → Solution → Thinking → Reasoning → Result`.

The current cleanup starts from upstream archive commit
`7293d0bda7381a206867c2b59ffd698899a6fac6`. Original `data/`, top-level
`prompt/*.txt`, and `log/` files are preserved byte-for-byte. The professional
runner and zero-shot templates added here are clearly marked as reconstructed from
the revised manuscript; they are not presented as recovered historical code.

## What is included

```text
LICENSE                       MIT license for original project code/documentation
data/                         Archived GSM8K train and MATH500 snapshots
prompt/                       Archived 8/5-shot prompt files
prompt/zero-shot/             Paper-derived Qwen3.5 zero-shot templates
src/vsession/                 Data, prompts, scoring, RSQ aggregation, runner, statistics
scripts/prepare_gsm8k.py      Freeze the canonical 1,319-item GSM8K test split
scripts/run_qwen35_9b_*.sh    Two-GPU, five-method Qwen3.5-9B example
scripts/validate_traces.py    V-Session trace checks before fine-tuning
eval/                         Accuracy and paired-statistics utilities
fine-tuning/                  Guarded Llama3-8B/LLaMA-Factory template
test/                         Backward-friendly MATH500 entry point
log/                          Preserved Qwen2.5-3B reference logs
tests/                        Unit tests that do not load a language model
```

## Protocols: do not mix them

The revised manuscript defines the new Qwen3.5 main experiment as **zero-shot**,
with no worked examples, full GSM8K test (1,319 items), full MATH500 (500 items),
and greedy pass@1 decoding. Five methods are compared: Direct/Base, CoT, PoT,
Plan-and-Solve, and V-Session.

The archived repository separately contains GSM8K 8-shot and MATH 5-shot evaluation
prefixes. Those artifacts are useful for historical comparisons but are not the
Qwen3.5 main-table protocol. The paper describes a separate 8-shot prompt for
converting gold GSM8K solutions into fine-tuning traces. That conversion prompt was
not archived and is not the same artifact as the retained evaluation prefixes.

One archived MATH 5-shot demonstration overlaps the included MATH500 snapshot.
Use `legacy-five-shot` only as an explicitly labeled historical protocol, not as a
paper-main result.

## Installation

Python 3.10 or newer is required. Exact historical package versions were not
archived, so the dependency ranges below are compatibility specifications rather
than a claim of the original environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For vLLM on a CUDA machine, use a fresh environment with a PyTorch/CUDA-compatible
vLLM wheel:

```bash
pip install -r requirements-vllm.txt
```

## Validate without loading a model

The dry-run path validates every source record, verifies the prompt placeholder,
renders examples, records hashes/configuration, and does not import PyTorch or vLLM:

```bash
python -m vsession.cli dry-run \
  --dataset math500 \
  --method vsession \
  --expected-count 500 \
  --limit 2 \
  --output-dir results/dry-run/math500-vsession
```

## Prepare GSM8K test data

The archived `data/GSM8K.json` and `data/GSM8K.jsonl` each contain 7,473 examples
and must not be used as the paper's 1,319-item test set. Prepare and fingerprint the
canonical test split with:

```bash
pip install -e '.[data]'
python scripts/prepare_gsm8k.py --output data/gsm8k_test.jsonl
```

For a stronger provenance record, pass an immutable dataset `--revision`.

## Run one paper-derived experiment

The default backend is vLLM. Prompts are sent as raw completions without a chat
template, which is appropriate for a Base checkpoint. The command records all
generation settings because the manuscript does not specify the exact checkpoint,
dtype, backend, maximum output length, stop strings, repetition penalty, batching,
or random seed.

```bash
python -m vsession.cli run \
  --dataset math500 \
  --dataset-path data/MATH.jsonl \
  --expected-count 500 \
  --method vsession \
  --model-path /path/to/Qwen3.5-9B-Base \
  --backend vllm \
  --dtype bfloat16 \
  --max-new-tokens 2048 \
  --output-dir results/paper-zero-shot-reconstructed/math500/vsession
```

`2048`, BF16, seed 42, vLLM, and repetition penalty 1.0 are transparent
engineering defaults, not paper-reported values.

## Five methods on two GPUs

The repository includes one Qwen3.5-9B Base example, as requested. It runs
Direct/CoT/PoT sequentially on GPU 0 and PS/V-Session sequentially on GPU 1:

```bash
MODEL_PATH=/path/to/Qwen3.5-9B-Base \
DATASET=math500 \
bash scripts/run_qwen35_9b_paper_two_gpu.sh
```

Set `DRY_RUN=1` to validate all five configurations without loading the model.
`GPU_IDS` accepts exactly two comma-separated device IDs and defaults to an existing
`CUDA_VISIBLE_DEVICES` value (otherwise `0,1`). Result paths derive from the model
directory name; set `MODEL_LABEL` or `RESULTS_ROOT` explicitly when needed. Existing
logs/results are protected unless `OVERWRITE=1` is intentionally supplied, and
interrupting the launcher terminates its child jobs.

## Historical MATH entry point

The familiar wrapper remains available. Its default is the revised paper zero-shot
V-Session template:

```bash
MODEL_NAME_OR_PATH=/path/to/model bash test/run_math_eval.sh
```

To deliberately use the archived MATH prefix (now correctly prepended), set
`PROMPT_MODE=legacy-five-shot`. Results are written as `predictions.jsonl`,
`metadata.json`, and `summary.json`; malformed data is no longer skipped silently.

## Scoring and statistics

The scorer prioritizes the manuscript's standard final fields:

- GSM8K V-Session: `#### <number>`
- MATH500 V-Session: `Final Answer: <answer>`

For paper-derived V-Session and ablation runs, the primary `correct` field requires
the strict terminal field; a fallback parse is retained separately for diagnostics.
Because the manuscript does not publish the parser for Direct/CoT/PoT/PS, those
baselines use labeled fallback extraction when necessary. GSM8K additionally records
the archived last-unsigned-digit score as `legacy_correct`. MATH equivalence preserves
tuple/list/interval/set structure and converts a small LaTeX whitelist through a
bounded Python AST; model-produced strings are never passed to `eval`, `sympify`, or
`parse_expr`.

```bash
python eval/compute_accuracy.py results/path/predictions.jsonl
python eval/paired_statistics.py \
  --reference results/no_symbols/predictions.jsonl \
  --candidate results/full/predictions.jsonl
```

The paired tool aligns exact item IDs, reports a paired percentile-bootstrap
interval, and performs a two-sided exact McNemar test. Its difference is always
`candidate − reference`; the example therefore reports Full − No Symbols, matching
the paper's ablation-table direction. Bootstrap count/seed and the McNemar
implementation are disclosed engineering choices because the paper does not specify
them.

The preserved sample logs produce:

- MATH500: 196/500 (39.20%)
- GSM8K1000: 787/1000 (78.70%) after treating both `acc:True` and `acc:1.0` as
  correct; the archived literal-only utility reported 784/1000 (78.40%).

## RSQ aggregation (external ratings only)

The repository implements the paper's equal-weight and weighted RSQ equations for
already-collected four-dimension ratings. It deliberately does not generate ratings
or call an LLM judge. Ratings use one JSON object per dimension:

```json
{"style":"vsession","item_id":"math-001","evaluator_id":"human-1","dimension":"structural_integrity","score":8.5}
```

```bash
python eval/compute_rsq.py ratings.jsonl --paper-panel --preset equal
```

`--paper-panel` requires a complete rectangular panel with the same eight evaluators
and item set for every style. Four paper-reported weight presets are available. For
multiple items, the implementation applies the published panel equation per item and
then averages items equally; the item index is absent from the displayed manuscript
formula, so this extension is explicitly recorded in the output.

An automatic RSQ judge cannot be reconstructed honestly: the manuscript prose calls
for four dimension scores, while the supplied appendix figure asks for one overall
score per response. The original 200 item IDs, evaluator-level ratings, four-dimension
rubric anchors, model revisions, and calibration parameters are also unavailable.
Consequently, this utility cannot reproduce the paper's reported RSQ table without
those external judgments. The binary checks in `structure.py` are format audits and
must not be presented as 0--10 RSQ Structural Integrity ratings.

## Ablations and trace validation

Paper-described stage and notation ablations are under
`prompt/zero-shot/math500/`. Exact ablation wording was not published, so these
files are explicitly described as prose-derived reconstructions. `No Result` still
requires an independent `Final Answer:` line, matching the paper's control.

Before using generated GSM8K V-Session traces for fine-tuning:

```bash
python scripts/validate_traces.py generated.jsonl validated.jsonl \
  --trace-field completion --answer-field answer --require-delimiters
```

The validator checks strict gold-answer agreement, stage presence/order/uniqueness,
delimiter balance/use, and known finish reasons. By default it writes only valid
records and returns status 3 if any record was rejected; `--allow-partial` opts into
a successful filtered run, while `--include-invalid` creates an audit-only output.
`--require-delimiters` conservatively requires a pair in every trace because the
paper's “when applicable” condition cannot be classified automatically. The
manuscript does not publish its arithmetic-expression grammar, so parseable-
calculation validation is reported as `not_checked` instead of being fabricated.

See [fine-tuning/README.md](fine-tuning/README.md) for the guarded Llama3-8B
LLaMA-Factory template and its reproducibility limits.

## Reproducibility boundary

See [PROVENANCE.md](PROVENANCE.md) for the upstream archive hash, preserved artifact
hashes, paper-known settings, and remaining unknowns. Do not label a run as an exact
paper reproduction unless those unknowns are resolved from original experiment
records.

## License

Unless otherwise noted, the original software and documentation in this repository
are licensed under the [MIT License](LICENSE).

The license does not replace or override the separate terms that apply to third-party
datasets, pretrained model weights, external frameworks, or other redistributed
artifacts. Users are responsible for complying with those upstream terms.

## Acknowledgements

This work uses GSM8K, MATH500, Hugging Face Transformers, vLLM, and the open-source
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) project.
