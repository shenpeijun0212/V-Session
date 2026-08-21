# V-Session: Structured Mathematical Reasoning

V-Session is a structured reasoning framework for mathematical problem solving. It organizes a
response into five stages:

1. **Goal** — identify the objective and relevant information.
2. **Solution** — choose an overall strategy.
3. **Thinking** — break the strategy into concrete steps.
4. **Reasoning** — carry out the derivation and calculations.
5. **Result** — report the final answer in the required format.

Formal calculations in V-Session use the Unicode delimiters `≪ ... ≫`. GSM8K responses end with
`#### <number>`, while MATH500 responses end with `Final Answer: <answer>`.

This repository provides prompts, inference runners, answer evaluation, statistical analysis,
trace validation, and fine-tuning utilities for V-Session experiments on GSM8K and MATH500.

## Features

- Five prompting methods: Direct, Chain-of-Thought (CoT), Program-of-Thought (PoT),
  Plan-and-Solve (PS), and V-Session.
- Zero-shot prompt templates for GSM8K and MATH500.
- Stage and notation ablations for MATH500.
- Raw-completion inference with vLLM, with Hugging Face Transformers as a fallback.
- Deterministic dataset validation, prompt hashing, and run metadata.
- GSM8K numeric scoring and conservative MATH500 symbolic equivalence checks.
- V-Session structure audits for stage order, final-answer format, and calculation delimiters.
- Paired bootstrap confidence intervals, exact McNemar tests, and RSQ aggregation.
- Validation of generated V-Session traces before supervised fine-tuning.

## Repository layout

```text
data/                         Benchmark data files
prompt/                       Few-shot and zero-shot prompt templates
src/vsession/                 Inference, data loading, scoring, statistics, and structure checks
scripts/prepare_gsm8k.py      Prepare the GSM8K test split
scripts/run_qwen35_9b_*.sh    Run the five methods on two GPUs
scripts/validate_traces.py    Validate generated V-Session traces
eval/                         Accuracy, RSQ, and paired-statistics commands
fine-tuning/                  LLaMA-Factory fine-tuning launcher
test/                         MATH500 command-line wrapper
tests/                        Model-free unit tests
log/                          Example evaluation logs
```

## Installation

Python 3.10 or newer is required.

For the core package, dataset preparation, Hugging Face backend, and development tools:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA inference with vLLM, use a clean environment and install a vLLM build compatible with
the machine's CUDA driver:

```bash
python3 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-vllm.txt
```

## Data preparation

`data/MATH.jsonl` contains the 500 MATH500 evaluation examples. The runner expects the GSM8K test
split at `data/gsm8k_test.jsonl`; create it with:

```bash
python scripts/prepare_gsm8k.py --output data/gsm8k_test.jsonl
```

The command downloads the 1,319-example `openai/gsm8k` test split and writes a SHA-256 metadata
file beside it. The included `data/GSM8K.json` and `data/GSM8K.jsonl` files contain the GSM8K
training split and should not be used for test-set accuracy.

## Validate a configuration

Use `dry-run` to validate the full dataset and render prompt samples without loading a model:

```bash
python -m vsession.cli dry-run \
  --dataset math500 \
  --method vsession \
  --expected-count 500 \
  --limit 2 \
  --output-dir results/dry-run/math500-vsession
```

The command writes `dry-run.json`, including the resolved settings, input hashes, and prompt
previews.

## Run an experiment

The default backend is vLLM. Prompts are passed to the selected checkpoint as raw completions
without a chat template.

```bash
CUDA_VISIBLE_DEVICES=0 python -m vsession.cli run \
  --dataset math500 \
  --dataset-path data/MATH.jsonl \
  --expected-count 500 \
  --method vsession \
  --model-path /path/to/Qwen3.5-9B-Base \
  --backend vllm \
  --dtype bfloat16 \
  --max-new-tokens 2048 \
  --output-dir results/math500/vsession
```

Use `--backend hf` for the Transformers backend. Common generation options include
`--request-batch-size`, `--tensor-parallel-size`, `--gpu-memory-utilization`, `--seed`,
`--repetition-penalty`, and `--stop`. Sampling is disabled by default; enable it explicitly with
`--do-sample --temperature <value>`.

Each completed run writes:

```text
metadata.json       Resolved configuration, hashes, and runtime information
predictions.jsonl   Item IDs, questions, prompt hashes, completions, scores, and structure audits
summary.json        Aggregate accuracy and generation statistics
```

Existing run artifacts are protected from accidental replacement. Use `--overwrite` only when an
existing output directory is intentionally being reused.

## Run all five methods on two GPUs

The launcher assigns Direct, CoT, and PoT to the first GPU, and PS and V-Session to the second.
Methods assigned to the same GPU run sequentially. This is method-level parallelism: each GPU
loads its own model instance.

```bash
MODEL_PATH=/path/to/Qwen3.5-9B-Base \
DATASET=math500 \
GPU_IDS=0,1 \
bash scripts/run_qwen35_9b_paper_two_gpu.sh
```

Set `DATASET=gsm8k` after preparing `data/gsm8k_test.jsonl`. To check all five configurations
without loading the model, add `DRY_RUN=1`. The launcher also accepts `PYTHON_BIN`, `DATASET_PATH`,
`MODEL_LABEL`, `RESULTS_ROOT`, `REQUEST_BATCH_SIZE`, `MAX_NEW_TOKENS`, `LIMIT`, and `OVERWRITE`.

## MATH500 ablations

The following additional `--method` values are available for MATH500:

```text
no_goal              Remove the Goal stage
no_solution          Remove the Solution stage
no_thinking          Remove the Thinking stage
no_reasoning         Remove the Reasoning stage
no_result            Remove the Result stage
vsession_no_symbols  Do not use calculation delimiters
vsession_ascii_symbols
                     Use << ... >> delimiters
vsession_compact     Use a compact five-stage response
```

Run an ablation with the same command used for a main experiment, changing `--method` and
`--output-dir`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vsession.cli run \
  --dataset math500 \
  --dataset-path data/MATH.jsonl \
  --expected-count 500 \
  --method no_goal \
  --model-path /path/to/Qwen3.5-9B-Base \
  --backend vllm \
  --output-dir results/math500/no_goal
```

## Accuracy and paired statistics

Recompute aggregate accuracy from the boolean `correct` field in an item-level prediction file:

```bash
python eval/compute_accuracy.py results/math500/vsession/predictions.jsonl --json
```

For V-Session and its ablations, the primary accuracy requires a correct answer in the strict
terminal field. Structure compliance is reported independently and does not replace answer
accuracy.

Compare two runs whose prediction files contain the same set of item IDs:

```bash
python eval/paired_statistics.py \
  --reference results/math500/vsession_no_symbols/predictions.jsonl \
  --candidate results/math500/vsession/predictions.jsonl \
  --output results/math500/full-vs-no-symbols.json
```

The reported accuracy difference is `candidate - reference`. The command uses paired percentile
bootstrap confidence intervals and a two-sided exact McNemar test, and records the input hashes,
bootstrap seed, and number of resamples.

## RSQ aggregation

RSQ ratings use JSONL with one row per style, item, evaluator, and dimension:

```json
{"style":"vsession","item_id":"math-001","evaluator_id":"human-1","dimension":"structural_integrity","score":8.5}
```

Each evaluator must provide scores from 0 to 10 for `structural_integrity`, `logical_rigor`,
`calculation_precision`, and `expression_clarity`. Aggregate the ratings with:

```bash
python eval/compute_rsq.py ratings.jsonl --paper-panel --preset equal \
  --output results/rsq.json
```

Available presets are `equal`, `structure-heavy`, `math-rigor-heavy`, and `clarity-heavy`.
The command aggregates the supplied ratings; it does not call a judge model or generate ratings.
`--paper-panel` requires every style-item group to use the same eight evaluators and every style
to use the same item set.

## Trace validation and fine-tuning

Validate generated GSM8K V-Session traces before adding them to a supervised fine-tuning dataset:

```bash
python scripts/validate_traces.py generated.jsonl validated.jsonl \
  --trace-field completion \
  --answer-field answer \
  --require-delimiters
```

By default, only valid records are written. The validator checks stage presence and order,
non-empty stage content, delimiter format, terminal answer agreement, and generation finish
reason. It does not evaluate the arithmetic inside `≪ ... ≫`. If any record is rejected, the
valid subset is still written and the command exits with status 3; use `--allow-partial` when a
successful filtered run is intended.

The fine-tuning launcher integrates with a LLaMA-Factory checkout and a dataset registered in
LLaMA-Factory. Inspect the generated command before training:

```bash
LLAMAFACTORY_ROOT=/path/to/LLaMA-Factory \
MODEL_NAME_OR_PATH=/path/to/Llama3-8B \
DATASET=gsm8k_vsession \
DRY_RUN=1 \
bash fine-tuning/run.sh
```

See [fine-tuning/README.md](fine-tuning/README.md) for the configurable batch size, learning rate,
warmup, epoch, DeepSpeed, output, and overwrite options.

## Tests

The test suite does not load a language model:

```bash
python -m pytest
ruff check .
ruff format --check .
```

## License

This project is licensed under the [MIT License](LICENSE).
