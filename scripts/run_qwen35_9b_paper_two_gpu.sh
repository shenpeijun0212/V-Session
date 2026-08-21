#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-}"
DATASET="${DATASET:-gsm8k}"
BACKEND="${BACKEND:-vllm}"
DTYPE="${DTYPE:-bfloat16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
REQUEST_BATCH_SIZE="${REQUEST_BATCH_SIZE:-16}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1}}"
MODEL_LABEL="${MODEL_LABEL:-}"

if [[ -z "$MODEL_LABEL" ]]; then
  if [[ -n "$MODEL_PATH" ]]; then
    trimmed_model_path="${MODEL_PATH%/}"
    MODEL_LABEL="${trimmed_model_path##*/}"
  else
    MODEL_LABEL="qwen3.5-9b-base"
  fi
fi
model_slug="${MODEL_LABEL//[!A-Za-z0-9_.-]/-}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results/paper-zero-shot-reconstructed/$model_slug}"

IFS=',' read -r -a gpu_ids <<< "$GPU_IDS"
if (( ${#gpu_ids[@]} != 2 )) || [[ -z "${gpu_ids[0]}" || -z "${gpu_ids[1]}" ]]; then
  echo "GPU_IDS must contain exactly two comma-separated GPU IDs" >&2
  exit 2
fi
if [[ "${gpu_ids[0]}" == "${gpu_ids[1]}" ]]; then
  echo "GPU_IDS must identify two distinct GPUs" >&2
  exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

case "$DATASET" in
  gsm8k)
    DATASET_PATH="${DATASET_PATH:-$REPO_ROOT/data/gsm8k_test.jsonl}"
    EXPECTED_COUNT=1319
    ;;
  math500)
    DATASET_PATH="${DATASET_PATH:-$REPO_ROOT/data/MATH.jsonl}"
    EXPECTED_COUNT=500
    ;;
  *) echo "DATASET must be gsm8k or math500" >&2; exit 2 ;;
esac

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "Dataset not found: $DATASET_PATH" >&2
  if [[ "$DATASET" == "gsm8k" ]]; then
    echo "Run: python scripts/prepare_gsm8k.py" >&2
  fi
  exit 2
fi
if [[ "$DRY_RUN" != "1" && -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH is required for inference" >&2
  exit 2
fi

run_method() {
  local gpu="$1"
  local method="$2"
  local command_name="run"
  if [[ "$DRY_RUN" == "1" ]]; then
    command_name="dry-run"
  fi
  local args=(
    -m vsession.cli "$command_name"
    --dataset "$DATASET"
    --dataset-path "$DATASET_PATH"
    --method "$method"
    --protocol paper-zero-shot-reconstructed
    --output-dir "$RESULTS_ROOT/$DATASET/$method"
    --expected-count "$EXPECTED_COUNT"
    --backend "$BACKEND"
    --dtype "$DTYPE"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --repetition-penalty "$REPETITION_PENALTY"
    --request-batch-size "$REQUEST_BATCH_SIZE"
    --seed "$SEED"
  )
  if [[ -n "$LIMIT" ]]; then
    args+=(--limit "$LIMIT")
  fi
  if [[ "$DRY_RUN" != "1" ]]; then
    args+=(--model-path "$MODEL_PATH")
  fi
  if [[ "$OVERWRITE" == "1" ]]; then
    args+=(--overwrite)
  fi
  mkdir -p "$RESULTS_ROOT/logs"
  local log_file="$RESULTS_ROOT/logs/${DATASET}-${method}.log"
  if [[ -e "$log_file" && "$OVERWRITE" != "1" ]]; then
    echo "Refusing to overwrite log: $log_file (set OVERWRITE=1 intentionally)" >&2
    return 2
  fi
  echo "GPU $gpu: $DATASET / $method"
  PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON_BIN" "${args[@]}" \
    > "$log_file" 2>&1
}

if [[ "$DRY_RUN" == "1" ]]; then
  for method in direct cot pot ps vsession; do
    run_method "${gpu_ids[0]}" "$method"
  done
  echo "All five dry-runs completed under $RESULTS_ROOT"
  exit 0
fi

child_pids=()
cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  for child_pid in "${child_pids[@]}"; do
    kill -TERM "$child_pid" 2>/dev/null || true
  done
  for child_pid in "${child_pids[@]}"; do
    wait "$child_pid" 2>/dev/null || true
  done
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

(
  for method in direct cot pot; do
    run_method "${gpu_ids[0]}" "$method"
  done
) &
pid_gpu0=$!
child_pids+=("$pid_gpu0")
(
  for method in ps vsession; do
    run_method "${gpu_ids[1]}" "$method"
  done
) &
pid_gpu1=$!
child_pids+=("$pid_gpu1")

status=0
wait "$pid_gpu0" || status=$?
wait "$pid_gpu1" || status=$?
child_pids=()
if (( status != 0 )); then
  echo "At least one evaluation failed; inspect $RESULTS_ROOT/logs" >&2
  exit "$status"
fi
echo "All five methods completed under $RESULTS_ROOT"
