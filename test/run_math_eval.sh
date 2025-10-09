
#!/usr/bin/env bash
# Nohup runner for math_eval.py with editable parameters
set -euo pipefail

########## Editable parameters ##########
MODEL_NAME_OR_PATH="/gemini/code/V-Session/model/Qwen/Qwen2.5-3B"
DATASET_PATH="../data/MATH.jsonl"
FEWSHOT_PATH="../Prompt/math/V-Session_1-shot.txt"
LIMIT=500
TEMPERATURE=0.0
TOP_P=1.0
REPETITION_PENALTY=1.15
MAX_NEW_TOKENS=2000
SEED=42
SAVE_DIR="../results"
DO_SAMPLE=false     # true for sampling; false = greedy
########################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="math_eval.out"
PID_FILE="math_eval.pid"
PY_FILE="math_eval.py"

EXTRA_ARGS=("$@")

echo "Starting $PY_FILE with nohup ..."
nohup python3 -u "$PY_FILE"   --model_name_or_path "$MODEL_NAME_OR_PATH"   --dataset_path "$DATASET_PATH"   --fewshot_prompt_path "$FEWSHOT_PATH"   --limit "$LIMIT"   --temperature "$TEMPERATURE"   --top_p "$TOP_P"   --repetition_penalty "$REPETITION_PENALTY"   --max_new_tokens "$MAX_NEW_TOKENS"   --seed "$SEED"   --save_dir "$SAVE_DIR"   $( $DO_SAMPLE && echo "--do_sample" )   "${EXTRA_ARGS[@]}" > "$LOG_FILE" 2>&1 &

PID=$!
echo $PID > "$PID_FILE"
echo "Started PID $PID"
echo "Logs: $LOG_FILE"
echo "To stop: kill $(cat "$PID_FILE") || true"
