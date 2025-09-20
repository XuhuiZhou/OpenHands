#!/usr/bin/env bash

# SWE-Interact evaluation with remote runtime
# This script runs the interactive SWE-bench evaluation using remote runtime with 32 workers
# Usage: ./run_interact_remote.sh [model_name]
# Example: ./run_interact_remote.sh llm.claude-sonnet-4-20250514

MODEL=${1:-"llm.claude-sonnet-4-20250514"}

CLI_AVAILABLE="false" \
USE_HINT_TEXT="false" \
SYSTEM_PROMPT_FILENAME="system_prompt_interactive.j2" \
ALLHANDS_API_KEY="ah-69ce5388-6069-4c76-9d8d-eae75dd553dc" \
RUNTIME=remote \
SANDBOX_REMOTE_RUNTIME_API_URL="https://runtime.eval.all-hands.dev" \
EVAL_DOCKER_IMAGE_PREFIX="us-central1-docker.pkg.dev/evaluation-092424/swe-bench-images" \
nohup bash ./evaluation/benchmarks/swe_bench/scripts/run_infer_interact.sh \
  $MODEL \
  HEAD \
  CodeActAgent \
  50 \
  100 \
  50 \
  cmu-lti/interactive-swe \
  test > swe_bench_interact_remote_32_${MODEL//llm./}.log 2>&1 &

# Get the PID of the background process
NOHUP_PID=$!

echo "SWE-Interact evaluation started with remote runtime and 32 workers using model: $MODEL"
echo "Monitor progress with: tail -f swe_bench_interact_remote_32_${MODEL//llm./}.log"
echo "Check if running with: ps aux | grep run_infer_interact"

# Wait for the nohup process to finish
wait $NOHUP_PID

# Run evaluation after the nohup process completes
echo "Running evaluation..."
./evaluation/benchmarks/swe_bench/scripts/eval_infer.sh ./evaluation/evaluation_outputs/outputs/cmu-lti__interactive-swe-test/CodeActAgent/${MODEL//llm./}_maxiter_100_N_v0.54.0-no-hint-run_1/output.jsonl "" cmu-lti/interactive-swe test
