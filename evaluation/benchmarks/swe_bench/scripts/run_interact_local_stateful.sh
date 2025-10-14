CLI_AVAILABLE="false" \
USE_HINT_TEXT="false" \
PURE_RAG_MODE="false" \
SYSTEM_PROMPT_FILENAME="system_prompt_tom_benchmark.j2" \
TOM_AGENT_MODEL="gpt-5-nano-2025-08-07" \
bash ./evaluation/benchmarks/swe_bench/scripts/run_infer_interact.sh llm.claude-sonnet-4-20250514 HEAD TomCodeActAgent 1 100 1 cmu-lti/stateful test 1 stateful gpt5nano
