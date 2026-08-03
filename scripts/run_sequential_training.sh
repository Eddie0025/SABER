#!/usr/bin/env bash

# SABER Integrated Training Pipeline (DoRA -> Gate -> GRPO)
# This script orchestrates the sequential training of all specialists.
# It enforces a strict conditional gate: only models that beat the base
# model on the generalized benchmark proceed to the GRPO RL phase.

set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error when substituting
set -o pipefail # Return value of a pipeline is the status of the last command to exit with a non-zero status

# --- Constants & Colors ---
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

LOG_FILE="logs/training_report.md"

# Initialize markdown report table
mkdir -p logs
echo "# SABER Training Report" > "$LOG_FILE"
echo "" >> "$LOG_FILE"
echo "| Specialist | DoRA Status | Base Score | Adapter Score | Delta | Gate | GRPO Status |" >> "$LOG_FILE"
echo "|---|---|---|---|---|---|---|" >> "$LOG_FILE"

# --- Master Specialist List ---
SPECIALISTS=(
    "cybersecurity"
    "python"
    "javascript"
    "sql"
    "science"
    "medical"
    "finance"
    "architecture_qa"
    "architecture_planner"
)

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}  SABER MASTER TRAINING PIPELINE INITIALIZING...${NC}"
echo -e "${BLUE}======================================================${NC}"

for SPEC in "${SPECIALISTS[@]}"; do
    # SAFETY GUARD: Ensure SPEC is not empty before rm -rf
    [[ -n "$SPEC" ]] || { echo -e "${RED}Error: SPEC variable is empty. Aborting to prevent accidental deletion.${NC}"; exit 1; }

    echo -e "\n${YELLOW}>>> [1/5] PURGING OLD CHECKPOINTS FOR: ${SPEC}...${NC}"
    rm -rf "models/${SPEC}_checkpoints" "models/${SPEC}_v2" "models/${SPEC}_grpo"
    
    echo -e "${YELLOW}>>> [2/5] LAUNCHING DORA SFT FOR: ${SPEC}...${NC}"
    if ! python -m saber.training.trainer --mode dora --specialist "$SPEC" --target_modules attn_only; then
        echo -e "${RED}>>> [!] DORA TRAINING FAILED OR SKIPPED (MOCK DATA). BYPASSING.${NC}"
        echo "| **${SPEC}** | ❌ Failed/Skipped | N/A | N/A | N/A | N/A | N/A |" >> "$LOG_FILE"
        continue
    fi
    
    echo -e "${YELLOW}>>> [3/5] LAUNCHING BENCHMARK GATE FOR: ${SPEC}...${NC}"
    # Run generalized evaluation. It outputs a JSON line at the end prefixed with "RESULT_PAYLOAD:"
    # containing {"base": 88.5, "adapter": 92.1, "pass": true, "is_structural": false}
    EVAL_OUTPUT=$(python scripts/eval_specialist.py --specialist "$SPEC" 2>&1) || true
    
    # Extract the payload line
    PAYLOAD=$(echo "$EVAL_OUTPUT" | grep "^RESULT_PAYLOAD:" | sed 's/RESULT_PAYLOAD://' || echo '{"error": "Eval crashed"}')
    
    echo -e "${BLUE}Eval Payload: $PAYLOAD${NC}"
    
    # Parse JSON payload (requires jq installed on DGX)
    BASE_SCORE=$(echo "$PAYLOAD" | jq -r '.base // "Error"')
    ADAPTER_SCORE=$(echo "$PAYLOAD" | jq -r '.adapter // "Error"')
    DELTA=$(echo "$PAYLOAD" | jq -r '.delta // "N/A"')
    PASS_GATE=$(echo "$PAYLOAD" | jq -r '.pass // "false"')
    STRUCTURAL=$(echo "$PAYLOAD" | jq -r '.is_structural // "false"')
    
    # Structural models (like Orchestrator) skip MCQ benchmarks
    if [ "$STRUCTURAL" = "true" ]; then
        echo -e "${YELLOW}>>> [!] STRUCTURAL MODEL DETECTED. FLAGGING FOR MANUAL OPEN-ENDED REVIEW.${NC}"
        echo "| **${SPEC}** | ✅ Complete | N/A | N/A | N/A | ⚠️ Manual | Pending |" >> "$LOG_FILE"
        continue
    fi
    
    # Conditional Gate Logic
    if [ "$PASS_GATE" = "true" ]; then
        echo -e "${GREEN}>>> [4/5] GATE PASSED! ADAPTER BEATS BASE (${ADAPTER_SCORE}% vs ${BASE_SCORE}%). PROCEEDING TO GRPO...${NC}"
        
        echo -e "${YELLOW}>>> [5/5] LAUNCHING GRPO (PROMETHEUS 2 REWARDS) FOR: ${SPEC}...${NC}"
        if python -m saber.training.trainer --mode grpo --specialist "$SPEC"; then
            echo "| **${SPEC}** | ✅ Complete | ${BASE_SCORE}% | ${ADAPTER_SCORE}% | ${DELTA}% | ✅ PASS | ✅ Complete |" >> "$LOG_FILE"
            echo -e "${GREEN}>>> GRPO COMPLETE FOR ${SPEC}.${NC}"
        else
            echo "| **${SPEC}** | ✅ Complete | ${BASE_SCORE}% | ${ADAPTER_SCORE}% | ${DELTA}% | ✅ PASS | ❌ Failed |" >> "$LOG_FILE"
            echo -e "${RED}>>> GRPO FAILED FOR ${SPEC}.${NC}"
        fi
    else
        echo -e "${RED}>>> [4/5] GATE FAILED! ADAPTER (${ADAPTER_SCORE}%) <= BASE (${BASE_SCORE}%). SKIPPING GRPO.${NC}"
        echo "| **${SPEC}** | ✅ Complete | ${BASE_SCORE}% | ${ADAPTER_SCORE}% | ${DELTA}% | ❌ FAIL | Skipped |" >> "$LOG_FILE"
    fi
done

echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN}  PIPELINE COMPLETE. SEE logs/training_report.md${NC}"
echo -e "${GREEN}======================================================${NC}"
