#!/usr/bin/env bash

# SABER Sequential Training Pipeline (DoRA SFT -> GRPO RL)
# Orchestrates end-to-end training for all specialists without intermediate gating.

set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error when substituting
set -o pipefail # Fail pipeline if any command fails

# --- Constants & Colors ---
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

LOG_FILE="logs/training_report.md"

# Initialize markdown report table
mkdir -p logs models results
echo "# SABER Training Report" > "$LOG_FILE"
echo "" >> "$LOG_FILE"
echo "| Specialist | DoRA SFT Status | DoRA Adapter | GRPO RL Status | Final Model |" >> "$LOG_FILE"
echo "|---|---|---|---|---|" >> "$LOG_FILE"

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
echo -e "${BLUE}  SABER MASTER TRAINING PIPELINE (DoRA -> GRPO)${NC}"
echo -e "${BLUE}======================================================${NC}"

for SPEC in "${SPECIALISTS[@]}"; do
    # SAFETY GUARD: Ensure SPEC is not empty before rm -rf
    [[ -n "$SPEC" ]] || { echo -e "${RED}Error: SPEC variable is empty. Aborting to prevent accidental deletion.${NC}"; exit 1; }

    echo -e "\n${YELLOW}>>> [1/3] PURGING OLD ARTIFACTS FOR: ${SPEC}...${NC}"
    rm -rf "models/${SPEC}_checkpoints" "models/${SPEC}_v2" "models/${SPEC}_grpo" "models/${SPEC}_grpo_final"
    
    echo -e "\n${YELLOW}>>> [2/3] LAUNCHING DORA SFT FOR: ${SPEC}...${NC}"
    DORA_STATUS="❌ Failed"
    DORA_PATH="N/A"
    GRPO_STATUS="Skipped"
    FINAL_PATH="N/A"

    if python -m saber.training.trainer --mode dora --specialist "$SPEC" --target_modules attn_only; then
        DORA_STATUS="✅ Complete"
        DORA_PATH="models/${SPEC}_v2"
        echo -e "${GREEN}>>> DORA SFT COMPLETE FOR ${SPEC}.${NC}"
        
        echo -e "\n${YELLOW}>>> [3/3] LAUNCHING GRPO RL (PROMETHEUS 2 REWARDS) FOR: ${SPEC}...${NC}"
        if python -m saber.training.trainer --mode grpo --specialist "$SPEC" --target_modules attn_only --kl_coef 0.04; then
            GRPO_STATUS="✅ Complete"
            FINAL_PATH="models/${SPEC}_grpo_final"
            echo -e "${GREEN}>>> GRPO RL COMPLETE FOR ${SPEC}.${NC}"
        else
            GRPO_STATUS="❌ Failed"
            echo -e "${RED}>>> GRPO RL FAILED FOR ${SPEC}.${NC}"
        fi
    else
        echo -e "${RED}>>> DORA SFT FAILED FOR ${SPEC}. SKIPPING GRPO.${NC}"
    fi

    # Append status row to log report
    echo "| **${SPEC}** | ${DORA_STATUS} | \`${DORA_PATH}\` | ${GRPO_STATUS} | \`${FINAL_PATH}\` |" >> "$LOG_FILE"
done

echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN}  ALL SPECIALISTS PROCESSED. REPORT SAVED TO logs/training_report.md${NC}"
echo -e "${GREEN}======================================================${NC}"

