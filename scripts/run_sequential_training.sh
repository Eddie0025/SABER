#!/usr/bin/env bash

# SABER Sequential Training Pipeline (DoRA SFT -> GRPO RL)
# Orchestrates end-to-end training for all specialists.
# Smartly resumes from where it left off without deleting successful checkpoints!

set -e
set -u
set -o pipefail

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
echo -e "${BLUE}  SABER MASTER TRAINING PIPELINE (RESUMABLE)${NC}"
echo -e "${BLUE}======================================================${NC}"

for SPEC in "${SPECIALISTS[@]}"; do
    [[ -n "$SPEC" ]] || continue

    # 1. Check if the entire process is completely finished for this specialist
    if [ -d "models/${SPEC}_grpo_final" ]; then
        echo -e "\n${GREEN}>>> FINAL GRPO MODEL ALREADY EXISTS FOR ${SPEC}. SKIPPING ENTIRELY.${NC}"
        echo "| **${SPEC}** | ✅ Skipped (Already Done) | \`models/${SPEC}_v2\` | ✅ Skipped (Already Done) | \`models/${SPEC}_grpo_final\` |" >> "$LOG_FILE"
        continue
    fi

    # 2. Handle DoRA SFT Phase
    if [ -d "models/${SPEC}_v2" ]; then
        echo -e "\n${GREEN}>>> DORA SFT ALREADY COMPLETE FOR ${SPEC}. SKIPPING DORA PHASE.${NC}"
        DORA_STATUS="✅ Skipped (Already Done)"
        DORA_PATH="models/${SPEC}_v2"
    else
        echo -e "\n${YELLOW}>>> [1/3] PURGING OLD ARTIFACTS FOR: ${SPEC}...${NC}"
        rm -rf "models/${SPEC}_checkpoints" "models/${SPEC}_v2" "models/${SPEC}_grpo" "models/${SPEC}_grpo_final"
        
        echo -e "\n${YELLOW}>>> [2/3] LAUNCHING DORA SFT FOR: ${SPEC}...${NC}"
        if python -m saber.training.trainer --mode dora --specialist "$SPEC" --target_modules attn_only; then
            DORA_STATUS="✅ Complete"
            DORA_PATH="models/${SPEC}_v2"
            echo -e "${GREEN}>>> DORA SFT COMPLETE FOR ${SPEC}.${NC}"
        else
            echo -e "${RED}>>> DORA SFT FAILED FOR ${SPEC}. SKIPPING GRPO.${NC}"
            echo "| **${SPEC}** | ❌ Failed | \`N/A\` | Skipped | \`N/A\` |" >> "$LOG_FILE"
            continue
        fi
    fi
    
    # 3. Handle GRPO RL Phase
    echo -e "\n${YELLOW}>>> [3/3] LAUNCHING GRPO RL (STANDARD OUTCOME REWARDS) FOR: ${SPEC}...${NC}"
    # Delete any stale GRPO intermediate checkpoints just in case
    rm -rf "models/${SPEC}_grpo"
    
    if python -m saber.training.trainer --mode grpo --specialist "$SPEC" --target_modules attn_only --kl_coef 0.04; then
        GRPO_STATUS="✅ Complete"
        FINAL_PATH="models/${SPEC}_grpo_final"
        echo -e "${GREEN}>>> GRPO RL COMPLETE FOR ${SPEC}.${NC}"
    else
        GRPO_STATUS="❌ Failed"
        FINAL_PATH="N/A"
        echo -e "${RED}>>> GRPO RL FAILED FOR ${SPEC}.${NC}"
    fi

    # Append status row to log report
    echo "| **${SPEC}** | ${DORA_STATUS} | \`${DORA_PATH}\` | ${GRPO_STATUS} | \`${FINAL_PATH}\` |" >> "$LOG_FILE"
done

echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN}  ALL SPECIALISTS PROCESSED. REPORT SAVED TO logs/training_report.md${NC}"
echo -e "${GREEN}======================================================${NC}"
