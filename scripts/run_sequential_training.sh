#!/bin/bash

# SABER Master Sequential Training Pipeline
# Executes DoRA training across all defined Domain Specialists sequentially.

# Array of specialists to train
SPECIALISTS=(
    "python"
    "javascript"
    "sql"
    "cybersecurity"
    "science"
    "finance"
    "medical"
    "architecture_qa"
    "architecture_planner"
)

# Colors for pretty terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}    SABER Master Sequential Training Pipeline       ${NC}"
echo -e "${BLUE}====================================================${NC}"
echo -e "${YELLOW}Total Specialists Scheduled: ${#SPECIALISTS[@]}${NC}\n"

for i in "${!SPECIALISTS[@]}"; do
    SPEC="${SPECIALISTS[$i]}"
    STEP=$((i + 1))
    TOTAL=${#SPECIALISTS[@]}
    
    echo -e "${GREEN}>>> [Step ${STEP}/${TOTAL}] Starting Training for Specialist: ${SPEC^^}...${NC}"
    
    # Execute the python training module
    python -m saber.training.trainer \
        --mode dora \
        --specialist "$SPEC" \
        --target_modules all
        
    # Check if the python script executed successfully
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}>>> [Step ${STEP}/${TOTAL}] Successfully completed training for ${SPEC^^}.${NC}\n"
    else
        echo -e "${RED}>>> [Step ${STEP}/${TOTAL}] Training for ${SPEC^^} encountered an error or was skipped due to mock data.${NC}\n"
    fi
done

echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}    Master Sequential Training Pipeline Complete!   ${NC}"
echo -e "${BLUE}====================================================${NC}"
