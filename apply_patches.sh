#!/usr/bin/env bash
set -e

echo "============================================================"
echo " SABER FOCUSED OPTIMIZATION PATCH RUNNER"
echo "============================================================"

PROGRESS_FILE=".patch_completed"
touch "$PROGRESS_FILE"

# Function to run training if not already completed
run_step() {
    local domain=$1
    local mode=$2
    local data=$3
    
    # Check if the domain is in the progress file, OR if the model was successfully compiled in the last 120 minutes
    local recently_trained=false
    if [ -f "models/${domain}_v2/adapter_model.safetensors" ]; then
        # Find if modified in the last 120 minutes (macOS/Linux compatible check)
        if [ "$(find models/${domain}_v2 -name "adapter_model.safetensors" -mmin -120 2>/dev/null)" ]; then
            recently_trained=true
        fi
    fi

    if grep -q "^$domain$" "$PROGRESS_FILE" || [ "$recently_trained" = true ]; then
        echo ">> Domain '$domain' recently trained or marked completed. Skipping."
        # Ensure it is recorded in the progress file for consistency
        if ! grep -q "^$domain$" "$PROGRESS_FILE"; then
            echo "$domain" >> "$PROGRESS_FILE"
        fi
    else
        echo ">> Training domain '$domain'..."
        python3 -m saber.training.trainer --domain "$domain" --data "$data" "$mode"
        echo "$domain" >> "$PROGRESS_FILE"
    fi
}

# 1. Generate all the targeted patches
echo "[1/3] Generating SFT and DPO patches..."
python3 scripts/generate_patches.py

# 2. Apply SFT Patches (Coverage gaps)
echo ""
echo "[2/3] Applying Continuous SFT Patches..."
python3 -m saber.training.trainer --domain orchestrator --data data/processed/orchestrator_patch.jsonl --patch-mode
python3 -m saber.training.trainer --domain science --data data/processed/science_patch.jsonl --patch-mode

echo ""
echo "============================================================"
echo " Optimization Patches Applied Successfully."
echo "============================================================"
