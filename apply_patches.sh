#!/usr/bin/env bash
set -e

echo "============================================================"
echo " SABER FOCUSED OPTIMIZATION PATCH RUNNER"
echo "============================================================"

PROGRESS_FILE=".patch_completed"
touch "$PROGRESS_FILE"

# Reinstall stable trl version to resolve FSDPModule import error
echo "[+] Restoring stable trl version..."
pip install -q trl==0.8.6

# Function to run training unconditionally
run_step() {
    local domain=$1
    local mode=$2
    local data=$3
    
    echo ">> Training domain '$domain'..."
    python3 -m saber.training.trainer --domain "$domain" --data "$data" "$mode"
}

# 1. Generate all the targeted patches
echo "[1/3] Generating SFT and DPO patches..."
python3 scripts/generate_patches.py

# 2. Apply SFT Patches (Coverage gaps)
echo ""
echo "[2/3] Applying Continuous SFT Patches..."
run_step "medical" "--patch-mode" "data/processed/medical_patch.jsonl"
run_step "orchestrator" "--patch-mode" "data/processed/orchestrator_patch.jsonl"
run_step "science" "--patch-mode" "data/processed/science_patch.jsonl"
run_step "coding" "--patch-mode" "data/processed/coding_patch.jsonl"
run_step "cyber" "--patch-mode" "data/processed/cybersecurity_patch.jsonl"
run_step "architecture" "--patch-mode" "data/processed/architecture_patch.jsonl"
run_step "finance" "--patch-mode" "data/processed/finance_patch.jsonl"
run_step "meta_reasoner" "--patch-mode" "data/processed/meta_reasoner_patch.jsonl"

echo ""
echo "============================================================"
echo " Optimization Patches Applied Successfully."
echo "============================================================"
