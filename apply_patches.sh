#!/usr/bin/env bash
set -e

echo "============================================================"
echo " SABER FOCUSED OPTIMIZATION PATCH RUNNER"
echo "============================================================"

# 1. Generate all the targeted patches
echo "[1/3] Generating SFT and DPO patches..."
python3 scripts/generate_patches.py

# 2. Apply SFT Patches (Coverage gaps)
echo ""
echo "[2/3] Applying Continuous SFT Patches..."
python3 -m saber.training.trainer --domain coding --data data/processed/coding_patch.jsonl --patch-mode
python3 -m saber.training.trainer --domain orchestrator --data data/processed/orchestrator_patch.jsonl --patch-mode
python3 -m saber.training.trainer --domain science --data data/processed/science_patch.jsonl --patch-mode
python3 -m saber.training.trainer --domain finance --data data/processed/finance_patch.jsonl --patch-mode

# 3. Apply DPO Patches (Hallucinations & Hedging Behavior)
echo ""
echo "[3/3] Applying DPO Patches..."
python3 -m saber.training.trainer --domain medical --data data/processed/medical_dpo_patch.jsonl --dpo-mode
python3 -m saber.training.trainer --domain meta_reasoner --data data/processed/meta_reasoner_dpo_patch.jsonl --dpo-mode

echo ""
echo "============================================================"
echo " Optimization Patches Applied Successfully."
echo "============================================================"
