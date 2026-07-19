#!/usr/bin/env bash
# =============================================================================
# SABER Training Script (GPU 1)
# =============================================================================
set -e

# Clickable file links: [run_training_gpu1.sh](file:///workspace/SABER/run_training_gpu1.sh)
echo "=========================================================="
echo "          SABER SFT Training Runner (GPU 1)"
echo "=========================================================="
echo ""

echo "[+] Step 1: Installing dependencies..."
pip install -q -r requirements.txt

if [ -n "$HF_TOKEN" ]; then
    echo "[+] Authenticating with Hugging Face Hub via python API..."
    python3 -c "from huggingface_hub import login; import os; login(token=os.getenv('HF_TOKEN'), add_to_git_credential=True)"
fi

echo ""
echo "[+] Step 2: Preparing SFT datasets (including 12K Meta-Reasoner data)..."
PYTHONPATH=. python3 -m saber.training.dataset_loader

echo ""
echo "[+] Step 3: Starting training loop from scratch..."
# CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python3 -m saber.training.trainer --domain cyber --gpu 1
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python3 -m saber.training.trainer --domain architecture --gpu 1
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python3 -m saber.training.trainer --domain meta_reasoner --gpu 1

echo ""
echo "=========================================================="
echo "          GPU 1 SFT TRAINING WORKSHOPS COMPLETE!"
echo "=========================================================="
