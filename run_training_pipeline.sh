#!/usr/bin/env bash
# =============================================================================
# SABER Training Pipeline v3.0 — 1× H100 SXM (Sequential)
#
# Trains all 8 specialist models sequentially on a single GPU.
# Models are ordered largest-to-smallest so the heaviest training
# runs first while the GPU is at peak thermal efficiency.
#
# Order:  Medical → Meta-Reasoner → Science → Finance → Coding
#         → Architecture → Cyber → Orchestrator
#
# Hardware:  1× H100 SXM (80 GB VRAM)  —  JarvisLabs EU1
# Strategy:  SFTTrainer + Packing + ChatML CoT + Early Stopping
# Budget:    ~₹1,200 (4-5 hours @ ₹286/hr)
#
# Usage:
#     cd /home/SABER        # JarvisLabs workspace
#     bash run_training_pipeline.sh
# =============================================================================

set -e

SECONDS=0   # Built-in bash timer

echo "============================================================"
echo "  SABER Training Pipeline v3.0"
echo "  Hardware: 1× H100 SXM (80 GB VRAM)"
echo "  Mode:     Sequential — all 8 models on GPU 0"
echo "  Strategy: SFTTrainer + Packing + ChatML CoT"
echo "============================================================"
echo ""

# ------------------------------------------------------------------
# Step 0: Install / upgrade dependencies
# ------------------------------------------------------------------
echo "[pipeline] Step 0: Installing dependencies..."
pip install -q --upgrade \
    transformers \
    datasets \
    peft \
    trl \
    accelerate \
    sentencepiece \
    protobuf
echo "[pipeline] Dependencies installed."
echo ""

# ------------------------------------------------------------------
# Step 1: Prepare datasets
# ------------------------------------------------------------------
# If you pre-uploaded data/processed/*.jsonl, this step will detect
# existing files and skip re-downloading (saves ~15-30 min).
# ------------------------------------------------------------------
echo "[pipeline] Step 1: Downloading and preparing datasets..."
python -m saber.training.dataset_loader
echo "[pipeline] Datasets ready."
echo ""

# ------------------------------------------------------------------
# Step 2: Create log directory
# ------------------------------------------------------------------
mkdir -p logs

# ------------------------------------------------------------------
# Helper: train one model and report time
# ------------------------------------------------------------------
train_model() {
    local domain=$1
    local step_num=$2
    local start=$SECONDS

    echo "------------------------------------------------------------"
    echo "[pipeline] Step $step_num: Training $domain on GPU 0..."
    echo "[pipeline]   Start time: $(date '+%H:%M:%S')"
    echo "------------------------------------------------------------"

    python -m saber.training.trainer \
        --domain "$domain" \
        --gpu 0 \
        --batch-size 4 \
        2>&1 | tee "logs/train_${domain}.log"

    local elapsed=$(( SECONDS - start ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))
    echo ""
    echo "[pipeline]   ✓ $domain COMPLETE in ${mins}m ${secs}s"
    echo ""
}

# ------------------------------------------------------------------
# Step 3-10: Train all 8 models sequentially (largest first)
# ------------------------------------------------------------------
# Ordered by total training steps (largest → smallest):
#   Medical:       5,193 steps  (~53K records, 3 epochs)
#   Meta-Reasoner: 3,168 steps  (~4.8K records @ 1414 tok avg, 4 epochs)
#   Science:       1,965 steps  (~23K records, 3 epochs)
#   Finance:       1,698 steps  (~14K records, 3 epochs)
#   Coding:        1,668 steps  (~21K records, 3 epochs)
#   Architecture:  1,284 steps  (~9.5K records, 3 epochs)
#   Cyber:           741 steps  (~12K records, 3 epochs)
#   Orchestrator:    395 steps  (~5K records, 5 epochs)
# ------------------------------------------------------------------

train_model "medical"       3
train_model "meta_reasoner" 4
train_model "science"       5
train_model "finance"       6
train_model "coding"        7
train_model "architecture"  8
train_model "cyber"         9
train_model "orchestrator"  10

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
TOTAL_MINS=$(( SECONDS / 60 ))
TOTAL_HRS=$(( TOTAL_MINS / 60 ))
REMAINING_MINS=$(( TOTAL_MINS % 60 ))

echo ""
echo "============================================================"
echo "  ALL 8 MODELS TRAINED SUCCESSFULLY!"
echo "============================================================"
echo "  Medical:       models/medical_v2/"
echo "  Meta-Reasoner: models/meta_reasoner_v2/"
echo "  Science:       models/science_v2/"
echo "  Finance:       models/finance_v2/"
echo "  Coding:        models/coding_v2/"
echo "  Architecture:  models/architecture_v2/"
echo "  Cyber:         models/cyber_v2/"
echo "  Orchestrator:  models/orchestrator_v2/"
echo "============================================================"
echo ""
echo "  Total training time: ${TOTAL_HRS}h ${REMAINING_MINS}m"
echo ""
echo "  Training logs saved to:"
echo "    logs/train_medical.log"
echo "    logs/train_meta_reasoner.log"
echo "    logs/train_science.log"
echo "    logs/train_finance.log"
echo "    logs/train_coding.log"
echo "    logs/train_architecture.log"
echo "    logs/train_cyber.log"
echo "    logs/train_orchestrator.log"
echo ""
echo "============================================================"
echo ""

# ------------------------------------------------------------------
# Autocut — Stop instance to save credits
# ------------------------------------------------------------------
if [ -n "$JARVIS_VM_ID" ]; then
    echo "[pipeline] Autocut — Detected JarvisLabs.ai VM ($JARVIS_VM_ID)."
    echo "[pipeline] Attempting to pause instance..."
    if command -v jl &> /dev/null; then
        jl pause $JARVIS_VM_ID
    else
        python3 -c "
try:
    from jarvislabs import Client
    Client().get_instance('$JARVIS_VM_ID').pause()
except Exception as e:
    print('Failed to pause via Python SDK:', e)
    print('Please ensure JL_API_KEY is exported and jarvislabs package is installed.')
"
    fi
elif [ -n "$RUNPOD_POD_ID" ]; then
    echo "[pipeline] Autocut — Detected RunPod instance ($RUNPOD_POD_ID)."
    runpodctl stop pod $RUNPOD_POD_ID
else
    echo "[!] Autocut skipped. Neither RUNPOD_POD_ID nor JARVIS_VM_ID found."
    echo "    Please stop your instance manually via the dashboard to avoid extra charges."
fi
