#!/usr/bin/env bash
# =============================================================================
# SABER Benchmark Script (GPU 0)
# =============================================================================
set -e

# Clickable file links: [run_benchmark_gpu0.sh](file:///workspace/SABER/run_benchmark_gpu0.sh)
echo "=========================================================="
echo "          SABER Evaluation Benchmark Runner (GPU 0)"
echo "=========================================================="
echo ""

echo "[+] Step 0: Installing dependencies..."
pip install -q -r requirements.txt

if [ -n "$HF_TOKEN" ]; then
    echo "[+] Authenticating with Hugging Face Hub via python API..."
    python3 -c "from huggingface_hub import login; import os; login(token=os.getenv('HF_TOKEN'), add_to_git_credential=True)"
fi

echo "[+] Step 1: Starting target benchmarks on GPU 0 (Science, Coding, Finance)..."
export SABER_KEEP_MODELS_LOADED=1
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_final_benchmark.py

echo ""
echo "=========================================================="
echo "          GPU 0 BENCHMARK WORK COMPLETED!"
echo "=========================================================="
