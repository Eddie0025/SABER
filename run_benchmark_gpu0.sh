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

echo "[+] Step 1: Starting target benchmarks on GPU 0 (Science, Coding, Finance)..."
export SABER_KEEP_MODELS_LOADED=1
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_final_benchmark.py

echo ""
echo "=========================================================="
echo "          GPU 0 BENCHMARK WORK COMPLETED!"
echo "=========================================================="
