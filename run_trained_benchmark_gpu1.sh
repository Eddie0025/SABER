#!/usr/bin/env bash
# =============================================================================
# SABER Benchmark Script for Newly Trained Specialists (GPU 1)
# =============================================================================
set -e

# Clickable file links: [run_trained_benchmark_gpu1.sh](file:///workspace/SABER/run_trained_benchmark_gpu1.sh)
echo "=========================================================="
echo "    SABER Evaluation Benchmark: Cyber, Arch, Meta (GPU 1)"
echo "=========================================================="
echo ""

echo "[+] Step 1: Starting target benchmarks on GPU 1 (Cyber, Architecture, Meta-Reasoner)..."
export SABER_KEEP_MODELS_LOADED=1
CUDA_VISIBLE_DEVICES=1 python3 scripts/run_trained_benchmark.py

echo ""
echo "=========================================================="
echo "          GPU 1 TRAINED BENCHMARK WORK COMPLETED!"
echo "=========================================================="
