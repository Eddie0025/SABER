#!/usr/bin/env bash
# =============================================================================
# SABER Benchmark Script for Newly Trained Specialists (GPU 0)
# =============================================================================
set -e

# Clickable file links: [run_trained_benchmark_gpu0.sh](file:///workspace/SABER/run_trained_benchmark_gpu0.sh)
echo "=========================================================="
echo "    SABER Evaluation Benchmark: Cyber, Arch, Meta (GPU 0)"
echo "=========================================================="
echo ""

echo "[+] Step 1: Starting target benchmarks on GPU 0 (Cyber, Architecture, Meta-Reasoner)..."
export SABER_KEEP_MODELS_LOADED=1
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_trained_benchmark.py

echo ""
echo "=========================================================="
echo "          GPU 0 TRAINED BENCHMARK WORK COMPLETED!"
echo "=========================================================="
