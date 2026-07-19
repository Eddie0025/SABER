#!/usr/bin/env bash
# =============================================================================
# RunPod Multi-GPU Automation Script for SABER (Dual H100 SXM Optimization)
# =============================================================================
set -e

echo "=========================================================="
# Clickable file links: [run_pod.sh](file:///workspace/SABER/run_pod.sh)
echo "          SABER Multi-GPU Orchestrator (v3.0)"
echo "=========================================================="
echo ""

# Step 0: Ensure dependencies are installed and up-to-date
echo "[+] Step 0: Installing requirements..."
pip install -q --upgrade \
    transformers \
    datasets \
    peft \
    trl \
    accelerate \
    sentencepiece \
    protobuf \
    "numpy<2.0.0"

# Step 1: Pre-generate SFT datasets (including the new 12K Meta-Reasoner dataset)
echo ""
echo "[+] Step 1: Generating datasets..."
PYTHONPATH=. python3 -m saber.training.dataset_loader

# Step 2: Set up background training loop for GPU 1
echo ""
echo "[+] Step 2: Starting background SFT training on GPU 1 (Cyber, Architecture, Meta-Reasoner)..."
mkdir -p logs

# Write background SFT runner script
cat << 'EOF' > run_training_gpu1.sh
#!/usr/bin/env bash
set -e
echo ">> [GPU 1] Training 'cyber' from scratch..."
PYTHONPATH=. python3 -m saber.training.trainer --domain cyber --gpu 1 > logs/train_cyber_gpu1.log 2>&1

echo ">> [GPU 1] Training 'architecture' from scratch..."
PYTHONPATH=. python3 -m saber.training.trainer --domain architecture --gpu 1 > logs/train_architecture_gpu1.log 2>&1

echo ">> [GPU 1] Training 'meta_reasoner' from scratch..."
PYTHONPATH=. python3 -m saber.training.trainer --domain meta_reasoner --gpu 1 > logs/train_meta_reasoner_gpu1.log 2>&1
echo ">> [GPU 1] Training complete!"
EOF
chmod +x run_training_gpu1.sh

# Fire SFT training in the background
./run_training_gpu1.sh &
TRAIN_PID=$!
echo "[+] Training thread started on GPU 1 (PID: $TRAIN_PID). Logs saved in logs/train_*.log"

# Step 3: Run target benchmarks on GPU 0 in the foreground
echo ""
echo "[+] Step 3: Executing active benchmarks on GPU 0 (Science, Coding, Finance)..."
export SABER_KEEP_MODELS_LOADED=1
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_final_benchmark.py

# Step 4: Wait for GPU 1 SFT jobs to complete
echo ""
echo "[+] Waiting for SFT training on GPU 1 to complete (PID: $TRAIN_PID)..."
wait $TRAIN_PID

echo "=========================================================="
echo "          SABER RUN COMPLETED SUCCESSFULLY!"
echo "=========================================================="
echo "All trained weights saved under: models/"
echo "Logs saved under: logs/"
echo "=========================================================="
