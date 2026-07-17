#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "          SABER Automation Entrypoint (v2.1)"
echo "=========================================================="

# ------------------------------------------------------------------
# Step 0: Install / upgrade all required dependencies
# ------------------------------------------------------------------
echo "[+] Step 0: Installing dependencies..."
pip install -q --upgrade \
    transformers \
    datasets \
    peft \
    accelerate \
    sentencepiece \
    protobuf \
    "numpy<2.0.0"
echo "[+] Dependencies installed."
echo ""

# ------------------------------------------------------------------
# Step 1: Prepare datasets (skipped if already generated)
# ------------------------------------------------------------------
echo "[+] Step 1: Downloading & preparing CoT datasets..."
PYTHONPATH=. python3 -m saber.training.dataset_loader

echo ""
echo "[+] Step 2: Creating log directory..."
mkdir -p logs

echo ""
echo "[+] Step 3: Launching sequential model training (Batch Size: 8)..."
# Sequential run optimized for 7B models on 80GB VRAM H100 GPU
for domain in medical meta_reasoner science finance coding architecture cyber orchestrator
do
    echo "----------------------------------------------------------"
    echo ">> Training domain: $domain"
    echo "----------------------------------------------------------"
    
    # Check if this domain is already trained and skipped
    if [ -f "models/${domain}_v2/adapter_config.json" ]; then
        echo "[+] Skipping $domain: trained model adapter weights already exist at models/${domain}_v2/"
        continue
    fi
    
    PYTHONPATH=. python3 -m saber.training.trainer \
        --domain "$domain" \
        --gpu 0 \
        2>&1 | tee "logs/train_${domain}.log"
done

echo ""
echo "=========================================================="
echo "          SABER PIPELINE SUCCESSFUL!"
echo "=========================================================="
echo "All trained weights saved under: models/"
echo "Logs saved under: logs/"
