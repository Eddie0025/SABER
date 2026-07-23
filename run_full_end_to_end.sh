#!/usr/bin/env bash
# =============================================================================
# SABER Full End-to-End Automated Pipeline (Phase 1 SFT + Phase 2 GRPO + Eval)
#
# Sequential execution on single GPU (H100 / A100):
#   1. Phase 1: High-Rank Weight-Decomposed LoRA (DoRA SFT) on all 7 domains
#   2. Verifies all 7 DoRA model adapter checkpoints
#   3. Phase 2: Verifiable-Fact-Augmented GRPO Reinforcement Learning
#   4. Phase 3: Unified 5-Mode Benchmark Evaluation
#
# Usage:
#   nohup bash run_full_end_to_end.sh > full_pipeline.log 2>&1 &
# =============================================================================

set -e

SECONDS=0
mkdir -p logs models data/offline_kb data/processed

DOMAINS=("science" "cyber" "finance" "coding" "architecture" "orchestrator" "meta_reasoner")

echo "========================================================================="
echo "   SABER FULL END-TO-END AUTOMATED PIPELINE (SFT → GRPO → EVAL)"
echo "   Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================="
echo ""

# ------------------------------------------------------------------
# Step 1 & 2: Dataset & Knowledge Base Verification
# ------------------------------------------------------------------
echo "[+] [Pipeline 1/4] Verifying Datasets and Knowledge Bases..."
if [ ! -f "data/processed/dataset_manifest.json" ]; then
    echo "[!] Datasets missing. Building datasets..."
    PYTHONPATH=. python3 scripts/1_build_datasets.py
fi

if [ ! -f "data/offline_kb/science_kb.db" ]; then
    echo "[!] Knowledge bases missing. Building SQLite KBs..."
    PYTHONPATH=. python3 scripts/2_build_kb.py
fi
echo "[✓] Datasets and Knowledge Bases verified."
echo ""

# ------------------------------------------------------------------
# Step 3: Phase 1 High-Rank DoRA SFT Training (All 7 Domains)
# ------------------------------------------------------------------
echo "========================================================================="
echo "   [Pipeline 2/4] PHASE 1: DORA SFT TRAINING (7 DOMAINS)"
echo "========================================================================="

for domain in "${DOMAINS[@]}"
do
    echo "------------------------------------------------------------------------- shadow"
    echo ">> [Phase 1 SFT] Training domain: $domain"
    echo "   Start time: $(date '+%H:%M:%S')"
    echo "-------------------------------------------------------------------------"

    PYTHONPATH=. python3 scripts/3_train_dora.py --domain "$domain" --epochs 3 \
        2>&1 | tee "logs/sft_${domain}.log"

    echo "[✓] Phase 1 SFT for $domain complete."
    echo ""
done

# ------------------------------------------------------------------
# Check Phase 1 Completion
# ------------------------------------------------------------------
echo "[+] Validating Phase 1 Checkpoints..."
MISSING_MODELS=0
for domain in "${DOMAINS[@]}"
do
    if [ ! -f "models/${domain}_v2/adapter_config.json" ]; then
        echo "[!] Error: Model checkpoint models/${domain}_v2/adapter_config.json is missing!"
        MISSING_MODELS=$((MISSING_MODELS + 1))
    fi
done

if [ $MISSING_MODELS -gt 0 ]; then
    echo "[!] ERROR: $MISSING_MODELS model checkpoints failed to train. Aborting GRPO phase."
    exit 1
fi
echo "[✓] All 7 DoRA model checkpoints verified successfully!"
echo ""

# ------------------------------------------------------------------
# Step 4: Phase 2 Verifiable-Fact GRPO Reinforcement Learning
# ------------------------------------------------------------------
echo "========================================================================="
echo "   [Pipeline 3/4] PHASE 2: VERIFIABLE-FACT GRPO REINFORCEMENT LEARNING"
echo "========================================================================="

for domain in "${DOMAINS[@]}"
do
    echo "-------------------------------------------------------------------------"
    echo ">> [Phase 2 GRPO] Running GRPO RL rollout for domain: $domain"
    echo "-------------------------------------------------------------------------"

    PYTHONPATH=. python3 scripts/4_train_grpo.py --domain "$domain" --generations 4 \
        2>&1 | tee "logs/grpo_${domain}.log"

    echo "[✓] Phase 2 GRPO for $domain complete."
    echo ""
done

# ------------------------------------------------------------------
# Step 5: Phase 3 Unified 5-Mode Benchmark Evaluation
# ------------------------------------------------------------------
echo "========================================================================="
echo "   [Pipeline 4/4] PHASE 3: UNIFIED BENCHMARK EVALUATION"
echo "========================================================================="

PYTHONPATH=. python3 scripts/5_run_benchmark.py 2>&1 | tee logs/benchmark_eval.log

# ------------------------------------------------------------------
# Summary Report
# ------------------------------------------------------------------
TOTAL_MINS=$(( SECONDS / 60 ))
TOTAL_HRS=$(( TOTAL_MINS / 60 ))
REMAINING_MINS=$(( TOTAL_MINS % 60 ))

echo ""
echo "========================================================================="
echo "   FULL SABER PIPELINE COMPLETED SUCCESSFULLY!"
echo "   Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "   Total Execution Time: ${TOTAL_HRS}h ${REMAINING_MINS}m"
echo "========================================================================="
echo "   Trained Adapters Output Directories:"
for domain in "${DOMAINS[@]}"
do
    echo "     - models/${domain}_v2/"
done
echo "========================================================================="
