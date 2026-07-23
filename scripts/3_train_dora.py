# -*- coding: utf-8 -*-
"""scripts/3_train_dora.py

Step 3: Phase 1 High-Rank Weight-Decomposed LoRA (DoRA SFT) Trainer.
Runs DoRA (r=64, alpha=128, use_dora=True, all linear modules) for specialist models.
Usage:
    python3 scripts/3_train_dora.py --domain science --epochs 3
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath('.'))
from saber.training.trainer import TrainerConfig, run_training

def main():
    parser = argparse.ArgumentParser(description="Phase 1: DoRA SFT Trainer")
    parser.add_argument("--domain", type=str, default="science", help="Domain to train (science, cyber, finance, medical, coding, architecture, meta_reasoner, orchestrator)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of SFT training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    print("=========================================================================")
    print(f"           STEP 3: PHASE 1 DoRA SFT TRAINING [{args.domain.upper()}]           ")
    print("=========================================================================")

    cfg = TrainerConfig(
        domain=args.domain,
        data_path=f"data/processed/{args.domain}.jsonl",
        output_dir=f"models/{args.domain}_v2",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=64,
        lora_alpha=128,
        use_dora=True,
    )

    saved_path = run_training(cfg)
    print(f"\n[Step 3 Complete] DoRA model checkpoint saved to: {saved_path}\n")

if __name__ == "__main__":
    main()
