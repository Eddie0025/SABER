#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SABER Master Evaluation Script

Runs all domain evaluations against a target model.
"""

import argparse
import os
from saber.evaluation.harness import run_lm_eval
from saber.evaluation.multi_judge import run_manual_evaluation
from saber.evaluation.code_eval import run_code_eval

def main():
    parser = argparse.ArgumentParser(description="Run SABER Evaluations")
    parser.add_argument("--model", type=str, required=True, help="Path or HF ID of the model to evaluate")
    parser.add_argument("--domain", type=str, choices=["medical", "cyber", "architecture", "science", "coding", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of eval examples (for dry run)")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    model = args.model
    domain = args.domain
    limit = args.limit

    print(f"============================================================")
    print(f" SABER Evaluation Framework")
    print(f" Target Model: {model}")
    print(f" Domain: {domain}")
    print(f"============================================================")

    if domain in ["medical", "all"]:
        print("\n--- Evaluating MEDICAL ---")
        run_lm_eval(model, "medmcqa,medqa_4options", limit)

    if domain in ["cyber", "all"]:
        print("\n--- Evaluating CYBERSECURITY ---")
        # CyberSecEval requires custom loading or via lm-eval if mapped.
        # Fallback to general security metrics if custom mapping isn't installed.
        run_lm_eval(model, "cyberseceval", limit) 

    if domain in ["science", "all"]:
        print("\n--- Evaluating SCIENCE ---")
        run_lm_eval(model, "math_algebra,arc_challenge", limit)

    if domain in ["coding", "all"]:
        print("\n--- Evaluating CODING ---")
        # Evalplus runs HumanEval natively
        # Note: we only do full evalplus if not limited, but for dry run we can just call the script
        if not limit:
            run_code_eval(model, "humaneval")
        else:
            print("[eval:code] Skipping EvalPlus for dry-run (limit flag set).")

    if domain in ["architecture", "all"]:
        print("\n--- Evaluating ARCHITECTURE ---")
        run_manual_evaluation(model)

    print("\n============================================================")
    print(" ALL EVALUATIONS COMPLETED.")
    print("============================================================")

if __name__ == "__main__":
    main()
