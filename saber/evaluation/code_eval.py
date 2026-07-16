# -*- coding: utf-8 -*-
"""SABER Evaluation — EvalPlus Wrapper"""

import os
import subprocess

def run_code_eval(model_path: str, dataset: str = "humaneval") -> None:
    """Wrapper to run EvalPlus for HumanEval+ or MBPP+ natively (no Docker)."""
    print(f"\n[eval:code] Running EvalPlus natively on {dataset}+...")
    
    # 1. Generate code solutions
    gen_cmd = [
        "python3", "-m", "evalplus.generate",
        "--model", model_path,
        "--dataset", dataset,
        "--backend", "hf",
        "--greedy"
    ]
    try:
        print("[eval:code] Generating samples...")
        subprocess.run(gen_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[!] Error generating code samples: {e}")
        return
        
    # 2. Evaluate natively
    eval_cmd = [
        "python3", "-m", "evalplus.evaluate",
        "--dataset", dataset,
        "--samples", f"{dataset}_samples.jsonl" # default output name from evalplus
    ]
    try:
        print("[eval:code] Evaluating samples...")
        # Evalplus by default runs safely, but user asked for native execution
        os.environ["EVALPLUS_LOCAL"] = "1" # force local execution if needed
        subprocess.run(eval_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[!] Error evaluating code samples: {e}")
