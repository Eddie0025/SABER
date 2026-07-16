# -*- coding: utf-8 -*-
"""SABER Evaluation — EleutherAI LM-Eval Wrapper"""

import os
import subprocess

def run_lm_eval(model_path: str, tasks: str, limit: int = None) -> None:
    """Wrapper to run lm_eval from EleutherAI."""
    print(f"\n[eval:harness] Running lm-eval on tasks: {tasks}")
    cmd = [
        "python3", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True",
        "--tasks", tasks,
        "--batch_size", "auto",
        "--output_path", f"logs/eval_{tasks.replace(',', '_')}.json"
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])
        
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[!] Error running lm-eval: {e}")
