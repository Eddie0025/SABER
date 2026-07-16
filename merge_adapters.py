#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SABER Utility: Merge LoRA Adapters

Merges a trained LoRA adapter into the base model weights.
This MUST be run before executing the evaluation suite to ensure
lm-eval and evalplus test the full model capabilities.

Usage:
    python merge_adapters.py --adapter models/medical_v2 --base Qwen/Qwen2.5-7B-Instruct --output models/medical_v2_merged
"""

import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters with base model")
    parser.add_argument("--adapter", type=str, required=True, help="Path to the LoRA adapter directory (e.g., models/medical_v2)")
    parser.add_argument("--base", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base model HuggingFace ID or path")
    parser.add_argument("--output", type=str, required=True, help="Output directory for merged model")
    args = parser.parse_args()

    print(f"============================================================")
    print(f" LoRA MERGE SCRIPT")
    print(f" Base Model:      {args.base}")
    print(f" Adapter Path:    {args.adapter}")
    print(f" Output Path:     {args.output}")
    print(f"============================================================")

    if not os.path.exists(args.adapter):
        print(f"[!] Error: Adapter path '{args.adapter}' does not exist.")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("\n[1/3] Loading base model...")
    # Load base model in bfloat16 for optimal memory on Ada/Ampere, else float16
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map="auto"
    )

    print("\n[2/3] Loading LoRA adapters & merging...")
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()

    print(f"\n[3/3] Saving merged weights to {args.output}...")
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    
    # Save tokenizer as well so the directory is fully stand-alone
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    tokenizer.save_pretrained(args.output)
    
    print("\n[+] Merge successful! You can now point run_evaluations.py to this output directory.")

if __name__ == "__main__":
    main()
