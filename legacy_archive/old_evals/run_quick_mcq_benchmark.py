import json
import os
import sys
import time
import random
import urllib.request
import re
from typing import Dict, Any, List

# Ensure saber module can be imported
sys.path.append(os.path.abspath('.'))

try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets: pip install datasets")
    sys.exit(1)

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["SABER_KEEP_MODELS_LOADED"] = "1"
os.environ["SABER_BENCHMARK_MODE"] = "1"

from saber.config import SaberConfig, VerificationTier
from saber.registry import SpecialistRegistry
from saber.audit import AuditLogger
from saber.orchestrator import Orchestrator

def parse_mcq_answer(text):
    if not text: return None
    match = re.search(r"ANSWER:\s*([A-D])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    matches = re.findall(r"(?:^|\s|\*|_)([A-D])(?:$|\s|\.|\*|_)", text[-100:], re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    return None

def main():
    print("==========================================================")
    print("      SABER Quick MCQ Benchmark (MMLU + SecBench)")
    print("==========================================================\n")

    config = SaberConfig()
    registry = SpecialistRegistry()
    registry.auto_discover()
    
    for domain, specialist in registry.all().items():
        model_path = f"models/{domain}_v2"
        if os.path.exists(model_path):
            specialist.load_model(model_path)
            print(f"[*] Loaded adapter for '{domain}': {model_path}")
        else:
            specialist.load_model("Qwen/Qwen2.5-7B")
            print(f"[*] Adapter '{domain}' not found; falling back to Qwen/Qwen2.5-7B")
            
    audit = AuditLogger()
    orch = Orchestrator(config=config, registry=registry, audit=audit)
    
    bench_cases = []
    
    # 1. Science: MMLU college_physics
    try:
        ds = load_dataset("cais/mmlu", "college_physics", split="test")
        cases = list(ds)
        random.seed(42)
        random.shuffle(cases)
        for row in cases[:15]:
            q = f"Question: {row['question']}\nOptions:\nA: {row['choices'][0]}\nB: {row['choices'][1]}\nC: {row['choices'][2]}\nD: {row['choices'][3]}"
            bench_cases.append({
                "question": q,
                "expected": ["A", "B", "C", "D"][row['answer']],
                "domain": "science",
                "dataset": "mmlu_physics"
            })
        print("[+] Loaded 15 MMLU Physics cases.")
    except Exception as e:
        print(f"[!] Failed to load MMLU physics: {e}")

    # 2. Coding: MMLU college_computer_science
    try:
        ds = load_dataset("cais/mmlu", "college_computer_science", split="test")
        cases = list(ds)
        random.seed(42)
        random.shuffle(cases)
        for row in cases[:15]:
            q = f"Question: {row['question']}\nOptions:\nA: {row['choices'][0]}\nB: {row['choices'][1]}\nC: {row['choices'][2]}\nD: {row['choices'][3]}"
            bench_cases.append({
                "question": q,
                "expected": ["A", "B", "C", "D"][row['answer']],
                "domain": "coding",
                "dataset": "mmlu_cs"
            })
        print("[+] Loaded 15 MMLU CS cases.")
    except Exception as e:
        print(f"[!] Failed to load MMLU CS: {e}")
        
    # 3. Finance: MMLU professional_accounting
    try:
        ds = load_dataset("cais/mmlu", "professional_accounting", split="test")
        cases = list(ds)
        random.seed(42)
        random.shuffle(cases)
        for row in cases[:15]:
            q = f"Question: {row['question']}\nOptions:\nA: {row['choices'][0]}\nB: {row['choices'][1]}\nC: {row['choices'][2]}\nD: {row['choices'][3]}"
            bench_cases.append({
                "question": q,
                "expected": ["A", "B", "C", "D"][row['answer']],
                "domain": "finance",
                "dataset": "mmlu_accounting"
            })
        print("[+] Loaded 15 MMLU Accounting cases.")
    except Exception as e:
        print(f"[!] Failed to load MMLU Accounting: {e}")

    # 4. Cyber: SecBench
    try:
        url = "https://raw.githubusercontent.com/secbench-git/SecBench/main/data/MCQs_2730.jsonl"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            lines = response.read().decode("utf-8").splitlines()
            valid_cases = []
            for line in lines:
                if not line.strip(): continue
                row = json.loads(line)
                if row.get("language") == "English" and len(row.get("answers", [])) == 4:
                    q_text = row.get("question")
                    choices = row["answers"]
                    label_char = row.get("label", "").upper().strip()
                    correct_idx = ord(label_char) - 65
                    if 0 <= correct_idx < 4:
                        correct_ans = choices[correct_idx]
                        shuffled_choices = list(choices)
                        random.seed(len(valid_cases))
                        random.shuffle(shuffled_choices)
                        correct_char = chr(65 + shuffled_choices.index(correct_ans))
                        
                        q = f"Question: {q_text}\nOptions:\nA: {shuffled_choices[0]}\nB: {shuffled_choices[1]}\nC: {shuffled_choices[2]}\nD: {shuffled_choices[3]}"
                        valid_cases.append({
                            "question": q,
                            "expected": correct_char,
                            "domain": "cyber",
                            "dataset": "secbench"
                        })
            random.seed(42)
            random.shuffle(valid_cases)
            for case in valid_cases[:15]:
                bench_cases.append(case)
            print("[+] Loaded 15 SecBench cases.")
    except Exception as e:
        print(f"[!] SecBench load failed: {e}")

    print(f"\n[+] Total benchmark cases compiled: {len(bench_cases)}")
    results = []
    
    prompt_no_cot = "\n\nAnswer the following multiple choice question. DO NOT think step by step. Simply output the final answer. The last line of your response MUST strictly follow the format: ANSWER: LETTER (where LETTER is A, B, C, or D)."
    prompt_cot = "\n\nAnswer the following multiple choice question. Think step by step before answering. The last line of your response MUST strictly follow the format: ANSWER: LETTER (where LETTER is A, B, C, or D)."

    modes = [
        ("Adapter (No CoT)", VerificationTier.TIER_0, prompt_no_cot),
        ("Adapter + CoT", VerificationTier.TIER_0, prompt_cot),
        ("Adapter + CoT + Sentinel 2 Check", VerificationTier.TIER_1, prompt_cot)
    ]
    
    live_summary = {}

    for idx, case in enumerate(bench_cases, 1):
        ds_name = case["dataset"]
        print(f"\n[{idx}/{len(bench_cases)}] Dataset: {ds_name} | Query: {case['question'][:60].strip().replace(chr(10), ' ')}...")
        
        case_res = {"question": case["question"], "expected": case["expected"], "dataset": ds_name, "runs": {}}
        
        for mode_name, tier, prompt_suffix in modes:
            q = case["question"] + prompt_suffix
            start = time.time()
            try:
                original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                try:
                    res = orch.process_query(q, tier=tier)
                    ans = res.get("answer", "").strip()
                finally:
                    sys.stdout.close()
                    sys.stdout = original_stdout
            except Exception as e:
                ans = f"[ERROR]: {e}"
                
            latency = time.time() - start
            extracted = parse_mcq_answer(ans)
            is_correct = 1.0 if extracted == case["expected"] else 0.0
            
            case_res["runs"][mode_name] = {
                "answer": ans,
                "extracted": extracted,
                "latency": round(latency, 2),
                "evaluation": {"accuracy": is_correct}
            }
            
            if ds_name not in live_summary: live_summary[ds_name] = {}
            if mode_name not in live_summary[ds_name]: live_summary[ds_name][mode_name] = {"acc_sum": 0.0, "acc_cnt": 0}
            
            live_summary[ds_name][mode_name]["acc_sum"] += is_correct
            live_summary[ds_name][mode_name]["acc_cnt"] += 1
            
        results.append(case_res)
        
        if idx % 5 == 0 or idx == len(bench_cases):
            print(f"\n[LIVE SCOREBOARD] Progress: {idx}/{len(bench_cases)}")
            print("| Dataset | Adapter (No CoT) | Adapter + CoT | Adapter + CoT + Sentinel 3 Check |")
            print("| :--- | :--- | :--- | :--- |")
            for ds, m_data in live_summary.items():
                cells = [ds]
                for m_name, _, _ in modes:
                    st = m_data.get(m_name, {"acc_sum":0, "acc_cnt":0})
                    pct = (st["acc_sum"] / st["acc_cnt"] * 100.0) if st["acc_cnt"] > 0 else 0.0
                    cells.append(f"{pct:.1f}%")
                print("| " + " | ".join(cells) + " |")

    with open("quick_mcq_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
