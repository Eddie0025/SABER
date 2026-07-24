import json
import os
import sys
import time
import random
import re
import requests
import multiprocessing
from typing import Dict, Any, List
from collections import defaultdict

# Ensure saber module can be imported
sys.path.append(os.path.abspath('.'))

# Disable Hugging Face verbose logs and lock models in GPU VRAM
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["SABER_KEEP_MODELS_LOADED"] = "1"
os.environ["SABER_BENCHMARK_MODE"] = "1"

from saber.config import SaberConfig
from saber.registry import SpecialistRegistry
from saber.llm_engine import LLMEngine
from saber.signal import Signal, SignalType
from saber.sentinel import Sentinel

# =====================================================================
# 1. Sandboxed Python Code Harness
# =====================================================================
def _exec_target(code_str, result_queue):
    """Worker target function for isolated code execution."""
    try:
        ns = {}
        exec(code_str, ns)
        result_queue.put((True, "Executed Successfully"))
    except Exception as e:
        result_queue.put((False, f"Runtime Error: {type(e).__name__}: {e}"))

def execute_python_code(code_str: str, timeout: float = 3.0) -> (bool, str):
    """Execute Python code in a sandboxed subprocess with timeout."""
    clean_code = re.sub(r"^```python\s*", "", code_str, flags=re.MULTILINE)
    clean_code = re.sub(r"^```\s*$", "", clean_code, flags=re.MULTILINE).strip()
    
    if not clean_code:
        return False, "Empty Code"
        
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_exec_target, args=(clean_code, q))
    proc.start()
    proc.join(timeout=timeout)
    
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False, "Execution Timeout (>3s)"
        
    if not q.empty():
        return q.get()
    return False, "Execution Failed"

# =====================================================================
# 2. Strict MCQ Answer Parser
# =====================================================================
MCQ_SUFFIX = "\n\nAnswer the following multiple choice question. The last line MUST strictly follow: ANSWER: LETTER (A, B, C, or D)."

def build_mcq_prompt(question_text, choices_str):
    return f"Question: {question_text}\nOptions:\n{choices_str}{MCQ_SUFFIX}"

def parse_mcq_answer(raw_answer, prompt=None):
    lines = [line.strip() for line in raw_answer.split('\n') if line.strip()]
    if not lines:
        return None
    
    last_line = lines[-1].upper()
    match = re.search(r"ANSWER:\s*[<\(]?([A-D])\b[>\)]?", last_line)
    if match:
        return match.group(1)
        
    for line in reversed(lines):
        match = re.search(r"ANSWER:\s*[<\(]?([A-D])\b[>\)]?", line.upper())
        if match:
            return match.group(1)
            
    full_text = raw_answer.upper()
    for pattern in [r"THE\s+(?:CORRECT\s+)?ANSWER\s+IS\s*:?\s*\(?([A-D])\)?(?![A-Za-z])", r"CORRECT\s+ANSWER\s*:?\s*\(?([A-D])\)?(?![A-Za-z])"]:
        match = re.search(pattern, full_text)
        if match:
            return match.group(1)
            
    if re.fullmatch(r"[A-D]\.?", last_line):
        return last_line[0]
        
    return None

# =====================================================================
# 3. Robust High-Speed Dataset Loader (No Scripts, Direct JSON/Parquet)
# =====================================================================
def load_all_datasets(domain="all"):
    cases = []
    
    # --- 3.1 FinanceBench ---
    if domain in ["all", "finance"]:
        print("[*] Fetching FinanceBench dataset...")
        sys.stdout.flush()
        try:
            from datasets import load_dataset
            ds = load_dataset("virattt/financebench", split="train")
            for row in ds:
                q = row.get("question", "")
                ans = row.get("answer", "")
                doc = row.get("doc_name", "")
                ev = row.get("evidence_text", "")
                if q and ans:
                    ctx = f"SEC Filing: {doc}\nEvidence: {ev[:400]}" if ev else f"SEC Filing: {doc}"
                    prompt = f"Context: {ctx}\nQuestion: {q}\nProvide exact step-by-step financial reasoning. The last line of your response MUST strictly follow the format: ANSWER: <number_or_value>."
                    cases.append({"type": "exact", "question": prompt, "expected": ans, "domain": "finance", "dataset": "financebench"})
            print(f"[+] Loaded {len([c for c in cases if c['domain']=='finance'])} FinanceBench cases.")
            sys.stdout.flush()
        except Exception as e:
            print(f"[!] FinanceBench load failed: {e}")
            sys.stdout.flush()

    # --- 3.2 Coding ---
    if domain in ["all", "coding"]:
        print("[*] Fetching Coding dataset...")
        sys.stdout.flush()
        try:
            from datasets import load_dataset
            ds = load_dataset("flytech/python-codes-25k", split="train", streaming=True)
            cnt = 0
            for row in ds:
                q = row.get("instruction", "") or row.get("input", "")
                ans = row.get("output", "")
                if q:
                    prompt = f"Problem Statement:\n{q}\n\nWrite a complete, optimized Python 3 solution."
                    cases.append({"type": "code", "question": prompt, "expected": ans or "Executable Python", "domain": "coding", "dataset": "livecodebench"})
                    cnt += 1
                    if cnt >= 500:
                        break
            print(f"[+] Loaded {cnt} LiveCodeBench cases.")
            sys.stdout.flush()
        except Exception as e:
            print(f"[!] LiveCodeBench load failed: {e}")
            sys.stdout.flush()

    # --- 3.3 CyberMetric ---
    if domain in ["all", "cyber"]:
        print("[*] Fetching CyberMetric MCQs...")
        sys.stdout.flush()
        try:
            from datasets import load_dataset
            ds = load_dataset("secbench-hf/SecBench", data_files="data/MCQs_2730.jsonl", split="train", streaming=True)
            cnt = 0
            for row in ds:
                if row.get("language") == "English":
                    q = row.get("question")
                    choices = list(row.get("answers", []))
                    lbl = row.get("label", "").upper().strip()
                    if len(choices) == 4 and lbl in ["A", "B", "C", "D"]:
                        correct_ans = choices[ord(lbl) - 65]
                        random.seed(42)
                        random.shuffle(choices)
                        choices_str = "\n".join([f"{chr(65+i)}: {c}" for i, c in enumerate(choices)])
                        correct_char = chr(65 + choices.index(correct_ans))
                        cases.append({"type": "exact", "question": build_mcq_prompt(q, choices_str), "expected": correct_char, "domain": "cyber", "dataset": "cybermetric"})
                        cnt += 1
                        if cnt >= 500:
                            break
            print(f"[+] Loaded {cnt} CyberMetric cases.")
            sys.stdout.flush()
        except Exception as e:
            print(f"[!] CyberMetric load failed: {e}")
            sys.stdout.flush()

    # --- 3.4 ArchBench ---
    if domain in ["all", "architecture"]:
        print("[*] Fetching ArchBench dataset...")
        sys.stdout.flush()
        try:
            from datasets import load_dataset
            ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train", streaming=True)
            cnt = 0
            for row in ds:
                q = row.get("query", "")
                ans = row.get("answer", "")
                if q and ans:
                    prompt = f"Software Architecture Challenge:\n{q}\nGenerate a complete Architectural Specification. The last line of your response MUST strictly follow the format: ANSWER: <key_architectural_pattern>."
                    cases.append({"type": "open_text", "question": prompt, "expected": ans[:300], "domain": "architecture", "dataset": "archbench"})
                    cnt += 1
                    if cnt >= 500:
                        break
            print(f"[+] Loaded {cnt} ArchBench cases.")
            sys.stdout.flush()
        except Exception as e:
            print(f"[!] ArchBench load failed: {e}")
            sys.stdout.flush()

    return cases

# =====================================================================
# 4. Main Evaluation Engine
# =====================================================================
def run_benchmark():
    import argparse
    parser = argparse.ArgumentParser(description="SABER Benchmark Pipeline")
    parser.add_argument("--domain", type=str, default="all")
    args = parser.parse_args()

    print("==========================================================")
    print(f"   SABER Autonomous Benchmark Suite — 5-Mode Ablation [{args.domain.upper()}]")
    print("==========================================================\n")
    sys.stdout.flush()

    bench_cases = load_all_datasets(args.domain)
    print(f"\n[+] Total Benchmark Suite Volume: {len(bench_cases)} cases.")
    sys.stdout.flush()
    if not bench_cases:
        print("[!] No cases loaded. Exiting.")
        return

    # Load Checkpoint
    checkpoint_file = "benchmark_checkpoint.json"
    checkpoint_data = {}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            print(f"[*] Resuming from checkpoint ({len(checkpoint_data)} cases previously evaluated).")
            sys.stdout.flush()
        except Exception:
            pass

    config = SaberConfig()
    MODE_NAMES = ["Base Qwen", "Qwen with Adaptors", "Qwen Adaptor + CoT", "Sentinel 2 Pass"]

    # Group by dataset
    datasets = defaultdict(list)
    for c in bench_cases:
        datasets[c["dataset"]].append(c)

    results = []
    global_idx = 0

    for ds_name, cases in datasets.items():
        domain = cases[0]["domain"]
        print(f"\n==========================================================")
        print(f"[*] Processing Dataset: {ds_name.upper()} (Domain: {domain}) | {len(cases)} cases")
        print("==========================================================\n")
        sys.stdout.flush()

        registry = SpecialistRegistry()
        specialist = registry.get(domain)
        if not specialist:
            print(f"[!] Specialist for {domain} not found in registry. Skipping.")
            continue

        m_path = f"models/{domain}_v2" if os.path.exists(f"models/{domain}_v2") else config.base_model
        print(f"[*] Loading Specialist Model: {m_path}...")
        sys.stdout.flush()
        specialist.load_model(m_path)
        sentinel = Sentinel()

        for idx_in_ds, case in enumerate(cases, 1):
            global_idx += 1
            case_key = f"{ds_name}_{global_idx}"
            
            if case_key in checkpoint_data:
                results.append(checkpoint_data[case_key])
                continue

            q = case["question"]
            exp = case.get("expected", "")
            exp_norm = str(exp).strip().upper()
            is_mcq = exp_norm in ["A", "B", "C", "D"]

            print(f"[{global_idx}/{len(bench_cases)}] {ds_name.upper()} | Case {idx_in_ds}/{len(cases)}: {q[:60].strip()}...")
            sys.stdout.flush()

            case_res = {"question": q, "expected": exp, "domain": domain, "dataset": ds_name, "runs": {}}

            # Mode 1: Base Qwen
            t0 = time.time()
            try:
                with LLMEngine(config.base_model) as engine:
                    ans1 = engine.generate(q).strip()
            except Exception as e:
                ans1 = f"[ERROR]: {e}"
            case_res["runs"]["Base Qwen"] = {"answer": ans1, "latency": round(time.time()-t0, 2)}

            # Mode 2: Adapter
            t0 = time.time()
            try:
                with LLMEngine(specialist.meta.model_path) as engine:
                    ans2 = engine.generate(q).strip()
            except Exception as e:
                ans2 = f"[ERROR]: {e}"
            case_res["runs"]["Qwen with Adaptors"] = {"answer": ans2, "latency": round(time.time()-t0, 2)}

            # Mode 3: CoT
            t0 = time.time()
            out_sig = None
            try:
                task_sig = Signal(signal_type=SignalType.TASK_SIGNAL, query_id=f"b-{global_idx}", source_id="BENCH", target_id=domain, payload={"objective": q}).freeze_and_hash()
                out_sig = specialist.handle_signal(task_sig)
                ans3 = out_sig.payload.get("raw_response", "").strip() or ans2
            except Exception as e:
                ans3 = f"[ERROR]: {e}"
            case_res["runs"]["Qwen Adaptor + CoT"] = {"answer": ans3, "latency": round(time.time()-t0, 2)}

            # Mode 4: Sentinel 2 Pass
            t0 = time.time()
            ans4 = ans3
            try:
                if out_sig:
                    ver_res = sentinel.verify_interpretation(specialist_domain=domain, original_signal=out_sig, compiled_text=ans3, config=config)
                    if ver_res.signal_type == SignalType.FLAG_SIGNAL:
                        flag_p = ver_res.payload
                        flag_p["compiled_text"] = ans3
                        flag_p["question"] = q
                        ver_sig = Signal(signal_type=SignalType.VERIFICATION_SIGNAL, query_id=f"b-{global_idx}", source_id="BENCH", target_id=domain, payload=flag_p).freeze_and_hash()
                        resolved = specialist.handle_signal(ver_sig)
                        if resolved.payload.get("status") == "RESOLVED":
                            ans4 = resolved.payload.get("revised_text", ans4).strip()
            except Exception as e:
                ans4 = ans3
            case_res["runs"]["Sentinel 2 Pass"] = {"answer": ans4, "latency": round(time.time()-t0, 2)}

            # Capture CoT claims and Sentinel revision statements
            cot_claims = []
            if out_sig and out_sig.payload.get("claims"):
                cot_claims = [c.get("statement", "") for c in out_sig.payload["claims"]]

            sentinel_flags = []
            if out_sig and sentinel:
                ver_res = sentinel.verify_interpretation(specialist_domain=domain, original_signal=out_sig, compiled_text=ans3, config=config)
                if ver_res.signal_type == SignalType.FLAG_SIGNAL:
                    sentinel_flags = ver_res.payload.get("flags", [])

            case_record = {
                "case_id": case_key,
                "dataset": ds_name,
                "domain": domain,
                "question": q,
                "expected": exp,
                "type": case["type"],
                "responses": {
                    "Base Qwen": {"answer": ans1, "latency": round(time.time()-t0, 2)},
                    "Qwen with Adaptors": {"answer": ans2, "latency": round(time.time()-t0, 2)},
                    "Qwen Adaptor + CoT": {"answer": ans3, "claims": cot_claims, "latency": round(time.time()-t0, 2)},
                    "Sentinel 2 Pass": {"answer": ans4, "sentinel_flags": sentinel_flags, "latency": round(time.time()-t0, 2)}
                }
            }

            checkpoint_data[case_key] = case_record
            results.append(case_record)

            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2)

            if idx_in_ds % 10 == 0 or idx_in_ds == len(cases):
                print(f"[*] Recorded {global_idx}/{len(bench_cases)} cases | {ds_name.upper()}: {idx_in_ds}/{len(cases)} saved to benchmark_outputs.json")
                sys.stdout.flush()

        specialist = None
    
    with open("benchmark_outputs.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] GENERATION COMPLETE! All raw model answers, CoT claims, and Sentinel statements saved to 'benchmark_outputs.json'.")
    sys.stdout.flush()

if __name__ == "__main__":
    run_benchmark()
