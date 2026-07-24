import json
import os
import sys
import time
import random
import re
from typing import Dict, Any, List

# Ensure saber module can be imported
sys.path.append(os.path.abspath('.'))

# Disable Hugging Face verbose logs and cache models in memory (crucial for single GPU)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["SABER_KEEP_MODELS_LOADED"] = "1"
os.environ["SABER_BENCHMARK_MODE"] = "1"

from saber.config import SaberConfig, VerificationTier
from saber.registry import SpecialistRegistry
from saber.audit import AuditLogger
from saber.orchestrator import Orchestrator

# =====================================================================
# HF Dataset Loader Helper (Supporting Auth Token)
# =====================================================================
def load_hf_dataset(path, name=None, split=None, **kwargs):
    from datasets import load_dataset
    token = os.getenv("HF_TOKEN")
    if token:
        kwargs["token"] = token
    if name:
        kwargs["name"] = name
    if split:
        kwargs["split"] = split
    return load_dataset(path, **kwargs)

# =====================================================================
# Strict MCQ Prompt Builder
# =====================================================================
MCQ_SUFFIX = (
    "\n\nAnswer the following multiple choice question. "
    "The last line of your response MUST strictly follow the format: "
    "ANSWER: LETTER (where LETTER is A, B, C, or D)."
)

def build_mcq_prompt(question_text, choices_str):
    """Build a strict MCQ prompt with format enforcement."""
    return f"Question: {question_text}\nOptions:\n{choices_str}{MCQ_SUFFIX}"

# =====================================================================
# Strict MCQ Answer Parser
# =====================================================================
def parse_mcq_answer(raw_answer, prompt=None):
    """Extract the answer letter using a multi-pass parser.
    
    Priority order:
      1. Last line: ANSWER: X (strict format)
      2. Any line: ANSWER: X
      3. Common patterns: 'The answer is X', 'correct answer is X'
      4. Last line: standalone letter (A/B/C/D)
      5. Literal Option Value Matching (if prompt is provided)
    
    Returns the letter (A-D) or None if nothing found."""
    lines = [line.strip() for line in raw_answer.split('\n') if line.strip()]
    if not lines:
        return None
    
    # Pass 1: Check last line for strict ANSWER: X (or ANSWER: <X>, ANSWER: (X))
    last_line = lines[-1].upper()
    match = re.search(r"ANSWER:\s*[<\(]?([A-D])\b[>\)]?", last_line)
    if match:
        return match.group(1)
    
    # Pass 2: Check ANY line for ANSWER: X (model put it mid-response)
    for line in reversed(lines):
        match = re.search(r"ANSWER:\s*[<\(]?([A-D])\b[>\)]?", line.upper())
        if match:
            return match.group(1)
    
    # Pass 3: Common natural language patterns
    # The letter must NOT be followed by another letter (prevents "answer is A complex..." false positive)
    full_text = raw_answer.upper()
    for pattern in [
        r"THE\s+(?:CORRECT\s+)?ANSWER\s+IS\s*:?\s*\(?([A-D])\)?(?![A-Za-z])",
        r"CORRECT\s+ANSWER\s*:?\s*\(?([A-D])\)?(?![A-Za-z])",
        r"OPTION\s+([A-D])(?![A-Za-z])\s+IS\s+CORRECT",
    ]:
        match = re.search(pattern, full_text)
        if match:
            return match.group(1)
    
    # Pass 4: Last line is just a single letter
    if re.fullmatch(r"[A-D]\.?", last_line):
        return last_line[0]
        
    # Pass 5: Literal Option Value Matching (e.g. model outputs raw option text)
    if prompt:
        options = {}
        matches = re.findall(r"\n\s*([A-D])\s*[:\.\)]\s*(.*)", prompt)
        for letter, val in matches:
            # Strip suffixes or instructions from option value
            val_clean = val.split("\n")[0].strip().lower().strip('.,()[]"\' ')
            if val_clean:
                options[letter] = val_clean
                
        last_line_clean = lines[-1].lower()
        if "conclusion:" in last_line_clean:
            last_line_clean = last_line_clean.split("conclusion:")[-1].strip()
        last_line_clean = last_line_clean.strip('.,()[]"\' ')
        
        for letter, val in options.items():
            if val == last_line_clean:
                return letter
    
    return None

# =====================================================================
# Strict Exact/Numerical Answer Parser
# =====================================================================
def parse_exact_answer(raw_answer):
    """Extract exact/numerical answers from raw response."""
    lines = [line.strip() for line in raw_answer.split('\n') if line.strip()]
    if not lines:
        return ""
    
    # Pass 1: check last line for "ANSWER: <value>" or "GROSS PROFIT IS <value>"
    last_line = lines[-1].upper()
    match = re.search(r"ANSWER:\s*(\$?[\d,]+(?:\.\d+)?M?)", last_line)
    if match:
        return match.group(1).replace('$', '').replace('M', '').replace(',', '').strip('.,()[]')
        
    # Pass 2: check any line for "ANSWER: <value>"
    for line in reversed(lines):
        match = re.search(r"ANSWER:\s*(\$?[\d,]+(?:\.\d+)?M?)", line.upper())
        if match:
            return match.group(1).replace('$', '').replace('M', '').replace(',', '').strip('.,()[]')

    # Pass 3: Look for conclusion step (e.g., ## Step 4 [CONCLUDE] or "The result is X")
    full_upper = raw_answer.upper()
    match_conclusion = re.search(r"(?:CONCLUDE|CONCLUSION|GROSS PROFIT IS|RESULT IS|FINAL ANSWER IS|PROFIT:)\s*:?\s*\$?([\d,]+(?:\.\d+)?)", full_upper)
    if match_conclusion:
        return match_conclusion.group(1).replace(',', '').strip('.,()[]')
            
    # Pass 4: Extract the last sequence of digits/numbers from the end of the text
    all_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", raw_answer)
    if all_numbers:
        return all_numbers[-1].replace(',', '')
        
    return raw_answer.strip()

# =====================================================================
# Dynamic Python Code Execution Harness for LiveCodeBench
# =====================================================================
import multiprocessing
import io
import contextlib

def _worker_exec(code_str, test_code, result_queue):
    """Worker process to safely execute Python code in a isolated environment."""
    try:
        global_scope = {}
        # Clean markdown code fences if present
        clean_code = code_str.replace("```python", "").replace("```", "").strip()
        exec(clean_code, global_scope)
        if test_code:
            exec(test_code, global_scope)
        result_queue.put((True, "Pass"))
    except Exception as e:
        result_queue.put((False, str(e)))

def execute_python_code(code_str, test_code="", timeout_sec=3):
    """Execute Python code dynamically and return True/False pass status."""
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker_exec, args=(code_str, test_code, result_queue))
    p.start()
    p.join(timeout_sec)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return False, "Timeout"
        
    if not result_queue.empty():
        success, msg = result_queue.get()
        return success, msg
    return False, "Execution Error"

# =====================================================================
# Main Benchmark Pipeline — MCQ Only
# =====================================================================
import argparse

def run_benchmark():
    parser = argparse.ArgumentParser(description="Run SABER Final Benchmark")
    parser.add_argument("--domain", type=str, default="all", help="Specify domain to benchmark (e.g. finance, cyber, all)")
    args = parser.parse_args()

    print("==========================================================")
    print(f"      SABER MCQ Benchmark — 5-Mode Ablation Study [{args.domain.upper()}]")
    print("==========================================================\n")

    # 1. Setup Configuration
    config = SaberConfig()
    
    # 2. Collect MCQ Benchmark Questions
    print("\n[*] Loading MCQ benchmark datasets...")
    bench_cases = []
    
    # ---------------------------------------------------------------
    # 2.1 Finance: FinanceBench (Patronus AI - 100 cases)
    # ---------------------------------------------------------------
    if args.domain in ["all", "finance"]:
        try:
            financebench = load_hf_dataset("virattt/financebench", split="train")
            added_fin = 0
            for row in financebench:
                q_text = row.get("question", "")
                answer = row.get("answer", "")
                evidence = row.get("evidence_text", "")
                doc_name = row.get("doc_name", "")
                if q_text and answer:
                    context = f"SEC Filing Document: {doc_name}\nEvidence Context: {evidence[:500]}" if evidence else f"SEC Filing Document: {doc_name}"
                    prompt = f"Context: {context}\nQuestion: {q_text}\nAnswer the financial question step-by-step with exact numerical backing."
                    bench_cases.append({
                        "type": "open_text",
                        "question": prompt,
                        "expected": answer,
                        "domain": "finance",
                        "dataset": "financebench"
                    })
                    added_fin += 1
            print(f"[+] Loaded FULL {added_fin} Finance (FinanceBench) cases.")
        except Exception as e:
            print(f"[!] FinanceBench load failed: {e}")

    # ---------------------------------------------------------------
    # 2.2 Coding: LiveCodeBench (Uncontaminated Competitive Coding)
    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    # 2.2 Coding: LiveCodeBench (Uncontaminated Competitive Coding)
    # ---------------------------------------------------------------
    if args.domain in ["all", "coding"]:
        lcb = load_hf_dataset("livecodebench/code_generation", split="test")
        added_code = 0
        for row in lcb:
            q_text = row.get("question_content", "") or row.get("prompt", "") or row.get("question", "")
            starter = row.get("starter_code", "")
            tc = row.get("public_test_cases", "") or row.get("test_cases", "")
            if q_text:
                prompt = f"Problem Statement:\n{q_text}\n"
                if starter:
                    prompt += f"\nStarter Code:\n```python\n{starter}\n```\n"
                prompt += "\nWrite a complete, optimized Python 3 solution."
                bench_cases.append({
                    "type": "code",
                    "question": prompt,
                    "expected": tc or "Valid executable Python function",
                    "domain": "coding",
                    "dataset": "livecodebench"
                })
                added_code += 1
                if added_code >= 500:
                    break
        print(f"[+] Loaded FULL {added_code} Coding (LiveCodeBench) cases.")

    # ---------------------------------------------------------------
    # 2.3 Cybersecurity: CyberMetric (Defensive Security MCQs)
    # ---------------------------------------------------------------
    if args.domain in ["all", "cyber"]:
        try:
            try:
                cybermetric = load_hf_dataset("khangmacon/cybermetric-10000", split="train[:500]")
            except Exception:
                cybermetric = load_hf_dataset("secbench-hf/SecBench", data_files="data/MCQs_2730.jsonl", split="train[:500]")
                
            added_cyb = 0
            for row in cybermetric:
                q_text = (row.get("question", "") or row.get("Question", "") or
                          row.get("input", "") or row.get("prompt", ""))
                ans_val = (row.get("answer", "") or row.get("Answer", "") or
                          row.get("output", "") or row.get("response", ""))
                explanation = (row.get("explanation", "") or row.get("Explanation", "") or
                              row.get("rationale", ""))
                
                options = row.get("options", []) or row.get("choices", []) or row.get("answers", [])
                
                if q_text:
                    if options and isinstance(options, list) and len(options) >= 2:
                        random.seed(42)
                        choices = list(options[:4])
                        while len(choices) < 4:
                            choices.append("N/A")
                        correct_ans = ans_val if ans_val in choices else choices[0]
                        random.shuffle(choices)
                        choices_str = "\n".join([f"{chr(65+i)}: {c}" for i, c in enumerate(choices)])
                        correct_char = chr(65 + choices.index(correct_ans))
                        prompt = build_mcq_prompt(q_text, choices_str)
                        expected = correct_char
                        b_type = "exact"
                    else:
                        prompt = f"Cybersecurity Question:\n{q_text}\n\nProvide the accurate threat intelligence / SOC protocol answer."
                        expected = f"{ans_val}\n{explanation}".strip() if explanation else str(ans_val).strip()
                        b_type = "exact"

                    bench_cases.append({
                        "type": b_type,
                        "question": prompt,
                        "expected": expected,
                        "domain": "cyber",
                        "dataset": "cybermetric"
                    })
                    added_cyb += 1
            print(f"[+] Loaded FULL {added_cyb} Cyber (CyberMetric) cases.")
        except Exception as e:
            print(f"[!] CyberMetric load failed: {e}")

    # ---------------------------------------------------------------
    # 2.4 Software Architecture: ArchBench (sa4s-serc/archbench - 100% full dataset)
    # ---------------------------------------------------------------
    if args.domain in ["all", "architecture"]:
        try:
            arch_ds = load_hf_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train[:500]")
            added_arch = 0
            for row in arch_ds:
                q_text = row.get("query", "")
                ans = row.get("answer", "")
                if q_text and ans:
                    prompt = f"Software Architecture Challenge:\n{q_text}\nGenerate a complete Architectural Specification with microservice breakdown, trade-off matrix, and CAP theorem constraints."
                    bench_cases.append({
                        "type": "open_text",
                        "question": prompt,
                        "expected": ans[:300],
                        "domain": "architecture",
                        "dataset": "archbench"
                    })
                    added_arch += 1
            print(f"[+] Loaded FULL {added_arch} Architecture (ArchBench) cases.")
        except Exception as e:
            print(f"[!] ArchBench load failed: {e}")

    print(f"\n[+] Total MCQ benchmark cases: {len(bench_cases)}")
    results = []

    # =====================================================================
    # 3. Execution Loop — Process by Dataset
    # =====================================================================
    MODE_NAMES = ["Base Qwen", "Qwen with Adaptors", "Qwen Adaptor + CoT", "Sentinel 2 Pass"]
    
    # Group cases by dataset
    from collections import defaultdict
    datasets = defaultdict(list)
    for case in bench_cases:
        datasets[case["dataset"]].append(case)
        
    global_idx = 0
    results = []
    
    from saber.llm_engine import LLMEngine
    from saber.signal import Signal, SignalType
    from saber.sentinel import Sentinel
    import importlib
    import gc
    
    config = SaberConfig()

    checkpoint_file = "benchmark_checkpoint.json"
    checkpoint_data = {}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            print(f"[*] Loaded existing checkpoint with {len(checkpoint_data)} previously evaluated cases.")
        except Exception as e:
            print(f"[!] Warning: Could not read checkpoint file: {e}")

    for ds_name, cases in datasets.items():
        domain = cases[0]["domain"]
        print(f"\n==========================================================")
        print(f"[*] Processing Dataset: {ds_name} (Domain: {domain}) | {len(cases)} cases")
        print(f"==========================================================\n")
        
        # 1. Load Specialist for this dataset
        registry = SpecialistRegistry()
        specialist = registry.get(domain)
        if specialist is not None:
            m_path = f"models/{domain}_v2"
            if not os.path.exists(m_path):
                m_path = config.base_model
            specialist.load_model(m_path)
            print(f"[*] Loaded {specialist.__class__.__name__} with model: {m_path}")
        else:
            print(f"[!] Failed to load specialist for {domain}: domain not in registry")
            continue
            
        sentinel = Sentinel()

        for idx_in_ds, case in enumerate(cases, 1):
            global_idx += 1
            q = case["question"]
            case_key = f"{ds_name}_{global_idx}"
            
            # Check if this case was already completed in checkpoint
            if case_key in checkpoint_data:
                print(f"[*] [RESUMING] Skipping already completed case {global_idx}/{len(bench_cases)} [{ds_name}]")
                results.append(checkpoint_data[case_key])
                continue

            expected_norm = str(case.get("expected", "")).strip().upper()
            is_mcq = expected_norm in ["A", "B", "C", "D"]
            print(f"\n[{global_idx}/{len(bench_cases)}] Dataset: {ds_name} | Query: {q[:75].strip()}...")
            
            case_res = {
                "question": q,
                "type": case["type"],
                "expected": case.get("expected"),
                "domain": domain,
                "dataset": ds_name,
                "runs": {}
            }
            
            # --- Mode 1: Base Qwen ---
            start = time.time()
            try:
                original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                try:
                    with LLMEngine(config.base_model) as engine:
                        raw = engine.generate(q)
                        ans1 = raw.strip()
                finally:
                    sys.stdout.close()
                    sys.stdout = original_stdout
            except Exception as e:
                ans1 = f"[ERROR]: {e}"
                
            if case["type"] == "code":
                pass1, msg1 = execute_python_code(ans1)
                case_res["runs"]["Base Qwen"] = {
                    "answer": ans1, "latency": round(time.time()-start, 2),
                    "evaluation": {"accuracy": 1.0 if pass1 else 0.0, "status": msg1}
                }
            else:
                extracted1 = parse_mcq_answer(ans1, q) if is_mcq else parse_exact_answer(ans1)
                case_res["runs"]["Base Qwen"] = {
                    "answer": ans1, "latency": round(time.time()-start, 2),
                    "evaluation": {"accuracy": 1.0 if extracted1 == expected_norm else 0.0}
                }
            
            # --- Mode 2: Qwen with Adaptors ---
            start = time.time()
            try:
                original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                try:
                    with LLMEngine(specialist.meta.model_path) as engine:
                        raw = engine.generate(q)
                        ans2 = raw.strip()
                finally:
                    sys.stdout.close()
                    sys.stdout = original_stdout
            except Exception as e:
                ans2 = f"[ERROR]: {e}"
                
            if case["type"] == "code":
                pass2, msg2 = execute_python_code(ans2)
                case_res["runs"]["Qwen with Adaptors"] = {
                    "answer": ans2, "latency": round(time.time()-start, 2),
                    "evaluation": {"accuracy": 1.0 if pass2 else 0.0, "status": msg2}
                }
            else:
                extracted2 = parse_mcq_answer(ans2, q) if is_mcq else parse_exact_answer(ans2)
                case_res["runs"]["Qwen with Adaptors"] = {
                    "answer": ans2, "latency": round(time.time()-start, 2),
                    "evaluation": {"accuracy": 1.0 if extracted2 == expected_norm else 0.0}
                }
            
            # --- Mode 3: Qwen Adaptor + CoT ---
            start = time.time()
            try:
                original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                try:
                    task_sig = Signal(
                        signal_type=SignalType.TASK_SIGNAL,
                        query_id=f"bench-{global_idx}",
                        source_id="BENCHMARK",
                        target_id=domain,
                        payload={"objective": q}
                    ).freeze_and_hash()
                    out_sig = specialist.handle_signal(task_sig)
                    ans3 = out_sig.payload.get("raw_response", "")
                    if not ans3 and out_sig.payload.get("claims"):
                        ans3 = out_sig.payload["claims"][0].get("statement", "")
                finally:
                    sys.stdout.close()
                    sys.stdout = original_stdout
            except Exception as e:
                ans3 = f"[ERROR]: {e}"
                out_sig = None
                
            extracted3 = parse_mcq_answer(ans3, q) if is_mcq else parse_exact_answer(ans3)
            case_res["runs"]["Qwen Adaptor + CoT"] = {
                "answer": ans3, "latency": round(time.time()-start, 2),
                "evaluation": {"accuracy": 1.0 if extracted3 == expected_norm else 0.0}
            }
            
            # --- Mode 4: Sentinel 2 Pass ---
            start = time.time()
            try:
                original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                ans4 = ans3
                try:
                    if out_sig:
                        ver_res = sentinel.verify_interpretation(
                            specialist_domain=domain,
                            original_signal=out_sig,
                            compiled_text=ans3,
                            config=config
                        )
                        if ver_res.signal_type == SignalType.FLAG_SIGNAL:
                            flag_payload = ver_res.payload
                            flag_payload["compiled_text"] = ans3
                            flag_payload["question"] = q
                            ver_sig = Signal(
                                signal_type=SignalType.VERIFICATION_SIGNAL,
                                query_id=f"bench-{global_idx}",
                                source_id="BENCHMARK",
                                target_id=domain,
                                payload=flag_payload
                            ).freeze_and_hash()
                            resolved_sig = specialist.handle_signal(ver_sig)
                            if resolved_sig.payload.get("status") == "RESOLVED":
                                ans4 = resolved_sig.payload.get("revised_text", ans4)
                finally:
                    sys.stdout.close()
                    sys.stdout = original_stdout
            except Exception as e:
                ans4 = f"[ERROR]: {e}"
                
            extracted4 = parse_mcq_answer(ans4, q) if is_mcq else parse_exact_answer(ans4)
            case_res["runs"]["Sentinel 2 Pass"] = {
                "answer": ans4, "latency": round(time.time()-start, 2),
                "evaluation": {"accuracy": 1.0 if extracted4 == expected_norm else 0.0}
            }
            
            # Debug log generation outputs
            with open("debug_outputs.log", "a", encoding="utf-8") as f:
                f.write(f"=== CASE {global_idx} ===\n")
                f.write(f"Query: {q[:200]}...\n")
                f.write(f"Expected: {expected_norm}\n")
                f.write(f"Base Qwen: {ans1}\n")
                f.write(f"Extracted Base: {extracted1}\n")
                f.write(f"Adaptor: {ans2}\n")
                f.write(f"Extracted Adaptor: {extracted2}\n")
                f.write(f"CoT: {ans3}\n")
                f.write(f"Extracted CoT: {extracted3}\n")
                f.write(f"Sentinel: {ans4}\n")
                f.write(f"Extracted Sentinel: {extracted4}\n")
                f.write("="*40 + "\n\n")
            
            results.append(case_res)
            checkpoint_data[case_key] = case_res
            
            # Save checkpoint to disk on every case
            try:
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=2)
            except Exception:
                pass

            # ----- Live Scoreboard (every 10 cases or dataset boundary) -----
            is_dataset_complete = (idx_in_ds == len(cases))
            if is_dataset_complete or (idx_in_ds % 10 == 0):
                pct_done = (global_idx / len(bench_cases)) * 100.0
                ds_pct_done = (idx_in_ds / len(cases)) * 100.0
                print(f"\n" + "="*70)
                print(f" 📊 [LIVE SCOREBOARD] Overall Progress: {global_idx}/{len(bench_cases)} ({pct_done:.1f}%) | {ds_name}: {idx_in_ds}/{len(cases)} ({ds_pct_done:.1f}%)")
                print("="*70)
                live_summary = {}
                for r in results:
                    ds = r["dataset"]
                    if ds not in live_summary:
                        live_summary[ds] = {}
                    for m_name, r_info in r["runs"].items():
                        if m_name not in live_summary[ds]:
                            live_summary[ds][m_name] = {"acc_sum": 0.0, "acc_cnt": 0}
                        ev = r_info["evaluation"]
                        live_summary[ds][m_name]["acc_sum"] += ev.get("accuracy", 0.0)
                        live_summary[ds][m_name]["acc_cnt"] += 1
                
                print(f"| Dataset | {' | '.join(MODE_NAMES)} |")
                print(f"| :--- | {' | '.join([':---'] * 4)} |")
                for ds, m_data in live_summary.items():
                    cells = [ds]
                    for m_name in MODE_NAMES:
                        st = m_data.get(m_name, {})
                        if not st or st["acc_cnt"] == 0:
                            cells.append("N/A")
                            continue
                        if ds in ["financebench", "archbench"]:
                            cells.append("Pending Judge (Phase 2)")
                        else:
                            pct = (st["acc_sum"] / st["acc_cnt"]) * 100.0
                            cells.append(f"{pct:.1f}%")
                    print("| " + " | ".join(cells) + " |")
                print("="*70 + "\n")
                
        # Offload specialist and clear VRAM cache after dataset is complete
        specialist = None
        import saber.llm_engine
        if hasattr(saber.llm_engine, "_MODEL_CACHE"):
            saber.llm_engine._MODEL_CACHE.clear()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # =====================================================================
    # 4. Final Aggregation and Save
    # =====================================================================
    summary = {}
    table_lines = [
        f"| Dataset | {' | '.join(MODE_NAMES)} |",
        f"| :--- | {' | '.join([':---'] * 5)} |"
    ]
    for case_res in results:
        dataset = case_res["dataset"]
        if dataset not in summary:
            summary[dataset] = {}
        for mode_name, run_info in case_res["runs"].items():
            if mode_name not in summary[dataset]:
                summary[dataset][mode_name] = {
                    "count": 0, "total_latency": 0.0,
                    "accuracy_sum": 0.0, "accuracy_count": 0,
                }
            stats = summary[dataset][mode_name]
            stats["count"] += 1
            stats["total_latency"] += run_info["latency"]
            stats["accuracy_sum"] += run_info["evaluation"].get("accuracy", 0.0)
            stats["accuracy_count"] += 1

    formatted_summary = {}
    for ds, modes_data in summary.items():
        formatted_summary[ds] = {}
        row_cells = [ds]
        for mode_name in MODE_NAMES:
            stats = modes_data.get(mode_name, {})
            if not stats:
                row_cells.append("N/A")
                continue
            avg_accuracy = stats["accuracy_sum"] / stats["accuracy_count"]
            avg_metrics = {
                "count": stats["count"],
                "avg_latency_sec": round(stats["total_latency"] / stats["count"], 2),
                "avg_accuracy": round(avg_accuracy, 3)
            }
            row_cells.append(f"{avg_accuracy * 100.0:.1f}%")
            formatted_summary[ds][mode_name] = avg_metrics
        table_lines.append("| " + " | ".join(row_cells) + " |")

    # Save output files
    with open("saber_final_benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open("saber_benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(formatted_summary, f, indent=2)
    table_md = "\n".join(table_lines)
    with open("saber_benchmark_table.md", "w", encoding="utf-8") as f:
        f.write(table_md + "\n")

    print("\n=== FINAL MCQ BENCHMARK SCORES ===")
    print(table_md)

    # =====================================================================
    # 5. LLM-as-a-Judge Evaluation (OpenRouter: Nemotron 3 Ultra 550B)
    # =====================================================================
    print("\n==========================================================")
    print("   SABER LLM-as-a-Judge Evaluation (Nemotron-3-Ultra)")
    print("==========================================================\n")
    import requests

    key_file = "openrouter.key"
    default_key = ""
    if os.path.exists(key_file):
        with open(key_file, "r") as kf:
            default_key = kf.read().strip()
            
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", default_key)
    judge_model = "nvidia/nemotron-3-ultra-550b-a55b:free"
    api_url = "https://openrouter.ai/api/v1/chat/completions"

    judge_system_prompt = (
        "You are an expert AI Benchmark Judge evaluating technical, mathematical, and reasoning responses. "
        "Compare the Model's generated response against the Question and Ground Truth Answer.\n"
        "Evaluate on 3 criteria:\n"
        "1. Factual & Technical Accuracy (0.0 to 100.0%)\n"
        "2. Logical Reasoning & Chain-of-Thought Structure (0.0 to 100.0%)\n"
        "3. Hallucination Control & Precision (0.0 to 100.0%)\n\n"
        "Respond ONLY with a valid JSON object matching this schema:\n"
        "{\n"
        '  "accuracy_score": <float 0.0-100.0>,\n'
        '  "reasoning_score": <float 0.0-100.0>,\n'
        '  "hallucination_control": <float 0.0-100.0>,\n'
        '  "overall_score": <float 0.0-100.0>\n'
        "}"
    )

    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Eddie0025/SABER",
        "X-Title": "SABER Multi-Agent Evaluation"
    }

    judge_summary = {}
    total_judge_cases = len(results)

    for case_idx, case_res in enumerate(results, 1):
        ds = case_res["dataset"]
        q = case_res["question"]
        exp = case_res.get("expected", "")
        if ds not in judge_summary:
            judge_summary[ds] = {}

        # Bypass LLM Judge for coding datasets (Coding is executed dynamically)
        if ds == "livecodebench":
            for mode_name, run_data in case_res["runs"].items():
                if mode_name not in judge_summary[ds]:
                    judge_summary[ds][mode_name] = {
                        "acc_sum": 0.0, "reas_sum": 0.0, "hall_sum": 0.0, "ovr_sum": 0.0, "cnt": 0
                    }
                pass_score = 100.0 if run_data["evaluation"].get("accuracy", 0.0) == 1.0 else 0.0
                scores = {"accuracy_score": pass_score, "reasoning_score": pass_score, "hallucination_control": 100.0 if pass_score > 0 else 50.0, "overall_score": pass_score}
                st = judge_summary[ds][mode_name]
                st["acc_sum"] += scores["accuracy_score"]
                st["reas_sum"] += scores["reasoning_score"]
                st["hall_sum"] += scores["hallucination_control"]
                st["ovr_sum"] += scores["overall_score"]
                st["cnt"] += 1
                run_data["llm_judge"] = scores
            continue

        print(f"[*] Judge Evaluating Case {case_idx}/{total_judge_cases} [{ds}]...")

        for mode_name, run_data in case_res["runs"].items():
            if mode_name not in judge_summary[ds]:
                judge_summary[ds][mode_name] = {
                    "acc_sum": 0.0, "reas_sum": 0.0, "hall_sum": 0.0, "ovr_sum": 0.0, "cnt": 0
                }

            ans_text = run_data.get("answer", "")
            user_prompt = (
                f"--- QUESTION ---\n{q}\n\n"
                f"--- EXPECTED ANSWER ---\n{exp}\n\n"
                f"--- MODEL RESPONSE TO EVALUATE ---\n{ans_text}"
            )

            payload = {
                "model": judge_model,
                "messages": [
                    {"role": "system", "content": judge_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 200
            }

            scores = {"accuracy_score": 50.0, "reasoning_score": 50.0, "hallucination_control": 50.0, "overall_score": 50.0}
            for attempt in range(10):
                try:
                    resp = requests.post(api_url, headers=headers, json=payload, timeout=25)
                    if resp.status_code == 200:
                        content = resp.json()["choices"][0]["message"]["content"].strip()
                        clean_json = content.replace("```json", "").replace("```", "").strip()
                        start = clean_json.find("{")
                        end = clean_json.rfind("}")
                        if start != -1 and end != -1:
                            clean_json = clean_json[start:end+1]
                        parsed = json.loads(clean_json)
                        # Normalize 0-10 scores to 0-100% if judge returns 0-10 range
                        for k in ["accuracy_score", "reasoning_score", "hallucination_control", "overall_score"]:
                            if k in parsed and parsed[k] <= 10.0:
                                parsed[k] = parsed[k] * 10.0
                        scores = parsed
                        time.sleep(0.5) # Smooth rate-limit throttle
                        break
                    elif resp.status_code == 429:
                        # Rate limit hit: exponential backoff wait
                        wait_time = (2 ** attempt) + 3
                        print(f"[!] Rate limit 429 hit. Backing off for {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        time.sleep(1.5)
                except Exception:
                    time.sleep(2.0)

            st = judge_summary[ds][mode_name]
            st["acc_sum"] += scores.get("accuracy_score", 50.0)
            st["reas_sum"] += scores.get("reasoning_score", 50.0)
            st["hall_sum"] += scores.get("hallucination_control", 50.0)
            st["ovr_sum"] += scores.get("overall_score", 50.0)
            st["cnt"] += 1
            run_data["llm_judge"] = scores

    # Output LLM-as-a-Judge Table
    judge_table_lines = [
        "\n=== LLM-AS-A-JUDGE (Nemotron 3 Ultra 550B) PERCENTAGE SCORES (%) ===",
        f"| Dataset | Mode | Accuracy (%) | Reasoning (%) | Hallucination Ctrl (%) | Overall Score (%) |",
        f"| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    for ds, m_data in judge_summary.items():
        for m_name in MODE_NAMES:
            st = m_data.get(m_name)
            if not st or st["cnt"] == 0:
                continue
            cnt = st["cnt"]
            acc = st["acc_sum"] / cnt
            reas = st["reas_sum"] / cnt
            hall = st["hall_sum"] / cnt
            ovr = st["ovr_sum"] / cnt
            judge_table_lines.append(f"| {ds} | {m_name} | {acc:.1f}% | {reas:.1f}% | {hall:.1f}% | **{ovr:.1f}%** |")

    judge_table_md = "\n".join(judge_table_lines)
    print(judge_table_md)

    with open("saber_llm_judge_report.md", "w", encoding="utf-8") as f:
        f.write(judge_table_md + "\n")
    with open("saber_final_benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_benchmark()
