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
def parse_mcq_answer(raw_answer):
    """Extract the answer letter using a multi-pass parser.
    
    Priority order:
      1. Last line: ANSWER: X (strict format)
      2. Any line: ANSWER: X
      3. Common patterns: 'The answer is X', 'correct answer is X'
      4. Last line: standalone letter (A/B/C/D)
    
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
    
    return None

# =====================================================================
# Main Benchmark Pipeline — MCQ Only
# =====================================================================
def run_benchmark():
    print("==========================================================")
    print("      SABER MCQ Benchmark — 5-Mode Ablation Study")
    print("==========================================================\n")

    # 1. Setup Configuration
    config = SaberConfig()
    
    # 2. Collect MCQ Benchmark Questions
    print("\n[*] Loading MCQ benchmark datasets...")
    bench_cases = []
    
    # ---------------------------------------------------------------
    # 2.1 Science: GPQA Diamond (198 cases)
    # ---------------------------------------------------------------
    try:
        gpqa = load_hf_dataset("idavidrein/gpqa", "gpqa_diamond", split="train")
        for row in gpqa:
            corr = row.get("correct_answer") or row.get("Correct Answer")
            inc1 = row.get("incorrect_answer1") or row.get("Incorrect Answer 1")
            inc2 = row.get("incorrect_answer2") or row.get("Incorrect Answer 2")
            inc3 = row.get("incorrect_answer3") or row.get("Incorrect Answer 3")
            q_text = row.get("question") or row.get("Question")
            
            if not corr or not q_text:
                continue
                
            choices = [corr, inc1, inc2, inc3]
            random.seed(42)
            random.shuffle(choices)
            choices_str = "\n".join([f"{chr(65+i)}: {c}" for i, c in enumerate(choices)])
            correct_char = chr(65 + choices.index(corr))
            
            bench_cases.append({
                "type": "exact",
                "question": build_mcq_prompt(q_text, choices_str),
                "expected": correct_char,
                "domain": "science",
                "dataset": "gpqa_diamond"
            })
        print(f"[+] Loaded {sum(1 for c in bench_cases if c['dataset'] == 'gpqa_diamond')} Science (GPQA Diamond) cases.")
    except Exception as e:
        print(f"[!] Critical Error loading GPQA: {e}")
        raise e

    # ---------------------------------------------------------------
    # 2.2 Science: MMLU-Pro (300 cases stratified)
    # ---------------------------------------------------------------
    try:
        mmlu_pro = load_hf_dataset("TIGER-Lab/MMLU-Pro", split="test[:300]")
        count_before = len(bench_cases)
        for row in mmlu_pro:
            choices = row.get("options", [])
            choices_str = "\n".join([f"{chr(65+i)}: {c}" for i, c in enumerate(choices)])
            bench_cases.append({
                "type": "exact",
                "question": build_mcq_prompt(row['question'], choices_str),
                "expected": row.get("answer", ""),
                "domain": "science",
                "dataset": "mmlu_pro"
            })
        print(f"[+] Loaded {len(bench_cases) - count_before} Science (MMLU-Pro) cases.")
    except Exception as e:
        print(f"[!] MMLU-Pro load failed: {e}")

    # ---------------------------------------------------------------
    # 2.3 Finance: FinQA Math (80 exact cases)
    # ---------------------------------------------------------------
    random.seed(42)
    for i in range(80):
        rev = random.randint(100, 5000)
        cogs = random.randint(50, int(rev * 0.6))
        gp = rev - cogs
        bench_cases.append({
            "type": "exact",
            "question": f"Context: Revenue: ${rev}M, COGS: ${cogs}M.\nQuestion: Calculate Gross Profit.",
            "expected": f"{gp}",
            "domain": "finance",
            "dataset": "finqa"
        })
    print(f"[+] Loaded 80 Finance (FinQA Math) cases.")

    # ---------------------------------------------------------------
    # 2.4 Cyber: SecBench (100 cases)
    # ---------------------------------------------------------------
    try:
        secbench = load_hf_dataset("secbench-hf/SecBench", data_files="data/MCQs_2730.jsonl", split="train")
        all_cyber = []
        for row in secbench:
            if row.get("language") == "English":
                q_text = row.get("question")
                choices = list(row.get("answers", []))
                label_char = row.get("label", "").upper().strip()
                if len(choices) != 4 or not label_char or not q_text:
                    continue
                if len(label_char) != 1 or label_char not in ["A", "B", "C", "D"]:
                    continue
                correct_idx = ord(label_char) - 65
                if not (0 <= correct_idx < 4):
                    continue
                correct_ans = choices[correct_idx]
                
                random.seed(42)
                random.shuffle(choices)
                choices_str = "\n".join([f"{chr(65+i)}: {c}" for i, c in enumerate(choices)])
                correct_char = chr(65 + choices.index(correct_ans))
                
                all_cyber.append({
                    "type": "exact",
                    "question": build_mcq_prompt(q_text, choices_str),
                    "expected": correct_char,
                    "domain": "cyber",
                    "dataset": "secbench"
                })
            
        if all_cyber:
            sliced_cyber = all_cyber[-100:]
            bench_cases.extend(sliced_cyber)
            print(f"[+] Loaded {len(sliced_cyber)} Cyber (SecBench) cases.")
        else:
            print("[!] Failed to load SecBench cases.")
    except Exception as e:
        print(f"[!] SecBench load failed: {e}")

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
    
    from saber.config import SaberConfig
    from saber.llm_engine import LLMEngine
    from saber.signal import Signal, SignalType
    from saber.sentinel import Sentinel
    import importlib
    import gc
    
    config = SaberConfig()

    for ds_name, cases in datasets.items():
        domain = cases[0]["domain"]
        print(f"\n==========================================================")
        print(f"[*] Processing Dataset: {ds_name} (Domain: {domain}) | {len(cases)} cases")
        print(f"==========================================================\n")
        
        # 1. Load Specialist for this dataset
        specialist_class_name = f"{domain.capitalize()}Specialist"
        try:
            mod = importlib.import_module(f"saber.specialists.{domain}")
            specialist_cls = getattr(mod, specialist_class_name)
            specialist = specialist_cls()
            m_path = f"models/{domain}_v2"
            if not os.path.exists(m_path):
                m_path = config.base_model
            specialist.load_model(m_path)
            print(f"[*] Loaded {specialist_class_name} with model: {m_path}")
        except Exception as e:
            print(f"[!] Failed to load specialist for {domain}: {e}")
            continue
            
        sentinel = Sentinel()

        for idx_in_ds, case in enumerate(cases, 1):
            global_idx += 1
            q = case["question"]
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
                
            extracted1 = parse_mcq_answer(ans1)
            expected_norm = str(case.get("expected", "")).strip().upper()
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
                
            extracted2 = parse_mcq_answer(ans2)
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
                
            extracted3 = parse_mcq_answer(ans3)
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
                
            extracted4 = parse_mcq_answer(ans4)
            case_res["runs"]["Sentinel 2 Pass"] = {
                "answer": ans4, "latency": round(time.time()-start, 2),
                "evaluation": {"accuracy": 1.0 if extracted4 == expected_norm else 0.0}
            }
            
            results.append(case_res)

            # ----- Live Scoreboard (every 10 cases or dataset boundary) -----
            is_dataset_complete = (idx_in_ds == len(cases))
            if is_dataset_complete or (idx_in_ds % 10 == 0):
                print(f"\n[LIVE UPDATE] Progress: {global_idx}/{len(bench_cases)} cases completed.")
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
                print(f"| :--- | {' | '.join([':---'] * 5)} |")
                for ds, m_data in live_summary.items():
                    cells = [ds]
                    for m_name in MODE_NAMES:
                        st = m_data.get(m_name, {})
                        if not st or st["acc_cnt"] == 0:
                            cells.append("N/A")
                            continue
                        pct = (st["acc_sum"] / st["acc_cnt"]) * 100.0
                        cells.append(f"{pct:.1f}%")
                    print("| " + " | ".join(cells) + " |")
                print("-" * 60)
                
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

if __name__ == "__main__":
    run_benchmark()
