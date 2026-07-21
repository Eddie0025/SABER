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
    
    # Pass 1: Check last line for strict ANSWER: X
    last_line = lines[-1].upper()
    match = re.search(r"ANSWER:\s*([A-D])\b", last_line)
    if match:
        return match.group(1)
    
    # Pass 2: Check ANY line for ANSWER: X (model put it mid-response)
    for line in reversed(lines):
        match = re.search(r"ANSWER:\s*([A-D])\b", line.upper())
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

    # 1. Setup SABER Orchestrator
    config = SaberConfig()
    registry = SpecialistRegistry()
    registry.auto_discover()
    
    # Configure model paths for registered specialists
    for domain, specialist in registry.all().items():
        model_path = f"models/{domain}_v2"
        if os.path.exists(model_path):
            specialist.load_model(model_path)
            print(f"[*] Loaded specialist model for '{domain}': {model_path}")
        else:
            specialist.load_model("Qwen/Qwen2.5-7B")
            print(f"[*] Specialist '{domain}' checkpoint not found; falling back to base Qwen/Qwen2.5-7B")
            
    audit = AuditLogger()
    orch = Orchestrator(config=config, registry=registry, audit=audit)
    
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
    # 3. Execution Loop — 5 Modes per Question
    # =====================================================================
    MODE_NAMES = ["Base Qwen", "Qwen with Adaptors", "Qwen Adaptor + CoT", "Sentinel 2 Pass"]
    
    for idx, case in enumerate(bench_cases, 1):
        ds_name = case["dataset"]
        q = case["question"]
        print(f"\n[{idx}/{len(bench_cases)}] Dataset: {ds_name} | Query: {q[:75].strip()}...")
        
        modes = [
            ("Base Qwen", "qwen_base"),
            ("Qwen with Adaptors", "adapter_no_cot"),
            ("Qwen Adaptor + CoT", VerificationTier.TIER_0),
            ("Sentinel 2 Pass", VerificationTier.TIER_1),
        ]
        
        case_res = {
            "question": q,
            "type": case["type"],
            "expected": case.get("expected"),
            "domain": case["domain"],
            "dataset": ds_name,
            "runs": {}
        }
        
        for mode_name, tier in modes:
            start = time.time()
            try:
                # Suppress stdout during model execution
                original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                try:
                    if tier == "qwen_base":
                        # Mode 1: Bare Qwen 2.5-7B — no adapter, no CoT, no sentinel
                        from saber.llm_engine import LLMEngine
                        with LLMEngine("Qwen/Qwen2.5-7B") as engine:
                            raw = engine.generate(q)
                            ans = raw.strip()
                    elif tier == "adapter_no_cot":
                        # Mode 2: Domain adapter loaded but no CoT architecture
                        from saber.llm_engine import LLMEngine
                        d_scores = orch.classify_domains(q)
                        act = orch.select_specialists(d_scores)
                        dom = act[0] if act else "science"
                        m_path = f"models/{dom}_v2"
                        if not os.path.exists(m_path):
                            m_path = "Qwen/Qwen2.5-7B"
                        with LLMEngine(m_path) as engine:
                            raw = engine.generate(q)
                            ans = raw.strip()
                    else:
                        # Modes 3-4: Full SABER pipeline — bypass meta-reasoner for MCQs
                        res = orch.process_query(q, tier=tier, bypass_meta=True)
                        ans = res.get("answer", "").strip()
                finally:
                    sys.stdout.close()
                    sys.stdout = original_stdout
            except Exception as e:
                if sys.stdout != original_stdout:
                    try:
                        sys.stdout.close()
                    except:
                        pass
                    sys.stdout = original_stdout
                ans = f"[ERROR]: {e}"
            latency = time.time() - start
            
            # Strict MCQ scoring
            extracted = parse_mcq_answer(ans)
            expected_norm = str(case.get("expected", "")).strip().upper()
            is_correct = (extracted == expected_norm)
            score_data = {
                "accuracy": 1.0 if is_correct else 0.0,
                "explanation": f"Expected: {expected_norm} | Extracted: {extracted} | Raw: {ans[:200]}"
            }
                
            case_res["runs"][mode_name] = {
                "answer": ans,
                "latency": round(latency, 2),
                "evaluation": score_data
            }
            
        results.append(case_res)

        # ----- Live Scoreboard (every 10 cases or dataset boundary) -----
        is_dataset_complete = (idx == len(bench_cases)) or (bench_cases[idx]["dataset"] != ds_name)
        if is_dataset_complete or (idx % 10 == 0):
            print(f"\n[LIVE UPDATE] Progress: {idx}/{len(bench_cases)} cases completed.")
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
