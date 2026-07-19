import json
import os
import sys
import time
import urllib.request
import urllib.error
import random

# Ensure saber module can be imported
sys.path.append(os.path.abspath('.'))

from saber.config import SaberConfig, VerificationTier
from saber.registry import SpecialistRegistry
from saber.audit import AuditLogger
from saber.orchestrator import Orchestrator

# =====================================================================
# Qwen LLM-as-a-Judge API Client (OpenAI / DashScope compatible)
# =====================================================================
def call_qwen_judge(prompt, question, student_answer, api_key=None):
    if not api_key:
        api_key = os.getenv("QWEN_API_KEY")
    
    if not api_key:
        return {
            "correctness": None, "relevance": None, "reasoning": None,
            "calibration": None, "red_herring": None, "confidence_in_judgment": "LOW",
            "explanation": "No QWEN_API_KEY provided; grading skipped."
        }

    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    system_prompt = (
        "You are an expert AI judge. Grade the student's answer based on the query and the following rubric.\n"
        "Rubric:\n"
        "1. Correctness (0-2):\n"
        "   2 = Factually accurate, no errors\n"
        "   1 = Substantially correct but contains minor errors/incomplete details\n"
        "   0 = Factual error, fabrication, or wrong conclusion\n"
        "2. Relevance / Answers-the-question (0-2):\n"
        "   2 = Directly and fully addresses the question asked\n"
        "   1 = Partially addresses the question but misses part\n"
        "   0 = Does not address the question asked\n"
        "3. Reasoning quality (0-2) [Only score for Differentiation, Mechanism, Multi-step; else null]:\n"
        "   2 = Clear, logical, step-by-step reasoning\n"
        "   1 = Correct/reasonable conclusion but shallow reasoning\n"
        "   0 = No real reasoning or confused/circular reasoning\n"
        "4. Calibration (0-2) [Only score for single precise expected fact/formula/term; else null]:\n"
        "   2 = States fact confidently and correctly, OR appropriately expresses uncertainty if genuinely unsure\n"
        "   1 = States plausible-but-imprecise answer\n"
        "   0 = Confidently states fabricated/wrong specific fact\n"
        "5. Red-herring resistance (0-2) [Only score for queries containing a distractor; else null]:\n"
        "   2 = Identifies the distractor as irrelevant and explains why\n"
        "   1 = Doesn't get misled but doesn't explicitly address distractor\n"
        "   0 = Incorporates the distractor as relevant, changing the answer incorrectly\n\n"
        "Output strictly valid JSON with no markdown tags matching this schema:\n"
        "{\n"
        "  \"correctness\": int or null,\n"
        "  \"relevance\": int or null,\n"
        "  \"reasoning\": int or null,\n"
        "  \"calibration\": int or null,\n"
        "  \"red_herring\": int or null,\n"
        "  \"confidence_in_judgment\": \"HIGH\" or \"LOW\",\n"
        "  \"explanation\": \"detailed breakdown...\"\n"
        "}"
    )
    
    user_content = (
        f"Question: {question}\n\n"
        f"Student Answer:\n{student_answer}\n"
    )
    
    data = {
        "model": "qwen2.5-72b-instruct", # DashScope compatible Qwen 72B model name
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            content = res_data["choices"][0]["message"]["content"].strip()
            # Clean JSON formatting wrappers
            clean_json = content.replace("```json", "").replace("```", "").strip()
            start = clean_json.find("{")
            end = clean_json.rfind("}")
            if start != -1 and end != -1:
                clean_json = clean_json[start:end+1]
            return json.loads(clean_json)
    except Exception as e:
        print(f"[!] Qwen API judge failed: {e}")
        return {
            "correctness": None, "relevance": None, "reasoning": None,
            "calibration": None, "red_herring": None, "confidence_in_judgment": "LOW",
            "explanation": f"API Error: {e}"
        }

# =====================================================================
# Main Benchmark Pipeline
# =====================================================================
def run_benchmark(api_key=None):
    from datasets import load_dataset
    
    # 1. Setup SABER Orchestrator
    config = SaberConfig()
    registry = SpecialistRegistry()
    registry.auto_discover()
    audit = AuditLogger()
    orch = Orchestrator(config=config, registry=registry, audit=audit)
    
    # 2. Collect Benchmark Questions
    print("\n[*] Loading benchmark datasets...")
    bench_cases = []
    
    # 2.1 MMLU Subsets (Exact choice matching)
    try:
        mmlu = load_dataset("cais/mmlu", "clinical_knowledge", split="test[:20]")
        for row in mmlu:
            choices = row["choices"]
            ans_idx = row["answer"]
            correct_ans = chr(65 + ans_idx)
            choices_str = "\n".join([f"{chr(65+i)}: {c}" for i, c in enumerate(choices)])
            
            bench_cases.append({
                "type": "exact",
                "question": f"Question: {row['question']}\nOptions:\n{choices_str}",
                "expected": correct_ans,
                "domain": "medical",
                "dataset": "mmlu"
            })
    except Exception as e:
        print(f"[!] MMLU benchmark load failed: {e}")
        
    # 2.2 SciQ Subsets (Exact choice matching)
    try:
        sciq = load_dataset("allenai/sciq", split="test[:20]")
        for row in sciq:
            bench_cases.append({
                "type": "exact",
                "question": row["question"],
                "expected": row["correct_answer"],
                "domain": "science",
                "dataset": "sciq"
            })
    except Exception as e:
        print(f"[!] SciQ benchmark load failed: {e}")

    # 2.3 MedMCQA Subsets (Exact choice matching)
    try:
        medmcqa = load_dataset("openlifescienceai/medmcqa", split="validation[:20]")
        for row in medmcqa:
            cop_idx = row["cop"]
            cop_char = chr(65 + cop_idx) if 0 <= cop_idx < 4 else str(cop_idx)
            text = f"Question: {row['question']}\nOptions:\nA: {row['opa']}\nB: {row['opb']}\nC: {row['opc']}\nD: {row['opd']}"
            bench_cases.append({
                "type": "exact",
                "question": text,
                "expected": cop_char,
                "domain": "medical",
                "dataset": "medmcqa"
            })
    except Exception as e:
        print(f"[!] MedMCQA benchmark load failed: {e}")

    # 2.4 Open-Ended Questions (from 30-case evaluation suites)
    eval_scripts = ["eval_medical_30.py", "eval_cyber_30.py", "eval_science_30.py", "eval_coding_30.py", "eval_architecture_30.py", "eval_finance_30.py", "eval_meta_reasoner_30.py"]
    for script in eval_scripts:
        script_path = os.path.join("scripts", script)
        if os.path.exists(script_path):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("eval_script", script_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                cases = getattr(module, "TEST_CASES")
                domain_name = script.replace("eval_", "").replace("_30.py", "")
                for c in cases[:5]: # Take first 5 cases from each domain for benchmark speed
                    bench_cases.append({
                        "type": "open_ended",
                        "question": c["question"],
                        "expected": None,
                        "domain": domain_name,
                        "dataset": f"eval_{domain_name}"
                    })
            except Exception as e:
                print(f"[!] Error loading {script}: {e}")

    print(f"Loaded {len(bench_cases)} total benchmark cases.")
    results = []

    # 3. Process each case across the 3 Sentinel Tiers
    for idx, case in enumerate(bench_cases, 1):
        print(f"\n[{idx}/{len(bench_cases)}] Running query: {case['question'][:80]}...")
        q = case["question"]
        
        # Output Modes
        modes = [
            ("Without Sentinel", VerificationTier.TIER_0),
            ("2-Check Sentinel", VerificationTier.TIER_1),
            ("4-Check Sentinel", VerificationTier.TIER_2)
        ]
        
        case_res = {
            "question": q,
            "type": case["type"],
            "expected": case["expected"],
            "domain": case["domain"],
            "dataset": case["dataset"],
            "runs": {}
        }
        
        for mode_name, tier in modes:
            print(f"  -> Mode: {mode_name}")
            start = time.time()
            try:
                res = orch.process_query(q, tier=tier)
                ans = res.get("answer", "").strip()
            except Exception as e:
                ans = f"[ERROR]: {e}"
            latency = time.time() - start
            
            score_data = {}
            if case["type"] == "exact":
                is_correct = False
                expected_norm = case["expected"].lower().strip()
                ans_norm = ans.lower().strip()
                if expected_norm in ans_norm:
                    is_correct = True
                score_data = {
                    "accuracy": 1.0 if is_correct else 0.0,
                    "explanation": f"Expected: {case['expected']} | Found: {ans}"
                }
            else:
                score_data = call_qwen_judge(q, q, ans, api_key=api_key)
                
            case_res["runs"][mode_name] = {
                "answer": ans,
                "latency": round(latency, 2),
                "evaluation": score_data
            }
            
        results.append(case_res)

    # 4. Aggregate and compute performance stats per dataset & mode
    summary = {}
    for case_res in results:
        dataset = case_res["dataset"]
        if dataset not in summary:
            summary[dataset] = {}
            
        for mode_name, run_info in case_res["runs"].items():
            if mode_name not in summary[dataset]:
                summary[dataset][mode_name] = {
                    "count": 0, "total_latency": 0.0,
                    "accuracy_sum": 0.0, "accuracy_count": 0,
                    "correctness_sum": 0.0, "correctness_count": 0,
                    "relevance_sum": 0.0, "relevance_count": 0,
                    "reasoning_sum": 0.0, "reasoning_count": 0,
                    "calibration_sum": 0.0, "calibration_count": 0,
                    "red_herring_sum": 0.0, "red_herring_count": 0
                }
            
            stats = summary[dataset][mode_name]
            stats["count"] += 1
            stats["total_latency"] += run_info["latency"]
            
            eval_res = run_info["evaluation"]
            if case_res["type"] == "exact":
                stats["accuracy_sum"] += eval_res.get("accuracy", 0.0)
                stats["accuracy_count"] += 1
            else:
                for metric in ["correctness", "relevance", "reasoning", "calibration", "red_herring"]:
                    val = eval_res.get(metric)
                    if val is not None:
                        stats[f"{metric}_sum"] += float(val)
                        stats[f"{metric}_count"] += 1

    # Format the aggregated metrics cleanly
    formatted_summary = {}
    for ds, modes_data in summary.items():
        formatted_summary[ds] = {}
        for mode_name, stats in modes_data.items():
            avg_metrics = {
                "count": stats["count"],
                "avg_latency_sec": round(stats["total_latency"] / stats["count"], 2)
            }
            if stats["accuracy_count"] > 0:
                avg_metrics["avg_accuracy"] = round(stats["accuracy_sum"] / stats["accuracy_count"], 3)
            
            for m in ["correctness", "relevance", "reasoning", "calibration", "red_herring"]:
                cnt = stats[f"{m}_count"]
                if cnt > 0:
                    avg_metrics[f"avg_{m}"] = round(stats[f"{m}_sum"] / cnt, 2)
            formatted_summary[ds][mode_name] = avg_metrics

    # Save outputs
    with open("saber_final_benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    with open("saber_benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(formatted_summary, f, indent=2)

    print("\n=== BENCHMARK SUMMARY (BY DATASET & MODE) ===")
    print(json.dumps(formatted_summary, indent=2))
    
    print(f"\n=========================================================")
    print(f" Benchmark Completed! Reports saved:")
    print(f" - Detailed: saber_final_benchmark_report.json")
    print(f" - Aggregated: saber_benchmark_summary.json")
    print(f"=========================================================")

if __name__ == "__main__":
    key = None
    if len(sys.argv) > 1:
        key = sys.argv[1]
    run_benchmark(api_key=key)
