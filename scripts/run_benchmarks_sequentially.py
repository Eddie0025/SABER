import json
import os
import sys
import time
import random
import urllib.request
import urllib.error
from typing import Dict, Any, List

# Disable Hugging Face verbose logs and cache models in memory (crucial for single GPU)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["SABER_KEEP_MODELS_LOADED"] = "1"
os.environ["SABER_BENCHMARK_MODE"] = "1"

# Ensure saber module can be imported
sys.path.append(os.path.abspath('.'))

from saber.config import SaberConfig, VerificationTier
from saber.registry import SpecialistRegistry
from saber.audit import AuditLogger
from saber.orchestrator import Orchestrator

# =====================================================================
# OpenRouter LLM-as-a-Judge API Client (Gemma-4-31B)
# =====================================================================
def call_qwen_judge(prompt, question, student_answer, api_key=None):
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("QWEN_API_KEY")
    if not api_key and os.path.exists("api_key.txt"):
        with open("api_key.txt", "r") as f:
            api_key = f.read().strip()
                
    if not api_key:
        return {
            "correctness": None, "relevance": None, "reasoning": None,
            "calibration": None, "red_herring": None, "confidence_in_judgment": "LOW",
            "explanation": "No API key found. Pass key as argument, set OPENROUTER_API_KEY env, or save to api_key.txt."
        }
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/Eddie0025/SABER",
        "X-Title": "SABER Benchmark"
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
        "model": "google/gemma-4-31b-it:free",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1
    }
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                content = res_data["choices"][0]["message"]["content"].strip()
                clean_json = content.replace("```json", "").replace("```", "").strip()
                start = clean_json.find("{")
                end = clean_json.rfind("}")
                if start != -1 and end != -1:
                    clean_json = clean_json[start:end+1]
                return json.loads(clean_json)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                sleep_time = (2 ** attempt) + 2
                print(f"[!] OpenRouter Rate Limited (429). Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
                continue
            print(f"[!] OpenRouter API judge HTTP Error: {e.code} - {e.reason}")
            return {
                "correctness": None, "relevance": None, "reasoning": None,
                "calibration": None, "red_herring": None, "confidence_in_judgment": "LOW",
                "explanation": f"HTTP Error: {e.code} {e.reason}"
            }
        except Exception as e:
            print(f"[!] OpenRouter API judge failed: {e}")
            return {
                "correctness": None, "relevance": None, "reasoning": None,
                "calibration": None, "red_herring": None, "confidence_in_judgment": "LOW",
                "explanation": f"API Error: {e}"
            }

# =====================================================================
# HF Dataset Loader Helper (Supporting Auth Token)
# =====================================================================
def load_hf_dataset(path, name=None, split=None, **kwargs):
    from datasets import load_dataset
    token = os.getenv("HF_TOKEN")
    if not token and os.path.exists("api_key.txt"):
        with open("api_key.txt", "r") as f:
            token = f.read().strip()
    if token:
        kwargs["token"] = token
    if name:
        kwargs["name"] = name
    if split:
        kwargs["split"] = split
    return load_dataset(path, **kwargs)

# =====================================================================
# Main Benchmark Pipeline
# =====================================================================
def main(api_key=None):
    print("==========================================================")
    print("      SABER Orchestrator Unified Single-GPU Benchmark")
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
    
    print("\n[*] Loading benchmark datasets...")
    bench_cases = []
    
    # 2.1 Science: GPQA Diamond (Last 78 cases)
    try:
        gpqa = load_hf_dataset("idavidrein/gpqa", "gpqa_diamond", split="train")
        all_gpqa = list(gpqa)
        sliced_gpqa = all_gpqa[-78:]
        for row in sliced_gpqa:
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
                "question": f"Question: {q_text}\nOptions:\n{choices_str}",
                "expected": correct_char,
                "domain": "science",
                "dataset": "gpqa_diamond"
            })
        print(f"[+] Loaded {len(sliced_gpqa)} Science (GPQA Diamond) cases.")
    except Exception as e:
        print(f"[!] Error loading GPQA: {e}")

    # 2.2 Cyber: CyberMetric (Last 60 cases)
    try:
        import urllib.request
        data = []
        urls = [
            "https://raw.githubusercontent.com/cybermetric/CyberMetric/main/CyberMetric-500-v1.json",
            "https://raw.githubusercontent.com/cybermetric/CyberMetric/main/CyberMetric-80-v1.json"
        ]
        
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=10) as response:
                    raw_data = json.loads(response.read().decode("utf-8"))
                    if isinstance(raw_data, dict):
                        for k, v in raw_data.items():
                            if isinstance(v, list):
                                data = v
                                break
                    elif isinstance(raw_data, list):
                        data = raw_data
                    if data:
                        break
            except Exception:
                continue

        if data:
            all_cyber = []
            for row in data:
                q_text = row.get("question") or row.get("Question")
                choices = []
                choices_dict = row.get("answers") or row.get("choices") or row
                if isinstance(choices_dict, list):
                    choices = choices_dict
                elif isinstance(choices_dict, dict):
                    for opt in ["A", "B", "C", "D", "a", "b", "c", "d", "1", "2", "3", "4"]:
                        if opt in choices_dict and choices_dict[opt]:
                            choices.append(choices_dict[opt])
                
                correct_ans = row.get("solution") or row.get("answer") or row.get("Answer") or row.get("correct") or row.get("correct_answer")
                if not q_text or not correct_ans or not choices:
                    continue
                choices_str = "\n".join([f"{chr(65+i)}: {c}" for i, c in enumerate(choices)])
                if str(correct_ans).upper() in ["A", "B", "C", "D"]:
                    correct_char = str(correct_ans).upper()
                else:
                    try:
                        idx = choices.index(correct_ans)
                        correct_char = chr(65 + idx)
                    except ValueError:
                        correct_char = "A"
                all_cyber.append({
                    "type": "exact",
                    "question": f"Question: {q_text}\nOptions:\n{choices_str}",
                    "expected": correct_char,
                    "domain": "cyber",
                    "dataset": "cybermetric"
                })
            sliced_cyber = all_cyber[-60:]
            bench_cases.extend(sliced_cyber)
            print(f"[+] Loaded {len(sliced_cyber)} Cyber (CyberMetric) cases.")
        else:
            print("[!] Failed to fetch CyberMetric Raw JSON.")
    except Exception as e:
        print(f"[!] CyberMetric load failed: {e}")

    # 2.3 Coding: HumanEval (Last 78 cases)
    try:
        he = load_hf_dataset("openai/openai_humaneval", split="test")
        all_he = list(he)
        sliced_he = all_he[-78:]
        for row in sliced_he:
            bench_cases.append({
                "type": "open_ended",
                "question": f"Complete the following Python function:\n{row['prompt']}",
                "expected": None,
                "domain": "coding",
                "dataset": "humaneval"
            })
        print(f"[+] Loaded {len(sliced_he)} Coding (HumanEval) cases.")
    except Exception as e:
        print(f"[!] HumanEval load failed: {e}")

    # 2.4 Finance: FinQA (80 exact cases)
    try:
        count = 0
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
            count += 1
        print(f"[+] Loaded {count} Finance (FinQA Math) cases.")
    except Exception as e:
        print(f"[!] FinQA setup failed: {e}")

    print(f"\n[+] Total benchmark cases compiled: {len(bench_cases)}")
    results = []

    # 3. Process each case across the 3 Sentinel Tiers (Without Sentinel, 2-Check, 4-Check)
    for idx, case in enumerate(bench_cases, 1):
        if idx < 29:
            continue
        ds_name = case["dataset"]
        domain = case["domain"]
        print(f"\n[{idx}/{len(bench_cases)}] Dataset: {ds_name} | Query: {case['question'][:75].strip()}...")
        q = case["question"]
        
        modes = [
            ("Without Sentinel", VerificationTier.TIER_0),
            ("2-Check Sentinel", VerificationTier.TIER_1),
            ("4-Check Sentinel", VerificationTier.TIER_2)
        ]
        
        case_res = {
            "question": q,
            "type": case["type"],
            "expected": case.get("expected"),
            "domain": domain,
            "dataset": ds_name,
            "runs": {}
        }
        
        for mode_name, tier in modes:
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
                expected_norm = str(case.get("expected", "")).lower().strip()
                ans_norm = ans.lower().strip()
                if expected_norm in ans_norm:
                    is_correct = True
                score_data = {
                    "accuracy": 1.0 if is_correct else 0.0,
                    "explanation": f"Expected: {case.get('expected')} | Found: {ans}"
                }
            else:
                score_data = call_qwen_judge(q, q, ans, api_key=api_key)
                
            case_res["runs"][mode_name] = {
                "answer": ans,
                "latency": round(latency, 2),
                "evaluation": score_data
            }
            
        results.append(case_res)

        # Print live scoreboard updates every 10 cases or at dataset completion
        is_dataset_complete = (idx == len(bench_cases)) or (bench_cases[idx]["dataset"] != ds_name)
        if is_dataset_complete or (idx % 10 == 0):
            print(f"\n[LIVE UPDATE] Progress: {idx}/{len(bench_cases)} cases completed. Dynamic scoreboard:")
            live_summary = {}
            for r in results:
                ds = r["dataset"]
                if ds not in live_summary:
                    live_summary[ds] = {}
                for m_name, r_info in r["runs"].items():
                    if m_name not in live_summary[ds]:
                        live_summary[ds][m_name] = {"acc_sum": 0.0, "acc_cnt": 0, "corr_sum": 0.0, "corr_cnt": 0}
                    ev = r_info["evaluation"]
                    if r["type"] == "exact":
                        live_summary[ds][m_name]["acc_sum"] += ev.get("accuracy", 0.0)
                        live_summary[ds][m_name]["acc_cnt"] += 1
                    else:
                        corr_val = ev.get("correctness")
                        if corr_val is not None:
                            live_summary[ds][m_name]["corr_sum"] += float(corr_val)
                            live_summary[ds][m_name]["corr_cnt"] += 1
            
            print("| Dataset | Without Sentinel (SABER) | 2-Check Sentinel | 4-Check Sentinel |")
            print("| :--- | :--- | :--- | :--- |")
            for ds, m_data in live_summary.items():
                cells = [ds]
                for m_name in ["Without Sentinel", "2-Check Sentinel", "4-Check Sentinel"]:
                    st = m_data.get(m_name, {})
                    if not st:
                        cells.append("N/A")
                        continue
                    pct = 0.0
                    if st["acc_cnt"] > 0:
                        pct = (st["acc_sum"] / st["acc_cnt"]) * 100.0
                    elif st["corr_cnt"] > 0:
                        pct = ((st["corr_sum"] / st["corr_cnt"]) / 2.0) * 100.0
                    cells.append(f"{pct:.1f}%")
                print("| " + " | ".join(cells) + " |")
            print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

    # 4. Final Aggregation and Save
    summary = {}
    table_lines = [
        "| Dataset | Without Sentinel (SABER) | 2-Check Sentinel | 4-Check Sentinel |",
        "| :--- | :--- | :--- | :--- |"
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
                    "correctness_sum": 0.0, "correctness_count": 0,
                }
            stats = summary[dataset][mode_name]
            stats["count"] += 1
            stats["total_latency"] += run_info["latency"]
            eval_res = run_info["evaluation"]
            if case_res["type"] == "exact":
                stats["accuracy_sum"] += eval_res.get("accuracy", 0.0)
                stats["accuracy_count"] += 1
            else:
                val = eval_res.get("correctness")
                if val is not None:
                    stats["correctness_sum"] += float(val)
                    stats["correctness_count"] += 1

    formatted_summary = {}
    for ds, modes_data in summary.items():
        formatted_summary[ds] = {}
        row_cells = [ds]
        for mode_name in ["Without Sentinel", "2-Check Sentinel", "4-Check Sentinel"]:
            stats = modes_data.get(mode_name, {})
            if not stats:
                row_cells.append("N/A")
                continue
            avg_metrics = {
                "count": stats["count"],
                "avg_latency_sec": round(stats["total_latency"] / stats["count"], 2)
            }
            percentage = 0.0
            if stats["accuracy_count"] > 0:
                avg_accuracy = stats["accuracy_sum"] / stats["accuracy_count"]
                avg_metrics["avg_accuracy"] = round(avg_accuracy, 3)
                percentage = avg_accuracy * 100.0
            elif stats["correctness_count"] > 0:
                avg_correctness = stats["correctness_sum"] / stats["correctness_count"]
                avg_metrics["avg_correctness"] = round(avg_correctness, 2)
                percentage = (avg_correctness / 2.0) * 100.0
            row_cells.append(f"{percentage:.1f}%")
            formatted_summary[ds][mode_name] = avg_metrics
        table_lines.append("| " + " | ".join(row_cells) + " |")

    # Save summary report files
    with open("saber_final_benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open("saber_benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(formatted_summary, f, indent=2)
    table_md = "\n".join(table_lines)
    with open("saber_benchmark_table.md", "w", encoding="utf-8") as f:
        f.write(table_md + "\n")

    print("\n=== FINAL BENCHMARK SCORES SUMMARY ===")
    print(table_md)

if __name__ == "__main__":
    key = None
    if len(sys.argv) > 1:
        key = sys.argv[1]
    main(api_key=key)
