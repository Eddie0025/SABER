# -*- coding: utf-8 -*-
"""saber.benchmark

Benchmark execution framework for SABER v2.0.
Runs questions through 5 distinct architectural modes to measure the exact
impact of Multi-Specialist routing and Verification.

Modes
-----
MODE_BASE           — Raw base model, zero-shot.
MODE_SELF_CRITIQUE  — Base model + self-reflection pass.
MODE_SPECIALIST     — Specialist model only, no verification.
MODE_SPECIALIST_VERIFY — Specialist + Sentinel verification loop.
MODE_SABER          — Full system: routing, multi-specialist, verification.

Output
------
CSV with columns: question_id, mode, accuracy, flags_raised,
corrections_applied, latency, tokens_consumed, final_confidence.
"""

import json
import time
import csv
import os
from enum import Enum
from typing import Dict, Any, List, Optional

from saber.config import SaberConfig, VerificationTier
from saber.meta_reasoner import MetaReasoner
from saber.registry import SpecialistRegistry
from saber.audit import AuditLogger
from saber.sentinel import Sentinel
from saber.signal import Signal, SignalType


class EvalMode(str, Enum):
    MODE_BASE = "MODE_BASE"
    MODE_SELF_CRITIQUE = "MODE_SELF_CRITIQUE"
    MODE_SPECIALIST = "MODE_SPECIALIST"
    MODE_SPECIALIST_VERIFY = "MODE_SPECIALIST_VERIFY"
    MODE_SABER = "MODE_SABER"


class BenchmarkEngine:
    """Orchestrates benchmark runs across all evaluation modes."""

    def __init__(self, config: SaberConfig, registry: SpecialistRegistry):
        self.config = config
        self.registry = registry
        self.audit = AuditLogger("data/benchmark/audit_benchmark.jsonl")
        self.meta_reasoner = MetaReasoner(config, registry, self.audit)
        self.sentinel = Sentinel()
        self.base_model = config.base_model

    # ------------------------------------------------------------------
    # Mode Implementations
    # ------------------------------------------------------------------

    def run_mode_base(self, question: str) -> Dict[str, Any]:
        """MODE_BASE: Direct zero-shot query to the base model."""
        from saber.llm_engine import LLMEngine
        start = time.time()
        with LLMEngine(self.base_model) as engine:
            ans = engine.generate(question, system_prompt="Answer the question directly and thoroughly.")
        latency = time.time() - start
        return {"answer": ans, "flags": 0, "corrections": 0, "latency": latency, "conf": 1.0}

    def run_mode_self_critique(self, question: str) -> Dict[str, Any]:
        """MODE_SELF_CRITIQUE: Base model answer + self-reflection rewrite."""
        from saber.llm_engine import LLMEngine
        start = time.time()
        with LLMEngine(self.base_model) as engine:
            initial_answer = engine.generate(
                question,
                system_prompt="Answer the question thoroughly."
            )
            critique_prompt = (
                f"Question: {question}\n\n"
                f"Initial Answer:\n{initial_answer}\n\n"
                "Critically review the above answer. Identify any factual errors, "
                "reasoning gaps, or missing evidence. Then produce a corrected, "
                "improved version of the answer. Output ONLY the improved answer."
            )
            revised_answer = engine.generate(
                critique_prompt,
                system_prompt="You are a critical reviewer. Fix all errors."
            )
        latency = time.time() - start
        return {"answer": revised_answer, "flags": 1, "corrections": 1, "latency": latency, "conf": 0.95}

    def run_mode_specialist(self, question: str, domain: str) -> Dict[str, Any]:
        """MODE_SPECIALIST: Direct query to the domain specialist, no verification."""
        from saber.llm_engine import LLMEngine
        start = time.time()
        specialist = self.registry.get(domain)
        if not specialist:
            # Fallback to base model if specialist not available
            return self.run_mode_base(question)

        model_name = getattr(specialist, "model_name", self.base_model)
        with LLMEngine(model_name) as engine:
            ans = engine.generate(
                question,
                system_prompt=f"You are an expert {domain} specialist. Provide a thorough, evidence-based answer."
            )
        latency = time.time() - start
        return {"answer": ans, "flags": 0, "corrections": 0, "latency": latency, "conf": 0.9}

    def run_mode_specialist_verify(self, question: str, domain: str, qid: str) -> Dict[str, Any]:
        """MODE_SPECIALIST_VERIFY: Specialist answer + Sentinel verification loop."""
        from saber.llm_engine import LLMEngine
        start = time.time()

        specialist = self.registry.get(domain)
        if not specialist:
            return self.run_mode_base(question)

        model_name = getattr(specialist, "model_name", self.base_model)
        with LLMEngine(model_name) as engine:
            ans = engine.generate(
                question,
                system_prompt=f"You are an expert {domain} specialist. Provide a thorough, evidence-based answer."
            )

        # Create a fake output signal for verification
        out_sig = Signal(
            signal_type=SignalType.OUTPUT_SIGNAL,
            query_id=qid,
            source_id=f"SPEC-{domain.upper()}",
            target_id="MANAGER",
            payload={"claims": [{"statement": ans, "confidence": 0.9}]}
        ).freeze_and_hash()

        # Run Sentinel verification
        flags_raised = 0
        corrections = 0
        compiled = ans

        for cycle in range(2):  # Max 2 verification passes
            ver_res = self.sentinel.verify_interpretation(
                specialist_domain=domain,
                original_signal=out_sig,
                compiled_text=compiled,
            )
            if ver_res.signal_type == SignalType.FLAG_SIGNAL:
                flags_raised += 1
                # Simple rewrite using base model
                fix = ver_res.payload.get("proposed_fix", "")
                reasoning = ver_res.payload.get("reasoning", "")
                with LLMEngine(self.base_model) as rewrite_engine:
                    compiled = rewrite_engine.generate(
                        f"Original: {compiled}\n\nError found: {reasoning}\nFix: {fix}\n\nRewrite the answer fixing this error.",
                        system_prompt="You are a fact-checker. Rewrite the answer to fix the identified error."
                    )
                corrections += 1
            else:
                break  # GREEN_CHIT

        latency = time.time() - start
        return {"answer": compiled, "flags": flags_raised, "corrections": corrections, "latency": latency, "conf": 0.85}

    def run_mode_saber(self, question: str, qid: str, domain: str) -> Dict[str, Any]:
        """MODE_SABER: Full system — routing, multi-specialist, verification."""
        start = time.time()

        # Determine which domains to activate
        domains_to_activate = []
        if domain == "cross_domain":
            # Activate all available specialists
            for d in ["cyber", "science", "medical"]:
                if self.registry.get(d):
                    domains_to_activate.append(d)
            if not domains_to_activate:
                domains_to_activate = ["cyber", "science", "medical"]
        else:
            domains_to_activate = [domain]

        res = self.meta_reasoner.execute(
            query=question,
            query_id=qid,
            activated_domains=domains_to_activate,
            verification_tier=VerificationTier.TIER_3,
        )
        latency = time.time() - start
        return {
            "answer": res.get("answer", ""),
            "flags": res.get("total_flags_raised", 0),
            "corrections": res.get("revision_count", 0),
            "latency": latency,
            "conf": res.get("confidence", 0.0),
        }

    # ------------------------------------------------------------------
    # Accuracy Scoring (LLM-as-a-Judge)
    # ------------------------------------------------------------------

    def score_accuracy(self, question: str, answer: str, ground_truth: str, reasoning_points: List[str]) -> float:
        """Use the base model as a judge to score the answer against ground truth."""
        from saber.llm_engine import LLMEngine

        points_str = "\n".join(f"- {p}" for p in reasoning_points)
        prompt = (
            f"Question: {question}\n\n"
            f"Ground Truth Answer:\n{ground_truth}\n\n"
            f"Key Reasoning Points that must be addressed:\n{points_str}\n\n"
            f"Candidate Answer:\n{answer}\n\n"
            "Score the Candidate Answer from 0.0 to 1.0 based on:\n"
            "- Factual correctness relative to the ground truth\n"
            "- Coverage of the key reasoning points\n"
            "- Logical coherence\n\n"
            "Reply with ONLY a single decimal number between 0.0 and 1.0."
        )
        try:
            with LLMEngine(self.base_model) as engine:
                score_str = engine.generate(prompt, system_prompt="You are an impartial accuracy scorer.").strip()
                # Extract the first float from the response
                import re
                match = re.search(r"(\d+\.?\d*)", score_str)
                if match:
                    return min(1.0, max(0.0, float(match.group(1))))
        except Exception:
            pass
        return 0.0

    # ------------------------------------------------------------------
    # Main Evaluation Loop
    # ------------------------------------------------------------------

    def evaluate(self, benchmark_file: str, output_csv: str, score_accuracy: bool = False, domain: Optional[str] = None):
        """Run benchmark questions through all 5 modes.

        Parameters
        ----------
        benchmark_file : str
            Path to the SABER_BENCHMARK_v1 JSONL file.
        output_csv : str
            Path to write the results CSV.
        score_accuracy : bool
            If True, use LLM-as-a-Judge to score accuracy (slow).
        domain : Optional[str]
            If specified, only evaluate questions for this domain.
        """
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        results = []

        with open(benchmark_file, "r") as f:
            questions = [json.loads(line) for line in f if line.strip()]

        if domain:
            questions = [q for q in questions if q.get("domain") == domain]
            print(f"[benchmark] Filtered to domain: {domain} ({len(questions)} questions found)")

        total = len(questions)
        for idx, data in enumerate(questions, 1):
            qid = data["question_id"]
            q = data["question"]
            dom = data.get("domain", "cyber")
            gt = data.get("ground_truth", "")
            rp = data.get("reasoning_points", [])

            print(f"[{idx}/{total}] Benchmarking {qid} ({dom})...")

            modes = [
                (EvalMode.MODE_BASE, lambda: self.run_mode_base(q)),
                (EvalMode.MODE_SELF_CRITIQUE, lambda: self.run_mode_self_critique(q)),
                (EvalMode.MODE_SPECIALIST, lambda: self.run_mode_specialist(q, dom)),
                (EvalMode.MODE_SPECIALIST_VERIFY, lambda: self.run_mode_specialist_verify(q, dom, qid)),
                (EvalMode.MODE_SABER, lambda: self.run_mode_saber(q, qid, dom)),
            ]

            for mode, runner in modes:
                try:
                    res = runner()
                    accuracy = 0.0
                    if score_accuracy and gt:
                        accuracy = self.score_accuracy(q, res["answer"], gt, rp)

                    results.append([
                        qid,
                        mode.value,
                        accuracy,
                        res["flags"],
                        res["corrections"],
                        round(res["latency"], 3),
                        res["conf"],
                    ])
                    print(f"  {mode.value}: latency={res['latency']:.1f}s flags={res['flags']} conf={res['conf']:.2f}")
                except Exception as e:
                    print(f"  {mode.value}: FAILED — {e}")
                    results.append([qid, mode.value, 0.0, 0, 0, 0.0, 0.0])

        # Write CSV
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["question_id", "mode", "accuracy", "flags_raised", "corrections_applied", "latency", "final_confidence"])
            writer.writerows(results)

        print(f"\nBenchmark complete! {len(results)} rows written to {output_csv}")
        return output_csv


if __name__ == "__main__":
    import argparse
    from saber.config import SaberConfig

    parser = argparse.ArgumentParser(description="SABER Benchmark Runner")
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Run benchmark only for this domain (e.g. medical, cyber, science, coding, architecture, finance)"
    )
    parser.add_argument(
        "--benchmark-file",
        type=str,
        default="data/benchmark/saber_benchmark_v1.jsonl",
        help="Path to benchmark questions JSONL file"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/benchmark/results.csv",
        help="Path to save results CSV"
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Disable LLM-as-a-Judge scoring to run benchmark faster"
    )

    args = parser.parse_args()

    config = SaberConfig.from_env()
    registry = SpecialistRegistry()
    registry.discover()
    engine = BenchmarkEngine(config, registry)
    
    engine.evaluate(
        args.benchmark_file,
        args.output_csv,
        score_accuracy=not args.no_score,
        domain=args.domain
    )
