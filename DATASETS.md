# SABER Dataset Curation & Filtering Strategy

This document defines the exact datasets, target record counts, and gold-quality filter criteria for training the SABER domain specialists (via DoRA/SFT).

## Domain Specialist Datasets

| Specialist | Dataset (Train Split Only) | Target Records | Gold-Quality Filter Criteria |
|---|---|---|---|
| **Cybersecurity** | `pAILabs/infosec-security-qa` | ~8,000 | Drop answers < 20 words; drop if no technical term overlap with question; dedupe near-identical questions (cosine > 0.95). |
| | MITRE ATT&CK STIX (parsed) | ~3,000 | Keep populated description > 50 words; drop deprecated/revoked; 1 QA pair per technique/sub-technique. |
| | CVEfixes | ~2,000 | Non-empty CVSS AND diff < 150 lines; drop uninformative commit messages (< 5 words). |
| **Finance** | FinQA (train) | ~6,251 | Keep official split; optionally drop if numerical answer cannot be reproduced by running provided program. |
| | ConvFinQA (train) | ~3,037 | Same executable-answer check as FinQA. |
| | TAT-QA (train) | ~13,215 | Drop "count" answers if > 20; preferentially keep table+text hybrid entries over text-only. |
| **Coding (Python)** | CodeAlpaca-20k | ~15,000 | Drop if code fails `ast.parse()`; drop instructions < 5 words; drop trivial code < 2 lines. |
| | Magicoder-OSS-Instruct-75K | ~10,000 | Keep if code executes without error in sandbox; prefer docstrings/comments; dedupe by code-body embedding similarity. |
| | APPS (train) | ~2,000 | Keep only "interview" and "competition" (drop "introductory"); require ≥1 passing test case in suite. |
| **Coding (JS)** | The Stack (JS/TS filtered) | ~10,000 | Drop files > 200 lines; drop minified/bundled; require valid syntax (esprima/acorn); drop if second LLM pass cannot verify instruction↔code correspondence. |
| **SQL** | Spider (train) | ~7,000 | Official curated split; optionally drop queries with execution errors against provided schema. |
| **Architecture** | Synthetic (System Design + Docs) | ~3,000–5,000 | Generate 2-3x target, drop answers < 100 words; require LLM verification of factual consistency; dedupe via embedding; manually spot-check 5%. |
| | Synthetic (Planner task decomp) | ~1,500–2,000 | Fully hand-curated/reviewed; verify sub-task dependency correctness (highest stakes dataset). |
| **Medical** | `medical-o1-reasoning-SFT` | ~15,000 | Apply ChatDoctor contamination check; dedupe against MedQA/MedMCQA overlap. |
| | MedQA-USMLE (train) | ~10,178 | Official curated split. |
| | MedMCQA (train) | ~15,000 | Keep only entries with non-empty `exp` (explanation) field to ensure CoT-quality training. |
| **Science** | SciQ (train) | ~11,679 | Keep only entries with non-empty `support` field (evidence passage). |
| | ARC (Challenge train) | ~1,119 | Use Challenge set only (Easy set is too trivial). |

## Core Infrastructure Datasets

| Component | Dataset | Target Records | Gold-Quality Filter Criteria |
|---|---|---|---|
| **Orchestrator** | Synthetic routing examples | ~2,000–3,000 | Generate across single and multi-domain queries equally; require human spot-check on ambiguous/multi-domain cases. |
| **Meta-Reasoner**| Synthetic contradiction-resolution | ~500–1,000 | Fully hand-curated cross-domain conflicts; verify resolution is factually correct. Quality matters more than size. |

## Universal Gold-Quality Filters (Applied to ALL datasets)

1. **Length bounds**: Discard suspiciously short answers (low-effort) or absurdly long answers (scrape artifacts).
2. **Deduplication**: Embedding-based near-duplicate removal (cosine > 0.9–0.95) within datasets and across datasets feeding the same specialist.
3. **Executability/Verifiability**: Code must run/parse; math/finance answers must be reproducible from reasoning.
4. **Benchmark Leakage Check**: Cross-check filtered sets against CyberMetric-80, FinanceBench, HumanEval/MBPP, and ArchBench via embedding similarity (to catch paraphrased leakage).
5. **Format Consistency**: Normalize all datasets into the exact ChatML-style schema prior to mixing, ensuring the Data Collator masking applies perfectly across all specialists.

---

## Training Hyperparameters & Checkpointing Strategy

**Target Hardware**: Single 80GB H100 GPU
**Pipeline Flow**: Sequential (train one adapter, validate, unload, train next)

| Specialist / Component | Post-Filter Size | Recommended Epochs | Reasoning / Risk Profile |
|---|---|---|---|
| **Cybersecurity** | ~13,000 | 3 | Medium size, mixed sources. Watch for overfit on CVEfixes. |
| **Finance** | ~22,500 | 2–3 | Repetitive structure; >3 epochs risks pattern memorization. |
| **Python** | ~27,000 | 2–3 | High surface diversity. >3 epochs risks snippet memorization. |
| **JavaScript** | ~10,000 | 3–4 | Synthetic-heavy. Needs more passes but overfits faster. |
| **SQL** | ~7,000 | 3–4 | Narrower pattern space (schema-grounded), can take more epochs. |
| **Architecture (Domain QA)** | ~4,000 | 3–4 | Small, synthetic. High overfit risk. |
| **Architecture (Planner)** | ~1,500–2,000 | 4–5 | Highest-stakes dataset. Needs more passes to learn decomposition. Extreme overfit risk. |
| **Medical** | ~35,000+ | 2–3 | Largest dataset, high-variance reasoning. Standard 2-3 epochs. |
| **Science** | ~12,800 | 3 | Standard MCQ+evidence structure. |
| **Orchestrator** | ~2,000–3,000 | 4–5 | Narrow task (routing), needs more passes. |
| **Meta-Reasoner** | ~500–1,000 | 5–6 | Smallest dataset, hand-curated. Extreme overfit risk. |

### Critical Stopping Criterion & Checkpointing Rules
The fixed epoch count is merely a guideline. The **actual stopping criterion** is driven by the `validate_dora.py` overfitting check (train loss vs. eval accuracy).

1. **Save Best, Not Last**: The pipeline must store the best model checkpoint based on eval score, not simply the final epoch checkpoint (Early Stopping).
2. **Frequency**: 
   - Standard datasets: Checkpoint every 1 epoch.
   - Small/High-Stakes datasets (Meta-Reasoner, Planner): Checkpoint every 0.5 epochs.
3. **Manual Validation Check**: For Meta-Reasoner and Planner decomposition, automated eval metrics are inherently softer. Manually inspect a handful of generations at each checkpoint rather than relying purely on automated metrics.
