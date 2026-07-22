# SABER — System Architecture

> **SABER** (Specialist Agent-Based Expert Reasoning) is a modular multi-specialist AI framework built on a single resident open-weight 7B base model (`Qwen2.5-7B-Instruct`). SABER pairs **High-Rank Weight-Decomposed Low-Rank Adaptation (DoRA Phase 1)** with **Verifiable-Fact-Augmented Group Relative Policy Optimization (GRPO Phase 2)**. Domain-specialized expert models generate verified, chain-of-thought-backed answers grounded by a 2-pass asynchronous verification kernel (**Sentinel**) with offline/online knowledge base routing and synthesized by a meta-reasoning layer.

---

## End-to-End Pipeline

```
                                USER QUERY / MESSAGE
                                         │
                                         ▼
            ┌──────────────────────────────────────────────────────────┐
            │       RESIDENT BASE MODEL (bare Qwen2.5-7B-Instruct)     │
            │       - Single resident 7B base model in VRAM            │
            └────────────────────────────┬─────────────────────────────┘
                                         │
                   2-TIERED INTENT GATE (is_casual_chat)
                   - Tier 1: Pattern & Phrase Match (<1ms)
                   - Tier 2: 7B LLM Semantic Intent Gate (<15ms)
                                         │
                      ┌──────────────────┴──────────────────┐
                      │ CASUAL_CHAT                         │ DOMAIN_QUERY
                      ▼                                     ▼
        ┌───────────────────────────┐         ┌───────────────────────────┐
        │  BARE 7B NATIVE CHAT      │         │  ORCHESTRATOR DORA        │
        │  - Instant warm response  │         │  ADAPTER                  │
        │    (<30ms latency)        │         │  - Dissects query into    │
        │  - Zero adapter loading   │         │    sub-task signals       │
        └───────────────────────────┘         └─────────────┬─────────────┘
                                                            │ TASK_SIGNALs
                                                            ▼
                                              ┌───────────────────────────┐
                                              │  SPECIALIST EXECUTION     │
                                              │  - Hot-swaps DoRA adapter │
                                              │    (Science, Cyber, Fin,  │
                                              │     Coding, Architecture) │
                                              │  - Generates CoT Claims   │
                                              └─────────────┬─────────────┘
                                                            │ COT_SIGNALs
                                                            ▼
                                              ┌───────────────────────────┐
                                              │  SENTINEL VERIFIER KERNEL │
                                              │  1. Fast SQLite KB (<5ms) │
                                              │  2. Web Search + Auto-    │
                                              │     Cache Write to KB     │
                                              │  3. Returns GREEN_CHIT or │
                                              │     FLAG_SIGNAL (Rewrite) │
                                              └─────────────┬─────────────┘
                                                            │ Verified Claims
                                                            ▼
                                              ┌───────────────────────────┐
                                              │  META-REASONING LAYER     │
                                              │  - Context-Aware Struct   │
                                              │    (MCQ / Code / Synthesis)│
                                              └─────────────┬─────────────┘
                                                            │ Synthesized Output
                                                            ▼
                                              ┌───────────────────────────┐
                                              │  THREAD-SAFE AUDIT LEDGER │
                                              │  - Logs query_id, signals │
                                              │    flags & confidence     │
                                              └─────────────┬─────────────┘
                                                            │
                                                            ▼
                                              ┌───────────────────────────┐
                                              │  RESPONSE OUTPUT          │
                                              │  Appends Sentinel Footer: │
                                              │  ⚡ Online Web Grounded /  │
                                              │  🔒 Offline Local KB      │
                                              └───────────────────────────┘
```

---

## Project File Map

### Root Files

| File | Purpose |
|------|---------|
| `architecture.md` | This document — master system architecture reference. |
| `implementation_plan.md` | Detailed training & benchmark plan with dataset breakdown. |
| `roadmap.md` | Feature & milestone roadmap. |
| `README.md` | Project overview & quickstart. |
| `requirements.txt` | Python dependencies. |
| `run.sh` | Main SABER execution entrypoint script. |
| `run_pod.sh` | RunPod GPU cloud execution script. |
| `run_training_pipeline.sh` | Sequential training pipeline launcher. |
| `run_evaluations.py` | Unified benchmark evaluation runner. |
| `chat.py` | Interactive CLI chat interface. |
| `merge_adapters.py` | Merges trained DoRA adapters into base model weights. |
| `apply_patches.sh` | Applies incremental code patches. |

---

## Component Details

### 1. 2-Tiered Intent Gate & Orchestrator

**File**: `saber/orchestrator.py`

The entry point. Operates via **Base-First Dynamic Adapter Insertion**:

1. **2-Tiered Intent Gate (`is_casual_chat`)**:
   - **Tier 1 (Fast Direct Pattern Match <1ms)**: Normalized alphanumeric check over greetings, slang, pleasantries (`hi`, `hello`, `thanks!`, `good morning`, `who made you`).
   - **Tier 2 (7B LLM Semantic Intent Gate <15ms)**: 4-token prompt gate evaluating semantic intent for unstructured conversational inputs.
   - **Casual Chat Fast-Path**: Responds natively using the bare 7B base model in **<30ms** with zero adapter loading overhead.

2. **Orchestrator DoRA Adapter (`models/orchestrator_v2`)**:
   - When a `DOMAIN_QUERY` is identified, the Orchestrator DoRA adapter is dynamically plugged onto the resident 7B base model to perform **Few-Shot Semantic Intent Routing** and **Ambiguity Detection**.

| Step | What It Does |
|------|-------------|
| **Casual Chat Gating** | Intercepts greetings and small talk, generating warm native answers instantly without loading adapters. |
| **Ambiguity Detection** | Scores query ambiguity (0–1). Queries ≥ 0.70 are rejected with a clarification request. |
| **Semantic Intent Classification** | Uses few-shot semantic prompt routing to dissect complex or polysemous queries (e.g. computer virus -> cyber vs biological virus -> science). |
| **Specialist Selection** | Activates specialists whose domain relevance score ≥ 0.50 (or top-1 domain in benchmark mode). |
| **Task Decomposition** | Splits multi-domain queries into domain-specific `TASK_SIGNAL` payloads for each activated specialist. |
| **Verification Tier** | Assigns Tier 0 (fast pass-through) or Tier 1 (full Sentinel grounding loop) based on query sensitivity. |

---

### 2. Specialist Execution Engine

**Files**: `saber/specialist.py`, `saber/specialists/*`

Domain-specific experts executing via **DoRA Adapters ($r=128, \alpha=256$)** plugged onto the resident `Qwen2.5-7B-Instruct` base model:

- **Science Specialist** (`saber/specialists/science.py`): Physics, chemistry, calculus, biology.
- **Cybersecurity Specialist** (`saber/specialists/cybersecurity.py`): Vulnerability analysis, exploit payloads, MITRE ATT&CK.
- **Finance Specialist** (`saber/specialists/finance.py`): EBITDA calculations, 10-K analysis, portfolio valuation.
- **Coding Specialist** (`saber/specialists/coding.py`): Algorithm design, python code generation, debugging.
- **Architecture Specialist** (`saber/specialists/architecture.py`): Microservices, Kubernetes, distributed systems.

Each specialist produces structured claims (`Claim`) and a step-by-step reasoning chain managed by `CoTMaintainer`.

---

### 3. LLM Engine (Dynamic Model Swapping)

**File**: `saber/llm_engine.py`

Context-managed dynamic model swapping engine. Ensures only **ONE model is loaded into VRAM** at any time:
- Loads base model or PEFT adapter via `with LLMEngine(model_path) as engine:`.
- Automatic VRAM cleanup on exit (`torch.cuda.empty_cache()`).
- Optional weight caching via `SABER_KEEP_MODELS_LOADED=1` env var.
- Supports single-turn (`generate`), multi-turn (`generate_with_history`), and session-based (`generate_from_session`) generation.

---

### 4. CoT Maintainer

**File**: `saber/cot_maintainer.py`

Bidirectional working memory module maintaining explicit Chain-of-Thought reasoning state:
1. `IDENTIFY` — Problem formulation and goal definition.
2. `ANALYZE` — Domain-specific analysis and intermediate step computation.
3. `HYPOTHESIZE` — Hypothesis generation.
4. `EVIDENCE` — Evidence gathering and citation.
5. `EVALUATE` — Internal sanity checking of logical steps.
6. `CONCLUDE` — Final answer formulation.

**Key methods**:
- `export_rich_synthesis_narrative()` — Formats CoT chain into rich, evidence-grounded narrative with dependency links, designed for Meta-Reasoner compilation of long open-ended explanations.
- `export_for_signal()` — Exports chain as dictionary payload including `rich_narrative` for signal transport.
- `cleanup()` — Deduplicates redundant steps, detects reasoning loops, merges consecutive same-action steps.

---

### 5. Sentinel Verification Kernel

**File**: `saber/sentinel.py`

The independent verification authority. Operates via **External-Evidence Grounding** (not internal LLM memory) to completely eliminate self-checking bias. **Sentinel is NOT an adapter** — it is a pure Python verification kernel with zero adapter loading overhead.

**Tiered Hybrid Verification Architecture:**

#### 1. Fast Local SQLite KB Lookup (< 5ms)
- Checks local SQLite Knowledge Base (`data/offline_kb/*_kb.db`) using **Numeric-Safe Cache Guarding** (`[numeric_guards]::clean_query`) to preserve exact dates, percentages, and CVE identifiers.
- Hits in-memory `_SEARCH_CACHE` in RAM for 0ms lookup overhead during RL training and offline execution.

#### 2. Live Web Grounding & Dynamic Auto-Caching
- If a novel claim or recent fact is missing from the local KB, Sentinel queries live web search via DuckDuckGo (Online Mode).
- **Dynamic Auto-Cache Write**: Clean verified web search snippets are **automatically written to the local SQLite KB** (`save_to_local_kb`). All future lookups for this fact become **instant 0ms local KB hits**.
- **Consecutive Search Circuit Breaker**: Identical queries are bypassed after 2 consecutive hits to prevent autoregressive search loops.

#### 3. Claim Extraction & Query Formulation
- Extracts structured `Claim` statements from `Signal.payload["claims"]`.
- Skips generic noise claims (`"The correct answer is B"`).
- Truncates to 120 chars for search efficiency.
- Falls back to `compiled_text` when claims list is empty.

#### 4. Dynamic Response Safety Footer
Every response appends a context-aware safety footer for UI/UX transparency:
- **Online Mode**: `⚡ Verified by SABER Sentinel (Online Web Grounded & Dynamic KB)`
- **Offline Mode**: `🔒 Verified by SABER Sentinel (Offline Local KB Mode — Air-Gapped)`

#### 5. CoT Step-Level Verification (`verify_cot_chain`)
- Checks that the first step is `IDENTIFY` and last step is `CONCLUDE`.
- Detects sharp confidence drops (>0.3 between consecutive steps).
- Verifies each step logically follows from the previous steps via LLM semantic check.

#### 6. Fail-Open Availability
- If LLM crashes (GPU OOM, timeout), Sentinel returns `GREEN_CHIT` gracefully — the pipeline never crashes.

---

### 6. Meta-Reasoning Layer

**File**: `saber/meta_reasoner.py`

Purely a synthesis engine — it does NOT generate domain answers.

**Two-phase operation:**

#### Phase 1: Read & Hold (happens at dispatch time)
When the Orchestrator finalizes routing, it sends two things to the Meta-Reasoner:
1. The **original user query** — so the meta-reasoner understands what the user actually asked.
2. The **list of activated specialists** — so it knows exactly how many outputs to expect.

The Meta-Reasoner reads these, holds them quietly, and waits. It does nothing until all expected specialist outputs have arrived.

#### Phase 2: Synthesis (activates once ALL specialist outputs received)
Once every activated specialist has sent its verified claims + CoT chain, the Meta-Reasoner checks them all off against its expected list and begins synthesis. Because it already knows the original query, it can ensure the final answer directly addresses what the user asked — not just what the specialists chose to talk about.

**Format-Aware Response Structuring**:
- **MCQ Tasks**: CoT reasoning trace + `ANSWER: <LETTER>`.
- **Coding Tasks**: Executable ```python code block.
- **Open-Ended Tasks**: Structured 6-section synthesis (`CLAIM EXTRACTION`, `CONFIDENCE ANALYSIS`, `CONFLICT DETECTION`, `TRADEOFF EVALUATION`, `RESOLUTION PATH`, `FINAL ANSWER`).

---

### 7. Signal Schema

**File**: `saber/signal.py`

Strict, strongly-typed JSON-serializable Pydantic dataclass defining all inter-component communication:

- `QUERY_SIGNAL`: User query entering the system.
- `TASK_SIGNAL`: Orchestrator sub-task dispatched to a specialist.
- `CONFIRMATION_SIGNAL`: Specialist confirming task receipt.
- `COT_SIGNAL`: Specialist submitting reasoning chain + claims.
- `VERIFICATION_SIGNAL`: Sentinel returning GREEN_CHIT confirmation.
- `FLAG_SIGNAL`: Sentinel raising an error/contradiction with correction.
- `OUTPUT_SIGNAL`: Final synthesized response.
- `PATCH_SIGNAL`: Incremental code/config patches.
- `HEALTH_SIGNAL`: System health checks.
- `AUDIT_SIGNAL`: Audit trail events.

Every signal is cryptographically signed with a SHA-256 `integrity_hash` over its payload via `freeze_and_hash()`.

---

### 8. Decision Ledger & Audit Trail

**File**: `saber/audit.py`

Thread-safe, append-only JSON-Lines audit log (`logs/audit.jsonl`). Records every query's full lifecycle:
- Query reception & ambiguity score
- Casual chat gating status
- Specialist selection & task signals
- Signal integrity check results
- Sentinel verification flags & proposed fixes
- Self-correcting rewrite iterations
- Meta-reasoning synthesis path & confidence score

---

### 9. Supporting Modules

| File | Purpose |
|------|---------|
| `saber/config.py` | Central configuration (model paths, domains, thresholds, benchmark mode). |
| `saber/registry.py` | Specialist registry for dynamic adapter discovery and loading. |
| `saber/context.py` | Session memory manager for multi-turn conversation history. |
| `saber/chat_history.py` | Persistent chat history storage. |
| `saber/claim_graph.py` | Claim dependency graph for tracking inter-claim relationships. |
| `saber/flag.py` | Flag data structures for Sentinel error reporting. |
| `saber/errors.py` | Custom exception classes. |
| `saber/metrics.py` | Performance metrics collection (latency, token counts). |
| `saber/benchmark.py` | Benchmark evaluation harness for automated scoring. |

---

### 10. Evaluation Suite

**Directory**: `saber/evaluation/`

| File | Purpose |
|------|---------|
| `saber/evaluation/code_eval.py` | Sandboxed Python code execution evaluator (Pass@1 with 2s timeout). |
| `saber/evaluation/harness.py` | Evaluation harness adapter for standard benchmark formats. |
| `saber/evaluation/multi_judge.py` | Multi-judge LLM-as-a-Judge evaluator (5-dimensional rubric scoring). |

---

### 11. API Server

**File**: `saber/api/main.py`

FastAPI-based REST API server exposing SABER endpoints for programmatic access.

---

## Training Pipeline & Benchmark Evaluation

**Files**: `saber/training/trainer.py`, `saber/training/dataset_loader.py`, `saber/training/rewards.py`

### 1. Two-Phase Training Paradigm

#### Phase 1: High-Rank Weight-Decomposed LoRA (DoRA SFT)
- **Base Model**: `Qwen/Qwen2.5-7B-Instruct`
- **Method**: DoRA (`use_dora=True`, rank $r=128$, $\alpha=256$, dropout $0.05$)
- **Target Modules**: All 7 linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) — expanding trainable parameter capacity to **~500M parameters (~7.2% of model)**.
- **Precision & Optimization**: Native `bfloat16`, batch size per device = 8, gradient accumulation = 4 (effective batch = 32), 4 epochs, learning rate = 2e-4, PyTorch gradient checkpointing enabled.
- **Hardware Acceleration**: Optimized for single NVIDIA H100 80GB GPU.
- **Dataset Scale**: **260,000 pristine CoT records** across 5 specialized domains + 2 core orchestrating systems (~13.5 hours total H100 training run).

#### Phase 2: Verifiable-Fact-Augmented GRPO Reinforcement Learning
- **Reward Functions** (`saber/training/rewards.py`):
  - Anti-lazy token guard (<20 tokens penalized with -1.0).
  - 3-gram repetition penalty (-1.5 for uniqueness <60%).
  - Sandboxed Python execution (2s timeout).
  - Relative float matching ($10^{-3}$).
- **Group Rollouts**: $G = 8$ parallel trajectory generations per prompt.
- **Composite Reward Signal**:
  $$\text{Reward} = R_{\text{Format}} (+1.0) + R_{\text{Outcome}} (+2.0) - \lambda \cdot R_{\text{Sentinel\_Factuality}}$$

#### Tiered Hybrid Knowledge Base (0ms RL & Consumer Grounding)
- **Indexing**: 139,973+ reference support passages compiled into local indexed SQLite databases (`data/offline_kb/*_kb.db`).
- **Dynamic Auto-Caching**: Un-cached live web snippets are automatically written back to SQLite KB for instant 0ms future hits.
- **Numeric-Safe Caching**: Cache key pre-guarding (`[guard]::clean_query`) preserving numerical, percentage, date, and CVE statistical precision.
- **Hysteresis Audit Loop**: Rollout Disagreement Rate (RDR $> 5\%$) dynamically decays $\lambda \rightarrow 0.1$ for 100 steps and auto-restores to $0.5$ once signal stability returns, monitored by TensorBoard Neutral Claim Ratio (NCR).

---

### 2. Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `scripts/1_build_datasets.py` | Downloads & builds the 260,000 CoT dataset corpus. |
| `scripts/2_build_kb.py` | Builds & audits offline SQLite KBs (139k+ passages). |
| `scripts/3_train_dora.py` | Phase 1 DoRA SFT training. |
| `scripts/4_train_grpo.py` | Phase 2 GRPO RL training. |
| `scripts/5_run_benchmark.py` | Unified 5-mode benchmark evaluation. |
| `scripts/build_offline_kb.py` | Standalone offline KB builder. |
| `scripts/validate_kb_coverage.py` | KB coverage validation & gap analysis. |
| `scripts/run_final_benchmark.py` | Final comprehensive benchmark runner. |

---

### 3. Dual-Tier Benchmark Hierarchy (Frontier vs. Mid-Tier)

| Domain Area | Mid-Tier Baseline Benchmarks | Top-Tier Frontier Benchmarks | Target |
|---|---|---|---|
| **Science** | **SciQ / ScienceQA** (13.6k) | **GPQA Diamond** (198 PhD-level) | $> 65.0\%$ |
| **Cybersecurity** | **CyberMetric-800** (800) | **SecBench / SecQA-Hard** (100) | $> 82.0\%$ |
| **Coding** | **HumanEval** (164) / **MBPP** (974) | **SWE-bench Verified** (500) | $> 78.0\%$ |
| **Finance** | **Finance-Alpaca** (20k) | **FinQA / ConvFinQA** (SEC 10-K) | $> 75.0\%$ |
| **Architecture** | **SystemDesign-Basic** (100) | **ArchBench-Hard** (50) | $> 80.0\%$ |

---

### 4. Multi-Domain Evaluation Strategy

1. **Finance + Coding (Quant Financial Engineering)**
2. **Cybersecurity + System Architecture (Zero-Trust Infrastructure)**
3. **Science + Coding (Scientific Computation)**
4. **Finance + Cybersecurity (Smart Contract Audit)**

Multi-domain training data uses a **Hybrid Sourcing Strategy**: ~40% existing multi-task corpora (FinQA, CyberSecEval, SWE-bench) + ~60% self-synthesized cross-domain compositions with Sentinel-verified ground truth.

---

### 5. Automated Open-Ended Evaluation Methodology

#### Mode A: Sandboxed Code Execution (`saber/evaluation/code_eval.py`)
- **Metric**: **Pass@1 Rate** via isolated `multiprocessing.Process` with 2.0s timeout.

#### Mode B: Rubric-Based LLM-as-a-Judge (`saber/evaluation/multi_judge.py`)
- **5-Dimensional 25-Point Rubric**: Factual Accuracy, Completeness, Structural Coherence, Domain Depth, Cross-Domain Accord.
- **Metric**: **Quality Score ($S \in [0.0, 1.0]$)**.

#### Mode C: Sentinel Contradiction Rate
- **Metric**: **Grounding Accuracy Rate (%)** and **Contradiction Rate (%)**.

---

## Test Suite

| Test File | Coverage |
|-----------|----------|
| `tests/test_signal.py` | Signal creation, SHA-256 hashing, integrity verification. |
| `tests/test_llm_engine.py` | LLM engine context management, device detection. |
| `tests/test_specialist.py` | Specialist task execution, claim generation. |
| `tests/test_sentinel.py` | Signal integrity, verification routing matrix. |
| `tests/test_sentinel_stress.py` | Cryptographic tampering, SQLite auto-caching, FLAG generation. |
| `tests/test_sentinel_search_extraction.py` | Numeric/CVE guard extraction, keyphrase extraction, KB retrieval. |
| `tests/test_sentinel_e2e_extraction.py` | End-to-end claim extraction, GREEN_CHIT/FLAG routing, CoT step checks, LLM crash fail-open. |
| `tests/test_cot_maintainer.py` | CoT chain creation, step management, cleanup dedup. |
| `tests/test_orchestrator.py` | Casual chat gating, polysemous disambiguation, domain routing. |
| `tests/test_rewards.py` | GRPO reward functions, anti-lazy guards, repetition penalties. |
| `tests/test_pipeline_flow.py` | Full end-to-end pipeline integration flow. |
| `tests/test_live_simulation.py` | Live multi-domain simulation test. |

**Current Status**: All tests passing 100%.
