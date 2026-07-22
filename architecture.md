# SABER — System Architecture

> **SABER** (Specialist Agent-Based Expert Reasoning) is a modular multi-specialist AI framework built on a single resident open-weight 7B base model (`Qwen2.5-7B-Instruct`). SABER pairs **High-Rank Weight-Decomposed Low-Rank Adaptation (DoRA Phase 1)** with **Verifiable-Fact-Augmented Group Relative Policy Optimization (GRPO Phase 2)**. Domain-specialized expert models generate verified, chain-of-thought-backed answers grounded by a 2-pass asynchronous verification kernel (**Sentinel**) with offline/online knowledge base routing and synthesized by a meta-reasoning layer.

---

## End-to-End Pipeline

```
                                USER MESSAGE
                                     │
                                     ▼
          ┌─────────────────────────────────────────────────────┐
          │  RESIDENT BASE MODEL (bare Qwen2.5-7B-Instruct)     │
          │  - Single resident 7B base model in VRAM            │
          └──────────────────────────┬──────────────────────────┘
                                     │
                 2-TIERED INTENT GATE (is_casual_chat)
                 - Tier 1: Pattern Match & Dictionary (<1ms)
                 - Tier 2: 7B LLM Semantic Intent Gate (<15ms)
                                     │
                  ┌──────────────────┴──────────────────┐
                  │ CASUAL_CHAT                         │ DOMAIN_QUERY
                  ▼                                     ▼
      [ Bare 7B Native Output ]           ┌───────────────────────────────────┐
      - Instant warm response (<30ms)     │ 1. Plug Orchestrator DoRA Adapter │
      - Zero adapter loading overhead     │    -> Dissect & route to domains  │
                                          │ 2. Hot-Swap Specialist Adapter    │
                                          │    -> Deep CoT Reasoning & KB     │
                                          │ 3. Sentinel Grounding & Synthesis │
                                          └─────────────────┬─────────────────┘
                                                            │
                                                            ▼
                                                     FINAL RESPONSE
                                              (With Dynamic Safety Footer)
```

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

### 3. CoT Maintainer

**File**: `saber/cot_maintainer.py`

Maintains the explicit Chain-of-Thought reasoning state for each specialist run:
1. `IDENTIFY` — Problem formulation and goal definition.
2. `ANALYZE` — Domain-specific analysis and intermediate step computation.
3. `VERIFY` — Internal sanity checking of logical steps.
4. `CONCLUDE` — Final answer formulation.

---

### 4. Sentinel Verification Kernel

**File**: `saber/sentinel.py`

The independent verification authority. Operates via **External-Evidence Grounding** (not internal LLM memory) to completely eliminate self-checking bias.

**Tiered Hybrid Verification Architecture:**

#### 1. Fast Local SQLite KB Lookup (< 5ms)
- Checks local SQLite Knowledge Base (`data/offline_kb/*_kb.db`) using **Numeric-Safe Cache Guarding** (`[numeric_guards]::clean_query`) to preserve exact dates, percentages, and CVE identifiers.
- Hits in-memory `_KB_CACHE` in RAM for 0ms lookup overhead during RL training and offline execution.

#### 2. Live Web Grounding & Dynamic Auto-Caching
- If a novel claim or recent fact is missing from the local KB, Sentinel queries live web search (Online Mode).
- **Dynamic Auto-Cache Write**: Clean verified web search snippets are **automatically written to the local SQLite KB** (`save_to_local_kb`). All future lookups for this fact become **instant 0ms local KB hits**!

#### 3. Dynamic Response Safety Footer
Every response appends a context-aware safety footer for UI/UX transparency:
- **Online Mode**: `⚡ Verified by SABER Sentinel (Online Web Grounded & Dynamic KB)`
- **Offline Mode**: `🔒 Verified by SABER Sentinel (Offline Local KB Mode — Air-Gapped)`

#### CoT Step-Level Verification
- Checks that the first step is `IDENTIFY` and last step is `CONCLUDE`.
- Detects sharp confidence drops (>0.3 between consecutive steps).
- Verifies each step logically follows from the previous steps.

---

### 5. Meta-Reasoning Layer

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
- **MCQ Tasks**: CoT reasoning trace + `ANSWER: <LETTER>`
- **Coding Tasks**: Executable ```python code block
- **Open-Ended Tasks**: Structured 6-section synthesis (`CLAIM EXTRACTION`, `CONFIDENCE ANALYSIS`, `CONFLICT DETECTION`, `TRADEOFF EVALUATION`, `RESOLUTION PATH`, `FINAL ANSWER`).

---

### 6. Signal Schema

**File**: `saber/signal.py`

Strict, strongly-typed JSON-serializable dataclass defining all inter-component communication:

- `QUERY_SIGNAL`: User query entering the system.
- `TASK_SIGNAL`: Orchestrator sub-task dispatched to a specialist.
- `CONFIRMATION_SIGNAL`: Specialist confirming task receipt.
- `COT_SIGNAL`: Specialist submitting reasoning chain + claims.
- `VERIFICATION_SIGNAL`: Meta-Reasoner requesting Sentinel check.
- `FLAG_SIGNAL`: Sentinel raising an error/contradiction with correction.
- `COMPILATION_SIGNAL`: Verified outputs delivered to Meta-Reasoner.
- `OUTPUT_SIGNAL`: Final synthesized response.

Every signal is cryptographically signed with a SHA-256 `integrity_hash` over its payload.

---

### 7. Decision Ledger & Audit Trail

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

## Training Pipeline

**Files**: `saber/training/trainer.py`, `saber/training/dataset_loader.py`, `scripts/build_offline_kb.py`, `scripts/validate_kb_coverage.py`

### 1. Two-Phase Training Paradigm

#### Phase 1: High-Rank Weight-Decomposed LoRA (DoRA SFT)
- **Base Model**: `Qwen/Qwen2.5-7B-Instruct`
- **Method**: DoRA (`use_dora=True`, rank $r=128$, $\alpha=256$, dropout $0.05$)
- **Target Modules**: All 7 linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) — expanding trainable parameter capacity to **~500M parameters (~7.2% of model)**.
- **Precision & Optimization**: Native `bfloat16`, batch size per device = 8, gradient accumulation = 4 (effective batch = 32), 4 epochs, learning rate = 2e-4, PyTorch gradient checkpointing enabled.
- **Hardware Acceleration**: Optimized for single NVIDIA H100 80GB GPU.
- **Dataset Scale**: **260,000 pristine CoT records** across 5 specialized domains + 2 core orchestrating systems (~13.5 hours total H100 training run).

#### Phase 2: Verifiable-Fact-Augmented GRPO Reinforcement Learning
- **Group Rollouts**: $G = 8$ parallel trajectory generations per prompt.
- **Composite Reward Signal**:
  $$\text{Reward} = R_{\text{Format}} (+1.0) + R_{\text{Outcome}} (+2.0) - \lambda \cdot R_{\text{Sentinel\_Factuality}}$$
  - **Definitive Tasks (Science, Cyber, Finance)**: Format (+1.0 for ANSWER: [A-D]), Outcome (+2.0 for ground truth match), Sentinel Factuality (-1.5 contradiction / +0.5 support / 0.0 neutral).
  - **Open-Ended Tasks (Coding, Architecture, Meta, Orchestrator)**: Sandboxed Code Execution (+2.0), CoT Completeness (+1.0), Token Repetition Penalty (-1.5).

#### Tiered Hybrid Knowledge Base (0ms RL & Consumer Grounding)
- **Indexing**: 139,973+ reference support passages compiled into local indexed SQLite databases (`data/offline_kb/*_kb.db`).
- **Dynamic Auto-Caching**: Un-cached live web snippets are automatically written back to SQLite KB for instant 0ms future hits.
- **Numeric-Safe Caching**: Cache key pre-guarding (`[guard]::clean_query`) preserving numerical, percentage, date, and CVE statistical precision.
- **Hysteresis Audit Loop**: Rollout Disagreement Rate (RDR $> 5\%$) dynamically decays $\lambda \rightarrow 0.1$ for 100 steps and auto-restores to $0.5$ once signal stability returns, monitored by TensorBoard Neutral Claim Ratio (NCR).

---

### Training ↔ Evaluation Separation (Zero Data Leakage)

| Domain | Trained On | Evaluated On | Overlap |
|--------|-----------|-------------|---------|
| Science | ScienceQA, SciQ, CAMEL-AI | **GPQA Diamond** (198 cases) | None |
| Cyber | MITRE STIX, CyberQA, Synthetic ATT&CK | **SecBench** (100 cases) | None |
| Coding | APPS, CodeContests, LeetCode, CodeFeedback | **HumanEval** (164 cases) | None |
| Finance | Finance-Alpaca, ConvFinQA, Finance-Instruct | **FinQA Math** (80 cases) | None |
