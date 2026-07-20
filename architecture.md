# SABER — System Architecture

> **SABER** (Specialist Agent-Based Expert Reasoning) is a modular multi-specialist AI framework built on a single open-weight 7B-parameter base model (Qwen 2.5-7B), where domain-specialized LoRA-fine-tuned expert models generate verified, chain-of-thought-backed answers that are synthesized into a single coherent response by a meta-reasoning layer.

---

## End-to-End Pipeline

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR (bare Qwen 2.5-7B)                                 │
│  The base 7B model — no LoRA adapter. Uses its general LLM       │
│  capabilities for all pre-routing checks.                        │
│                                                                  │
│  1. Ambiguity Detection                                         │
│  2. Query Completeness Check (asks follow-ups if info missing)  │
│  3. Few-Shot Domain Classification with confidence gate          │
│  4. Specialist Selection                                        │
│  5. Verification Tier Assignment                                │
│                                                                  │
│  Once routing is finalized:                                      │
│  ──► Pings 0.5B Chatbot with routing decision                   │
│  ──► Dispatches tasks to specialists                            │
└───────┬──────────────────────────────────────────┬───────────────┘
        │                                          │
        │ routing decision                         │ TASK_SIGNALs
        ▼                                          ▼
┌───────────────────────────┐   ┌──────────────────────────────────────────┐
│  CHATBOT (Qwen 2.5-0.5B) │   │  SPECIALIST EXECUTION (sequential)       │
│  Runs on CPU              │   │                                          │
│                           │   │  For each activated specialist:          │
│  Receives routing info    │   │    ┌────────────────────────────────┐    │
│  from Orchestrator and    │   │    │  1. Confirm task               │    │
│  gives user a context-    │   │    │  2. Load LoRA adapter          │    │
│  aware status message:    │   │    │  3. Generate answer with CoT   │    │
│                           │   │    │  4. Parse into Claims          │    │
│  "Routing to our cyber    │   │    │  5. Sentinel verification      │    │
│   security specialist     │   │    │     cycles (tier-dependent)    │    │
│   — analyzing your MITRE  │   │    │  6. If FLAG → rewrite          │    │
│   ATT&CK question..."     │   │    │  7. If GREEN_CHIT → pass       │    │
│                           │   │    └────────────────────────────────┘    │
│  Replaced when real       │   │                                          │
│  answer arrives           │   │  Output: Verified claims + CoT chains    │
└───────────────────────────┘   └──────────────────┬───────────────────────┘
                                                   │
                                    all verified specialist outputs
                                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  META-REASONING LAYER (synthesis — runs AFTER all specialists)   │
│                                                                  │
│  Receives: All specialist claims + CoT chains (post-sentinel)    │
│                                                                  │
│  1. CLAIM EXTRACTION — list key claims from each specialist      │
│  2. CONFIDENCE ANALYSIS — weight specialist confidence levels    │
│  3. CONFLICT DETECTION — identify contradictions between specs   │
│  4. TRADEOFF EVALUATION — analyse disagreements                  │
│  5. RESOLUTION PATH — reconcile conflicting views                │
│  6. FINAL ANSWER — produce one coherent, detailed answer         │
│                                                                  │
│  Also: Disagreement detection (confidence spread > 0.3)          │
│  Also: Final confidence scoring with flag/cycle penalty          │
└──────────────────────┬───────────────────────────────────────────┘
                       │ final synthesized answer
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  DECISION LEDGER (written to audit log)                          │
│  - Full specialist responses, all flags, all corrections,        │
│    verification history, disagreements, meta-reasoning path,     │
│    final confidence score                                        │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
                 Final Answer → User (replaces chatbot message)
```

---

## Component Details

### 0. Chatbot (Qwen 2.5-0.5B)

**Model**: `Qwen/Qwen2.5-0.5B` (350M parameters)

A lightweight model that **loads with the chat interface** and runs on CPU. It does NOT fire immediately when the user sends a query — it waits for the **Orchestrator to finalize routing** and then receives a ping with the routing decision.

Once pinged, the chatbot generates a **context-aware status message** based on which specialists were activated (e.g., *"Routing to our cybersecurity specialist — analyzing your MITRE ATT&CK question..."*). This message is displayed to the user while the specialists work in the background.

- **Runs on CPU** — no GPU contention with the specialist pipeline.
- **Loads once at startup** — always warm, instant response once pinged.
- **Informed by routing** — knows which specialists were activated, so the message is specific, not generic.
- **Replaced transparently** — when the backend pipeline returns the real answer, it replaces the chatbot message in the UI.
- **Also handles simple greetings** — if the user sends a non-domain message ("hi", "thanks", etc.), the chatbot handles it directly without invoking the full pipeline.

This model does NOT participate in any reasoning, routing, or verification.

### 1. Orchestrator (bare Qwen 2.5-7B)

**File**: `saber/orchestrator.py`

The entry point. It is the **bare Qwen 2.5-7B base model** — no LoRA adapter, no domain fine-tuning. It uses its general-purpose LLM capabilities to handle all pre-routing checks: ambiguity detection, query completeness verification, and domain classification. It does NOT generate domain answers — it only validates, classifies, and routes.

| Step | What It Does |
|------|-------------|
| **Ambiguity Detection** | Scores query ambiguity (0–1) based on word count, pronoun density, and domain keyword coverage. Queries ≥ 0.70 are rejected with a clarification request. |
| **Query Completeness Check** | Uses the 7B model to assess whether the query contains enough information for the specialists to produce a useful answer. If critical information is missing, the system asks targeted follow-up questions before proceeding (see below). |
| **Domain Classification** | Few-shot LLM classification using the 7B base model (see below). |
| **Confidence Gate** | If the classifier's confidence is < 95%, the query is re-classified with expanded context (full specialist descriptions + example queries). Routing must be near-100% accurate — wrong routing makes everything downstream useless. |
| **Specialist Selection** | Activates specialists whose relevance score ≥ 0.30 (configurable threshold). In benchmark mode, forces exactly one specialist (highest score). |
| **Tier Assignment** | Assigns verification depth: Tier 0 (no checks), Tier 1 (2 checks), Tier 2 (4 checks), Tier 3 (6 checks). |

#### Query Completeness Check

Before any routing happens, the Orchestrator uses the 7B model to determine if the query has enough detail to produce a meaningful answer. The model evaluates:

- **Missing constraints** — e.g., *"Calculate the force"* but no mass or acceleration values provided.
- **Underspecified scope** — e.g., *"How do I secure my system?"* — which system? What threat model? What's already in place?
- **Missing context** — e.g., *"What's wrong with this code?"* but no code attached.

If the query is incomplete, the system returns **targeted follow-up questions** to the user — not a generic "please provide more detail" but specific questions about what's missing (e.g., *"What mass and acceleration values should I use?"*). The pipeline does NOT proceed until the user provides enough information. This prevents specialists from generating answers based on assumptions that may be wrong.

Once the query is deemed complete, it moves to domain classification.

#### Domain Classification — Few-Shot LLM Routing

The Orchestrator uses the Qwen 2.5-7B base model to classify queries into specialist domains. The classification prompt includes:

1. **Explicit domain descriptions** — each specialist's capabilities, not just names.
2. **Few-shot examples** — 2-3 example queries per domain showing correct classification, covering edge cases and ambiguous phrasing.
3. **Structured output** — the model outputs a JSON list of activated domains with a confidence score per domain.

If the highest confidence score is **below 95%**, the Orchestrator re-runs classification with an **expanded prompt** that includes full specialist metadata, more few-shot examples, and asks the model to reason step-by-step before deciding. This two-pass approach ensures near-perfect routing accuracy — if the model isn't sure on the fast pass, it gets a second chance with more context.

No keyword heuristic fallback is used. Routing accuracy is mission-critical and only the LLM has the semantic understanding to get it right on ambiguous queries.

### 2. Domain Specialists

**Files**: `saber/specialist.py` (base class), `saber/specialists/*.py` (implementations)

Each specialist is a separate Python class with its own LoRA adapter. Currently 6 specialists:

| Specialist | Domain | LoRA Adapter |
|-----------|--------|-------------|
| ScienceSpecialist | `science` | `models/science_v2/` |
| CybersecuritySpecialist | `cyber` | `models/cyber_v2/` |
| CodingSpecialist | `coding` | `models/coding_v2/` |
| FinanceSpecialist | `finance` | `models/finance_v2/` |
| MedicalSpecialist | `medical` | `models/medical_v2/` |
| ArchitectureSpecialist | `architecture` | `models/architecture_v2/` |

**How a specialist processes a query:**

1. Receives a `TASK_SIGNAL` from the pipeline.
2. Confirms understanding via `CONFIRMATION_SIGNAL`.
3. Loads its domain LoRA adapter via `LLMEngine` (context manager — only one model in memory at a time).
4. Generates its answer with the **CoT Maintainer** plugged in during execution — the reasoning chain is built step by step (`IDENTIFY → ANALYZE → HYPOTHESIZE → EVIDENCE → EVALUATE → CONCLUDE`).
5. Parses the raw LLM output into structured **Claim** objects (Pydantic models with `statement`, `confidence`, `domain`, `status`).
6. Packages claims + full CoT chain into a `COT_SIGNAL`.
7. The Sentinel runs its verification checks on this specialist's output (see §4).
8. If verification passes (GREEN_CHITs), the output is finalized. If flags are raised, the answer is rewritten.

### 3. Chain-of-Thought (CoT) Maintainer

**File**: `saber/cot_maintainer.py`

A bidirectional working memory module **plugged into each specialist during execution**. The specialist reads from and writes to it as it reasons.

**Step Types**: `IDENTIFY` → `ANALYZE` → `HYPOTHESIZE` → `EVIDENCE` → `EVALUATE` → `CONCLUDE`

Each step records:
- `step_number`, `action`, `content`, `confidence`
- `evidence_refs` (sources cited)
- `depends_on` (which prior steps this step builds on — forms a DAG)
- `timestamp`

**Post-Processing Cleanup** (runs after specialist finishes):
- Deduplication: Steps with >85% text similarity are merged (keeps higher-confidence version).
- Loop detection: Prevents autoregressive repetition.
- Consecutive action consolidation: Merges adjacent steps with the same action type.

The cleaned CoT chain is exported into the COT_SIGNAL and travels with the claims through the rest of the pipeline.

### 4. Sentinel Verification Kernel

**File**: `saber/sentinel.py`

The independent verification authority. **Critically uses the unbiased base model** (not the fine-tuned LoRA adapter) to prevent self-checking bias.

**Two levels of verification on each specialist's output:**

#### Level 1: Signal Integrity (Cryptographic)
- Recomputes SHA-256 hash over the Signal payload.
- Compares against `integrity_hash` set when the Signal was created.
- Mismatch → immediate `FLAG_SIGNAL` with severity `CRITICAL` (tampering or corruption).

#### Level 2: Semantic Verification (LLM + Web Grounding)
- **Online mode**: Queries DuckDuckGo for each specialist claim. Search results are injected as "ground truth" into the verification prompt.
- **Offline mode**: Falls back to logical consistency checks only.
- **Targeted verification routing**: Different aspects get checked by different reviewers:
  - Cyber content → `technical_accuracy` by cyber, `logical_reasoning` by science
  - Science content → `factual_accuracy` + `mathematical_reasoning` by science
  - Medical content → `clinical_accuracy` by medical, `logical_reasoning` by science
- Runs the configured number of verification cycles (tier-dependent).
- Each cycle: if all checks return `CONFIRMED` → `GREEN_CHIT`. If errors found → `FLAG_SIGNAL` with structured JSON (issue_type, severity, evidence, reasoning, proposed_fix).
- Flagged answers are **rewritten by the LLM** to integrate corrections seamlessly.

#### CoT Step-Level Verification
- Checks that the first step is `IDENTIFY` and last step is `CONCLUDE`.
- Detects sharp confidence drops (>0.3 between consecutive steps).
- Verifies each step logically follows from the previous steps (LLM check).

### 5. Meta-Reasoning Layer

**File**: `saber/meta_reasoner.py`

**Runs AFTER all specialists have generated and verified their outputs.** It does NOT generate domain answers — it is purely a synthesis engine.

It receives the verified claims and CoT chains from all activated specialists and produces one coherent final answer through a structured 6-section analysis:

1. **CLAIM EXTRACTION** — Lists the key claims from each specialist.
2. **CONFIDENCE ANALYSIS** — Assesses specialist confidence levels and assigns weights.
3. **CONFLICT DETECTION** — Identifies contradictions or gaps between specialists. Writes "None" if clean.
4. **TRADEOFF EVALUATION** — Analyses tradeoffs when specialists disagree.
5. **RESOLUTION PATH** — Explains how conflicting views are reconciled.
6. **FINAL ANSWER** — The single, coherent, detailed answer to the user's query.

**Disagreement Detection**: If specialist confidence scores have a spread > 0.3, it flags a disagreement and logs it.

**Confidence Scoring**:
```
final_confidence = avg(specialist_claim_confidences) − (0.02 × total_flags_raised) − (0.01 × verification_cycles_run)
```

### 6. Signal Schema

**File**: `saber/signal.py`

All inter-component communication uses typed, Pydantic-validated Signal objects. No raw strings or ad-hoc dicts.

**Signal Types** in lifecycle order:
```
QUERY_SIGNAL         →  User query enters system
TASK_SIGNAL          →  Orchestrator assigns task to specialist
CONFIRMATION_SIGNAL  →  Specialist acknowledges task
COT_SIGNAL           →  Specialist returns claims + CoT chain
VERIFICATION_SIGNAL  →  Sentinel check result (GREEN_CHIT)
FLAG_SIGNAL          →  Sentinel flags an error
PATCH_SIGNAL         →  Correction applied
AUDIT_SIGNAL         →  Internal audit events
HEALTH_SIGNAL        →  Specialist health updates
```

Every Signal has:
- `signal_id` (UUID), `signal_type`, `query_id`, `source_id`, `target_id`
- `timestamp`, `version` (2.0.0), `priority`, `confidence`
- `integrity_hash` (SHA-256 of payload)
- `payload` (arbitrary dict)

### 7. LLM Engine

**File**: `saber/llm_engine.py`

Context-managed dynamic model loader. Only one model in memory at any time.

```python
with LLMEngine("models/science_v2") as engine:
    response = engine.generate(prompt, system_prompt=system_prompt)
# Model auto-unloaded, GPU cache cleared
```

- Auto-detects PEFT/LoRA adapters (reads `adapter_config.json`).
- Device fallback: CUDA → MPS → CPU.
- Weight caching mode for benchmarks (`SABER_KEEP_MODELS_LOADED=1`).
- Generation: `temperature=0.3`, `top_p=0.9`, `repetition_penalty=1.05`.
- Handles Qwen ChatML format with `<|im_end|>` stop tokens.

### 8. Audit Logger & Decision Ledger

**File**: `saber/audit.py`

Thread-safe, append-only JSONL logger. Every event in the system is recorded.

**Decision Ledger** (one per query):
- Query text + selected specialists
- Initial specialist responses (claims + confidence)
- All flags raised + all corrections applied
- Per-cycle verification history
- Specialist disagreement records
- Meta-reasoning path (all 6 sections)
- Final resolution text + final confidence

**Failure Taxonomy** (`saber/errors.py`):
| Category | When |
|----------|------|
| `ROUTING_FAILURE` | No specialists activated |
| `SPECIALIST_FAILURE` | Specialist crash or task rejection |
| `VERIFICATION_FAILURE` | Signal integrity or semantic check fails |
| `CONSENSUS_FAILURE` | No valid specialist outputs |
| `SYNTHESIS_FAILURE` | Meta-reasoning synthesis fails |
| `SYSTEM_FAILURE` | Infrastructure failures |

---

## Training Pipeline

**Files**: `saber/training/trainer.py`, `saber/training/dataset_loader.py`

- **Base Model**: Qwen 2.5-7B-Instruct
- **Method**: LoRA (rank=16, alpha=32, dropout=0.05, targets: q_proj, v_proj, k_proj, o_proj)
- **Trainer**: TRL SFTTrainer with data packing
- **Config**: bf16, batch=8, grad_accum=4 (effective batch=32), 3 epochs, lr=2e-4
- **Hardware**: 3× RTX 6000 Ada / H100 80GB
- **Data Processing**: Quality filtering + 30% CoT injection + ChatML formatting

### Training ↔ Evaluation Separation (Zero Data Leakage)

| Domain | Trained On | Evaluated On | Overlap |
|--------|-----------|-------------|---------|
| Science | ScienceQA, SciQ, CAMEL-AI | **GPQA Diamond** (198 cases) | None |
| Cyber | MITRE STIX, CyberQA, Synthetic ATT&CK | **SecBench** (100 cases) | None |
| Coding | APPS, CodeContests, LeetCode, CodeFeedback | **HumanEval** (164 cases) | None |
| Finance | Finance-Alpaca, ConvFinQA, Finance-Instruct | **FinQA Math** (80 cases) | None |
