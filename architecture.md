# SABER — System Architecture

> **SABER** (Specialist Agent-Based Expert Reasoning) is a modular multi-specialist AI framework built on a single resident open-weight base model (`Qwen2.5-7B-Instruct`). Domain-specialized expert models generate verified, chain-of-thought-backed answers grounded by a 2-pass verification kernel (**Sentinel**) with offline knowledge base semantic retrieval. The **CoT Maintainer** is a core infrastructure component — a passive scratchpad/buffer that stores model reasoning and re-injects it as context in multi-step operations.

---

## End-to-End Pipeline

```
                                USER QUERY / MESSAGE
                                         │
                                         ▼
            ┌──────────────────────────────────────────────────────────┐
            │       RESIDENT BASE MODEL (bare Qwen2.5-7B-Instruct)     │
            │       - Single resident 7B base model in RAM/VRAM        │
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
        │  BARE 7B NATIVE CHAT      │         │  ORCHESTRATOR ADAPTER     │
        │  - Instant warm response  │         │  - Routes query to the    │
        │    (<30ms latency)        │         │    correct specialist     │
        │  - Zero adapter loading   │         │    via TASK_SIGNAL        │
        └───────────────────────────┘         └─────────────┬─────────────┘
                                                            │ TASK_SIGNAL
                                            ┌───────────────┴───────────────┐
                                      CYBER, FINANCE, ETC.                  CODING ONLY
                                            ▼                               ▼
                              ┌───────────────────────────┐   ┌───────────────────────────┐
                              │  SPECIALIST EXECUTION     │   │  CODING SECTOR PLANNER    │
                              │  - Hot-swaps DoRA adapter │   │  - Decomposes into sub-   │
                              │    adapter for domain     │   │    tasks (Python, JS, SQL)│
                              │  - Model generates natural│   │  - Dispatches sequentially│
                              │    reasoning and writes it│   │    to Language Specialists│
                              │    to CoT Maintainer      │   │  - Monitors Code Sentinel │
                              └──────┬────────────┬───────┘   └─────────────┬─────────────┘
                                     │            │                         │
                              writes │     reads  │ (multi-step)            │ writes/reads 
                                     ▼            ▼                         ▼
                              ┌───────────────────────────┐   ┌───────────────────────────┐
                              │  CoT MAINTAINER           │   │  SHARED MEMORY            │
                              │  Pure passive buffer.     │   │  - Plan & Code Blocks     │
                              │  Stores what the model    │   │  - Language specs write   │
                              │  writes. Re-injects it    │   │    code sequentially      │
                              │  on subsequent steps.     │   │  - CoT maintained per spec│
                              └─────────────┬─────────────┘   └─────────────┬─────────────┘
                                            │ reasoning chain               │ code + tests
                                            ▼                               ▼
                              ┌───────────────────────────┐   ┌───────────────────────────┐
                              │  SENTINEL VERIFIER KERNEL │   │  CODE SENTINEL            │
                              │  Pure Python — no adapter │   │  - Runs unit tests on     │
                              │  1. Semantic search via   │   │    sandboxed code         │
                              │    BAAI/bge-base-en-v1.5  │   │  - Generates adversarial  │
                              │     offline SQLite KB     │   │    tests (via base LLM)   │
                              │  2. Cosine sim threshold  │   │  - Returns CONFIRMED or   │
                              │  3. FLAG triggers rewrite │   │    FLAG (triggers rewrite)│
                              └─────────────┬─────────────┘   └─────────────┬─────────────┘
                                            │ Verified output               │ Verified code
                                            ▼                               ▼
                              ┌───────────────────────────┐   ┌───────────────────────────┐
                              │  META REASONER            │   │  RESPONSE OUTPUT          │
                              │  Synthesizes final answer │   │  (Bypasses Meta Reasoner) │
                              │  Appends Sentinel Footer: │   │  Appends Sentinel Footer: │
                              │  ⚡ Offline KB Verified    │   │  ⚡ Code Sentinel Verified │
                              └───────────────────────────┘   └───────────────────────────┘
```

---

## Core Design Principles

1. **Domain-Agnostic by Default**: System prompts dynamically inject the domain (e.g., `You are a {domain} AI specialist.`). No component hardcodes a specific domain like "cybersecurity expert."

2. **CoT Maintainer is Infrastructure, Not Prompt Engineering**: The CoT Maintainer lives in the pipeline layer. The model never sees it explicitly. It stores model output, and on multi-step operations, those stored notes are injected back as context. The model references its own prior written work instead of relying on internal memory.

3. **Sentinel is Not an Adapter**: Sentinel is a pure Python verification kernel with zero adapter loading overhead. It does not call the LLM unless a FLAG is raised.

4. **Benchmark = Real System**: Benchmarks run the real SABER pipeline with no extra prompt engineering or special instructions. The same system prompt used for real user queries is used during evaluation.

---

## Component Details

### 1. 2-Tiered Intent Gate & Orchestrator

**File**: `saber/orchestrator.py`

The entry point. Operates via **Base-First Dynamic Adapter Insertion**:

1. **Tier 1 (Fast Direct Pattern Match <1ms)**: Normalized alphanumeric check over greetings, slang, pleasantries.
2. **Tier 2 (7B LLM Semantic Intent Gate <15ms)**: 4-token prompt gate evaluating semantic intent for unstructured conversational inputs.
3. **Casual Chat Fast-Path**: Responds natively using the bare 7B base model with zero adapter loading overhead.

When a `DOMAIN_QUERY` is identified, the Orchestrator adapter is dynamically plugged onto the resident base model to perform semantic intent routing and domain selection.

| Step | What It Does |
|------|-------------|
| **Casual Chat Gating** | Intercepts greetings and small talk. Generates warm native answers instantly. |
| **Ambiguity Detection** | Scores query ambiguity (0–1). Queries ≥ 0.70 rejected with clarification request. |
| **Semantic Intent Classification** | Few-shot semantic routing to disambiguate polysemous queries. |
| **Specialist Selection** | Activates specialists whose domain relevance score ≥ 0.50. |
| **Task Decomposition** | Splits multi-domain queries into domain-specific `TASK_SIGNAL` payloads. |

---

### 2. Specialist Execution Engine

**Files**: `saber/specialist.py`, `saber/specialists/*`

Domain-specific experts executing via **DoRA Adapters** plugged onto the resident `Qwen2.5-7B-Instruct` base model:

- **Cybersecurity Specialist** (`saber/specialists/cybersecurity.py`): CVE analysis, MITRE ATT&CK, incident response, threat modeling.
- **Finance Specialist** (`saber/specialists/finance.py`): 10-K analysis, EBITDA, portfolio valuation.
- **Coding Sector** (`saber/coding/*`): Hierarchical multi-agent sub-system with planner, language specialists (Python, JS, SQL), shared memory, and Code Sentinel. See Section 3 below.
- **Architecture Specialist** (`saber/specialists/architecture.py`): Microservices, distributed systems, cloud infrastructure.

**System Prompt Pattern** (domain-agnostic):
```
You are a {domain} AI specialist.
```

No hardcoded domain names. No extra CoT instructions in the prompt. The model reasons naturally. The CoT Maintainer captures that reasoning passively.

---

### 3. Coding Sector (Hierarchical Multi-Agent Sub-System)

**Directory**: `saber/coding/`

When the Orchestrator routes a query to the coding domain, it does **not** hit a single specialist. Instead it enters a **hierarchical sub-system** — the Coding Sector — which operates as a self-contained multi-agent pipeline with its own planner, language-specific specialists, shared memory, and a specialized **Code Sentinel**.

> **Coding output bypasses the Meta Reasoner.** After Code Sentinel unit test clearance, the assembled code goes directly to the user. Code needs to pass tests, not be synthesized across domains.

#### Coding Sector Pipeline

```
    Orchestrator
         │ TASK_SIGNAL (domain=coding)
         ▼
┌─────────────────────────────────────────────────────────┐
│              CODING SECTOR PLANNER                      │
│  (Mini-Orchestrator / Planning Adapter)                 │
│                                                         │
│  1. Analyzes the coding query                           │
│  2. Decomposes into sub-tasks with language tags        │
│  3. Writes PLAN to Shared Memory                        │
│  4. Dispatches sub-tasks sequentially to Language        │
│     Specialists (hot-swapping adapters one at a time)   │
│  5. On Code Sentinel failure → re-dispatches with       │
│     failure context (max 2 retries per specialist)      │
└───────────┬─────────────────────────────────────────────┘
            │ Writes PLAN
            ▼
┌─────────────────────────────────────────────────────────┐
│                   SHARED MEMORY                         │
│  (Persistent Collaborative Workspace)                   │
│                                                         │
│  ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌──────────┐ │
│  │  PLAN    │ │ CODE_BLOCKS│ │ THOUGHTS │ │  TESTS   │ │
│  │ Planner  │ │ Each spec  │ │ Each spec│ │ & RESULTS│ │
│  │ writes,  │ │ writes its │ │ writes   │ │ Code     │ │
│  │ all read │ │ code, all  │ │ notes,   │ │ Sentinel │ │
│  │          │ │ can see    │ │ all can  │ │ see all  │ │
│  │          │ │ all code   │ │ see all  │ │ verdicts │ │
│  └──────────┘ └────────────┘ └──────────┘ └──────────┘ │
│                                                         │
│  Persisted to: data/coding_memory/{task_id}.json        │
└───────────┬─────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│       LANGUAGE SPECIALISTS (Sequential Execution)       │
│       Hot-swap adapters one at a time                   │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   PYTHON     │  │  JAVASCRIPT  │  │     SQL      │  │
│  │  SPECIALIST  │  │  SPECIALIST  │  │  SPECIALIST  │  │
│  │ - Own DoRA   │  │ - Own DoRA   │  │ - Own DoRA   │  │
│  │ - Own CoT    │  │ - Own CoT    │  │ - Own CoT    │  │
│  │ - Reads plan │  │ - Reads plan │  │ - Reads plan │  │
│  │ - Reads all  │  │ - Reads all  │  │ - Reads all  │  │
│  │   other code │  │   other code │  │   other code │  │
│  │ - Writes own │  │ - Writes own │  │ - Writes own │  │
│  │   code +     │  │   code +     │  │   code +     │  │
│  │   thoughts   │  │   thoughts   │  │   thoughts   │  │
│  │ - Writes own │  │ - Writes own │  │ - Writes own │  │
│  │   unit tests │  │   unit tests │  │   unit tests │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└───────────┬─────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│              CODE SENTINEL                              │
│  (Unit Test Verification — NOT KB Lookup)               │
│                                                         │
│  Phase 1: Run specialist-written unit tests             │
│  Phase 2: Generate adversarial unit tests via base LLM  │
│           and run them                                  │
│                                                         │
│  CONFIRMED → Assembled code goes directly to user       │
│              (bypasses Meta Reasoner)                    │
│  FLAG → Returns to Planner with failure context          │
│         for specialist rewrite                          │
└───────────┬─────────────────────────────────────────────┘
            │ (bypasses Meta Reasoner)
            ▼
         USER RESPONSE
         ⚡ Code Sentinel Verified (X/Y tests passed)
```

#### 3a. Coding Sector Planner

**File**: `saber/coding/planner.py`

Mini-orchestrator for the coding domain. Receives `TASK_SIGNAL` from the main Orchestrator, decomposes the coding task, and manages the entire coding sub-pipeline.

| Step | What It Does |
|------|-------------|
| **Query Analysis** | Determines which languages are needed and estimates complexity. |
| **Task Decomposition** | Splits into sub-tasks with language tags and dependency ordering. |
| **Plan Writing** | Writes the full plan to Shared Memory for all specialists to read. |
| **Sequential Dispatch** | Dispatches sub-tasks one at a time, hot-swapping adapters. Respects dependency order (e.g., backend before frontend). |
| **Failure Handling** | On Code Sentinel FLAG, re-dispatches to failing specialist with the test failure context. Max 2 retries per specialist. |
| **Synthesis** | Assembles all code blocks from Shared Memory into a structured response after all tests pass. |

#### 3b. Shared Memory

**File**: `saber/coding/shared_memory.py`

Persistent collaborative workspace scoped to a coding task. All language specialists and the planner read and write here. Persisted to `data/coding_memory/{task_id}.json` after task completion for post-hoc review.

| Section | Writer | Reader | Purpose |
|---------|--------|--------|---------|
| **PLAN** | Planner | All specialists | Decomposed task plan with sub-task assignments. |
| **CODE_BLOCKS** | Each specialist | All specialists + Code Sentinel | Code written by each specialist. Cross-language visibility enables interface matching. |
| **THOUGHTS** | Each specialist | All specialists | Append-only reasoning notes and design decisions. |
| **TESTS & RESULTS** | Specialists + Code Sentinel | Planner | Unit tests written by specialists and Code Sentinel verdicts. |

**Schemas**:
```python
@dataclass
class SubTask:
    subtask_id: str           # "st_001"
    description: str
    language: str             # "python" | "javascript" | "sql"
    depends_on: List[str]     # Other subtask_ids this depends on
    status: str               # "pending" | "in_progress" | "done" | "failed"

@dataclass
class CodeBlock:
    subtask_id: str
    specialist: str           # "python" | "javascript" | "sql"
    code: str                 # The actual code
    tests: str                # Unit tests written by this specialist
    filename: str             # Suggested filename (e.g. "main.py")
    version: int              # Incremented on rewrites after test failure

@dataclass
class Thought:
    specialist: str
    subtask_id: str
    content: str              # Reasoning note visible to all specialists
    timestamp: str
```

#### 3c. Language Specialists

**Directory**: `saber/coding/specialists/`

Three language-specific specialists (extensible), each with its own DoRA adapter and its own CoT Maintainer:

| Specialist | File | Adapter |
|-----------|------|--------|
| Python | `saber/coding/specialists/python_spec.py` | `models/coding/python_v1` |
| JavaScript | `saber/coding/specialists/javascript_spec.py` | `models/coding/javascript_v1` |
| SQL | `saber/coding/specialists/sql_spec.py` | `models/coding/sql_v1` |

Each specialist's context window includes:
1. The **plan** from Shared Memory (knows what to build)
2. **Code** already written by other specialists (knows what interfaces to match)
3. **Thoughts** from other specialists (knows design decisions)
4. Its own **CoT Maintainer** history (private reasoning chain for multi-step reasoning)

**Execution is sequential**: Only one adapter loaded at a time. The Planner dispatches sub-tasks in dependency order, hot-swapping adapters between specialists. On a 16GB Mac, this prevents OOM.

**System prompt pattern**:
```
You are a {language} programming specialist working as part of a team.

PROJECT PLAN:
{plan_from_shared_memory}

YOUR ASSIGNED TASK:
{subtask_description}

CODE FROM OTHER TEAM MEMBERS:
{other_specialists_code_blocks}

NOTES FROM OTHER TEAM MEMBERS:
{other_specialists_thoughts}

Write your code, then write unit tests for your code.
Write your reasoning and any notes for other team members.
```

#### 3d. Code Sentinel

**File**: `saber/coding/code_sentinel.py`

Specialized verification kernel for code. **Fundamentally different from the Main Sentinel** — verifies code through unit test execution, not through semantic KB search.

| | Main Sentinel (Sections 6–7) | Code Sentinel |
|---|---|---|
| **Verifies by** | Semantic KB search via `BAAI/bge-base-en-v1.5` | Running unit tests |
| **Ground truth** | Offline KB passages | Do the tests pass? |
| **LLM calls** | Only on FLAG (rewrite) | Base LLM for adversarial test generation |
| **Flags when** | Factual contradiction with KB | Unit test failure |
| **Used by** | Cyber, Finance, Architecture specialists | Coding Sector only |

**Phase 1 — Specialist Unit Tests**:
- Takes each specialist's unit tests from Shared Memory (`CodeBlock.tests`)
- Writes code + tests to a temp directory
- Runs via `subprocess.run()` with 10s timeout
- Records pass/fail + stdout/stderr to Shared Memory

**Phase 2 — Adversarial Unit Tests**:
- Uses the **base model (no adapter)** to generate edge-case tests the specialist didn't write
- Focus: empty inputs, boundary values, null/None, type mismatches, off-by-one
- Runs these adversarial tests against the specialist's code
- This is the only phase where Code Sentinel calls the LLM

**Verdict**:
- **CONFIRMED**: All tests pass → Planner assembles code → goes **directly to user** (bypasses Meta Reasoner). Footer: `⚡ Code Sentinel Verified (X/Y tests passed)`
- **FLAG**: Test failure → returns to Planner with: which specialist failed, the failing test code, the stack trace/error output. Planner re-dispatches to that specialist with failure context for rewrite (max 2 retries).

---

### 4. MLX Engine (Apple Silicon Inference)

**File**: `saber/mlx_engine.py`

Context-managed MLX inference engine optimized for Apple Silicon (M-series Macs):
- Loads base model or DoRA adapter via `with MLXEngine(model_path, adapter_path) as engine:`.
- Automatic Metal cache cleanup on exit (`mx.clear_cache()`).
- Uses `mlx_lm.load` and `mlx_lm.generate` for Apple Silicon-optimized inference.
- Supports single-turn generation with optional system prompt via ChatML template.

---

### 5. CoT Maintainer

**File**: `saber/cot_maintainer.py`

**What it is**: A passive scratchpad and buffer. It is a core SABER infrastructure component that lives in the pipeline, not in the prompt.

**How it works**:
1. **The model writes**: The specialist model generates its natural reasoning as output.
2. **CoT Maintainer stores it**: The `add_step()` method records each `ReasoningStep` (action, content, confidence, evidence refs, dependencies) into a `CoTChain`.
3. **On subsequent steps, the stored notes are re-injected**: `read_summary()` formats the stored steps and injects them back into the next generation call as context. The model references its own prior written work.
4. **The maintainer does zero processing**: No parsing, no inference, no scoring. Pure storage and retrieval.

**The model never knows the difference** between receiving its own stored reasoning back vs. any other context. The CoT Maintainer is transparent to the model.

**Key methods**:
| Method | Purpose |
|--------|---------|
| `begin_chain(domain, query_id)` | Start a new reasoning chain for a task. |
| `add_step(action, content, confidence, ...)` | Store one reasoning step written by the model. |
| `read_summary()` | Format all stored steps as text for re-injection into the next prompt. |
| `conclude(conclusion, confidence)` | Mark chain complete, add a CONCLUDE step. |
| `cleanup()` | Deduplicate and merge redundant steps (similarity > 0.85). |
| `export_for_signal()` | Export chain as a dictionary payload for Sentinel verification. |
| `reset()` | Archive the current chain and reset for the next task. |

**Reasoning Step Schema**:
```python
class ReasoningStep:
    step_number: int
    action: str        # IDENTIFY | ANALYZE | HYPOTHESIZE | EVIDENCE | EVALUATE | CONCLUDE
    content: str
    confidence: float
    evidence_refs: List[str]
    depends_on: List[int]   # step numbers this step builds on
    timestamp: str
```

**NOT applicable to any specific domain**: The CoT Maintainer is domain-agnostic infrastructure used by every specialist equally.

---

### 6. Sentinel Verification Kernel

**File**: `saber/sentinel.py`, `scripts/run_sentinel_offline_kb.py`

The independent verification authority. **Sentinel is NOT an adapter** — it is a pure Python verification kernel with zero adapter loading overhead. It never calls the LLM for verification; it uses embedding-based semantic search against the offline KB.

#### Step 1: Semantic KB Lookup (BAAI/bge-base-en-v1.5)

**Model**: `BAAI/bge-base-en-v1.5` micro embedding model (stored locally at `models/bge-base-en-v1.5`)

- The Sentinel base model (or extraction heuristic) extracts verifiable claims/facts from the Specialist's generated CoT.
- Takes the extracted claim (up to 500 chars) as the query string.
- Generates a query embedding using `SentenceTransformer.encode()`.
- Opens the offline SQLite KB (`data/offline_kb/{domain}_kb_v2.db`).
- Loads all passage embeddings from the `embeddings` table (stored as `float32` blobs).
- Computes cosine similarity between the query embedding and every passage embedding.
- Returns the highest-scoring passage **only if score > 0.4** (relevance threshold).

```
KB Schema:
  knowledge table:  id, domain, passage, source, created_at
  embeddings table: id (FK), embedding_blob (float32 numpy array)
```

#### Step 2: Verification Decision

**If no KB hit** (score < 0.4):
- No web search fallback (offline mode).
- CoT answer is passed through unchanged.
- Response footer: `⚡ Verified by SABER Sentinel (No KB hit — passed through)`

**If KB hit** (score ≥ 0.4):
- The KB passage is used as ground truth.
- Sentinel (base model, no adapter) evaluates whether the specialist's CoT answer contradicts the KB passage.
- If reasoning is sound → responds with `CONFIRMED`.
- If a contradiction is found → responds with a JSON flag:
  ```json
  {
    "issue_type": "FACTUAL_ERROR | REASONING_ERROR",
    "reasoning": "...",
    "proposed_fix": "..."
  }
  ```

#### Step 3: Rewrite on FLAG

If Sentinel raises a flag:
- The specialist adapter (e.g., `models/cyber_v2`) is hot-loaded.
- A rewrite prompt is constructed containing: the original question, the error explanation, and the proposed fix.
- The adapter generates a corrected answer.
- Response footer: `⚡ Verified by SABER Sentinel (Offline KB) [Rewritten]`

#### Sentinel Is Domain-Agnostic

Sentinel's system prompt:
```
You are the SABER SENTINEL verifying {domain} content.
```

The domain is injected dynamically. Sentinel works identically for cyber, finance, coding, architecture, or any other domain.

---

### 7. Offline Knowledge Base

**Location**: `data/offline_kb/{domain}_kb_v2.db`

- **cyber_kb_v2.db**: 11,580 passages drawn from MITRE ATT&CK STIX, CyberMetric, NVD advisories, and the pAILabs infosec-security-qa dataset.
- Passages are stored in format-agnostic text form (facts and concepts, not rigid Q&A pairs).
- All passages are pre-embedded using `BAAI/bge-base-en-v1.5` and stored as `float32` binary blobs.
- Semantic search at inference time: O(n) cosine similarity scan over all passages (fast for <50k entries).

---

### 8. Signal Schema

**File**: `saber/signal.py`

Strongly-typed Pydantic dataclass defining all inter-component communication:

| Signal | Purpose |
|--------|---------|
| `QUERY_SIGNAL` | User query entering the system. |
| `TASK_SIGNAL` | Orchestrator sub-task dispatched to a specialist. |
| `CONFIRMATION_SIGNAL` | Specialist confirming task receipt. |
| `COT_SIGNAL` | Specialist submitting reasoning chain + claims. |
| `VERIFICATION_SIGNAL` | Sentinel returning GREEN_CHIT confirmation. |
| `FLAG_SIGNAL` | Sentinel raising an error/contradiction with correction. |
| `OUTPUT_SIGNAL` | Final synthesized response. |
| `AUDIT_SIGNAL` | Audit trail events. |
| `CODE_PLAN_SIGNAL` | Coding Planner's decomposed task plan written to Shared Memory. |
| `CODE_DISPATCH_SIGNAL` | Planner dispatching a sub-task to a language specialist. |
| `CODE_BLOCK_SIGNAL` | Language specialist submitting code + tests to Shared Memory. |
| `CODE_SENTINEL_VERDICT` | Code Sentinel's unit test execution results (CONFIRMED/FLAG). |
| `CODE_REWRITE_SIGNAL` | Planner re-dispatching to specialist after Code Sentinel FLAG. |

Every signal is cryptographically signed with a SHA-256 `integrity_hash` over its payload via `freeze_and_hash()`.

---

### 9. Decision Ledger & Audit Trail

**File**: `saber/audit.py`

Thread-safe, append-only JSON-Lines audit log (`logs/audit.jsonl`). Records every query's full lifecycle:
- Query reception & ambiguity score
- Casual chat gating status
- Specialist selection & task signals
- Signal integrity check results
- Sentinel KB hit/miss & verification flags
- Self-correcting rewrite iterations

---

## Training Pipeline

**Files**: `saber/training/trainer.py`, `saber/training/dataset_loader.py`

### Training Method: DoRA SFT

- **Base Model**: `Qwen/Qwen2.5-7B-Instruct`
- **Method**: Weight-Decomposed DoRA (DoRA), rank `r=64`, `α=128`, dropout `0.05`
- **Target Modules**: All 7 linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Precision**: `bfloat16`, batch size 8, warmup ratio 0.03, 3 epochs
- **Hardware**: NVIDIA H100 80GB SXM (cloud training)

### Known Issue: Data Collator (To Fix Before Next Training Run)

The current trainer uses `DataCollatorForLanguageModeling(mlm=False)`, which computes cross-entropy loss over the **entire sequence** (system prompt + question + answer). This means the model spends its gradient budget learning to predict prompt tokens it already knows, instead of learning domain-specific answers. This explains why the DoRA adapter fails to improve over the base model on benchmarks.

**Remediation Plan**:

1. **Replace Collator**: Swap `DataCollatorForLanguageModeling(mlm=False)` with `trl.DataCollatorForCompletionOnlyLM`.
2. **Apply Response Template**: Use the exact ChatML response template `"<|im_start|>assistant\n"` so loss is masked (-100) on everything except the assistant's answer tokens. (Note: `DataCollatorForCompletionOnlyLM` does exact substring matching).
3. **Verify Template Rendering**: Verify that the chat template/tokenizer setup actually produces `"<|im_start|>assistant\n"` verbatim as the turn boundary.
4. **Build Verification Script**: Create a debug snippet (via `--debug_collator` flag or standalone script) that:
   - Pulls one batch from the training dataloader after collation.
   - Decodes `input_ids` back to text.
   - Decodes only the tokens where `labels != -100`.
   - Prints them side-by-side to visually confirm the "labels" text is ONLY the assistant's answer.
5. **Preserve Hyperparameters**: Do not change the DoRA config (r=64, alpha=128, dropout=0.05), target modules, learning rate, batch size, epochs, or dataset loading logic in `dataset_loader.py`. This is strictly a collator/loss-masking fix.
6. **Pre-Training Validation**: Run the verification script on 3 random samples from the cyber training set and visually confirm the output before kicking off the real training run.

### Training Data Sources (Cyber Domain)

- MITRE ATT&CK STIX knowledge graph entries
- CyberQA & Trendyol cybersecurity QA pairs
- CyberMetric benchmark questions (MCQ with CoT rationale)
- Synthetic MITRE incident response scenarios (hand-crafted)
- pAILabs infosec-security-qa (11k+ entries)

---

## Benchmark Evaluation

### CyberMetric-80 (Official Dataset)

**Source**: [https://github.com/CyberMetric/CyberMetric](https://github.com/CyberMetric/CyberMetric) — `CyberMetric-80-v1.json`

80 expert-validated multiple-choice questions covering cryptography, network security, NIST standards, PCI DSS, and incident response.

### Evaluation Modes

| Mode | Description |
|------|-------------|
| **Base QWEN** | Raw `Qwen2.5-7B-Instruct` baseline with no adapter. |
| **Cyber Adapter** | Base model + `cyber_v2` DoRA adapter, no CoT. |
| **Adapter + CoT** | Adapter + passive CoT Maintainer (model reasons naturally, maintainer buffers and re-injects). No extra prompt engineering. |
| **Sentinel (Offline KB)** | Adapter + CoT + Sentinel verification via `BAAI/bge-base-en-v1.5` semantic search against `cyber_kb_v2.db`. Rewrites on FLAG. |

### Grader

**Script**: `scripts/eval_cybermetric.py`

Rigorous regex parser that extracts the final answer letter from model output. Handles:
- `ANSWER: X`, `Final Answer: X`, `The correct option is X`
- Chinese language answer patterns
- Direct prefix format `A. ...`
- Does NOT use ungrounded fallback (avoids false positives from strings like "Section D")

### Partial Results (CyberMetric-80, as of last run)

| Mode | Correct | Total | Accuracy |
|------|---------|-------|----------|
| Base QWEN | 75 | 80 | **93.75%** |
| Cyber Adapter | 75 | 80 | **93.75%** |
| Adapter + CoT | 67 | 80 | **83.75%** (8 unparsed — formatting drift) |
| Sentinel (Offline KB) | ~94% on completed cases | 80 | Running |

> **Note on Adapter Performance**: The Adapter scores identical to Base QWEN because the training data collator was not masking prompt tokens (see Known Issue above). The adapter trained to predict the prompt itself rather than learning domain-specific answers. This is the primary issue to fix before re-training.

---

## File Map

### Root Files

| File | Purpose |
|------|---------|
| `architecture.md` | This document — master system architecture reference. |
| `requirements.txt` | Python dependencies. |
| `run.sh` | Main SABER execution entrypoint script. |
| `chat.py` | Interactive CLI chat interface. |

### `saber/` Module

| File | Purpose |
|------|---------|
| `saber/orchestrator.py` | 2-tiered intent gate & domain routing. |
| `saber/specialist.py` | Specialist execution engine. |
| `saber/mlx_engine.py` | Apple Silicon MLX inference engine. |
| `saber/cot_maintainer.py` | Passive CoT storage buffer (infrastructure). |
| `saber/sentinel.py` | Verification kernel core logic. |
| `saber/signal.py` | Strongly-typed inter-component signal schema. |
| `saber/audit.py` | Thread-safe append-only audit trail. |
| `saber/config.py` | Central configuration (model paths, thresholds). |
| `saber/context.py` | Session memory manager for multi-turn history. |
| `saber/meta_reasoner.py` | Multi-specialist output synthesis engine (coding bypasses this). |
| `saber/coding/planner.py` | Coding Sector mini-orchestrator and task decomposer. |
| `saber/coding/shared_memory.py` | Persistent collaborative workspace for coding specialists. |
| `saber/coding/code_sentinel.py` | Unit test verification kernel for code (not KB-based). |
| `saber/coding/specialists/python_spec.py` | Python language specialist. |
| `saber/coding/specialists/javascript_spec.py` | JavaScript language specialist. |
| `saber/coding/specialists/sql_spec.py` | SQL language specialist. |
| `saber/training/trainer.py` | DoRA SFT training pipeline. |
| `saber/training/dataset_loader.py` | Dataset download & preprocessing. |
| `saber/training/rewards.py` | GRPTO reward functions. |

### `scripts/`

| Script | Purpose |
|--------|---------|
| `scripts/run_mac_benchmark.py` | Full 3-pass benchmark runner (Base, Adapter, CoT) on Apple Silicon. |
| `scripts/run_sentinel_offline_kb.py` | Sentinel offline KB verification pass (4th mode). |
| `scripts/eval_cybermetric.py` | Regex grader for CyberMetric MCQ results. |
| `scripts/build_offline_kb.py` | Builds the offline SQLite KB with embeddings. |

### `data/`

| Path | Purpose |
|------|---------|
| `data/offline_kb/cyber_kb_v2.db` | 11,580-passage cybersecurity knowledge base with pre-computed `BAAI/bge-base-en-v1.5` embeddings. |
| `data/coding_memory/` | Persistent Shared Memory traces from Coding Sector tasks (one JSON per task). |

### `models/`

| Path | Purpose |
|------|---------|
| `models/cyber_v2` | Cybersecurity specialist DoRA adapter. |
| `models/coding/python_v1` | Python language specialist DoRA adapter. |
| `models/coding/javascript_v1` | JavaScript language specialist DoRA adapter. |
| `models/coding/sql_v1` | SQL language specialist DoRA adapter. |
| `models/bge-base-en-v1.5` | Local `BAAI/bge-base-en-v1.5` micro embedding model for Sentinel KB search. |
| `qwen-mlx-4bit/` | 4-bit quantized base model for Apple Silicon (MLX format). |
