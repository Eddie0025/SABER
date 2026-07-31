# SABER Development & Change Log

*This document maintains a persistent record of all architectural decisions, codebase changes, and system states to ensure context is never lost during development.*

---

## 2026-07-31

### 1. Repository State Reset
- **Action**: Purged the entire `saber/` codebase (except `architecture.md`) to start a clean, from-scratch rewrite of the multi-agent pipeline.
- **Reason**: The previous monolithic architecture was insufficient for coding tasks. 

### 2. Coding Sector Architecture
- **Action**: Designed and documented the **Coding Sector** in `architecture.md`.
- **Details**:
  - **Planner**: The Software Architecture Specialist now acts as the Coding Sector Planner (Mini-Orchestrator), decomposing coding tasks into sub-tasks (Python, JS, SQL).
  - **Shared Memory**: Created a JSON-backed collaborative workspace (`data/coding_memory/{task_id}.json`) where the planner writes the plan, and language specialists read/write their code blocks and reasoning thoughts.
  - **Sequential Execution**: Language specialists (Python, SQL, JS) execute sequentially to prevent OOM errors on Apple Silicon.
  - **Code Sentinel**: A specialized verification kernel that tests code by *running unit tests* and generating adversarial tests (via the base model), rather than relying on semantic KB search.
  - **Routing Bypass**: Output from the Coding Sector bypasses the Meta Reasoner and goes straight to the user upon Sentinel confirmation.

### 3. Sentinel KB Model Upgrade
- **Action**: Replaced the micro embedding model for the Main Sentinel.
- **Change**: Upgraded from `all-MiniLM-L6-v2` to `BAAI/bge-base-en-v1.5`.
- **Cleanup**: Deleted the old model directory `models/all-MiniLM-L6-v2` from disk.
- **Correction**: Updated `architecture.md` to clarify that the Sentinel extracts facts/claims from the specialist's CoT to use as the search query, rather than using the raw user prompt.

### 4. Training Pipeline "Known Issues" Logged
- **Action**: Documented a critical fix required for the data collator before the next training run.
- **Details**: The trainer was computing cross-entropy loss over the entire sequence. We documented a 6-step remediation plan in `architecture.md` to swap to `trl.DataCollatorForCompletionOnlyLM` using the `<|im_start|>assistant\n` template.
- **Action**: Updated all references of "LoRA" and "LoRA/DoRA" to **DoRA** across `architecture.md`, and fixed a typo changing GRPO to **GRPTO**.

### 5. Post-Training Validation Suite
- **Action**: Defined comprehensive read-only validation scripts for DoRA and GRPO training phases.
- **Details**:
  - `validate_dora.py`: Checks collator sanity (loss masking), weight deltas, adapter application eval, base vs adapter benchmark delta, and overfitting.
  - `validate_grpo.py`: Checks reward variance, KL divergence tracking, pre/post GRPO benchmark delta, reward hacking (top 10 rollouts), and cross-domain regression.
  - Outputs are standardized JSON reports (`validate_dora_report.json` and `validate_grpo_report.json`).

### 6. Dataset Curation Strategy Defined
- **Action**: Formalized the training data requirements and filtering criteria in `DATASETS.md`.
- **Details**: Specified exact dataset splits, target sizes, and gold-quality filtering logic (executability checks, length bounds, deduplication, benchmark leakage checks, and format consistency) for all 8 domain specialists, the orchestrator, and the meta-reasoner.

### 7. Core System Scaffolding Plan (Pending Execution)
- **Action**: Drafted `implementation_plan.md` to scaffold the core specialists and the meta reasoner.
- **Status**: Awaiting execution.
