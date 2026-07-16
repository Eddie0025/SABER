# -*- coding: utf-8 -*-
"""Run CYBER-001 through all 5 SABER benchmark modes sequentially.

Prints live output for each mode as it completes.
"""

import sys
import time
import os

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

CYBER_001 = (
    "A Windows Server 2022 domain controller begins making encrypted outbound "
    "connections to an IP address in a foreign country every 12 minutes.\n\n"
    "Security logs show:\n"
    "- A PowerShell script executed from a user's Downloads folder.\n"
    "- A new local administrator account was created.\n"
    "- Multiple ZIP archives containing HR records were created shortly before "
    "the outbound connections began.\n"
    "- EDR did not detect malware.\n\n"
    "As a cybersecurity analyst:\n"
    "1. Identify the likely attack stages.\n"
    "2. Map the activity to MITRE ATT&CK techniques.\n"
    "3. Assess severity and business impact.\n"
    "4. Identify attacker objectives.\n"
    "5. Recommend immediate containment actions.\n"
    "6. List forensic evidence that should be collected.\n"
    "7. Recommend long-term remediation measures.\n\n"
    "Provide technical reasoning for every conclusion."
)

OUTPUT_FILE = "/Users/adityavir/.gemini/antigravity-ide/brain/f0e0302e-3bd0-4295-aa04-85662d5a792a/cyber001_5modes_results.md"


def p(msg):
    """Print and flush immediately."""
    print(msg, flush=True)
    sys.stdout.flush()


def main():
    total_start = time.time()

    p("=" * 60)
    p("CYBER-001 — 5 MODE BENCHMARK")
    p("=" * 60)
    p("")

    # ---- Setup ----
    p("[SETUP] Importing SABER modules...")
    from saber.config import SaberConfig, VerificationTier
    from saber.registry import SpecialistRegistry
    from saber.benchmark import BenchmarkEngine
    p("[SETUP] Done.")

    # Build config & registry
    config = SaberConfig(
        verification_tier=VerificationTier.TIER_2,
        model_dir="models",
    )
    registry = SpecialistRegistry()

    p("[SETUP] Discovering specialists...")
    count = registry.auto_discover("saber.specialists")
    p(f"[SETUP] Discovered {count} specialists: {registry.list_domains()}")

    # Load the cyber_v1 model path into the cyber specialist
    cyber_spec = registry.get("cyber")
    if cyber_spec:
        cyber_spec.load_model("models/cyber_v1")
        p(f"[SETUP] Cyber specialist model set to: {cyber_spec.meta.model_path}")
    else:
        p("[SETUP] WARNING: No cyber specialist found!")

    # Create benchmark engine
    engine = BenchmarkEngine(config, registry)
    p("[SETUP] BenchmarkEngine ready.")
    p("")

    # ---- Results file ----
    with open(OUTPUT_FILE, "w") as f:
        f.write("# CYBER-001 — 5 Mode Benchmark Results\n\n")
        f.write("Running CYBER-001 (Incident Response) through all 5 SABER evaluation modes.\n\n")

    results = []

    # ---- MODE 1: BASE ----
    p("=" * 60)
    p("[1/5] MODE_BASE — Raw base model, zero-shot")
    p("=" * 60)
    start = time.time()
    try:
        res = engine.run_mode_base(CYBER_001)
        elapsed = time.time() - start
        p(f"  Latency: {elapsed:.1f}s | Flags: {res['flags']} | Conf: {res['conf']:.2f}")
        p("")
        p(res["answer"])
        p("")
        results.append(("MODE_BASE", res, elapsed))
        _append_result(OUTPUT_FILE, "MODE_BASE", "Raw base model, zero-shot", res, elapsed)
    except Exception as e:
        p(f"  FAILED: {e}")
        results.append(("MODE_BASE", None, time.time() - start))

    # ---- MODE 2: SELF_CRITIQUE ----
    p("=" * 60)
    p("[2/5] MODE_SELF_CRITIQUE — Base model + self-reflection rewrite")
    p("=" * 60)
    start = time.time()
    try:
        res = engine.run_mode_self_critique(CYBER_001)
        elapsed = time.time() - start
        p(f"  Latency: {elapsed:.1f}s | Flags: {res['flags']} | Conf: {res['conf']:.2f}")
        p("")
        p(res["answer"])
        p("")
        results.append(("MODE_SELF_CRITIQUE", res, elapsed))
        _append_result(OUTPUT_FILE, "MODE_SELF_CRITIQUE", "Base model + self-reflection rewrite", res, elapsed)
    except Exception as e:
        p(f"  FAILED: {e}")
        results.append(("MODE_SELF_CRITIQUE", None, time.time() - start))

    # ---- MODE 3: SPECIALIST ----
    p("=" * 60)
    p("[3/5] MODE_SPECIALIST — Cyber specialist only, no verification")
    p("=" * 60)
    start = time.time()
    try:
        res = engine.run_mode_specialist(CYBER_001, "cyber")
        elapsed = time.time() - start
        p(f"  Latency: {elapsed:.1f}s | Flags: {res['flags']} | Conf: {res['conf']:.2f}")
        p("")
        p(res["answer"])
        p("")
        results.append(("MODE_SPECIALIST", res, elapsed))
        _append_result(OUTPUT_FILE, "MODE_SPECIALIST", "Cyber specialist only, no verification", res, elapsed)
    except Exception as e:
        p(f"  FAILED: {e}")
        results.append(("MODE_SPECIALIST", None, time.time() - start))

    # ---- MODE 4: SPECIALIST_VERIFY ----
    p("=" * 60)
    p("[4/5] MODE_SPECIALIST_VERIFY — Specialist + Sentinel verification")
    p("=" * 60)
    start = time.time()
    try:
        res = engine.run_mode_specialist_verify(CYBER_001, "cyber", "CYBER-001")
        elapsed = time.time() - start
        p(f"  Latency: {elapsed:.1f}s | Flags: {res['flags']} | Conf: {res['conf']:.2f}")
        p("")
        p(res["answer"])
        p("")
        results.append(("MODE_SPECIALIST_VERIFY", res, elapsed))
        _append_result(OUTPUT_FILE, "MODE_SPECIALIST_VERIFY", "Specialist + Sentinel verification loop", res, elapsed)
    except Exception as e:
        p(f"  FAILED: {e}")
        results.append(("MODE_SPECIALIST_VERIFY", None, time.time() - start))

    # ---- MODE 5: SABER ----
    p("=" * 60)
    p("[5/5] MODE_SABER — Full pipeline: routing, multi-specialist, verification")
    p("=" * 60)
    start = time.time()
    try:
        res = engine.run_mode_saber(CYBER_001, "CYBER-001", "cyber")
        elapsed = time.time() - start
        p(f"  Latency: {elapsed:.1f}s | Flags: {res['flags']} | Conf: {res['conf']:.2f}")
        p("")
        p(res["answer"])
        p("")
        results.append(("MODE_SABER", res, elapsed))
        _append_result(OUTPUT_FILE, "MODE_SABER", "Full SABER pipeline", res, elapsed)
    except Exception as e:
        p(f"  FAILED: {e}")
        results.append(("MODE_SABER", None, time.time() - start))

    # ---- Summary ----
    total_elapsed = time.time() - total_start
    p("")
    p("=" * 60)
    p("SUMMARY")
    p("=" * 60)
    p(f"{'Mode':<30} {'Latency':>10} {'Flags':>8} {'Conf':>8}")
    p("-" * 60)
    for mode, res, elapsed in results:
        if res:
            p(f"{mode:<30} {elapsed:>9.1f}s {res['flags']:>8} {res['conf']:>7.2f}")
        else:
            p(f"{mode:<30} {elapsed:>9.1f}s {'FAIL':>8} {'N/A':>8}")
    p("-" * 60)
    p(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # Append summary to file
    with open(OUTPUT_FILE, "a") as f:
        f.write("\n## Summary\n\n")
        f.write(f"| Mode | Latency | Flags | Confidence |\n")
        f.write(f"|------|---------|-------|------------|\n")
        for mode, res, elapsed in results:
            if res:
                f.write(f"| {mode} | {elapsed:.1f}s | {res['flags']} | {res['conf']:.2f} |\n")
            else:
                f.write(f"| {mode} | {elapsed:.1f}s | FAIL | N/A |\n")
        f.write(f"\n**Total time:** {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)\n")

    p(f"\nResults written to: {OUTPUT_FILE}")
    p("Done!")


def _append_result(path, mode_name, description, res, elapsed):
    """Append a single mode result to the markdown file."""
    with open(path, "a") as f:
        f.write(f"## {mode_name}\n")
        f.write(f"**Description:** {description}\n\n")
        f.write(f"**Latency:** {elapsed:.2f}s | **Flags:** {res['flags']} | **Corrections:** {res['corrections']} | **Confidence:** {res['conf']:.2f}\n\n")
        f.write(f"### Answer\n\n{res['answer']}\n\n---\n\n")


if __name__ == "__main__":
    main()
