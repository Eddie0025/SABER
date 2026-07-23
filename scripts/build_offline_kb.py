# -*- coding: utf-8 -*-
"""scripts/build_offline_kb.py

Extracts reference support passages and context from processed dataset JSONL files
and compiles them into indexed local SQLite databases per domain for 0ms offline Sentinel verification.
"""

import json
import os
import re
import sqlite3
import sys

# Ensure saber package is in path
sys.path.append(os.path.abspath('.'))

DATA_DIR = "data/processed"
KB_DIR = "data/offline_kb"


def normalize_query_guard(text: str) -> str:
    """Extract numeric guards and normalize text to prevent false cache collisions."""
    clean = " ".join(text.lower().split())
    numerics = re.findall(r'\b\d+(?:\.\d+)?%?\b|cve-\d+-\d+', clean)
    guard = "_".join(numerics)
    return f"[{guard}]::{clean}"


def build_domain_kb(domain: str):
    jsonl_path = os.path.join(DATA_DIR, f"{domain}.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"[build_offline_kb] WARNING: {jsonl_path} does not exist. Skipping.")
        return

    os.makedirs(KB_DIR, exist_ok=True)
    db_path = os.path.join(KB_DIR, f"{domain}_kb.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE passage_kb (
            id TEXT PRIMARY KEY,
            query_guard TEXT,
            question_text TEXT,
            support_passage TEXT,
            label_text TEXT
        )
    """)
    cursor.execute("CREATE INDEX idx_guard ON passage_kb(query_guard)")

    count = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            q_id = rec.get("id", f"{domain}_{count}")
            text = rec.get("text", "")
            label = rec.get("label", "")
            support = rec.get("support", rec.get("context", label))

            guard = normalize_query_guard(text)
            cursor.execute(
                "INSERT OR REPLACE INTO passage_kb (id, query_guard, question_text, support_passage, label_text) VALUES (?, ?, ?, ?, ?)",
                (q_id, guard, text, support, label)
            )
            count += 1

    conn.commit()
    conn.close()
    print(f"[build_offline_kb] Successfully built SQLite KB for '{domain}' with {count} passages -> {db_path}")


def main():
    domains = ["science", "cyber", "finance", "coding", "architecture", "meta_reasoner", "orchestrator"]
    for d in domains:
        build_domain_kb(d)

if __name__ == "__main__":
    main()
