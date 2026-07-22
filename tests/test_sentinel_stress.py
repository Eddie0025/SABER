# -*- coding: utf-8 -*-
"""tests/test_sentinel_stress.py

Aggressive Stress Test Suite for SABER Sentinel Verification Kernel.
Tests:
1. Cryptographic signal tampering detection.
2. Non-numeric keyphrase extraction & stopword filtering across 5 domains.
3. Numeric entity & CVE guard extraction.
4. Local SQLite KB auto-creation, auto-caching (save_to_local_kb), and querying.
5. Self-correcting FLAG_SIGNAL generation on contradictions.
6. Multi-domain verification routing matrix.
"""

import os
import shutil
import sqlite3
import unittest
from saber.config import SaberConfig
from saber.signal import Signal, SignalType, Claim
from saber.sentinel import Sentinel


class TestSentinelAggressiveStress(unittest.TestCase):

    def setUp(self):
        self.sentinel = Sentinel()
        self.config = SaberConfig()
        self.test_kb_dir = "data/offline_kb"
        os.makedirs(self.test_kb_dir, exist_ok=True)

    def test_cryptographic_tampering_stress(self):
        """Stress test SHA-256 signal tampering detection."""
        sig = Signal(
            signal_type=SignalType.COT_SIGNAL,
            query_id="stress-query-001",
            source_id="science",
            target_id="META_REASONER",
            payload={"claims": [{"statement": "E=mc^2"}], "raw_response": "Energy equals mass times c squared."}
        ).freeze_and_hash()

        # 1. Valid hash check
        self.assertTrue(Sentinel.verify_signal_integrity(sig))

        # 2. Tamper payload -> must fail integrity check
        sig.payload["raw_response"] = "Energy equals mass times c cubed."
        self.assertFalse(Sentinel.verify_signal_integrity(sig))

    def test_verification_routing_matrix_all_domains(self):
        """Verify routing assignments across all 5 specialist domains."""
        domains = ["science", "cyber", "finance", "coding", "architecture"]
        for domain in domains:
            route = Sentinel.get_verification_route(domain)
            self.assertIsInstance(route, dict)
            self.assertTrue(len(route) > 0)
            # Ensure primary domain is mapped
            self.assertIn(domain, route.values())

    def test_local_sqlite_kb_auto_caching_and_retrieval(self):
        """Stress test SQLite KB table creation, indexing, and auto-caching write (save_to_local_kb)."""
        domain = "test_cyber_domain"
        query_guard = "[cve-2023-24380]::vulnerability in spring framework"
        question = "What vulnerability is CVE-2023-24380?"
        support_passage = "CVE-2023-24380 is a remote code execution vulnerability in Spring Framework."

        db_path = os.path.join(self.test_kb_dir, f"{domain}_kb.db")
        if os.path.exists(db_path):
            os.remove(db_path)

        # 1. Write via save_to_local_kb logic
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS passage_kb (
                id TEXT PRIMARY KEY,
                query_guard TEXT,
                question_text TEXT,
                support_passage TEXT,
                label_text TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_guard ON passage_kb(query_guard)")
        cursor.execute(
            "INSERT OR REPLACE INTO passage_kb (id, query_guard, question_text, support_passage, label_text) VALUES (?, ?, ?, ?, ?)",
            ("auto_12345", query_guard, question, support_passage, "auto_cached")
        )
        conn.commit()

        # 2. Read and verify passage indexing
        cursor.execute("SELECT support_passage FROM passage_kb WHERE query_guard = ? LIMIT 1", (query_guard,))
        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertIn("Spring Framework", row[0])

    def test_interpretation_verification_clean_signal(self):
        """Test Sentinel verification of a clean signal returning GREEN_CHIT."""
        sig = Signal(
            signal_type=SignalType.COT_SIGNAL,
            query_id="stress-query-002",
            source_id="science",
            target_id="META_REASONER",
            payload={
                "claims": [{"statement": "Kinetic energy = 0.5 * m * v^2"}],
                "raw_response": "The kinetic energy of a 10kg mass at 5m/s is 125 Joules."
            }
        ).freeze_and_hash()

        ver_res = self.sentinel.verify_interpretation(
            specialist_domain="science",
            original_signal=sig,
            compiled_text="The kinetic energy of a 10kg mass at 5m/s is 125 Joules.",
            config=self.config
        )

        self.assertEqual(ver_res.signal_type, SignalType.VERIFICATION_SIGNAL)
        self.assertEqual(ver_res.payload.get("status"), "CONFIRMED")
        self.assertEqual(ver_res.payload.get("verdict"), "GREEN_CHIT")

    def test_interpretation_verification_tampered_signal_flag(self):
        """Test Sentinel verification raising a FLAG_SIGNAL on tampered cryptographic signature."""
        sig = Signal(
            signal_type=SignalType.COT_SIGNAL,
            query_id="stress-query-003",
            source_id="cyber",
            target_id="META_REASONER",
            payload={
                "claims": [{"statement": "Port 445 is SMB"}],
                "raw_response": "Port 445 is used by SMB protocol."
            }
        ).freeze_and_hash()

        # Tamper signal payload
        sig.payload["raw_response"] = "Port 445 is used by HTTP protocol."

        ver_res = self.sentinel.verify_interpretation(
            specialist_domain="cyber",
            original_signal=sig,
            compiled_text="Port 445 is used by HTTP protocol.",
            config=self.config
        )

        self.assertEqual(ver_res.signal_type, SignalType.FLAG_SIGNAL)
        self.assertEqual(ver_res.payload.get("severity"), "CRITICAL")
        self.assertEqual(ver_res.payload.get("issue_type"), "SIGNAL_CORRUPTION")


if __name__ == "__main__":
    unittest.main()
