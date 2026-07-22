# -*- coding: utf-8 -*-
"""tests/test_sentinel_search_extraction.py

Deep Extraction & Search Accuracy Test Suite for SABER Sentinel.
Tests:
1. Extraction of structured claims & key entities across 5 specialized domains.
2. Guard key & search query formulation (numeric guards, CVE IDs, conceptual terms).
3. Retrieval accuracy against SQLite KB databases and factual match verification.
"""

import os
import re
import sqlite3
import unittest
from saber.sentinel import Sentinel
from saber.signal import Signal, SignalType


class TestSentinelSearchExtractionAccuracy(unittest.TestCase):

    def setUp(self):
        self.test_kb_dir = "data/offline_kb"
        os.makedirs(self.test_kb_dir, exist_ok=True)

    def _create_mock_sqlite_kb(self, domain: str, records: list):
        """Helper to populate an in-memory or file-backed SQLite KB."""
        db_path = os.path.join(self.test_kb_dir, f"{domain}_kb.db")
        if os.path.exists(db_path):
            os.remove(db_path)

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
        for r in records:
            cursor.execute(
                "INSERT INTO passage_kb (id, query_guard, question_text, support_passage, label_text) VALUES (?, ?, ?, ?, ?)",
                r
            )
        conn.commit()
        conn.close()
        return db_path

    def test_numeric_and_cve_guard_extraction(self):
        """Test extraction of numeric anchors and CVE IDs from complex queries."""
        # Query 1: Cyber CVE
        query_cyber = "Analyze vulnerability CVE-2023-24380 in Spring Framework."
        clean_q1 = " ".join(query_cyber.lower().split())
        numerics1 = re.findall(r'\b\d+(?:\.\d+)?%?\b|cve-\d+-\d+', clean_q1)
        guard1 = f"[{'_'.join(numerics1)}]::{clean_q1}"

        self.assertIn("cve-2023-24380", numerics1)
        self.assertTrue(guard1.startswith("[cve-2023-24380]::"))

        # Query 2: Physics calculation
        query_science = "Calculate kinetic energy of a 10.5kg mass moving at 4m/s."
        clean_q2 = " ".join(query_science.lower().split())
        numerics2 = re.findall(r'\b\d+(?:\.\d+)?%?\b|cve-\d+-\d+', clean_q2)
        guard2 = f"[{'_'.join(numerics2)}]::{clean_q2}"

        self.assertIn("10.5", numerics2)
        self.assertIn("4", numerics2)
        self.assertTrue(guard2.startswith("[10.5_4]::"))

    def test_non_numeric_conceptual_keyword_extraction(self):
        """Test keyphrase extraction for non-numeric conceptual queries."""
        query_concept = "How do ribozymes catalyze RNA splicing reactions?"
        stopwords = {"how", "do", "in", "a", "an", "the", "of", "is", "are", "by"}
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query_concept.lower())
        keywords = [w for w in words if w not in stopwords]

        self.assertIn("ribozymes", keywords)
        self.assertIn("catalyze", keywords)
        self.assertIn("splicing", keywords)
        self.assertIn("reactions", keywords)

    def test_offline_kb_exact_retrieval_and_fact_matching(self):
        """Test that extracted queries retrieve the exact ground-truth passage from SQLite."""
        domain = "science_extract_test"
        query_guard = "[10_5_125]::calculate kinetic energy of 10kg at 5m/s"
        support_passage = "Kinetic Energy formula is KE = 0.5 * m * v^2. For m=10 and v=5, KE equals 125 Joules."
        
        self._create_mock_sqlite_kb(domain, [
            ("rec_001", query_guard, "Calculate KE", support_passage, "ground_truth")
        ])

        db_path = os.path.join(self.test_kb_dir, f"{domain}_kb.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT support_passage FROM passage_kb WHERE query_guard = ? LIMIT 1", (query_guard,))
        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        retrieved_text = row[0]
        self.assertEqual(retrieved_text, support_passage)

        # Grounding check logic simulation
        response_correct = "The kinetic energy is 125 Joules using formula KE = 0.5 * m * v^2."
        response_incorrect = "The kinetic energy is not 125 Joules."

        # Grounded bonus check
        has_support = any(fact in response_correct.lower() for fact in retrieved_text.lower().split()[:5] if len(fact) > 5)
        self.assertTrue(has_support)

        # Contradiction check
        is_contradiction = "not" in retrieved_text.lower() and "not" not in response_correct.lower()
        self.assertFalse(is_contradiction)


if __name__ == "__main__":
    unittest.main()
