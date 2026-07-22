# -*- coding: utf-8 -*-
"""tests/test_orchestrator.py

Unit tests for SABER Orchestrator ambiguity detection and domain classification.
"""

import unittest
from unittest.mock import MagicMock, patch
from saber.config import SaberConfig
from saber.registry import SpecialistRegistry
from saber.audit import AuditLogger
from saber.orchestrator import Orchestrator
from saber.specialists.science import ScienceSpecialist
from saber.specialists.cybersecurity import CyberSpecialist


class TestOrchestrator(unittest.TestCase):

    def setUp(self):
        self.engine_patcher = patch("saber.llm_engine.LLMEngine.__enter__")
        self.mock_enter = self.engine_patcher.start()
        mock_engine = MagicMock()
        mock_engine.generate.return_value = "ANSWER: A"
        self.mock_enter.return_value = mock_engine

        self.config = SaberConfig()
        self.registry = SpecialistRegistry()
        self.registry.register(ScienceSpecialist())
        self.registry.register(CyberSpecialist())
        self.audit = AuditLogger("data/test_audit.jsonl")
        self.orchestrator = Orchestrator(self.config, self.registry, self.audit)

    def tearDown(self):
        self.engine_patcher.stop()

    def test_ambiguity_detection(self):
        amb_score_vague = self.orchestrator.detect_ambiguity("it is this")
        self.assertGreater(amb_score_vague, 0.4)

        amb_score_clear = self.orchestrator.detect_ambiguity(
            "Calculate the kinetic energy of a 10kg mass moving at 5m/s in physics."
        )
        self.assertLess(amb_score_clear, 0.3)

    def test_domain_classification(self):
        scores_science = self.orchestrator.classify_domains("Calculate velocity, mass, and kinetic energy in physics.")
        self.assertGreater(scores_science.get("science", 0.0), 0.6)

        scores_cyber = self.orchestrator.classify_domains("Analyze malware vulnerability CVE-2023-1234 and threat intelligence.")
        self.assertGreater(scores_cyber.get("cyber", 0.0), 0.6)

    def test_polysemous_virus_disambiguation(self):
        # Computer virus -> cyber domain
        scores_cyber_virus = self.orchestrator._heuristic_classify_domains("How does a computer virus spread over network ports?")
        self.assertEqual(scores_cyber_virus.get("cyber"), 1.0)
        self.assertEqual(scores_cyber_virus.get("science"), 0.0)

        # Biological virus -> science domain
        scores_bio_virus = self.orchestrator._heuristic_classify_domains("How does an RNA virus replicate inside a host cell capsid?")
        self.assertEqual(scores_bio_virus.get("science"), 1.0)
        self.assertEqual(scores_bio_virus.get("cyber"), 0.0)

    def test_casual_chat_gating(self):
        self.assertTrue(self.orchestrator.is_casual_chat("hi"))
        self.assertTrue(self.orchestrator.is_casual_chat("good morning"))
        self.assertTrue(self.orchestrator.is_casual_chat("thanks!"))
        self.assertFalse(self.orchestrator.is_casual_chat("calculate kinetic energy in physics"))


if __name__ == "__main__":
    unittest.main()
