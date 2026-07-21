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
        self.assertGreater(scores_science.get("science", 0.0), 0.5)

        scores_cyber = self.orchestrator.classify_domains("Analyze malware vulnerability CVE-2023-1234 and threat intelligence.")
        self.assertGreater(scores_cyber.get("cyber", 0.0), 0.5)


if __name__ == "__main__":
    unittest.main()
