# -*- coding: utf-8 -*-
"""tests/test_cot_maintainer.py

Unit tests for SABER CoTMaintainer step-by-step reasoning chains.
"""

import unittest
from saber.cot_maintainer import CoTMaintainer, ReasoningStep


class TestCoTMaintainer(unittest.TestCase):

    def test_chain_lifecycle(self):
        cot = CoTMaintainer()
        cot.begin_chain("science", "query-cot-01")

        s1 = cot.add_step("IDENTIFY", "Identify mass m=5kg and velocity v=10m/s", 0.95)
        self.assertEqual(s1, 1)

        s2 = cot.add_step("ANALYZE", "Apply momentum formula p = m * v", 0.95, depends_on=[1])
        self.assertEqual(s2, 2)

        cot.conclude("Calculated momentum p = 50 kg m/s", 0.95)
        self.assertTrue(cot._current_chain.is_complete)
        self.assertEqual(cot._current_chain.final_conclusion, "Calculated momentum p = 50 kg m/s")

    def test_export_for_signal(self):
        cot = CoTMaintainer()
        cot.begin_chain("cyber", "query-cot-02")
        cot.add_step("IDENTIFY", "Identify CVE-2023-1234", 0.90)
        cot.conclude("Threat mapped to T1059", 0.90)

        exported = cot.export_for_signal()
        self.assertEqual(exported["domain"], "cyber")
        self.assertEqual(len(exported["steps"]), 2)


if __name__ == "__main__":
    unittest.main()
