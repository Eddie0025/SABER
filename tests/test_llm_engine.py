# -*- coding: utf-8 -*-
"""tests/test_llm_engine.py

Unit tests for SABER LLMEngine initialization and device resolution.
"""

import unittest
from saber.llm_engine import LLMEngine


class TestLLMEngine(unittest.TestCase):

    def test_engine_device_detection(self):
        engine = LLMEngine("Qwen/Qwen2.5-7B-Instruct")
        self.assertIn(engine.device, ["cuda", "mps", "cpu"])

    def test_benchmark_mode_token_override(self):
        import os
        os.environ["SABER_BENCHMARK_MODE"] = "1"
        engine = LLMEngine("Qwen/Qwen2.5-7B-Instruct", max_new_tokens=2048)
        self.assertEqual(engine.max_new_tokens, 512)
        os.environ.pop("SABER_BENCHMARK_MODE", None)


if __name__ == "__main__":
    unittest.main()
