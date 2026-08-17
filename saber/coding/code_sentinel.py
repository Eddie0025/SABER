import os
import re
import uuid
import logging
import tempfile
import subprocess
from typing import Dict, Tuple, Optional

logger = logging.getLogger("SABER_CodeSentinel")

# Maximum time allowed for test execution
TEST_TIMEOUT_SECONDS = 10
MAX_RETRIES = 2


class CodeSentinel:
    """
    Code Sentinel — Unit test verification kernel for the Coding Sector.
    
    Fundamentally different from the main Sentinel:
    - Main Sentinel verifies via semantic KB search.
    - Code Sentinel verifies by RUNNING unit tests.
    
    Pipeline:
    1. Phase 1: Run specialist-written unit tests via subprocess.
    2. Phase 2: (Future) Generate adversarial edge-case tests via base LLM.
    3. Return CONFIRMED or FLAG with failure context.
    """

    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="saber_code_sentinel_")

    def _extract_code_blocks(self, output: str) -> Tuple[str, str]:
        """
        Extract the main code and test code from the specialist's output.
        Expects code blocks delimited by ```language ... ```
        """
        code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", output, re.DOTALL)

        main_code = ""
        test_code = ""

        if len(code_blocks) >= 2:
            main_code = code_blocks[0].strip()
            test_code = code_blocks[1].strip()
        elif len(code_blocks) == 1:
            main_code = code_blocks[0].strip()

        return main_code, test_code

    def verify_python(self, specialist_output: str) -> Dict:
        """
        Run Python code + tests in an isolated subprocess.
        
        Returns:
            {
                "status": "CONFIRMED" | "FLAG",
                "tests_passed": int,
                "tests_total": int,
                "stdout": str,
                "stderr": str,
                "failure_context": str (if FLAG)
            }
        """
        main_code, test_code = self._extract_code_blocks(specialist_output)

        if not main_code:
            return {
                "status": "FLAG",
                "tests_passed": 0,
                "tests_total": 0,
                "stdout": "",
                "stderr": "No code block found in specialist output.",
                "failure_context": "Could not extract code from the specialist's response.",
            }

        if not test_code:
            # No tests provided — pass through (can't verify without tests)
            return {
                "status": "CONFIRMED",
                "tests_passed": 0,
                "tests_total": 0,
                "stdout": "No tests provided by specialist.",
                "stderr": "",
                "failure_context": "",
            }

        # Write code + tests to a temp file
        test_id = uuid.uuid4().hex[:8]
        test_file = os.path.join(self.temp_dir, f"test_{test_id}.py")

        combined = f"{main_code}\n\n# --- TESTS ---\n{test_code}"
        with open(test_file, "w") as f:
            f.write(combined)

        # Execute
        try:
            result = subprocess.run(
                ["python3", test_file],
                capture_output=True,
                text=True,
                timeout=TEST_TIMEOUT_SECONDS,
                cwd=self.temp_dir,
            )

            if result.returncode == 0:
                # Parse test count from pytest/unittest output if possible
                tests_total = test_code.count("def test_")
                return {
                    "status": "CONFIRMED",
                    "tests_passed": tests_total,
                    "tests_total": tests_total,
                    "stdout": result.stdout[:2000],
                    "stderr": result.stderr[:2000],
                    "failure_context": "",
                }
            else:
                return {
                    "status": "FLAG",
                    "tests_passed": 0,
                    "tests_total": test_code.count("def test_"),
                    "stdout": result.stdout[:2000],
                    "stderr": result.stderr[:2000],
                    "failure_context": f"Tests failed with exit code {result.returncode}.\n\nSTDERR:\n{result.stderr[:1000]}",
                }

        except subprocess.TimeoutExpired:
            return {
                "status": "FLAG",
                "tests_passed": 0,
                "tests_total": 0,
                "stdout": "",
                "stderr": f"Test execution timed out after {TEST_TIMEOUT_SECONDS}s.",
                "failure_context": f"Code execution exceeded the {TEST_TIMEOUT_SECONDS}s timeout limit.",
            }
        except Exception as e:
            return {
                "status": "FLAG",
                "tests_passed": 0,
                "tests_total": 0,
                "stdout": "",
                "stderr": str(e),
                "failure_context": f"Unexpected error running tests: {e}",
            }

    def verify(self, specialist_output: str, language: str = "python") -> Dict:
        """
        Route verification to the appropriate language handler.
        Currently supports Python. JS and SQL are future extensions.
        """
        if language == "python":
            return self.verify_python(specialist_output)
        else:
            # For unsupported languages, pass through
            logger.info(f"Code Sentinel does not yet support '{language}'. Passing through.")
            return {
                "status": "CONFIRMED",
                "tests_passed": 0,
                "tests_total": 0,
                "stdout": f"Language '{language}' not yet supported by Code Sentinel.",
                "stderr": "",
                "failure_context": "",
            }
