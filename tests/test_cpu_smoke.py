"""Exercise the runnable skeleton examples through the public CLI."""

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CpuSmokeTest(unittest.TestCase):
    def run_example(self, task, source_dir, expected_exit):
        with TemporaryDirectory() as temporary:
            reports_dir = Path(temporary) / "reports"
            completed = subprocess.run(
                [
                    sys.executable, "-m", "cannbench", "run", task,
                    "--suite", "examples", "--source-dir", str(source_dir),
                    "--reports-dir", str(reports_dir),
                ],
                cwd=ROOT, env={**os.environ, "PYTHONUTF8": "1"},
                capture_output=True, encoding="utf-8", timeout=30, check=False,
            )
            self.assertEqual(completed.returncode, expected_exit, completed.stderr)
            self.assertEqual(
                json.loads(completed.stdout)["execution"],
                {"exit_code": expected_exit, "reports": ["example_result.json"]},
            )
            return json.loads((reports_dir / "example_result.json").read_text(encoding="utf-8"))

    def test_classifier_and_relu_examples(self):
        for task in ("example.model.classifier", "example.operator.relu"):
            with self.subTest(task=task):
                report = self.run_example(task, ROOT / "tasks" / task / "answer", 0)
                self.assertTrue(report["correct"])
                self.assertTrue(report["complete"])
                self.assertGreater(report["score"], 0)

    def test_candidate_import_failure_produces_report(self):
        failures = (
            ('raise ModuleNotFoundError("candidate dependency unavailable")\n', "ModuleNotFoundError"),
            ("def predict(:\n", "SyntaxError"),
            ("raise SystemExit(0)\n", "SystemExit"),
        )
        with TemporaryDirectory() as temporary:
            source_dir = Path(temporary)
            for source, error_type in failures:
                with self.subTest(error_type=error_type):
                    (source_dir / "solution.py").write_text(source, encoding="utf-8")
                    report = self.run_example("example.operator.relu", source_dir, 1)
                    self.assertFalse(report["correct"])
                    self.assertFalse(report["complete"])
                    self.assertEqual(report["cases"], [])
                    self.assertEqual(report["measurements"], [])
                    self.assertIsNone(report["score"])
                    self.assertEqual(report["error"]["stage"], "candidate_import")
                    self.assertEqual(report["error"]["type"], error_type)
                    self.assertTrue(report["error"]["message"])
