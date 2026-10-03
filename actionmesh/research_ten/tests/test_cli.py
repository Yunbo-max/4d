import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CommandLineTests(unittest.TestCase):
    def test_controls_write_finite_results_and_preserve_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"controls"
            args = [sys.executable, "-m", "research_ten", "controls", "--methods", "6",
                    "--output", str(output)]
            result = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr+result.stdout)
            report = json.loads((output/"report.json").read_text())
            self.assertEqual(report["methods"]["6"]["status"], "completed")
            row = json.loads((output/"method-06.json").read_text())
            self.assertEqual(row["evidence_type"], "constructed_control")
            original = (output/"method-06.json").read_bytes()
            retry = subprocess.run(args, capture_output=True, text=True)
            self.assertNotEqual(retry.returncode, 0)
            self.assertEqual((output/"method-06.json").read_bytes(), original)

    def test_list_reports_external_input_requirements(self):
        result = subprocess.run([sys.executable, "-m", "research_ten", "list"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = json.loads(result.stdout)
        self.assertEqual(set(rows), {str(i) for i in range(1, 11)})
        self.assertIn("natural long video", " ".join(rows["10"]["requires"]))


if __name__ == "__main__":
    unittest.main()
