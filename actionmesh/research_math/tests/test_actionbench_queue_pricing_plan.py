import json
from pathlib import Path
import tempfile
import unittest

from prepare_actionbench_queue_pricing import (
    pricing_output_path,
    validate_source_records,
)


class ActionBenchQueuePricingPlanTests(unittest.TestCase):
    def test_output_is_the_single_canonical_pricing_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.assertEqual(
                pricing_output_path(root),
                root / "inputs/actionbench-full128-queue/pricing.json")

    def test_plan_source_validation_requires_completed_admission_without_queue_claim(self):
        admission = {
            "kind": "actionbench-complete-unit-admission",
            "version": "1.0.0",
            "status": "admitted_engineering_complete_unit",
            "eligible_for_queue_pricing": True,
            "queue_approved": False,
            "queue_generated": False,
            "scientific_effect_qualification": False,
            "native_scientific_qualification": False,
            "candidate_methods_tested": False,
        }
        validate_source_records(admission)
        for key, value in (("status", "generated_unexecuted"),
                           ("eligible_for_queue_pricing", False),
                           ("queue_approved", True),
                           ("queue_generated", True)):
            changed = json.loads(json.dumps(admission))
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_source_records(changed)


if __name__ == "__main__":
    unittest.main()
