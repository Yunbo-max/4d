"""Local-only contracts for prospective C04 held-out semantic evidence.

These byte fixtures test admission binding, not scientific validity or native
ActionBench performance. Web authors this source without running it.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import prepare_c04_native_scoring as planner
from research_math import c04_native_comparison as comparison


class C04SemanticAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name, record, digest_key=None):
        if digest_key:
            record[digest_key] = planner.canonical_record_digest(record, digest_key)
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record) + "\n")
        return comparison.file_ref(self.root, path)

    def fixture(self):
        implementation = self.write("code/implementation.json", {"version": 1})
        artifact = self.write("outputs/confirmation/candidate.json", {
            "target_parameters": {"radius": .2}})
        family = self.write("development/families.json", {
            "development_ids": ["fit", "held"],
            "confirmation_ids": ["confirmation"],
            "family_by_uid": {"fit": "family-fit", "held": "family-held",
                              "confirmation": "family-confirmation"}})
        spec = self.write("development/method-spec.json", {
            "kind": "c04-motion-surrogate-specification", "version": 1,
            "candidate_id": planner.CANDIDATE_ID,
            "implementation_ref": implementation,
            "surrogate": "normalized_predicted_kinetic_energy_v1",
            "uncertainty_set": "positive_diagonal_acceleration_weighted_gradient_ellipsoid_v1",
            "parameters": {"radius": .2}}, "spec_digest")
        source = self.write("development/raw/native-semantic-findings.json", {
            "uid": "held", "raw_values": [1.0, 2.0]})
        held_artifact = self.write("development/held-candidate.json", {
            "candidate_id": planner.CANDIDATE_ID, "uid": "held",
            "target_parameters": {"radius": .2},
            "implementation_sha256": implementation["sha256"],
            "recorded_at": "2026-10-08T00:30:00Z"})
        job = {"trial_id": "held-construction", "input_refs": [],
               "code_refs": [implementation], "seed": 42, "group": "engineering",
               "arm_role": "three-method-artifacts", "command": [
                   "python", "-m", "research_math.robust_motion_candidate",
                   "--uid", "held", "--radius", "0.2"]}
        native = {"evidence_mode": "developmental", "provenance": {}, "jobs": [job]}
        native_ref = self.write("development/native-plan.json", native, "plan_digest")
        harness = {"tasks": [{"plan_ref": native_ref}]}
        harness_ref = self.write("development/harness-plan.json", harness, "plan_digest")
        harness_report = self.write("development/harness-report.json", {
            "status": "completed", "plan_digest": harness["plan_digest"]})
        receipt = self.write("development/receipt.json", {
            "status": "completed", "plan_digest": native["plan_digest"],
            "provenance": {}, "attempts": [{**job, "status": "completed",
                "exit_code": 0, "output_refs": [source, held_artifact],
                "started_at": "2026-10-08T00:00:00Z",
                "completed_at": "2026-10-08T00:40:00Z"}]})
        evidence = self.write("development/evidence.json", {
            "kind": "c04-held-out-semantic-evidence", "version": 1,
            "candidate_id": planner.CANDIDATE_ID, "uid": "held",
            "asset_family": "family-held", "source_refs": [source],
            "receipt_ref": receipt, "native_plan_ref": native_ref,
            "harness_plan_ref": harness_ref, "harness_report_ref": harness_report,
            "candidate_artifact_ref": held_artifact,
            "surrogate": "normalized_predicted_kinetic_energy_v1",
            "uncertainty_set": "positive_diagonal_acceleration_weighted_gradient_ellipsoid_v1",
            "findings": {"engineering_fixture": True},
            "recorded_at": "2026-10-08T01:00:00Z"}, "evidence_digest")
        review = {
            "kind": "c04-motion-uncertainty-semantic-review", "version": 1,
            "candidate_id": planner.CANDIDATE_ID, "status": "accepted",
            "reviewer": "engineering-fixture-reviewer",
            "reviewed_at": "2026-10-08T02:00:00Z", "method_spec_ref": spec,
            "family_split_ref": family,
            "surrogate": "normalized_predicted_kinetic_energy_v1",
            "uncertainty_set": "positive_diagonal_acceleration_weighted_gradient_ellipsoid_v1",
            "probability_coverage_claimed": False,
            "fit_ids": ["fit"], "held_out_ids": ["held"],
            "evidence_refs": [evidence]}
        review_ref = self.write("development/review.json", review, "review_digest")
        frozen = {"candidate_artifact_ref": artifact,
                  "semantic_review_ref": review_ref,
                  "roles": [{"role": "robust_conic_protection",
                             "implementation_ref": implementation}],
                  "input_refs": [implementation, artifact, family, spec, source,
                                 receipt, evidence, review_ref, native_ref, harness_ref,
                                 harness_report, held_artifact]}
        return frozen, family, review

    def validate(self, frozen, family):
        return planner.require_c04_semantic_review(
            self.root, frozen, family_split_ref=family,
            protocol_frozen_at="2026-10-08T03:00:00Z",
            authorized_at="2026-10-08T04:00:00Z")

    def update_review(self, frozen, review):
        ref = self.write("development/review.json", review, "review_digest")
        frozen["semantic_review_ref"] = ref
        frozen["input_refs"] = [ref if row["path"] == ref["path"] else row
                                for row in frozen["input_refs"]]

    def test_prospective_spec_has_no_dependency_on_confirmation_output(self):
        frozen, family, _ = self.fixture()
        closure = self.validate(frozen, family)
        self.assertNotIn(frozen["candidate_artifact_ref"], closure)
        self.assertTrue(any(row["path"] == "development/receipt.json"
                            for row in closure))
        self.assertTrue(any(row["path"] == "development/raw/native-semantic-findings.json"
                            for row in closure))

    def test_missing_review_unbound_receipt_and_changed_parameters_reject(self):
        frozen, family, _ = self.fixture()
        missing = {**frozen, "semantic_review_ref": None}
        with self.assertRaisesRegex(ValueError, "held-out semantic review"):
            self.validate(missing, family)
        frozen["input_refs"] = [row for row in frozen["input_refs"]
                                if row["path"] != "development/receipt.json"]
        with self.assertRaisesRegex(ValueError, "missing from frozen"):
            self.validate(frozen, family)
        frozen, family, _ = self.fixture()
        frozen["candidate_artifact_ref"] = self.write(
            "outputs/confirmation/candidate.json", {"target_parameters": {"radius": 1.0}})
        with self.assertRaisesRegex(ValueError, "algorithm/parameters"):
            self.validate(frozen, family)

    def test_review_cannot_use_confirmation_ids_probability_claim_or_late_review(self):
        for mutation in ("confirmation", "probability", "late"):
            frozen, family, review = self.fixture()
            if mutation == "confirmation":
                review["held_out_ids"] = ["confirmation"]
            elif mutation == "probability":
                review["probability_coverage_claimed"] = True
            else:
                review["reviewed_at"] = "2026-10-08T03:30:00Z"
            self.update_review(frozen, review)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.validate(frozen, family)

    def test_declared_semantic_evidence_must_be_a_receipt_output(self):
        frozen, family, review = self.fixture()
        receipt_record = comparison.read_json(self.root / "development/receipt.json")
        receipt_record["attempts"][0]["output_refs"] = [
            row for row in receipt_record["attempts"][0]["output_refs"]
            if row["path"] == "development/held-candidate.json"]
        receipt = self.write("development/receipt.json", receipt_record)
        evidence_ref = review["evidence_refs"][0]
        evidence = comparison.read_json(self.root / evidence_ref["path"])
        evidence["receipt_ref"] = receipt
        review["evidence_refs"] = [self.write(
            evidence_ref["path"], evidence, "evidence_digest")]
        self.update_review(frozen, review)
        with self.assertRaisesRegex(ValueError, "not receipt-bound"):
            self.validate(frozen, family)


if __name__ == "__main__":
    unittest.main()
