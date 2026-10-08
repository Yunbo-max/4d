"""Authored G01 contract checks; Local executes them through the CPU harness.

The repository-wide test binds the actual 15 selected specifications and the
released ActionBench population.  Temporary records below are engineering
contract fixtures only; they are never benchmark observations or scientific
family assignments.  Web authored this file without importing project modules
or running the tests.
"""
from copy import deepcopy
import hashlib
import io
import importlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest


try:
    g01 = importlib.import_module("research_math.g01_design")
    g01_planner = importlib.import_module("prepare_g01_acceptance")
except ModuleNotFoundError:
    g01 = None
    g01_planner = None


class G01DesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repository = Path(__file__).resolve().parents[3]
        cls.design_path = (
            cls.repository / "docs/research-math-20261006/longgoal-20261007/"
            "G01_DESIGN.json"
        )

    def setUp(self):
        self.assertIsNotNone(g01, "G01 design validator is missing")
        self.assertIsNotNone(g01_planner, "G01 Local acceptance builder is missing")

    def test_real_stage_design_binds_all_selected_specs_and_native_population(self):
        design = g01.validate_design(self.repository, self.design_path)
        self.assertEqual(len(design["candidates"]), 15)
        self.assertEqual(design["benchmark"]["population_size"], 128)
        self.assertEqual(design["statistics"]["confirmatory_test_count"], 99)
        self.assertEqual(sum(
            row["approved_control_implementation_refs"] is not None
            for row in design["candidates"]), 12)
        self.assertEqual(design["statistics"]["independent_unit"], "asset_family")
        self.assertEqual(design["cohort"]["minimum_confirmation_families"], 30)
        self.assertEqual(design["criteria"]["minimum_effect"], {
            "cd_3d": 0.002, "cd_motion": 0.003})
        self.assertEqual(design["criteria"]["noninferiority_margin"], {
            "cd_3d": 0.002, "cd_4d": 0.004, "cd_motion": 0.003})

    def test_unreviewed_semantic_rewrite_is_rejected_by_frozen_digest(self):
        design = json.loads(self.design_path.read_text())
        design["fairness"]["cost"] = "weakened after source review"
        design["design_digest"] = g01.canonical_record_digest(
            design, "design_digest")
        with self.assertRaisesRegex(ValueError, "independently frozen"):
            g01.validate_design_record(self.repository, design)

    def test_shared_b0_profiles_pin_the_actual_context_producer(self):
        design = g01.validate_design(self.repository, self.design_path)
        for suffix in ('c06', 'c11', 'c12'):
            row = next(item for item in design['candidates']
                       if item['candidate_id'] == '4d-math-20261006-' + suffix)
            paths = {ref['path'] for ref in g01.approved_native_code_refs(design, row)}
            self.assertIn('actionmesh/research_math/native_context_runner.py', paths)

    def test_changed_candidate_spec_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "G01_DESIGN.json"
            design = json.loads(self.design_path.read_text())
            design["candidates"][0]["spec_ref"]["sha256"] = "0" * 64
            design["design_digest"] = g01.canonical_record_digest(
                design, "design_digest")
            copied.write_text(json.dumps(design))
            with self.assertRaisesRegex(ValueError, "reference digest"):
                g01.validate_design(self.repository, copied)

    def test_removed_or_substituted_transitive_implementation_is_rejected(self):
        design = json.loads(self.design_path.read_text())
        row = next(item for item in design["candidates"]
                   if item["candidate_id"] == "4d-math-20261006-c11")
        row["approved_control_implementation_refs"] = [
            ref for ref in row["approved_control_implementation_refs"]
            if not ref["path"].endswith("integrable_gradient_candidate.py")]
        design["design_digest"] = g01.canonical_record_digest(
            design, "design_digest")
        with self.assertRaisesRegex(ValueError, "independently frozen"):
            g01.validate_design_record(self.repository, design)

    def test_c15_preparer_and_shared_support_are_frozen_and_extra_code_is_rejected(self):
        design = g01.validate_design(self.repository, self.design_path)
        row = next(item for item in design["candidates"]
                   if item["candidate_id"] == "4d-math-20261006-c15")
        implementation = g01.approved_native_code_refs(design, row)
        paths = {ref["path"] for ref in implementation}
        self.assertIn("actionmesh/prepare_protected_lowrank_candidate.py", paths)
        self.assertIn("actionmesh/prepare_c14_native_scoring.py", paths)
        expected = sorted(
            implementation + design["benchmark"]["official_scorer"]["source_refs"],
            key=lambda ref: ref["path"])
        rogue = deepcopy(expected)
        rogue.append({"path": "actionmesh/rogue.py", "sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "exact frozen closure"):
            g01.validate_native_job_code_refs(
                implementation,
                design["benchmark"]["official_scorer"]["source_refs"], rogue)

    def test_null_or_missing_numeric_criterion_is_rejected(self):
        design = json.loads(self.design_path.read_text())
        design["criteria"]["minimum_effect"]["cd_motion"] = None
        design["design_digest"] = g01.canonical_record_digest(
            design, "design_digest")
        with self.assertRaisesRegex(ValueError, "numeric criteria"):
            g01.validate_design_record(self.repository, design)

    def test_required_contrasts_cover_every_declared_non_treatment_arm(self):
        design = json.loads(self.design_path.read_text())
        row = design["candidates"][0]
        row["confirmatory_contrasts"] = row["confirmatory_contrasts"][:-1]
        design["design_digest"] = g01.canonical_record_digest(
            design, "design_digest")
        with self.assertRaisesRegex(ValueError, "contrast inventory"):
            g01.validate_design_record(self.repository, design)

    def _family_fixture(self, directory, *, uid_derived=False,
                        unexposed_family_count=None):
        root = Path(directory)
        population = json.loads((
            self.repository / "actionmesh/research_overnight/assets/"
            "actionbench_population.json").read_text())
        exposed = json.loads((
            self.repository / "docs/research-math-20261006/evidence/"
            "exposure-and-contract.json").read_text())
        exposed_ids = set(exposed["overnight_development_uids"])
        exposed_ids.update(exposed["earlier_inspected_uids"])
        extractor = root / "family_extractor.py"
        extractor.write_text("# fixture implementation of the frozen JSON pointer extractor\n")
        extractor_ref = g01.file_ref(root, extractor)
        geometry = root / "geometry.bin"
        geometry.write_bytes(b"source-derived geometry identity fixture")
        geometry_ref = g01.file_ref(root, geometry)
        units = {}
        family_by_uid = {}
        fresh_index = 0
        for index, uid in enumerate(population["uids"]):
            if uid in exposed_ids:
                family_key = "asset-exposed"
            elif uid_derived:
                family_key = uid
                fresh_index += 1
            else:
                family_index = (fresh_index if unexposed_family_count is None
                                else fresh_index % unexposed_family_count)
                family_key = "asset-group-%03d" % family_index
                fresh_index += 1
            metadata_source = root / ("metadata-%03d.json" % index)
            metadata_source.write_text(json.dumps({
                "native": {"asset_family": family_key},
            }))
            metadata_source_ref = g01.file_ref(root, metadata_source)
            metadata = {
                "kind": "g01-upstream-family-metadata-receipt",
                "version": "1.0.0", "benchmark_revision": population["revision"],
                "uid": uid, "metadata_source_ref": metadata_source_ref,
                "family_key_pointers": ["/native/asset_family"],
                "extractor_ref": extractor_ref,
                "extracted_family_keys": [family_key],
                "author": "fixture-metadata-author",
                "created_at": "2026-10-08T00:00:00Z",
            }
            metadata["receipt_digest"] = g01.canonical_record_digest(
                metadata, "receipt_digest")
            metadata_path = root / ("metadata-receipt-%03d.json" % index)
            metadata_path.write_text(json.dumps(metadata))
            geometry_receipt = {
                "kind": "g01-geometry-identity-receipt", "version": "1.0.0",
                "benchmark_revision": population["revision"], "uid": uid,
                "geometry_refs": [geometry_ref], "extractor_ref": extractor_ref,
                "geometry_sha256": [geometry_ref["sha256"]],
                "author": "fixture-geometry-author",
                "created_at": "2026-10-08T00:00:00Z",
            }
            geometry_receipt["receipt_digest"] = g01.canonical_record_digest(
                geometry_receipt, "receipt_digest")
            geometry_receipt_path = root / ("geometry-receipt-%03d.json" % index)
            geometry_receipt_path.write_text(json.dumps(geometry_receipt))
            basis = {
                "upstream_family_keys": [family_key],
                "metadata_content_sha256": [metadata_source_ref["sha256"]],
                "geometry_content_sha256": [geometry_ref["sha256"]],
                "extractor_content_sha256": [extractor_ref["sha256"]],
            }
            family_id = hashlib.sha256(g01._canonical_bytes(basis)).hexdigest()
            units[uid] = {
                "family_basis": basis, "family_id": family_id,
                "metadata_receipt_ref": g01.file_ref(root, metadata_path),
                "geometry_receipt_ref": g01.file_ref(root, geometry_receipt_path),
            }
            family_by_uid[uid] = family_id
        derivation = {
            "kind": "g01-family-derivation", "version": "1.0.0",
            "benchmark_revision": population["revision"],
            "units": units,
            "assignment_rule":
                "family_id=sha256(canonical_family_basis); uid_and_outcomes_forbidden",
            "author": "fixture-derivation-author",
            "derived_at": "2026-10-08T00:00:00Z",
        }
        derivation["derivation_digest"] = g01.canonical_record_digest(
            derivation, "derivation_digest")
        derivation_path = root / "derivation.json"
        derivation_path.write_text(json.dumps(derivation))
        review = {
            "kind": "g01-family-derivation-review", "version": "1.0.0",
            "derivation_ref": g01.file_ref(root, derivation_path),
            "outcome": "verified",
            "checks": ["source_identity", "mapping_reproduction",
                       "split_independence", "exposure_containment"],
            "reviewer": "fixture-independent-reviewer",
            "reviewed_at": "2026-10-08T00:01:00Z",
        }
        review["review_digest"] = g01.canonical_record_digest(
            review, "review_digest")
        review_path = root / "review.json"
        review_path.write_text(json.dumps(review))
        evidence = {
            "kind": "g01-family-evidence", "version": "1.0.0",
            "benchmark_revision": population["revision"],
            "population_ref": g01.file_ref(
                self.repository,
                self.repository / "actionmesh/research_overnight/assets/"
                "actionbench_population.json"),
            "family_by_uid": family_by_uid,
            "derivation_ref": g01.file_ref(root, derivation_path),
            "review_ref": g01.file_ref(root, review_path),
            "frozen_at": "2026-10-08T00:02:00Z",
        }
        evidence["evidence_digest"] = g01.canonical_record_digest(
            evidence, "evidence_digest")
        return evidence, exposed_ids

    def test_family_split_is_deterministic_and_keeps_exposure_out_of_confirmation(self):
        design = g01.validate_design(self.repository, self.design_path)
        with tempfile.TemporaryDirectory() as directory:
            evidence, exposed_ids = self._family_fixture(directory)
            one = g01.derive_family_split(
                self.repository, design, evidence, evidence_root=Path(directory))
            reversed_evidence = deepcopy(evidence)
            reversed_evidence["family_by_uid"] = dict(reversed(
                list(reversed_evidence["family_by_uid"].items())))
            reversed_evidence["evidence_digest"] = g01.canonical_record_digest(
                reversed_evidence, "evidence_digest")
            two = g01.derive_family_split(
                self.repository, design, reversed_evidence,
                evidence_root=Path(directory))
            self.assertEqual(one, two)
            self.assertTrue(exposed_ids.issubset(set(one["development_exposed_ids"])))
            self.assertFalse(exposed_ids & set(one["confirmation_ids"]))
            self.assertEqual(len(one["d1_families"]), 12)
            self.assertEqual(len(one["d2_families"]), 12)

    def test_unreviewed_or_incomplete_family_evidence_is_rejected(self):
        design = g01.validate_design(self.repository, self.design_path)
        with tempfile.TemporaryDirectory() as directory:
            evidence, _ = self._family_fixture(directory)
            evidence["family_by_uid"].pop(next(iter(evidence["family_by_uid"])))
            evidence["evidence_digest"] = g01.canonical_record_digest(
                evidence, "evidence_digest")
            with self.assertRaisesRegex(ValueError, "complete released population"):
                g01.derive_family_split(
                    self.repository, design, evidence,
                    evidence_root=Path(directory))

    def test_uid_derived_family_labels_are_rejected(self):
        design = g01.validate_design(self.repository, self.design_path)
        with tempfile.TemporaryDirectory() as directory:
            evidence, _ = self._family_fixture(directory, uid_derived=True)
            with self.assertRaisesRegex(ValueError, "UID-derived family"):
                g01.derive_family_split(
                    self.repository, design, evidence,
                    evidence_root=Path(directory))

    def test_family_label_unrelated_to_metadata_and_category_swap_are_rejected(self):
        design = g01.validate_design(self.repository, self.design_path)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence, _ = self._family_fixture(directory)
            derivation_path = g01.resolve_ref(root, evidence["derivation_ref"])
            derivation = json.loads(derivation_path.read_text())
            uid = next(iter(derivation["units"]))
            unit = derivation["units"][uid]
            metadata_path = g01.resolve_ref(root, unit["metadata_receipt_ref"])
            metadata = json.loads(metadata_path.read_text())
            metadata["extracted_family_keys"] = ["irrelevant-self-attested-label"]
            metadata["receipt_digest"] = g01.canonical_record_digest(
                metadata, "receipt_digest")
            metadata_path.write_text(json.dumps(metadata))
            unit["metadata_receipt_ref"] = g01.file_ref(root, metadata_path)
            derivation["derivation_digest"] = g01.canonical_record_digest(
                derivation, "derivation_digest")
            derivation_path.write_text(json.dumps(derivation))
            evidence["derivation_ref"] = g01.file_ref(root, derivation_path)
            evidence["evidence_digest"] = g01.canonical_record_digest(
                evidence, "evidence_digest")
            with self.assertRaisesRegex(ValueError, "pinned upstream metadata"):
                g01.derive_family_split(
                    self.repository, design, evidence, evidence_root=root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence, _ = self._family_fixture(directory)
            derivation_path = g01.resolve_ref(root, evidence["derivation_ref"])
            derivation = json.loads(derivation_path.read_text())
            uid = next(iter(derivation["units"]))
            unit = derivation["units"][uid]
            metadata = json.loads(g01.resolve_ref(
                root, unit["metadata_receipt_ref"]).read_text())
            geometry_path = g01.resolve_ref(root, unit["geometry_receipt_ref"])
            geometry = json.loads(geometry_path.read_text())
            metadata_source = g01.resolve_ref(root, metadata["metadata_source_ref"])
            copied_metadata = root / "metadata-bytes-disguised-as-geometry.bin"
            copied_metadata.write_bytes(metadata_source.read_bytes())
            copied_ref = g01.file_ref(root, copied_metadata)
            geometry["geometry_refs"] = [copied_ref]
            geometry["geometry_sha256"] = [copied_ref["sha256"]]
            geometry["receipt_digest"] = g01.canonical_record_digest(
                geometry, "receipt_digest")
            geometry_path.write_text(json.dumps(geometry))
            unit["geometry_receipt_ref"] = g01.file_ref(root, geometry_path)
            derivation["derivation_digest"] = g01.canonical_record_digest(
                derivation, "derivation_digest")
            derivation_path.write_text(json.dumps(derivation))
            evidence["derivation_ref"] = g01.file_ref(root, derivation_path)
            evidence["evidence_digest"] = g01.canonical_record_digest(
                evidence, "evidence_digest")
            with self.assertRaisesRegex(ValueError, "digests overlap"):
                g01._validate_family_evidence(
                    self.repository, design, evidence, evidence_root=root)

    def test_too_few_confirmation_clusters_are_rejected(self):
        design = g01.validate_design(self.repository, self.design_path)
        with tempfile.TemporaryDirectory() as directory:
            evidence, _ = self._family_fixture(
                directory, unexposed_family_count=53)
            with self.assertRaisesRegex(ValueError, "insufficient unexposed families"):
                g01.derive_family_split(
                    self.repository, design, evidence,
                    evidence_root=Path(directory))

    def test_b_star_freeze_covers_all_candidates_and_only_declared_controls(self):
        design = g01.validate_design(self.repository, self.design_path)
        selections = {
            row["candidate_id"]: {
                "role": next(role for role in row["control_roles"]
                             if role.lower() != "b_star"),
                "selection_stage": "D1_only",
                "metric": row["primary_metric"],
            }
            for row in design["candidates"]
        }
        g01.validate_b_star_selections(design, selections)
        first = design["candidates"][0]
        selections[first["candidate_id"]]["role"] = next(
            role for role in first["control_roles"] if role.lower() == "b_star")
        with self.assertRaisesRegex(ValueError, "selectable declared control"):
            g01.validate_b_star_selections(design, selections)

    def test_b_star_freeze_recomputes_complete_d1_minimum(self):
        design = deepcopy(g01.validate_design(self.repository, self.design_path))
        fixture_candidate_refs = next(
            row["approved_control_implementation_refs"]
            for row in design["candidates"]
            if row["approved_control_implementation_refs"] is not None)
        for row in design["candidates"]:
            if row["approved_control_implementation_refs"] is None:
                row["approved_control_implementation_refs"] = \
                    fixture_candidate_refs
        with tempfile.TemporaryDirectory(dir=self.repository) as directory:
            root = Path(directory)
            split = {
                "kind": "g01-family-split", "version": "1.0.0",
                "benchmark_revision": design["benchmark"]["revision"],
                "design_digest": design["design_digest"],
                "family_evidence_digest": "1" * 64,
                "split_salt": design["cohort"]["split_salt"],
                "development_exposed_families": ["exposed"],
                "development_exposed_ids": ["exposed-id"],
                "d1_families": ["d1-family"], "d1_ids": ["d1-id"],
                "d2_families": ["d2-family"], "d2_ids": ["d2-id"],
                "confirmation_families": ["confirmation-family"],
                "confirmation_ids": ["confirmation-id"],
                "family_by_uid": {
                    "exposed-id": "exposed", "d1-id": "d1-family",
                    "d2-id": "d2-family",
                    "confirmation-id": "confirmation-family",
                },
            }
            split["split_digest"] = g01.canonical_record_digest(
                split, "split_digest")
            split_path = root / "split.json"
            split_path.write_text(json.dumps(split))
            split_ref = g01.file_ref(self.repository, split_path)

            result_refs = []
            selections = {}
            first_result = first_receipt = first_receipt_path = None
            first_admission = first_admission_path = None
            for row in design["candidates"]:
                selectable = [role for role in row["control_roles"]
                              if role.lower() != "b_star"]
                implementation_refs = g01.approved_native_code_refs(design, row)
                selections[row["candidate_id"]] = {
                    "role": selectable[0], "selection_stage": "D1_only",
                    "metric": row["primary_metric"],
                }
                for index, role in enumerate(selectable):
                    receipt_refs = []
                    for seed in design["randomness"]["generation_seeds"]:
                        stem = row["candidate_id"] + "-" + role + "-" + str(seed)
                        sequence_path = root / (stem + "-sequence.npz")
                        sequence_path.write_bytes(("sequence-" + stem).encode())
                        sequence_ref = g01.file_ref(
                            self.repository, sequence_path)
                        identity = {
                            "kind": "g01-generation-identity", "version": "1.0.0",
                            "benchmark_id": "facebook/actionbench",
                            "benchmark_revision": design["benchmark"]["revision"],
                            "uid": "d1-id", "seed": seed,
                            "producer_source_refs": implementation_refs,
                        }
                        identity["identity_digest"] = g01.canonical_record_digest(
                            identity, "identity_digest")
                        identity_path = root / (stem + "-identity.json")
                        identity_path.write_text(json.dumps(identity))
                        native_plan = {
                            "schema_id": "research-native-plan",
                            "schema_version": 1,
                            "run_id": stem,
                            "provenance": {"g01_control_unit": {
                                "candidate_id": row["candidate_id"],
                                "control_role": role, "uid": "d1-id",
                                "generation_seed": seed,
                            }},
                            "limits": {"max_attempts": 1,
                                       "max_retries_per_trial": 0},
                            "jobs": [{"trial_id": stem, "arm_role": role,
                                      "code_refs": implementation_refs +
                                      design["benchmark"]["official_scorer"]["source_refs"],
                                      "output_paths": [sequence_ref["path"]]}],
                        }
                        native_plan["plan_digest"] = g01.canonical_record_digest(
                            native_plan, "plan_digest")
                        native_plan_path = root / (stem + "-native-plan.json")
                        native_plan_path.write_text(json.dumps(native_plan))
                        native_plan_ref = g01.file_ref(
                            self.repository, native_plan_path)
                        native_receipt = {
                            "plan_digest": native_plan["plan_digest"],
                            "status": "completed",
                            "attempts": [{"status": "completed",
                                          "output_refs": [sequence_ref]}],
                        }
                        native_receipt_path = root / (stem + "-native-receipt.json")
                        native_receipt_path.write_text(json.dumps(native_receipt))
                        harness_plan = {
                            "schema_id": "research-harness-plan",
                            "schema_version": 1, "batch_id": stem,
                            "tasks": [{"task_id": stem,
                                       "plan_ref": native_plan_ref}],
                        }
                        harness_plan["plan_digest"] = g01.canonical_record_digest(
                            harness_plan, "plan_digest")
                        harness_plan_path = root / (stem + "-harness-plan.json")
                        harness_plan_path.write_text(json.dumps(harness_plan))
                        harness_report_path = root / (stem + "-harness-report.json")
                        harness_report_path.write_text(json.dumps({
                            "plan_digest": harness_plan["plan_digest"],
                            "status": "completed",
                        }))
                        harness_attempt = {
                            "kind": "g01-native-harness-attempt", "version": "1.0.0",
                            "design_digest": design["design_digest"],
                            "candidate_id": row["candidate_id"], "control_role": role,
                            "uid": "d1-id", "generation_seed": seed,
                            "status": "completed_once", "max_attempts": 1,
                            "max_retries": 0,
                            "implementation_refs": implementation_refs,
                            "sequence_ref": sequence_ref,
                            "harness_plan_ref": g01.file_ref(
                                self.repository, harness_plan_path),
                            "harness_report_ref": g01.file_ref(
                                self.repository, harness_report_path),
                            "native_plan_ref": native_plan_ref,
                            "native_receipt_ref": g01.file_ref(
                                self.repository, native_receipt_path),
                            "approved_plan_digest": harness_plan["plan_digest"],
                            "completed_at": "2026-10-08T00:00:00Z",
                        }
                        harness_attempt["attempt_digest"] = g01.canonical_record_digest(
                            harness_attempt, "attempt_digest")
                        harness_path = root / (stem + "-harness.json")
                        harness_path.write_text(json.dumps(harness_attempt))
                        harness_ref = g01.file_ref(self.repository, harness_path)
                        producer = {
                            "kind": "g01-generation-producer-receipt",
                            "version": "1.0.0", "design_digest": design["design_digest"],
                            "candidate_id": row["candidate_id"], "control_role": role,
                            "uid": "d1-id", "seed": seed,
                            "status": "completed_once", "sequence_ref": sequence_ref,
                            "implementation_refs": implementation_refs,
                            "harness_attempt_ref": harness_ref,
                            "completed_at": "2026-10-08T00:00:00Z",
                        }
                        producer["receipt_digest"] = g01.canonical_record_digest(
                            producer, "receipt_digest")
                        producer_receipt_path = root / (stem + "-producer.json")
                        producer_receipt_path.write_text(json.dumps(producer))
                        producer_ref = g01.file_ref(
                            self.repository, producer_receipt_path)
                        admission = {
                            "kind": "g01-control-native-admission", "version": "1.0.0",
                            "design_digest": design["design_digest"],
                            "candidate_id": row["candidate_id"], "control_role": role,
                            "uid": "d1-id", "generation_seed": seed,
                            "candidate_spec_ref": row["spec_ref"],
                            "sequence_ref": sequence_ref,
                            "implementation_refs": implementation_refs,
                            "generation_identity_ref": g01.file_ref(
                                self.repository, identity_path),
                            "producer_receipt_ref": producer_ref,
                            "harness_attempt_ref": harness_ref,
                            "official_scorer_refs":
                                design["benchmark"]["official_scorer"]["source_refs"],
                            "status": "admitted", "author": "fixture-admission-author",
                            "admitted_at": "2026-10-08T00:01:00Z",
                        }
                        admission_review = {
                            "kind": "g01-control-native-admission-review",
                            "version": "1.0.0",
                            "admission_payload_digest":
                                g01._control_admission_payload_digest(admission),
                            "outcome": "verified",
                            "checks": ["candidate_role_identity",
                                       "implementation_closure", "single_attempt",
                                       "sequence_identity", "official_scorer_closure"],
                            "reviewer": "fixture-admission-reviewer",
                            "reviewed_at": "2026-10-08T00:02:00Z",
                        }
                        admission_review["review_digest"] = \
                            g01.canonical_record_digest(
                                admission_review, "review_digest")
                        admission_review_path = root / (stem + "-admission-review.json")
                        admission_review_path.write_text(json.dumps(admission_review))
                        admission["review_ref"] = g01.file_ref(
                            self.repository, admission_review_path)
                        admission["admission_digest"] = g01.canonical_record_digest(
                            admission, "admission_digest")
                        admission_path = root / (stem + "-admission.json")
                        admission_path.write_text(json.dumps(admission))
                        admission_ref = g01.file_ref(self.repository, admission_path)
                        generation = {
                            "kind": "g01-qualified-generation-unit",
                            "version": "1.0.0",
                            "design_digest": design["design_digest"],
                            "uid": "d1-id", "seed": seed,
                            "status": "qualified_complete",
                            "sequence_ref": sequence_ref,
                            "producer_receipt_ref": producer_ref,
                            "generation_identity_ref": g01.file_ref(
                                self.repository, identity_path),
                            "completed_at": "2026-10-08T00:00:00Z",
                        }
                        generation["receipt_digest"] = g01.canonical_record_digest(
                            generation, "receipt_digest")
                        generation_path = root / (stem + "-generation.json")
                        generation_path.write_text(json.dumps(generation))
                        comparison = {
                            "kind": "g01-b-star-comparison-request",
                            "version": "1.0.0",
                            "design_digest": design["design_digest"],
                            "candidate_id": row["candidate_id"], "uid": "d1-id",
                            "inference_seed": seed, "scoring_seed": 44,
                            "roles": [{"role": role,
                                       "sequence_ref": sequence_ref,
                                       "native_admission_ref": admission_ref}],
                        }
                        comparison["request_digest"] = hashlib.sha256(
                            g01._canonical_bytes(comparison)).hexdigest()
                        comparison_path = root / (stem + "-comparison.json")
                        comparison_path.write_text(json.dumps(comparison))
                        scoring = {
                            "kind": "g01-b-star-scoring-request",
                            "version": "1.0.0",
                            "design_digest": design["design_digest"],
                            "candidate_id": row["candidate_id"], "uid": "d1-id",
                            "comparison_request_digest":
                                comparison["request_digest"],
                            "metrics": ["cd_3d", "cd_4d", "cd_motion"],
                            "official_scorer_refs":
                                design["benchmark"]["official_scorer"]["source_refs"],
                        }
                        scoring["request_digest"] = hashlib.sha256(
                            g01._canonical_bytes(scoring)).hexdigest()
                        scoring_path = root / (stem + "-scoring.json")
                        scoring_path.write_text(json.dumps(scoring))
                        raw_bytes = ("official-raw-" + stem).encode()
                        raw_path = root / (stem + "-raw-evidence.tar")
                        with tarfile.open(raw_path, mode="w:") as archive:
                            info = tarfile.TarInfo("official.json")
                            info.size = len(raw_bytes)
                            archive.addfile(info, io.BytesIO(raw_bytes))
                        raw_ref = g01.file_ref(self.repository, raw_path)
                        manifest = {
                            "kind": "g01-b-star-official-raw-manifest",
                            "version": "1.0.0",
                            "request_digest": scoring["request_digest"],
                            "comparison_request_digest":
                                comparison["request_digest"],
                            "files": [{
                                "path": "official.json", "bytes": len(raw_bytes),
                                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                            }],
                            "bundle_ref": {
                                "archive": {"sha256": raw_ref["sha256"]},
                                "raw_file_count": 1,
                            },
                        }
                        manifest_path = root / (stem + "-raw-manifest.json")
                        manifest_path.write_text(json.dumps(manifest))
                        manifest_ref = g01.file_ref(
                            self.repository, manifest_path)
                        score_path = root / (stem + "-score.json")
                        scored = {
                            "kind": "g01-b-star-official-score-result",
                            "version": "1.0.0",
                            "design_digest": design["design_digest"],
                            "candidate_id": row["candidate_id"],
                            "uid": "d1-id", "status": "execution_completed",
                            "request_digest": scoring["request_digest"],
                            "comparison_request_digest":
                                comparison["request_digest"],
                            "readout": {"roles": [{
                                "role": role, "status": "success",
                                "metrics": {
                                    "cd_3d": float(index + 1),
                                    "cd_4d": float(index + 1),
                                    "cd_motion": float(index + 1),
                                },
                            }]},
                            "raw_bundle": {
                                "manifest": {"sha256": manifest_ref["sha256"]},
                                "archive": {"sha256": raw_ref["sha256"]},
                            },
                            "completed_at": "2026-10-08T00:03:00Z",
                        }
                        scored["result_digest"] = g01.canonical_record_digest(
                            scored, "result_digest")
                        score_path.write_text(json.dumps(scored))
                        receipt = {
                            "kind": "g01-b-star-d1-native-receipt",
                            "version": "1.0.0",
                            "design_digest": design["design_digest"],
                            "split_ref": split_ref,
                            "candidate_id": row["candidate_id"],
                            "control_role": role,
                            "uid": "d1-id", "generation_seed": seed,
                            "metric": row["primary_metric"],
                            "official_scorer_seed": 44,
                            "status": "qualified_complete",
                            "value": float(index + 1),
                            "sequence_ref": sequence_ref,
                            "generation_receipt_ref": g01.file_ref(
                                self.repository, generation_path),
                            "native_admission_ref": admission_ref,
                            "comparison_request_ref": g01.file_ref(
                                self.repository, comparison_path),
                            "scoring_request_ref": g01.file_ref(
                                self.repository, scoring_path),
                            "result_ref": g01.file_ref(
                                self.repository, score_path),
                            "raw_manifest_ref": manifest_ref,
                            "raw_evidence_ref": raw_ref,
                            "completed_at": "2026-10-08T00:04:00Z",
                        }
                        receipt["receipt_digest"] = g01.canonical_record_digest(
                            receipt, "receipt_digest")
                        receipt_path = root / (
                            row["candidate_id"] + "-" + role + "-" + str(seed)
                            + "-receipt.json")
                        receipt_path.write_text(json.dumps(receipt))
                        if first_receipt is None:
                            first_receipt = deepcopy(receipt)
                            first_receipt_path = receipt_path
                            first_admission = deepcopy(admission)
                            first_admission_path = admission_path
                        receipt_refs.append(g01.file_ref(
                            self.repository, receipt_path))
                    result = {
                        "kind": "g01-b-star-d1-control-result",
                        "version": "1.0.0",
                        "design_digest": design["design_digest"],
                        "split_ref": split_ref,
                        "candidate_id": row["candidate_id"],
                        "control_role": role,
                        "metric": row["primary_metric"],
                        "unit_receipt_refs": receipt_refs,
                        "completed_at": "2026-10-08T00:05:00Z",
                    }
                    result["result_digest"] = g01.canonical_record_digest(
                        result, "result_digest")
                    if first_result is None:
                        first_result = deepcopy(result)
                    result_path = root / (row["candidate_id"] + "-" + role + ".json")
                    result_path.write_text(json.dumps(result))
                    result_refs.append(g01.file_ref(self.repository, result_path))

            freeze = {
                "kind": "g01-b-star-freeze", "version": "1.0.0",
                "design_digest": design["design_digest"],
                "split_ref": split_ref,
                "selected_by_candidate": selections,
                "d1_result_refs": result_refs,
                "selection_rule":
                    "min_primary_family_mean_over_all_qualified_controls_on_D1_lexical_tie_break",
                "author": "fixture-freeze-author",
                "frozen_at": "2026-10-08T00:06:00Z",
            }
            review = {
                "kind": "g01-b-star-freeze-review", "version": "1.0.0",
                "selection_payload_digest": g01._b_star_payload_digest(freeze),
                "outcome": "verified",
                "checks": ["complete_control_inventory", "D1_only",
                           "native_metric_identity", "deterministic_selection",
                           "no_confirmation_access"],
                "reviewer": "fixture-independent-reviewer",
                "reviewed_at": "2026-10-08T00:07:00Z",
            }
            review["review_digest"] = g01.canonical_record_digest(
                review, "review_digest")
            review_path = root / "review.json"
            review_path.write_text(json.dumps(review))
            freeze["review_ref"] = g01.file_ref(self.repository, review_path)
            freeze["freeze_digest"] = g01.canonical_record_digest(
                freeze, "freeze_digest")
            g01.validate_b_star_freeze(
                self.repository, design, freeze, expected_split=split)

            rogue_source = root / "coherent-rogue-implementation.py"
            rogue_source.write_text("# coherently substituted but not design-approved\n")
            rogue_refs = [g01.file_ref(self.repository, rogue_source)]
            rogue_identity = json.loads(g01.resolve_ref(
                self.repository,
                first_admission["generation_identity_ref"]).read_text())
            rogue_identity["producer_source_refs"] = rogue_refs
            rogue_identity["identity_digest"] = g01.canonical_record_digest(
                rogue_identity, "identity_digest")
            rogue_identity_path = root / "rogue-identity.json"
            rogue_identity_path.write_text(json.dumps(rogue_identity))
            base_harness = json.loads(g01.resolve_ref(
                self.repository, first_admission["harness_attempt_ref"]).read_text())
            rogue_native_plan = json.loads(g01.resolve_ref(
                self.repository, base_harness["native_plan_ref"]).read_text())
            rogue_native_plan["jobs"][0]["code_refs"] = rogue_refs + \
                design["benchmark"]["official_scorer"]["source_refs"]
            rogue_native_plan["plan_digest"] = g01.canonical_record_digest(
                rogue_native_plan, "plan_digest")
            rogue_native_plan_path = root / "rogue-native-plan.json"
            rogue_native_plan_path.write_text(json.dumps(rogue_native_plan))
            rogue_native_plan_ref = g01.file_ref(
                self.repository, rogue_native_plan_path)
            rogue_native_receipt = json.loads(g01.resolve_ref(
                self.repository, base_harness["native_receipt_ref"]).read_text())
            rogue_native_receipt["plan_digest"] = rogue_native_plan["plan_digest"]
            rogue_native_receipt_path = root / "rogue-native-receipt.json"
            rogue_native_receipt_path.write_text(json.dumps(rogue_native_receipt))
            rogue_harness_plan = json.loads(g01.resolve_ref(
                self.repository, base_harness["harness_plan_ref"]).read_text())
            rogue_harness_plan["tasks"][0]["plan_ref"] = rogue_native_plan_ref
            rogue_harness_plan["plan_digest"] = g01.canonical_record_digest(
                rogue_harness_plan, "plan_digest")
            rogue_harness_plan_path = root / "rogue-harness-plan.json"
            rogue_harness_plan_path.write_text(json.dumps(rogue_harness_plan))
            rogue_harness_report = json.loads(g01.resolve_ref(
                self.repository, base_harness["harness_report_ref"]).read_text())
            rogue_harness_report["plan_digest"] = rogue_harness_plan["plan_digest"]
            rogue_harness_report_path = root / "rogue-harness-report.json"
            rogue_harness_report_path.write_text(json.dumps(rogue_harness_report))
            base_harness["implementation_refs"] = rogue_refs
            base_harness["harness_plan_ref"] = g01.file_ref(
                self.repository, rogue_harness_plan_path)
            base_harness["harness_report_ref"] = g01.file_ref(
                self.repository, rogue_harness_report_path)
            base_harness["native_plan_ref"] = rogue_native_plan_ref
            base_harness["native_receipt_ref"] = g01.file_ref(
                self.repository, rogue_native_receipt_path)
            base_harness["approved_plan_digest"] = rogue_harness_plan["plan_digest"]
            base_harness["attempt_digest"] = g01.canonical_record_digest(
                base_harness, "attempt_digest")
            rogue_harness_path = root / "rogue-harness-attempt.json"
            rogue_harness_path.write_text(json.dumps(base_harness))
            rogue_harness_ref = g01.file_ref(self.repository, rogue_harness_path)
            rogue_producer = json.loads(g01.resolve_ref(
                self.repository, first_admission["producer_receipt_ref"]).read_text())
            rogue_producer["implementation_refs"] = rogue_refs
            rogue_producer["harness_attempt_ref"] = rogue_harness_ref
            rogue_producer["receipt_digest"] = g01.canonical_record_digest(
                rogue_producer, "receipt_digest")
            rogue_producer_path = root / "rogue-producer.json"
            rogue_producer_path.write_text(json.dumps(rogue_producer))
            rogue_admission = deepcopy(first_admission)
            rogue_admission["implementation_refs"] = rogue_refs
            rogue_admission["generation_identity_ref"] = g01.file_ref(
                self.repository, rogue_identity_path)
            rogue_admission["producer_receipt_ref"] = g01.file_ref(
                self.repository, rogue_producer_path)
            rogue_admission["harness_attempt_ref"] = rogue_harness_ref
            rogue_review = json.loads(g01.resolve_ref(
                self.repository, first_admission["review_ref"]).read_text())
            rogue_review["admission_payload_digest"] = \
                g01._control_admission_payload_digest(rogue_admission)
            rogue_review["review_digest"] = g01.canonical_record_digest(
                rogue_review, "review_digest")
            rogue_review_path = root / "rogue-admission-review.json"
            rogue_review_path.write_text(json.dumps(rogue_review))
            rogue_admission["review_ref"] = g01.file_ref(
                self.repository, rogue_review_path)
            rogue_admission["admission_digest"] = g01.canonical_record_digest(
                rogue_admission, "admission_digest")
            rogue_admission_path = root / "rogue-admission.json"
            rogue_admission_path.write_text(json.dumps(rogue_admission))
            admitted_row = next(row for row in design["candidates"]
                                if row["candidate_id"] ==
                                first_receipt["candidate_id"])
            with self.assertRaisesRegex(ValueError, "native admission"):
                g01._validate_control_admission(
                    self.repository, design, admitted_row,
                    first_receipt["control_role"], first_receipt["uid"],
                    first_receipt["generation_seed"],
                    first_receipt["sequence_ref"],
                    g01.file_ref(self.repository, rogue_admission_path))

            first = design["candidates"][0]
            alternate = [role for role in first["control_roles"]
                         if role.lower() != "b_star"][1]
            freeze["selected_by_candidate"][first["candidate_id"]]["role"] = alternate
            freeze["freeze_digest"] = g01.canonical_record_digest(
                freeze, "freeze_digest")
            with self.assertRaisesRegex(ValueError, "deterministic D1 minimum"):
                g01.validate_b_star_freeze(
                    self.repository, design, freeze, expected_split=split)

            missing = deepcopy(first_result)
            missing["unit_receipt_refs"] = missing["unit_receipt_refs"][:-1]
            missing["result_digest"] = g01.canonical_record_digest(
                missing, "result_digest")
            with self.assertRaisesRegex(ValueError, "full UID x seed denominator"):
                g01._validate_b_star_result(
                    self.repository, design, split_ref, split, missing)

            replay = deepcopy(first_result)
            replay["unit_receipt_refs"][1] = replay["unit_receipt_refs"][0]
            replay["result_digest"] = g01.canonical_record_digest(
                replay, "result_digest")
            with self.assertRaisesRegex(ValueError, "duplicate native D1"):
                g01._validate_b_star_result(
                    self.repository, design, split_ref, split, replay)

            failed = deepcopy(first_receipt)
            failed["status"] = "execution_error"
            failed["receipt_digest"] = g01.canonical_record_digest(
                failed, "receipt_digest")
            first_receipt_path.write_text(json.dumps(failed))
            failed_result = deepcopy(first_result)
            failed_result["unit_receipt_refs"][0] = g01.file_ref(
                self.repository, first_receipt_path)
            failed_result["result_digest"] = g01.canonical_record_digest(
                failed_result, "result_digest")
            with self.assertRaisesRegex(ValueError, "invalid native D1 receipt"):
                g01._validate_b_star_result(
                    self.repository, design, split_ref, split, failed_result)

            wrong_scorer = deepcopy(first_receipt)
            wrong_scorer["official_scorer_seed"] = 45
            wrong_scorer["receipt_digest"] = g01.canonical_record_digest(
                wrong_scorer, "receipt_digest")
            first_receipt_path.write_text(json.dumps(wrong_scorer))
            wrong_result = deepcopy(first_result)
            wrong_result["unit_receipt_refs"][0] = g01.file_ref(
                self.repository, first_receipt_path)
            wrong_result["result_digest"] = g01.canonical_record_digest(
                wrong_result, "result_digest")
            with self.assertRaisesRegex(ValueError, "invalid native D1 receipt"):
                g01._validate_b_star_result(
                    self.repository, design, split_ref, split, wrong_result)

            wrong_admission = deepcopy(first_admission)
            wrong_admission["control_role"] = "substituted-unapproved-role"
            wrong_admission["admission_digest"] = g01.canonical_record_digest(
                wrong_admission, "admission_digest")
            first_admission_path.write_text(json.dumps(wrong_admission))
            substituted = deepcopy(first_receipt)
            substituted["native_admission_ref"] = g01.file_ref(
                self.repository, first_admission_path)
            substituted["receipt_digest"] = g01.canonical_record_digest(
                substituted, "receipt_digest")
            first_receipt_path.write_text(json.dumps(substituted))
            substituted_result = deepcopy(first_result)
            substituted_result["unit_receipt_refs"][0] = g01.file_ref(
                self.repository, first_receipt_path)
            substituted_result["result_digest"] = g01.canonical_record_digest(
                substituted_result, "result_digest")
            with self.assertRaisesRegex(ValueError, "native admission"):
                g01._validate_b_star_result(
                    self.repository, design, split_ref, split, substituted_result)

            rogue_source = root / "rogue-implementation.py"
            rogue_source.write_text("# unapproved substituted implementation\n")
            wrong_source = deepcopy(first_admission)
            wrong_source["implementation_refs"] = [g01.file_ref(
                self.repository, rogue_source)]
            wrong_source["admission_digest"] = g01.canonical_record_digest(
                wrong_source, "admission_digest")
            first_admission_path.write_text(json.dumps(wrong_source))
            substituted["native_admission_ref"] = g01.file_ref(
                self.repository, first_admission_path)
            substituted["receipt_digest"] = g01.canonical_record_digest(
                substituted, "receipt_digest")
            first_receipt_path.write_text(json.dumps(substituted))
            substituted_result["unit_receipt_refs"][0] = g01.file_ref(
                self.repository, first_receipt_path)
            substituted_result["result_digest"] = g01.canonical_record_digest(
                substituted_result, "result_digest")
            with self.assertRaisesRegex(ValueError, "generation identity"):
                g01._validate_b_star_result(
                    self.repository, design, split_ref, split, substituted_result)

            admission_review_path = g01.resolve_ref(
                self.repository, first_admission["review_ref"])
            late_review = json.loads(admission_review_path.read_text())
            late_review["reviewed_at"] = "2026-10-08T00:04:00Z"
            late_review["review_digest"] = g01.canonical_record_digest(
                late_review, "review_digest")
            admission_review_path.write_text(json.dumps(late_review))
            post_score_admission = deepcopy(first_admission)
            post_score_admission["review_ref"] = g01.file_ref(
                self.repository, admission_review_path)
            post_score_admission["admission_digest"] = \
                g01.canonical_record_digest(
                    post_score_admission, "admission_digest")
            first_admission_path.write_text(json.dumps(post_score_admission))
            post_score = deepcopy(first_receipt)
            post_score["native_admission_ref"] = g01.file_ref(
                self.repository, first_admission_path)
            post_score["receipt_digest"] = g01.canonical_record_digest(
                post_score, "receipt_digest")
            first_receipt_path.write_text(json.dumps(post_score))
            post_score_result = deepcopy(first_result)
            post_score_result["unit_receipt_refs"][0] = g01.file_ref(
                self.repository, first_receipt_path)
            post_score_result["result_digest"] = g01.canonical_record_digest(
                post_score_result, "result_digest")
            with self.assertRaisesRegex(ValueError, "official score result"):
                g01._validate_b_star_result(
                    self.repository, design, split_ref, split, post_score_result)

    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            g01.loads_json_strict('{"kind":"one","kind":"two"}')

    def test_local_acceptance_job_is_cpu_only_single_attempt_and_fail_closed(self):
        command, limits, resources = g01_planner.g01_job_contract(
            python="/env/bin/python",
            design="../docs/G01_DESIGN.json",
            family_evidence="../inputs/g01/family-evidence.json",
            output_split="../inputs/g01/family-split.json",
            b_star_selections=None)
        self.assertEqual(command[:4], [
            "/env/bin/python", "-m", "research_math.g01_design", "--root"])
        self.assertIn("--family-evidence", command)
        self.assertIn("--output-split", command)
        self.assertNotIn("--b-star-selections", command)
        self.assertEqual(limits["max_attempts"], 1)
        self.assertEqual(limits["max_retries_per_trial"], 0)
        self.assertEqual(limits["max_confirmation_trials"], 0)
        self.assertEqual(resources["gpu_count"], 0)
        self.assertFalse(resources["allow_gpu_share"])


if __name__ == "__main__":
    unittest.main()
