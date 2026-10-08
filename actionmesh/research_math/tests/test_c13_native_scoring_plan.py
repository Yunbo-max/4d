"""Static-fixture contracts for the admitted C13 scoring plan builder.

No harness, scorer, scientific project code, or GPU is invoked by these tests.
Web authors them unexecuted; Local runs them through the common harness.
"""
import importlib
import unittest


try:
    planner = importlib.import_module('prepare_c13_native_scoring')
except ModuleNotFoundError:
    planner = None


class C13NativeScoringPlanTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(planner, 'C13 native scoring plan builder is missing')

    def contract(self):
        names = {
            'treatment': 'group_acceleration',
            'baseline': 'b_star',
            'b0': 'b0',
            'gaussian': 'gaussian',
            'quadratic_acceleration': 'quadratic_acceleration',
        }
        refs = {
            role: [{'path': role + '.py', 'sha256': 'a' * 64}]
            for role in names
        }
        refs['baseline'] = refs['gaussian']
        return {
            'benchmark_id': 'facebook/actionbench', 'benchmark_revision': 'rev',
            'primary_metric': 'cd_motion',
            'metrics': [{'name': name, 'output_path': ['readout', name]}
                        for name in ('cd_3d', 'cd_4d', 'cd_motion')],
            'contrasts': {'treatment': 'group_acceleration', 'baseline': 'b_star',
                          'controls': ['b0', 'gaussian', 'quadratic_acceleration']},
            'arm_requirements': {
                role: {'name': name, 'revision': name + '-method',
                       'implementation_refs': refs[role]}
                for role, name in names.items()},
            'scorer': {'kind': 'official', 'source_refs': [], 'code_refs': []},
        }

    def comparison(self):
        return {'roles': [
            {'role': role, 'method_id': role + '-method',
             **({'alias_of': 'gaussian'} if role == 'b_star' else {}),
             **({} if role == 'b_star' else {
                 'report_ref': {'path': role + '-report.json',
                                'sha256': chr(97 + index) * 64}})}
            for index, role in enumerate(
                ('b0', 'b_star', 'gaussian', 'quadratic_acceleration',
                 'group_acceleration'))
        ]}

    def admission(self):
        ref = lambda name, char: {'path': name + '.json', 'sha256': char * 64}
        value = {
            'kind': 'c13-scientific-dispatch-admission', 'version': 1,
            'candidate_id': '4d-math-20261006-c13', 'uid': 'uid',
            'benchmark_revision': 'rev', 'comparison_digest': 'b' * 64,
            'protocol_ref': ref('protocol', 'c'),
            'method_batch_ref': ref('method-batch', 'd'),
            'natural_gate_0_ref': ref('natural-gate-0', 'e'),
            'importance_decision_ref': ref('importance-decision', 'f'),
            'family_split_ref': ref('family-split', '1'),
            'outcome_criteria_ref': ref('outcome-criteria', '2'),
            'gpu_resume_authorization_ref': ref('gpu-resume', '3'),
            'status': 'admitted', 'admitted_at': '2026-10-08T00:00:00Z',
        }
        value['admission_digest'] = planner.canonical_record_digest(value)
        return value

    def test_contract_requires_group_as_treatment_and_b_star_as_baseline(self):
        planner.require_c13_contract(self.contract(), benchmark_revision='rev')
        changed = self.contract()
        changed['contrasts']['baseline'] = 'gaussian'
        with self.assertRaisesRegex(ValueError, 'five-role'):
            planner.require_c13_contract(changed, benchmark_revision='rev')

    def test_contract_requires_all_three_named_controls(self):
        changed = self.contract()
        changed['contrasts']['controls'] = ['b0', 'gaussian']
        changed['arm_requirements'].pop('quadratic_acceleration')
        with self.assertRaisesRegex(ValueError, 'five-role'):
            planner.require_c13_contract(changed, benchmark_revision='rev')

    def test_contract_requires_official_scorer(self):
        changed = self.contract()
        changed.pop('scorer')
        with self.assertRaisesRegex(ValueError, 'official'):
            planner.require_c13_contract(changed, benchmark_revision='rev')

    def test_arm_binding_uses_frozen_method_ids_and_every_implementation(self):
        refs = planner.bind_contract_arms(self.contract(), self.comparison())
        self.assertEqual({ref['path'] for ref in refs}, {
            'treatment.py', 'b0.py', 'gaussian.py',
            'quadratic_acceleration.py',
        })
        changed = self.contract()
        changed['arm_requirements']['treatment']['revision'] = 'other'
        with self.assertRaisesRegex(ValueError, 'method identity'):
            planner.bind_contract_arms(changed, self.comparison())

    def test_one_attempt_preserves_protocol_mode_and_treatment_role(self):
        job, limits = planner.scientific_job_contract(
            {'evidence_mode': 'prospective_confirmatory'}, wall_seconds=900)
        self.assertEqual(job, {'arm_role': 'treatment'})
        self.assertEqual(limits['max_development_trials'], 0)
        self.assertEqual(limits['max_confirmation_trials'], 1)
        self.assertEqual(limits['max_retries_per_trial'], 0)

    def test_admission_digest_and_exact_scope_are_required(self):
        admission = self.admission()
        planner.require_admission_core(
            admission, uid='uid', benchmark_revision='rev',
            comparison_digest='b' * 64,
            protocol_ref=admission['protocol_ref'],
            method_batch_ref=admission['method_batch_ref'])
        changed = dict(admission, status='gpu_stopped')
        with self.assertRaisesRegex(ValueError, 'admission'):
            planner.require_admission_core(
                changed, uid='uid', benchmark_revision='rev',
                comparison_digest='b' * 64,
                protocol_ref=admission['protocol_ref'],
                method_batch_ref=admission['method_batch_ref'])

    def test_method_boundary_must_be_design_verified_for_c13(self):
        ready = {'workflow_boundary': {
            'action': 'dispatch', 'candidate_id': '4d-math-20261006-c13',
            'required_stage': 'design_verified', 'ready': True}}
        planner.require_method_boundary(ready)
        blocked = {'workflow_boundary': {**ready['workflow_boundary'], 'ready': False,
                                         'reason_codes': ['DESIGN_REVIEW_REQUIRED']}}
        with self.assertRaisesRegex(ValueError, 'design-verified'):
            planner.require_method_boundary(blocked)

    def test_outputs_require_result_manifest_and_archive(self):
        outputs = planner.scoring_outputs()
        self.assertEqual(outputs, [
            'actionmesh/c13-scoring-output/result.json',
            'actionmesh/c13-scoring-output/raw-manifest.json',
            'actionmesh/c13-scoring-output/raw-evidence.tar',
        ])


if __name__ == '__main__':
    unittest.main()
