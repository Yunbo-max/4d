"""Static-fixture contracts for the admitted C01 scoring plan builder.

No harness, scorer, scientific project code, or GPU is invoked by these tests.
Web authors them unexecuted; Local runs them through the common harness.
"""
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


try:
    planner = importlib.import_module('prepare_c01_native_scoring')
    launcher = importlib.import_module('launch_c01_native_scoring')
except ModuleNotFoundError:
    planner = None
    launcher = None


class C01NativeScoringPlanTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(planner, 'C01 native scoring plan builder is missing')
        self.assertIsNotNone(launcher, 'C01 finalized controller launch gate is missing')

    def contract(self):
        names = {
            'treatment': 'self_map_subtraction',
            'baseline': 'b_star',
            'b0': 'b0',
            'raw_uncorrected': 'raw_uncorrected',
            'mean_bias': 'mean_bias',
        }
        refs = {
            role: [{'path': role + '.py', 'sha256': 'a' * 64}]
            for role in names
        }
        refs['baseline'] = refs['raw_uncorrected']
        return {
            'benchmark_id': 'facebook/actionbench', 'benchmark_revision': 'rev',
            'primary_metric': 'cd_motion',
            'metrics': [{'name': name, 'output_path': ['readout', name]}
                        for name in ('cd_3d', 'cd_4d', 'cd_motion')],
            'contrasts': {'treatment': 'self_map_subtraction', 'baseline': 'b_star',
                          'controls': ['b0', 'raw_uncorrected', 'mean_bias']},
            'arm_requirements': {
                role: {'name': name, 'revision': name + '-method',
                       'implementation_refs': refs[role]}
                for role, name in names.items()},
            'scorer': {'kind': 'official', 'source_refs': [], 'code_refs': []},
        }

    def comparison(self):
        roles = []
        ref_hashes = {
            'b0': 'a' * 64, 'raw_uncorrected': 'a' * 64,
            'mean_bias': 'a' * 64,
            'self_map_subtraction': 'a' * 64,
        }
        for index, role in enumerate(
                ('b0', 'b_star', 'raw_uncorrected', 'mean_bias',
                 'self_map_subtraction')):
            row = {'role': role, 'method_id': role + '-method'}
            if role == 'b_star':
                row['alias_of'] = 'raw_uncorrected'
            else:
                row.update(
                    report_ref={'path': role + '-report.json',
                                'sha256': chr(97 + index) * 64},
                    implementation_sha256=ref_hashes[role])
            roles.append(row)
        return {'roles': roles}

    def admission(self):
        ref = lambda name, char: {'path': name + '.json', 'sha256': char * 64}
        value = {
            'kind': 'c01-scientific-dispatch-admission', 'version': 1,
            'candidate_id': '4d-math-20261006-c01', 'uid': 'uid',
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

    def test_analysis_core_binding_is_acyclic_and_content_sensitive(self):
        base = {'protocol_id': 'c01', 'criteria': [{'metric': 'cd_motion', 'min_effect': 1.0}]}
        first = planner.analysis_protocol_core_digest(base)
        finalized = dict(base, analysis_plan_ref={'path': 'analysis.json', 'sha256': 'a' * 64},
                         protocol_digest='b' * 64, frozen_at='2026-10-08T00:00:00Z')
        self.assertEqual(first, planner.analysis_protocol_core_digest(finalized))
        changed = dict(finalized, criteria=[{'metric': 'cd_motion', 'min_effect': 2.0}])
        self.assertNotEqual(first, planner.analysis_protocol_core_digest(changed))

    def test_contract_requires_corotational_treatment_and_b_star_baseline(self):
        planner.require_c14_contract(self.contract(), benchmark_revision='rev')
        changed = self.contract()
        changed['contrasts']['baseline'] = 'raw_uncorrected'
        with self.assertRaisesRegex(ValueError, 'five-role'):
            planner.require_c14_contract(changed, benchmark_revision='rev')

    def test_contract_requires_all_three_named_controls(self):
        changed = self.contract()
        changed['contrasts']['controls'] = ['b0', 'raw_uncorrected']
        changed['arm_requirements'].pop('mean_bias')
        with self.assertRaisesRegex(ValueError, 'five-role'):
            planner.require_c14_contract(changed, benchmark_revision='rev')

    def test_contract_requires_official_scorer(self):
        changed = self.contract()
        changed.pop('scorer')
        with self.assertRaisesRegex(ValueError, 'official'):
            planner.require_c14_contract(changed, benchmark_revision='rev')

    def test_arm_binding_uses_frozen_method_ids_and_every_implementation(self):
        refs = planner.bind_contract_arms(self.contract(), self.comparison())
        self.assertEqual({ref['path'] for ref in refs}, {
            'treatment.py', 'b0.py', 'raw_uncorrected.py',
            'mean_bias.py',
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

    def test_method_boundary_must_be_design_verified_for_c01(self):
        ready = {'workflow_boundary': {
            'action': 'dispatch', 'candidate_id': '4d-math-20261006-c01',
            'required_stage': 'design_verified', 'ready': True}}
        planner.require_method_boundary(ready)
        blocked = {'workflow_boundary': {**ready['workflow_boundary'], 'ready': False,
                                         'reason_codes': ['DESIGN_REVIEW_REQUIRED']}}
        with self.assertRaisesRegex(ValueError, 'design-verified'):
            planner.require_method_boundary(blocked)

    def test_outputs_require_result_manifest_and_archive(self):
        outputs = planner.scoring_outputs()
        self.assertEqual(outputs, [
            'actionmesh/c01-scoring-output/result.json',
            'actionmesh/c01-scoring-output/raw-manifest.json',
            'actionmesh/c01-scoring-output/raw-evidence.tar',
        ])

    def test_authorization_reservation_is_recoverable_then_consumed_once(self):
        ref = {'path': 'inputs/c01/gpu-resume.json', 'sha256': 'a' * 64}
        request_ref = {'path': 'inputs/c01/request.json', 'sha256': 'b' * 64}
        method_ref = {'path': 'inputs/c01/methods.json', 'sha256': 'c' * 64}
        environment_ref = {
            'path': 'inputs/native-runtime/environment.json',
            'sha256': 'd' * 64}
        authorization = {'authorization_nonce': 'e' * 64,
                         'expires_at': '2026-10-09T00:00:00+00:00'}
        command = ['python', '-m', 'research_math.c01_native_scoring', 'score']
        class Validator:
            @staticmethod
            def validate_plan(root, value):
                return value

            @staticmethod
            def plan_digest(value):
                return planner.canonical_record_digest(value, 'plan_digest')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reserved = planner.reserve_authorization(
                root, ref, authorization, request_ref=request_ref,
                method_batch_ref=method_ref, environment_ref=environment_ref,
                run_id='c01-native-scoring-001', group='confirmation',
                contract_digest='f' * 64)
            resumed = planner.reserve_authorization(
                root, ref, authorization, request_ref=request_ref,
                method_batch_ref=method_ref, environment_ref=environment_ref,
                run_id='c01-native-scoring-001', group='confirmation',
                contract_digest='f' * 64)
            self.assertEqual(reserved, resumed)
            self.assertEqual(
                reserved['path'],
                'inputs/c01/authorization-consumption/' + 'a' * 64
                + '.reservation.json')
            ticket = planner.issue_launch_ticket(
                root, ref, authorization, reservation_ref=reserved,
                request_ref=request_ref, method_batch_ref=method_ref,
                environment_ref=environment_ref,
                run_id='c01-native-scoring-001', group='confirmation',
                contract_digest='f' * 64, gpu_uuid='GPU-fixture')
            native_path = root/'plans/c01/native.json'
            native_path.parent.mkdir(parents=True)
            native = {
                'run_id': 'c01-native-scoring-001',
                'jobs': [{'trial_id': 'c01-five-role-official-scoring',
                          'group': 'confirmation', 'command': command,
                          'input_refs': [request_ref, ref, reserved, ticket]}],
            }
            native['plan_digest'] = planner.canonical_record_digest(
                native, 'plan_digest')
            native_path.write_text(json.dumps(native))
            native_ref = planner.scoring.file_ref(root, native_path)
            harness_path = root/'plans/c01/harness.json'
            harness = {
                'batch_id': 'c01-native-scoring-001',
                'tasks': [{'task_id': 'c01-five-role-official-scoring',
                           'plan_ref': native_ref}],
            }
            harness['plan_digest'] = planner.canonical_record_digest(
                harness, 'plan_digest')
            harness_path.write_text(json.dumps(harness))
            harness_ref = planner.scoring.file_ref(root, harness_path)
            with self.assertRaisesRegex(ValueError, 'plan pair'):
                planner.consume_authorization(
                    root, ref, authorization, reservation_ref=reserved,
                    launch_ticket_ref=ticket,
                    request_ref=request_ref, method_batch_ref=method_ref,
                    environment_ref=environment_ref,
                    run_id='c01-native-scoring-001', group='confirmation',
                    contract_digest='f' * 64,
                    native_plan_ref=native_ref,
                    native_plan_digest=native['plan_digest'],
                    harness_plan_ref=harness_ref,
                    harness_plan_digest=harness['plan_digest'],
                    expected_command=['different'], native_validator=Validator,
                    harness_validator=Validator)
            consumed = planner.consume_authorization(
                root, ref, authorization, reservation_ref=reserved,
                launch_ticket_ref=ticket,
                request_ref=request_ref,
                method_batch_ref=method_ref, environment_ref=environment_ref,
                run_id='c01-native-scoring-001', group='confirmation',
                contract_digest='f' * 64,
                native_plan_ref=native_ref,
                native_plan_digest=native['plan_digest'],
                harness_plan_ref=harness_ref,
                harness_plan_digest=harness['plan_digest'],
                expected_command=command, native_validator=Validator,
                harness_validator=Validator)
            self.assertEqual(
                consumed['path'],
                'inputs/c01/authorization-consumption/' + 'a' * 64 + '.json')
            finalized = planner.scoring.read_json(root/consumed['path'])
            owner, claim = launcher.claim_launch(
                root, root/consumed['path'], finalized,
                expected_harness_digest=harness['plan_digest'])
            with self.assertRaisesRegex(RuntimeError, 'live controller owner'):
                launcher.claim_launch(
                    root, root/consumed['path'], finalized,
                    expected_harness_digest=harness['plan_digest'])
            owner.close()
            resumed, same_claim = launcher.claim_launch(
                root, root/consumed['path'], finalized,
                expected_harness_digest=harness['plan_digest'])
            self.assertEqual(claim, same_claim)
            resumed.close()
            with self.assertRaises(FileExistsError):
                planner.consume_authorization(
                    root, ref, authorization, reservation_ref=reserved,
                    launch_ticket_ref=ticket,
                    request_ref=request_ref,
                    method_batch_ref=method_ref,
                    environment_ref=environment_ref,
                    run_id='c01-native-scoring-001', group='confirmation',
                    contract_digest='f' * 64,
                    native_plan_ref=native_ref,
                    native_plan_digest=native['plan_digest'],
                    harness_plan_ref=harness_ref,
                    harness_plan_digest=harness['plan_digest'],
                    expected_command=command, native_validator=Validator,
                    harness_validator=Validator)
            with self.assertRaises(FileExistsError):
                planner.reserve_authorization(
                    root, ref, authorization, request_ref=request_ref,
                    method_batch_ref=method_ref,
                    environment_ref=environment_ref,
                    run_id='c01-native-scoring-001', group='confirmation',
                    contract_digest='f' * 64)

    def test_installed_stage_rewrites_hash_bound_launch_ticket(self):
        import run_experiments
        from research_math import c01_native_scoring as scoring

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'actionmesh').mkdir()
            request_path = root/'inputs/c01/request.json'
            request_path.parent.mkdir(parents=True)
            request_path.write_text('{"fixture":true}\n')
            request_ref = scoring.file_ref(root, request_path)
            authorization_path = root/'inputs/c01/authorization.json'
            authorization = {
                'authorization_nonce': 'e' * 64, 'gpu_uuid': 'GPU-fixture',
                'status': 'authorized', 'request_ref': request_ref,
                'method_batch_ref': {'path': 'inputs/methods.json',
                                     'sha256': 'b' * 64},
                'environment_ref': {'path': 'inputs/environment.json',
                                    'sha256': 'c' * 64},
                'run_id': 'c01-native-scoring-001',
                'task_id': 'c01-five-role-official-scoring',
                'group': 'confirmation', 'contract_digest': 'f' * 64,
                'expires_at': '2099-01-01T00:00:00+00:00',
            }
            authorization_path.write_text(json.dumps(authorization))
            authorization_ref = scoring.file_ref(root, authorization_path)
            reservation_path = root/'inputs/c01/reservation.json'
            reservation = {
                'authorization_ref': authorization_ref,
                'request_ref': request_ref,
                'run_id': 'c01-native-scoring-001', 'state': 'reserved'}
            reservation_path.write_text(json.dumps(reservation))
            reservation_ref = scoring.file_ref(root, reservation_path)
            ticket_path = root/'inputs/c01/launch-ticket.json'
            ticket = {
                'kind': 'c01-staged-launch-ticket', 'version': 1,
                'authorization_ref': authorization_ref,
                'authorization_nonce': 'e' * 64,
                'request_ref': request_ref,
                'method_batch_ref': authorization['method_batch_ref'],
                'environment_ref': authorization['environment_ref'],
                'run_id': authorization['run_id'],
                'task_id': authorization['task_id'],
                'group': authorization['group'],
                'contract_digest': authorization['contract_digest'],
                'reservation_ref': reservation_ref, 'gpu_uuid': 'GPU-fixture',
                'expected_output': 'actionmesh/c01-scoring-output',
                'execution_contract': {
                    'purpose': 'scientific',
                    'evidence_mode': 'prospective_confirmatory',
                    'max_attempts': 1, 'max_confirmation_trials': 1,
                    'max_retries_per_trial': 0},
                'issued_at': '2026-10-08T00:00:00+00:00',
                'expires_at': authorization['expires_at'],
            }
            ticket['ticket_digest'] = scoring.canonical_digest(ticket)
            ticket_path.write_text(json.dumps(ticket))
            ticket_ref = scoring.file_ref(root, ticket_path)
            job = {
                'command': ['python', '-m', 'research_math.c01_native_scoring',
                            'score', '--root', '..', '--request', str(request_path),
                            '--output', 'c01-scoring-output',
                            '--gpu-uuid', 'GPU-fixture',
                            '--launch-ticket', str(ticket_path)],
                'cwd': 'actionmesh',
                'input_refs': [request_ref, authorization_ref,
                               reservation_ref, ticket_ref],
                'code_refs': [],
            }
            attempt = root/'attempt'; attempt.mkdir()
            command, _, workspace = run_experiments._stage(root, job, attempt)
            staged_request = Path(command[command.index('--request') + 1])
            staged_ticket = Path(command[command.index('--launch-ticket') + 1])
            self.assertEqual(staged_request, workspace/'inputs/c01/request.json')
            self.assertEqual(staged_ticket,
                             workspace/'inputs/c01/launch-ticket.json')
            with self.assertRaisesRegex(ValueError, 'controller claim injection'):
                scoring.validate_execution_authorization(
                    workspace, staged_request,
                    workspace/'actionmesh/c01-scoring-output',
                    'GPU-fixture', staged_ticket)

    def test_final_consumption_claim_and_staged_ticket_close_one_chain(self):
        import _autoresearch as core
        import run_experiments
        import run_harness
        from research_math import c01_native_scoring as scoring

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'actionmesh').mkdir()
            def pinned(relative, value):
                path = root/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
                return path, scoring.file_ref(root, path)
            request_path, request_ref = pinned(
                'inputs/c01/request.json', {'request_digest': '1' * 64})
            _, method_ref = pinned('inputs/c01/method.json', {'fixture': 'method'})
            _, environment_ref = pinned(
                'inputs/native-runtime/environment.json', {'fixture': 'environment'})
            authorization = {
                'kind': 'c01-gpu-resume-authorization', 'version': 1,
                'authorization_nonce': 'e' * 64, 'gpu_uuid': 'GPU-fixture',
                'status': 'authorized', 'scope': 'single_c01_scoring_attempt',
                'request_ref': request_ref, 'method_batch_ref': method_ref,
                'environment_ref': environment_ref,
                'run_id': 'c01-native-scoring-001',
                'task_id': 'c01-five-role-official-scoring',
                'group': 'confirmation', 'contract_digest': 'f' * 64,
                'authorized_at': '2026-10-08T00:00:00+00:00',
                'expires_at': '2099-01-01T00:00:00+00:00'}
            authorization['authorization_digest'] = planner.canonical_record_digest(
                authorization, 'authorization_digest')
            authorization_path, authorization_ref = pinned(
                'inputs/c01/authorization.json', authorization)
            reservation_ref = planner.reserve_authorization(
                root, authorization_ref, authorization, request_ref=request_ref,
                method_batch_ref=method_ref, environment_ref=environment_ref,
                run_id=authorization['run_id'], group=authorization['group'],
                contract_digest=authorization['contract_digest'])
            ticket_ref = planner.issue_launch_ticket(
                root, authorization_ref, authorization,
                reservation_ref=reservation_ref, request_ref=request_ref,
                method_batch_ref=method_ref, environment_ref=environment_ref,
                run_id=authorization['run_id'], group=authorization['group'],
                contract_digest=authorization['contract_digest'],
                gpu_uuid='GPU-fixture')
            ticket_path = root/ticket_ref['path']
            source_path, source_ref = pinned(
                'inputs/c01/native/source.json', {'fixture': 'published source'})
            _, labels_ref = pinned(
                'inputs/c01/native/labels.json', {'fixture': 'labels'})
            _, implementation_ref = pinned(
                'inputs/c01/native/implementation.json', {'fixture': 'code'})
            sampling = {'policy': 'native', 'parameters': {'repetitions': 1}}
            budget = {'seconds': 900}
            _, sample_ref = pinned('inputs/c01/native/samples.json', {
                'benchmark_id': 'c01-fixture',
                'benchmark_revision': 'fixture-v1', 'split': 'confirmation',
                'sample_ids': ['uid'], 'denominator': 1,
                'predictions_per_sample': 1,
                'labels_or_tests_ref': labels_ref,
                'sampling': sampling, 'budget': budget})
            metrics = [{'name': 'accuracy',
                        'output_path': ['metrics', 'accuracy']}]
            scorer_contract = {
                'kind': 'official', 'identity': 'fixture-official-scorer',
                'revision': 'fixture-v1', 'source_refs': [source_ref],
                'code_refs': [implementation_ref],
                'command': ['fixture-scorer', '{predictions}', '{samples}',
                            '{labels}'],
                'cwd': '.', 'output': {'format': 'json', 'source': 'stdout'},
                'denominator_path': ['denominator']}
            _, definition_ref = pinned('inputs/c01/native/definition.json', {
                'benchmark_id': 'c01-fixture',
                'benchmark_revision': 'fixture-v1',
                'published_at': '2026-10-08T00:00:00Z',
                'source_url': 'https://example.org/c01-fixture',
                'primary_metric': 'accuracy', 'metrics': metrics,
                'splits': [{'name': 'confirmation',
                            'sample_manifest_ref': sample_ref,
                            'labels_or_tests_ref': labels_ref}],
                'prediction_format': {
                    'format': 'json', 'records_path': ['predictions'],
                    'id_path': ['sample_id']},
                'sampling': sampling, 'budget': budget,
                'scorer': scorer_contract, 'publication_refs': [source_ref]})
            contrasts = {
                'treatment': 'self_map_subtraction', 'baseline': 'b_star',
                'controls': ['b0', 'raw_uncorrected', 'mean_bias']}
            role_names = {
                'treatment': 'self_map_subtraction', 'baseline': 'b_star',
                'b0': 'b0', 'raw_uncorrected': 'raw_uncorrected',
                'mean_bias': 'mean_bias'}
            qualification = {
                'metric': 'accuracy', 'operator': 'ge', 'threshold': 0.0,
                'reference_ref': source_ref}
            native_contract = {
                'schema_id': 'native-eval-contract', 'schema_version': '1.0.0',
                'benchmark_id': 'c01-fixture',
                'benchmark_revision': 'fixture-v1', 'split': 'confirmation',
                'published_source_refs': [source_ref],
                'native_definition_ref': definition_ref,
                'sample_manifest_ref': sample_ref,
                'labels_or_tests_ref': labels_ref,
                'primary_metric': 'accuracy', 'metrics': metrics,
                'prediction_format': {
                    'format': 'json', 'records_path': ['predictions'],
                    'id_path': ['sample_id']},
                'sampling': sampling, 'budget': budget,
                'scorer': scorer_contract, 'contrasts': contrasts,
                'arm_requirements': {
                    role: {'name': name, 'revision': 'fixture-v1',
                           'implementation_refs': [implementation_ref]}
                    for role, name in role_names.items()},
                'baseline_qualification': qualification,
                'control_qualifications': [
                    {**qualification, 'role': role}
                    for role in contrasts['controls']]}
            protocol = {
                'protocol_id': 'c01-controller-fixture',
                'evidence_mode': 'prospective_confirmatory',
                'frozen_at': '2026-10-08T00:00:00Z',
                'required_groups': ['confirmation'],
                'seed_policy': {'seeds': [1], 'allow_extra': False},
                'contrasts': contrasts, 'criteria': [],
                'native_eval_contract': native_contract}
            protocol['protocol_digest'] = core.protocol_hash(protocol)
            _, protocol_ref = pinned('inputs/c01/native/protocol.json', protocol)
            command = [
                'python', '-m', 'research_math.c01_native_scoring', 'score',
                '--root', '..', '--request', str(request_path.resolve()),
                '--output', 'c01-scoring-output', '--gpu-uuid', 'GPU-fixture',
                '--launch-ticket', str(ticket_path.resolve())]
            native = run_experiments.make_plan(
                root, run_id=authorization['run_id'], purpose='scientific',
                evidence_mode='prospective_confirmatory',
                protocol_ref=protocol_ref,
                jobs=[{
                    'trial_id': authorization['task_id'],
                    'group': authorization['group'], 'arm_role': 'treatment',
                    'command': command, 'cwd': 'actionmesh',
                    'input_refs': [sample_ref, labels_ref, request_ref,
                                   authorization_ref, reservation_ref, ticket_ref],
                    'code_refs': [implementation_ref],
                    'output_paths': planner.scoring_outputs(), 'seed': 1}],
                provenance={
                    'git_revision': 'fixture-v1', 'model_revision': 'none',
                    'data_revision': 'fixture-v1',
                    'environment_digest': environment_ref['sha256']},
                limits={'max_attempts': 1, 'max_development_trials': 0,
                        'max_confirmation_trials': 1,
                        'max_retries_per_trial': 0,
                        'wall_time_seconds': 900,
                        'attempt_timeout_seconds': 600})
            native_path, native_ref = pinned('plans/c01/native.json', native)
            harness = run_harness.make_plan(
                root, batch_id=authorization['run_id'],
                pool_dir=root/'controller-pool',
                tasks=[{
                    'task_id': authorization['task_id'],
                    'idea_id': planner.CANDIDATE_ID, 'depends_on': [],
                    'priority': 1, 'plan_ref': native_ref,
                    'resources': {
                        'cpu_cores': 1, 'ram_mib': 1024, 'gpu_count': 1,
                        'gpu_peak_mib': None, 'allow_gpu_share': False,
                        'memory_profile_ref': None,
                        'exclusive_keys': [authorization['task_id']]}}],
                limits={'total_wall_seconds': 900, 'window_seconds': 900,
                        'max_parallel_tasks': 1, 'cpu_cores': 1,
                        'ram_mib': 1024, 'max_gpu_task_seconds': 900},
                gpus={'uuids': ['GPU-fixture'], 'safety_margin_mib': 1024,
                      'max_tasks_per_gpu': 1})
            harness_path, harness_ref = pinned('plans/c01/harness.json', harness)
            consumption_ref = planner.consume_authorization(
                root, authorization_ref, authorization,
                reservation_ref=reservation_ref, launch_ticket_ref=ticket_ref,
                request_ref=request_ref, method_batch_ref=method_ref,
                environment_ref=environment_ref, run_id=authorization['run_id'],
                group=authorization['group'],
                contract_digest=authorization['contract_digest'],
                native_plan_ref=native_ref,
                native_plan_digest=native['plan_digest'],
                harness_plan_ref=harness_ref,
                harness_plan_digest=harness['plan_digest'],
                expected_command=command, native_validator=run_experiments,
                harness_validator=run_harness)
            consumption_path = root/consumption_ref['path']
            consumption, _, _ = planner.validate_final_consumption(
                root, consumption_path,
                expected_harness_digest=harness['plan_digest'],
                native_validator=run_experiments,
                harness_validator=run_harness)
            copied = root/'inputs/c01/copied-consumption.json'
            copied.write_bytes(consumption_path.read_bytes())
            with self.assertRaisesRegex(ValueError, 'Canonical'):
                planner.validate_final_consumption(
                    root, copied, expected_harness_digest=harness['plan_digest'],
                    native_validator=run_experiments,
                    harness_validator=run_harness)
            owner, claim_ref = launcher.claim_launch(
                root, consumption_path, consumption,
                expected_harness_digest=harness['plan_digest'])
            with self.assertRaisesRegex(RuntimeError, 'live controller owner'):
                launcher.claim_launch(
                    root, consumption_path, consumption,
                    expected_harness_digest=harness['plan_digest'])
            owner.close()
            owner, recovered_claim_ref = launcher.claim_launch(
                root, consumption_path, consumption,
                expected_harness_digest=harness['plan_digest'])
            self.assertEqual(claim_ref, recovered_claim_ref)
            job = native['jobs'][0]
            attempt = root/'attempt-final'; attempt.mkdir()
            staged_command, _, workspace = run_experiments._stage(root, job, attempt)
            staged_request = Path(staged_command[
                staged_command.index('--request') + 1])
            staged_ticket = Path(staged_command[
                staged_command.index('--launch-ticket') + 1])
            injected = {
                'C01_CONTROLLER_ROOT': str(root),
                'C01_LAUNCH_CLAIM_PATH': str((root/claim_ref['path']).resolve()),
                'C01_LAUNCH_CLAIM_SHA256': claim_ref['sha256'],
                'C01_CONSUMPTION_PATH': str(consumption_path.resolve())}
            try:
                with mock.patch.dict(os.environ, injected, clear=False):
                    scoring.validate_execution_authorization(
                        workspace, staged_request,
                        workspace/'actionmesh/c01-scoring-output',
                        'GPU-fixture', staged_ticket)
                    with mock.patch.dict(
                            os.environ, {'C01_LAUNCH_CLAIM_SHA256': '0' * 64},
                            clear=False), self.assertRaisesRegex(
                                ValueError, 'controller claim|finalized claim/consumption closure mismatch'):
                        scoring.validate_execution_authorization(
                            workspace, staged_request,
                            workspace/'actionmesh/c01-scoring-output',
                            'GPU-fixture', staged_ticket)
                    claim_path = root/claim_ref['path']
                    original_claim = claim_path.read_bytes()
                    changed_claim = scoring.read_json(claim_path)
                    changed_claim['claimed_at'] = '2099-01-01T00:00:00+00:00'
                    claim_path.write_text(json.dumps(changed_claim))
                    with mock.patch.dict(
                            os.environ, {'C01_LAUNCH_CLAIM_SHA256':
                                         scoring.digest(claim_path)},
                            clear=False), self.assertRaisesRegex(
                                ValueError, 'controller claim|finalized claim/consumption closure mismatch'):
                        scoring.validate_execution_authorization(
                            workspace, staged_request,
                            workspace/'actionmesh/c01-scoring-output',
                            'GPU-fixture', staged_ticket)
                    claim_path.write_bytes(original_claim)
                    original_consumption = consumption_path.read_bytes()
                    changed_consumption = scoring.read_json(consumption_path)
                    changed_consumption['group'] = 'development'
                    consumption_path.write_text(json.dumps(changed_consumption))
                    with self.assertRaises(ValueError):
                        scoring.validate_execution_authorization(
                            workspace, staged_request,
                            workspace/'actionmesh/c01-scoring-output',
                            'GPU-fixture', staged_ticket)
                    with self.assertRaises(ValueError):
                        planner.validate_final_consumption(
                            root, consumption_path,
                            expected_harness_digest=harness['plan_digest'],
                            native_validator=run_experiments,
                            harness_validator=run_harness)
                    consumption_path.write_bytes(original_consumption)
                    original_harness = harness_path.read_bytes()
                    harness_path.write_bytes(original_harness + b' ')
                    with self.assertRaises(ValueError):
                        scoring.validate_execution_authorization(
                            workspace, staged_request,
                            workspace/'actionmesh/c01-scoring-output',
                            'GPU-fixture', staged_ticket)
                    with self.assertRaises(ValueError):
                        planner.validate_final_consumption(
                            root, consumption_path,
                            expected_harness_digest=harness['plan_digest'],
                            native_validator=run_experiments,
                            harness_validator=run_harness)
                    harness_path.write_bytes(original_harness)
                original_native = native_path.read_bytes()
                native_path.write_bytes(original_native + b' ')
                with mock.patch.dict(os.environ, injected, clear=False):
                    with self.assertRaises(ValueError):
                        scoring.validate_execution_authorization(
                            workspace, staged_request,
                            workspace/'actionmesh/c01-scoring-output',
                            'GPU-fixture', staged_ticket)
                    with self.assertRaises(ValueError):
                        planner.validate_final_consumption(
                            root, consumption_path,
                            expected_harness_digest=harness['plan_digest'],
                            native_validator=run_experiments,
                            harness_validator=run_harness)
                native_path.write_bytes(original_native)
            finally:
                owner.close()


if __name__ == '__main__':
    unittest.main()
