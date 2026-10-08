"""CPU plan and genuine cross-root staging contracts; no scientific result claim."""
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import prepare_robust_motion_candidate as planner
from research_math.robust_motion_candidate import FLOAT_PARAMETERS,INTEGER_PARAMETERS


class RobustMotionPlanTests(unittest.TestCase):
    def project(self,root):
        source=Path(planner.__file__).resolve().parent
        for relative in ('research_math/__init__.py','research_math/robust_motion_candidate.py',
                         'research_math/protected_geometry_candidate.py','research_ten/__init__.py',
                         'research_ten/m01_elasticity.py'):
            target=root/'actionmesh'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source/relative,target)
        case=root/'inputs/native-case';case.mkdir(parents=True)
        path=case/'sequence.npz';path.write_bytes(b'opaque engineering staging fixture; not native acceptance')
        (case/'report.json').write_text(json.dumps({'status':'completed','uid':'fixture','seed':42,
            'sha256':{'sequence.npz':hashlib.sha256(path.read_bytes()).hexdigest()}}))
        return path

    def build(self,root,path,**overrides):
        params={k:.1 for k in FLOAT_PARAMETERS};params.update({k:64 for k in INTEGER_PARAMETERS})
        options=dict(root=root,source_sequence=path,run_id='c04-stage-fixture',parameters=params,
                     coordinate_lower=-1.,coordinate_upper=1.,bounds_policy='preserve_and_report',
                     max_artifact_bytes=1024*1024,plan_dir=root/'plans',wall_seconds=600)
        options.update(overrides);return planner.build_plans(**options)

    def test_single_attempt_cpu_and_transitive_closure(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=self.project(root);native,outer=self.build(root,path)
            self.assertEqual(native['limits']['max_attempts'],1)
            self.assertEqual(native['limits']['max_retries_per_trial'],0)
            self.assertEqual(outer['tasks'][0]['resources']['gpu_count'],0)
            self.assertEqual(outer['limits']['max_gpu_task_seconds'],0)
            self.assertEqual(len(native['jobs'][0]['code_refs']),5)
            self.assertIn('actionmesh/c04-robust-output/artifact.tar',native['jobs'][0]['output_paths'])

    def test_real_stage_remaps_source_across_roots_and_pins_report(self):
        native_module=importlib.import_module('run_experiments')
        with tempfile.TemporaryDirectory() as temp,tempfile.TemporaryDirectory() as other:
            root=Path(temp);path=self.project(root);native,_=self.build(root,path)
            attempt=Path(other)/'attempt';attempt.mkdir()
            command,cwd,workspace=native_module._stage(root,native['jobs'][0],attempt)
            staged=Path(command[command.index('--source-sequence')+1])
            self.assertTrue(staged.is_relative_to(workspace))
            self.assertNotEqual(staged,path)
            self.assertEqual(staged.read_bytes(),path.read_bytes())
            self.assertEqual(staged.with_name('report.json').read_bytes(),path.with_name('report.json').read_bytes())
            self.assertEqual(cwd,workspace/'actionmesh')
            self.assertEqual(command[command.index('--source-sequence-ref')+1],'inputs/native-case/sequence.npz')

    def test_changed_receipt_and_invalid_floor_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=self.project(root);path.write_bytes(b'changed')
            with self.assertRaises(ValueError):self.build(root,path)
            self.assertFalse((root/'plans').exists())
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=self.project(root)
            params={k:.1 for k in FLOAT_PARAMETERS};params.update({k:64 for k in INTEGER_PARAMETERS});params['shape_floor']=0.
            with self.assertRaises(ValueError):self.build(root,path,parameters=params)




class C04AcceptancePreparationPlanTests(unittest.TestCase):
    """Metadata fixtures test real staging, never substitute native acceptance."""
    def fixture(self,root):
        import prepare_c04_native_acceptance as acceptance
        project=Path(__file__).resolve().parents[3]
        for source in acceptance.acceptance_sources(project):
            target=root/source.relative_to(project)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        def put(relative,value):
            path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
            if isinstance(value,dict):path.write_text(json.dumps(value))
            else:path.write_bytes(value)
            return path
        ref=lambda path:acceptance._ref(root,path)
        source=put('inputs/real-source/sequence.npz',b'engineering metadata fixture only')
        source_report=put('inputs/real-source/report.json',{'status':'completed','uid':'fixture','seed':42,
                          'sha256':{'sequence.npz':ref(source)['sha256']}})
        common=put('retained/common-target.npz',b'opaque local-namespace common target')
        candidate=put('retained/candidate.json',{'candidate_id':'4d-math-20261006-c04',
            'status':'incomplete','native_qualified':False,'scientific_admission':False,
            'source_refs':{'sequence':ref(source),'report':ref(source_report)},
            'common_target':{'path':'common-target.npz','sha256':ref(common)['sha256']}})
        manifest=put('retained/manifest.json',{'scope':'engineering fixture; no native claim'})
        members=[candidate,manifest,common]
        for role in ('deterministic_protection','strength_matched_repair','robust_conic_protection'):
            members.append(put('retained/'+role+'/report.json',{'status':'error','candidate_arm':role}))
        archive=put('retained/artifact.tar',b'opaque engineering transport fixture')
        put('retained/artifact-archive.json',{'kind':'c04-terminal-artifact-archive','version':1,
            'terminal_candidate_status':'incomplete','archive':{'path':'artifact.tar',
            'sha256':ref(archive)['sha256'],'size_bytes':archive.stat().st_size},
            'members':[{'path':p.relative_to(candidate.parent).as_posix(),'sha256':ref(p)['sha256'],
                        'size_bytes':p.stat().st_size} for p in members]})
        basis=put('freeze/development-basis.json',{'scope':'engineering prospective metadata fixture'})
        decision=put('freeze/decision.json',{'selection_basis_refs':[ref(basis)]})
        raw=put('review/raw-receipt.json',{'scope':'engineering metadata fixture'})
        spec=put('review/method-spec.json',{'implementation_ref':ref(root/'actionmesh/research_math/robust_motion_candidate.py')})
        # Candidate JSON has an internal common_target path. Only its explicit
        # project ref is pinned here: recursive file traversal would be wrong.
        evidence=put('review/evidence.json',{'receipt_ref':ref(raw),'candidate_artifact_ref':ref(candidate),
                    'source_refs':{'sequence':ref(source),'report':ref(source_report)}})
        review=put('review/review.json',{'method_spec_ref':ref(spec),'evidence_refs':[ref(evidence)]})
        freeze=put('freeze/freeze.json',{'candidate_artifact_ref':ref(candidate),
            'source_sequence_ref':ref(source),'source_report_ref':ref(source_report),
            'b_star_decision_ref':ref(decision),'semantic_review_ref':ref(review),'roles':[]})
        gt=put('native/data/fixture/surfaces.npy',b'opaque GT file for engineering staging')
        population=put('native/population.json',{'scope':'engineering metadata'})
        admission=put('native/admission.json',{'scope':'engineering metadata'})
        semantics=put('native/semantics.json',{'scope':'engineering metadata'})
        repo=root/'retained-official-repo'
        for name in acceptance.official_source_names(root):
            put('retained-official-repo/actionbench/'+name,b'# engineering source identity fixture\n')
        return dict(candidate=candidate,freeze=freeze,basis=basis,raw=raw,gt=gt,population=population,
                    admission=admission,semantics=semantics,repo=repo)

    def test_comparison_and_scoring_preparation_all_refs_survive_real_cross_root_stage(self):
        import ast
        import os
        import run_experiments
        import prepare_c04_native_acceptance as acceptance
        skill=Path(os.environ.get('RESEARCH_AUTOPILOT_SKILL_DIR',
                   str(Path(run_experiments.__file__).resolve().parent.parent)))
        with tempfile.TemporaryDirectory() as temp,tempfile.TemporaryDirectory() as other:
            root=Path(temp);f=self.fixture(root)
            native,outer=acceptance.build_plans(root,skill_dir=skill,artifact_candidate=f['candidate'],
                plan_dir=root/'plans',run_id='c04-preparation-stage',wall_seconds=300,ram_mib=4096,
                freeze=f['freeze'],ground_truth=f['gt'],population=f['population'],
                dataset_admission=f['admission'],dataset_semantics=f['semantics'],repo_root=f['repo'],timeout_seconds=60)
            job=native['jobs'][0]
            self.assertEqual(job['output_paths'],['actionmesh/c04-comparison-request.json','actionmesh/c04-scoring-request.json'])
            self.assertEqual(outer['tasks'][0]['resources']['gpu_count'],0)
            self.assertEqual(native['limits']['max_retries_per_trial'],0)
            ast.parse(job['command'][2])
            self.assertNotIn(str(root),job['command'][2])
            self.assertIn('comparison.make_request',job['command'][2])
            self.assertIn('candidate.validate_candidate_artifact',job['command'][2])
            self.assertNotIn('loadTestsFromName',job['command'][2])
            self.assertEqual(job['arm_role'],'request_preparation')
            self.assertIn('scoring.make_scoring_request',job['command'][2])
            self.assertNotIn('scoring.score(',job['command'][2])
            names={r['path'] for r in job['input_refs']}
            self.assertIn('review/raw-receipt.json',names)
            self.assertIn('freeze/development-basis.json',names)
            self.assertNotIn('common-target.npz',names)
            self.assertIn('retained/common-target.npz',names)
            attempt=Path(other)/'attempt';attempt.mkdir()
            command,cwd,workspace=run_experiments._stage(root,job,attempt)
            # program occupies argv[2]; every following physical path is independent.
            for actual,original in zip(command[3:-1],job['command'][3:-1]):
                staged=Path(actual)
                self.assertTrue(staged.is_relative_to(workspace))
                self.assertEqual(staged.read_bytes(),Path(original).read_bytes())
            self.assertEqual(command[-1],'60')
            self.assertEqual(Path(command[-2]).parent.parent,workspace/'retained-official-repo')
            for record in job['input_refs']:
                staged=workspace/record['path']
                self.assertEqual(hashlib.sha256(staged.read_bytes()).hexdigest(),record['sha256'])
            original_basis=(workspace/'freeze/development-basis.json').read_bytes()
            f['basis'].write_text('{}')
            self.assertEqual((workspace/'freeze/development-basis.json').read_bytes(),original_basis)
            self.assertEqual(cwd,workspace/'actionmesh')

    def test_comparison_only_declares_one_output_and_partial_scoring_arguments_fail(self):
        import run_experiments
        import prepare_c04_native_acceptance as acceptance
        skill=Path(run_experiments.__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);f=self.fixture(root)
            options=dict(root=root,skill_dir=skill,artifact_candidate=f['candidate'],
                         plan_dir=root/'plans',run_id='c04-comparison-stage',wall_seconds=300,ram_mib=4096)
            native,_=acceptance.build_plans(**options,freeze=f['freeze'])
            self.assertEqual(native['jobs'][0]['output_paths'],['actionmesh/c04-comparison-request.json'])
            self.assertEqual(native['jobs'][0]['command'][-1],str(f['freeze']))
            options['plan_dir']=root/'rejected-plans'
            with self.assertRaises(ValueError):acceptance.build_plans(**options,ground_truth=f['gt'])
            self.assertFalse(options['plan_dir'].exists())


if __name__=='__main__':unittest.main()
