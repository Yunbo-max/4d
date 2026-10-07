import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from research_math.complete_unit_runner import (
    collect_native_sequence,
    require_full128_budget,
    run_stage,
    validate_full128_mode,
    verify_prerequisites,
)

class CompleteUnitRunnerTests(unittest.TestCase):
    def test_full128_mode_requires_exact_argument_bundle_and_canonical_price(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();pricing=root/'inputs/actionbench-full128-queue/pricing.json'
            valid=SimpleNamespace(uid='uid',window_id='full128-window-01',pricing=pricing,root=root)
            self.assertTrue(validate_full128_mode(valid))
            for key in ('uid','window_id','pricing','root'):
                changed=SimpleNamespace(**vars(valid));setattr(changed,key,None)
                with self.subTest(key=key),self.assertRaises(ValueError):
                    validate_full128_mode(changed)
            changed=SimpleNamespace(**vars(valid));changed.pricing=root/'other.json'
            with self.assertRaisesRegex(ValueError,'Canonical'):
                validate_full128_mode(changed)

    def test_full128_wall_limit_must_equal_admitted_unit_price(self):
        manifest={'selected_unit':{'unit_timeout_seconds':1664}}
        require_full128_budget(SimpleNamespace(wall_seconds=1664),manifest)
        with self.assertRaisesRegex(ValueError,'queue price'):
            require_full128_budget(SimpleNamespace(wall_seconds=1663),manifest)

    def test_full128_prerequisite_check_returns_selected_not_template_manifest(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();output=root/'output';output.mkdir()
            paths={name:root/f'{name}.json' for name in
                   ('contract','population','snapshot_contract','snapshot_admission',
                    'dataset_semantics','unit_manifest')}
            snapshot={'snapshots':{key:{'key':key} for key in
                                   ('actionmesh','triposg','dinov2','rmbg')}}
            template={'prerequisite_receipts':{},'generation':{
                'runtime_profile':'fp16-lowram-v1','seed':42,'fast':False,
                'low_ram':True,'dtype':'float16',
                'config':'actionmesh/configs/actionmesh_lowram.yaml',
                'repair_parent_receipt_sha256':
                '7b4223a3fdcf4a738c1cec646ecb8f8172ece850a46aa8936aa96d8626cd62f9'}}
            values={'contract':{},'population':{},'snapshot_contract':{'source':{}},
                    'snapshot_admission':snapshot,'dataset_semantics':{},
                    'unit_manifest':template}
            for name,path in paths.items():path.write_text(json.dumps(values[name]))
            import hashlib
            template['prerequisite_receipts']={
                'snapshot_admission_sha256':hashlib.sha256(paths['snapshot_admission'].read_bytes()).hexdigest(),
                'dataset_semantics_sha256':hashlib.sha256(paths['dataset_semantics'].read_bytes()).hexdigest()}
            paths['unit_manifest'].write_text(json.dumps(template))
            pricing=root/'inputs/actionbench-full128-queue/pricing.json'
            pricing.parent.mkdir(parents=True)
            pricing.write_text(json.dumps({'unit_timeout_seconds':1664}))
            source=root/'source';weights=root/'weights';source.mkdir();weights.mkdir()
            (source/'pretrained_weights').mkdir()
            for key,name in {'actionmesh':'ActionMesh','triposg':'TripoSG',
                             'dinov2':'dinov2','rmbg':'RMBG'}.items():
                target=weights/name;target.mkdir();(source/'pretrained_weights'/name).symlink_to(target)
            selected={'status':'frozen_engineering_full128_unit',
                      'selected_unit':{'uid':'uid-b','unit_timeout_seconds':1664},
                      'generation':template['generation']}
            args=SimpleNamespace(**paths,root=root,pricing=pricing,uid='uid-b',
                window_id='full128-window-01',source_root=source,dataset_root=root,
                weights_root=weights,gpu_uuid='GPU-test',wall_seconds=1664)
            with patch('research_math.actionbench_full128_unit.freeze_unit',return_value=selected), \
                    patch('research_math.snapshot_admission.verify_source'), \
                    patch('research_math.snapshot_admission.snapshot_manifest',
                          side_effect=lambda path,expected:expected), \
                    patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'GPU-test'}):
                self.assertIs(verify_prerequisites(args,output),selected)
            args.wall_seconds=1663
            with patch('research_math.actionbench_full128_unit.freeze_unit') as freezer, \
                    self.assertRaisesRegex(ValueError,'queue price'):
                verify_prerequisites(args,root/'rejected-output')
            freezer.assert_not_called()
            self.assertFalse((root/'rejected-output/revalidated-unit-manifest.json').exists())

    def test_failed_subprocess_retains_exit_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(RuntimeError):
                run_stage('generation',[sys.executable,'-c','print("before failure");raise SystemExit(7)'],root,root,5)
            record=json.loads((root/'generation.execution.json').read_text())
            self.assertEqual(record['exit_code'],7)
            self.assertEqual(record['status'],'failed')
            self.assertIn('before failure',(root/'generation.stdout.log').read_text())

    def test_timeout_retains_typed_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(RuntimeError):
                run_stage('generation',[sys.executable,'-c','import time;time.sleep(2)'],root,root,.1)
            record=json.loads((root/'generation.execution.json').read_text())
            self.assertEqual(record['status'],'timeout')
            self.assertIsNone(record['exit_code'])

    def test_timeout_terminates_descendant_process(self):
        import psutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pid_path=root/'child.pid'
            program=('import subprocess,sys,time,pathlib;'
                     'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(10)"]);'
                     'pathlib.Path(sys.argv[1]).write_text(str(p.pid));time.sleep(10)')
            try:
                with self.assertRaises(RuntimeError):
                    run_stage('generation',[sys.executable,'-c',program,str(pid_path)],root,root,1)
                self.assertTrue(pid_path.exists())
                pid=int(pid_path.read_text())
                live=psutil.pid_exists(pid) and psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE
                self.assertFalse(live,'Timed-out scoring descendant remains running')
            finally:
                if pid_path.exists():
                    try:psutil.Process(int(pid_path.read_text())).kill()
                    except psutil.NoSuchProcess:pass

    def test_official_export_requires_all_glbs_and_exact_agreement(self):
        import numpy as np
        import trimesh
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'native';output.mkdir()
            mesh=trimesh.creation.icosphere(subdivisions=0)
            vertices=np.repeat(np.asarray(mesh.vertices,dtype=np.float32)[None],16,axis=0)
            faces=np.asarray(mesh.faces,dtype=np.int32)
            transformed=vertices[:,:,[2,0,1]].copy();transformed[:,:,0]*=-1
            np.save(output/'deformations_vertices.npy',transformed)
            np.save(output/'deformations_faces.npy',faces)
            for i in range(16):trimesh.Trimesh(vertices=vertices[i],faces=faces,process=False).export(output/f'mesh_{i:02d}.glb')
            collect_native_sequence(output,'uid')
            with np.load(output/'sequence.npz') as saved:np.testing.assert_array_equal(saved['vertices'],vertices)
            (output/'mesh_15.glb').unlink()
            with self.assertRaises((ValueError,FileNotFoundError)):collect_native_sequence(output,'uid')
