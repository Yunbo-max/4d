import json
from pathlib import Path
import sys
import tempfile
import unittest
from research_math.complete_unit_runner import run_stage, collect_native_sequence

class CompleteUnitRunnerTests(unittest.TestCase):
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
