import unittest
from pathlib import Path
from research_math.complete_unit_contract import (
    complete_unit_output_paths,
    generation_argv,
    require_three_scores,
)

class CompleteUnitContractTests(unittest.TestCase):
    def test_default_generation_does_not_silently_enable_fallback(self):
        command=generation_argv(Path('/source'),Path('/data/imgs'),Path('/out'))
        self.assertNotIn('--fast',command)
        self.assertNotIn('--low_ram',command)
        for flag,value in {'--seed':'42','--dtype':'bfloat16','--stage_0_steps':'100',
                           '--stage_1_steps':'30','--face_decimation':'40000',
                           '--guidance_scales':'7.5','--anchor_idx':'0'}.items():
            self.assertEqual(command[command.index(flag)+1],value)

    def test_all_three_distinct_successful_full_frame_rows_required(self):
        rows=[{'case_id':'uid-'+arm,'uid':'uid','status':'success','n_frames':16,
               'cd_3d':.1,'cd_4d':.2,'cd_motion':.3}
              for arm in ('native','world_gaussian','body_gaussian')]
        self.assertEqual(len(require_three_scores({'cases':rows},'uid')),3)
        for broken in (rows[:2],[rows[0]]*3,rows+[rows[0]]):
            with self.assertRaises(ValueError):require_three_scores({'cases':broken},'uid')
        for field,value in [('status','error'),('n_frames',15),('uid','other'),('cd_3d',float('nan'))]:
            bad=[dict(row) for row in rows];bad[0][field]=value
            with self.assertRaises(ValueError):require_three_scores({'cases':bad},'uid')

    def test_complete_success_inventory_binds_every_arm_and_raw_official_output(self):
        uid='000-000_sample'
        paths=complete_unit_output_paths(uid)
        root='actionmesh/unit-output/'
        expected={root+name for name in (
            'result.json','device-samples.jsonl','host-samples.jsonl',
            'revalidated-unit-manifest.json','generation.stdout.log',
            'generation.stderr.log','generation.execution.json',
            'official-scoring.stdout.log','official-scoring.stderr.log',
            'official-scoring.execution.json','official-scores.json',
            'final-integrity/revalidated-unit-manifest.json',
            'native-generation/deformations_vertices.npy',
            'native-generation/deformations_faces.npy',
            'native-generation/sequence.npz','native-generation/report.json',
            'native-generation/grid_normal.mp4','controls/manifest.json',
            'controls/controls.json','controls/native/sequence.npz',
            'controls/native/report.json','controls/world_gaussian/sequence.npz',
            'controls/world_gaussian/report.json',
            'controls/body_gaussian/sequence.npz',
            'controls/body_gaussian/report.json','controls/body_gaussian/poses.npz')}
        expected.update(root+f'native-generation/mesh_{i:02d}.glb' for i in range(16))
        source=root+'official-scores.json.official/official-source/'
        expected.update(source+name for name in (
            'benchmark.py','chamfer.py','icp.py','sample_mesh.py',
            'sample_point_cloud.py','evaluate_dataset.py'))
        for arm in ('native','world_gaussian','body_gaussian'):
            case=uid+'-'+arm
            base=root+'official-scores.json.official/'+case+'/'
            expected.update(base+name for name in (
                'execution.json','export-manifest.json','official.csv',
                'official.summary.json','official.backend.json','stdout.log','stderr.log'))
            expected.update(base+'predictions/'+uid+f'/mesh_{i:05d}.glb'
                            for i in range(16))
        self.assertEqual(len(expected),117)
        self.assertEqual(set(paths),expected)
