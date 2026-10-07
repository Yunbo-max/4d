import unittest
from pathlib import Path
from research_math.complete_unit_contract import generation_argv, require_three_scores

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

    def test_explicit_repair_uses_full_quality_fp16_low_ram(self):
        command=generation_argv(Path('/source'),Path('/data/imgs'),Path('/out'),profile='fp16-lowram-v1')
        self.assertIn('--low_ram',command)
        self.assertNotIn('--fast',command)
        for flag,value in {'--seed':'42','--dtype':'float16','--stage_0_steps':'100',
                           '--stage_1_steps':'30','--face_decimation':'40000'}.items():
            self.assertEqual(command[command.index(flag)+1],value)

    def test_profile_validation_rejects_silent_or_mixed_repairs(self):
        from research_math.complete_unit_contract import validate_generation_profile
        base={'seed':42,'dtype':'bfloat16','fast':False,'low_ram':False,
              'config':'actionmesh/configs/actionmesh.yaml'}
        self.assertEqual(validate_generation_profile(base),'default')
        repair={**base,'runtime_profile':'fp16-lowram-v1','dtype':'float16','low_ram':True,
                'config':'actionmesh/configs/actionmesh_lowram.yaml',
                'repair_parent_receipt_sha256':'7b4223a3fdcf4a738c1cec646ecb8f8172ece850a46aa8936aa96d8626cd62f9'}
        self.assertEqual(validate_generation_profile(repair),'fp16-lowram-v1')
        for key,value in [('runtime_profile','unreviewed'),('repair_parent_receipt_sha256','0'*64),
                          ('seed',44),('fast',True),('dtype','bfloat16'),('low_ram',False),
                          ('config','actionmesh/configs/actionmesh.yaml')]:
            with self.assertRaises(ValueError):validate_generation_profile({**repair,key:value})
        with self.assertRaises(ValueError):validate_generation_profile({**base,'dtype':'float16'})
        with self.assertRaises(ValueError):generation_argv(Path('/s'),Path('/d'),Path('/o'),profile='unknown')
