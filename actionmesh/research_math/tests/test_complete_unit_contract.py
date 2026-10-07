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
