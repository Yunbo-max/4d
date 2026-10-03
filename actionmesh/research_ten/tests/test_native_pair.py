import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from research_ten.case_io import digest, load_census_case
from research_ten.native_pair import scheduler_parameters, ARMS


class NativePlanTests(unittest.TestCase):
    def config(self):
        return dict(stage_1_steps=30, model={"scheduler": dict(
            _target_="actionmesh.scheduler.scheduler.SchedulerFlow", _partial_=True,
            num_inference_steps=30, num_train_timesteps=1000, shift=3.,
            is_additive=True, split_cfg_batch=True)})

    def test_native_update_sign_and_schedule_are_explicit(self):
        got = scheduler_parameters(self.config())
        self.assertIs(got["is_additive"], True)
        self.assertIs(got["split_cfg_batch"], True)
        self.assertEqual(got["num_inference_steps"], 30)
        self.assertEqual(got["shift"], 3.)
        for key in ("is_additive", "num_train_timesteps"):
            conf = self.config()
            del conf["model"]["scheduler"][key]
            with self.assertRaises(ValueError):
                scheduler_parameters(conf)
        conf = self.config()
        conf["model"]["scheduler"]["is_additive"] = False
        with self.assertRaises(ValueError):
            scheduler_parameters(conf)

    def test_all_interventions_have_matching_three_branch_controls(self):
        self.assertEqual(ARMS, ("scalar", "projection", "norm_matched", "random"))
        conf = self.config()
        conf["stage_1_steps"] = 2
        with self.assertRaises(ValueError):
            scheduler_parameters(conf)


class CensusCacheTests(unittest.TestCase):
    def fixture(self, root):
        rest = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], np.float32)
        faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int64)
        latents = np.zeros((16, 2048, 64), np.float32)
        times = np.arange(16, dtype=np.float32)
        tracks = np.broadcast_to(rest, (16, 4, 3)).copy()
        query = np.column_stack((rest, np.tile([0., 0., 1.], (4, 1)))).astype(np.float32)
        np.savez_compressed(root/"prepared.npz", timesteps=times, anchor_latent=latents[:1],
            anchor_timesteps=times[:1], anchor_vertices=rest, anchor_faces=faces,
            anchor_query_features=query, query_vertex_ids=np.arange(4), seed=np.asarray(43),
            context=np.zeros((16, 2, 3), np.float32))
        np.savez_compressed(root/"denoised.npz", latents=latents, timesteps=times, seed=np.asarray(43))
        np.savez_compressed(root/"sequence.npz", vertices=tracks, faces=faces, timesteps=times,
            frame_indices=np.arange(16), query_vertex_ids=np.arange(4))
        report = dict(status="completed", frames=16, uid="another-census-asset", seed=43,
            sha256={name: digest(root/name) for name in ("prepared.npz", "denoised.npz", "sequence.npz")})
        (root/"report.json").write_text(json.dumps(report))

    def test_other_cohort_assets_and_repeat_seed_are_not_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            report, hashes, latents, tracks, _, _ = load_census_case(root)
            self.assertEqual(report["seed"], 43)
            self.assertEqual(report["uid"], "another-census-asset")
            self.assertEqual(latents.shape, (16, 2048, 64))
            self.assertEqual(tracks.shape, (16, 4, 3))
            self.assertEqual(hashes["sequence.npz"], digest(root/"sequence.npz"))

    def test_changed_cache_hash_fails_even_if_geometry_is_finite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            with (root/"sequence.npz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(ValueError):
                load_census_case(root)


if __name__ == "__main__":
    unittest.main()
