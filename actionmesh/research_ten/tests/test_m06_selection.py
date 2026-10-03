"""Independent fixtures for candidate gating, feature extraction and selection."""
import unittest
import numpy as np
from research_ten import m06_selection as selection


class SelectionTests(unittest.TestCase):
    def candidates(self):
        # The static high-identity candidate must fail the common action gate.
        return selection.CandidateFeatures(
            ids=("static", "distorted", "usable"),
            action=np.array([0.1, 0.9, 0.8]),
            text=np.array([0.2, 0.95, 0.7]),
            appearance=np.array([1., 0.2, 0.8]),
            proportion=np.array([1., 0.2, 0.9]),
            trackability=np.array([1., 0.8, 0.9]),
        )

    def test_action_gate_blocks_static_and_feasibility_differs_from_text(self):
        f = self.candidates()
        got = selection.select_video(f, action_threshold=0.5)
        self.assertEqual(got["selected_id"], "usable")
        self.assertEqual(got["eligible_indices"], [1, 2])
        self.assertEqual(selection.select_video(f, action_threshold=0.5,
                         policy="text")["selected_id"], "distorted")

    def test_all_ineligible_abstains_instead_of_selecting_static(self):
        got = selection.select_video(self.candidates(), action_threshold=0.99)
        self.assertEqual(got["status"], "no_eligible_candidate")
        self.assertIsNone(got["selected_index"])

    def test_random_control_respects_same_action_gate(self):
        f = self.candidates()
        for seed in range(10):
            self.assertIn(selection.select_video(f, action_threshold=.5,
                          policy="random", seed=seed)["selected_index"], [1, 2])

    def test_validity_gate_applies_to_all_policies(self):
        f = self.candidates()
        for policy in ("feasibility", "text", "appearance", "random"):
            got = selection.select_video(f, action_threshold=.5,
                                         policy=policy, valid=[True, False, True])
            self.assertEqual(got["selected_id"], "usable")

    def test_missing_and_invalid_scores_rejected(self):
        with self.assertRaises(ValueError):
            selection.CandidateFeatures(("a",), [.5], [.5], [.5], [np.nan], [.5])
        with self.assertRaises(ValueError):
            selection.select_video(self.candidates(), weights=[1, -1, 1])

    def test_large_finite_weights_keep_ranking_and_normalization(self):
        f = selection.CandidateFeatures(("poor", "good"), [1.,1.], [1.,1.],
                    [.1,.9], [.1,.9], [.1,.9])
        with np.errstate(over="raise", invalid="raise"):
            got = selection.select_video(f, weights=[1e308, 1e308, 1.])
        self.assertEqual(got["selected_id"], "good")
        self.assertAlmostEqual(sum(got["weights"]), 1.)

    def test_translation_and_uniform_camera_scale_do_not_change_proportion(self):
        reference = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        tracks = np.stack([reference, 3*reference + [5., 8.], reference+[2., 3.]])[None]
        got = selection.track_features(tracks, np.ones((1, 3, 4), bool), reference)
        np.testing.assert_allclose(got["proportion"], [1.], atol=1e-12)
        np.testing.assert_allclose(got["trackability"], [1.])
        self.assertEqual(got["valid"].tolist(), [True])

    def test_nonuniform_distortion_lowers_shape_score(self):
        reference = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        tracks = np.stack([reference, reference*[5., 1.]])[None]
        got = selection.track_features(tracks, np.ones((1, 2, 4), bool), reference)
        self.assertLess(float(got["proportion"][0]), .95)

    def test_no_tracks_are_invalid_not_perfectly_stable(self):
        tracks = np.full((1, 3, 4, 2), np.nan)
        got = selection.track_features(tracks, np.zeros((1, 3, 4), bool),
                    np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]))
        self.assertEqual(got["valid"].tolist(), [False])
        np.testing.assert_equal(got["trackability"], [0.])

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            selection.CandidateFeatures(("a", "a"), [1, 1], [1, 1],
                                        [1, 1], [1, 1], [1, 1])

    def test_pipeline_scores_every_candidate_but_reconstructs_selected_one_only(self):
        generated, reconstructed = [], []
        def generate(seed):
            generated.append(seed)
            return {"seed": seed}
        def score(video):
            return dict(action=.9, text=.8, appearance=.9,
                        proportion=.5 if video["seed"] == 1 else .9, trackability=.9)
        def reconstruct(video):
            reconstructed.append(video["seed"])
            return "mesh-" + str(video["seed"])
        got = selection.run_pipeline(generate, score, reconstruct, seeds=[1, 2])
        self.assertEqual(generated, [1, 2])
        self.assertEqual(reconstructed, [2])
        self.assertEqual(got["output"], "mesh-2")
        self.assertEqual(got["counts"], {"video_generations": 2, "video_scores": 2, "backend_calls": 1})

    def test_pipeline_never_calls_backend_when_action_gate_fails(self):
        def forbidden(_):
            self.fail("ineligible candidate was sent to backend")
        got = selection.run_pipeline(lambda seed: seed,
                lambda video: dict(action=0., text=1., appearance=1., proportion=1., trackability=1.),
                forbidden, seeds=[1, 2])
        self.assertIsNone(got["output"])
        self.assertEqual(got["counts"]["backend_calls"], 0)

    def test_invalid_selection_seed_fails_before_expensive_generation(self):
        generated = []
        def generate(seed):
            generated.append(seed)
            return seed
        with self.assertRaises(ValueError):
            selection.run_pipeline(generate,
                lambda video: dict(action=1., text=1., appearance=1., proportion=1., trackability=1.),
                lambda video: video, seeds=[1, 2], policy="random", selection_seed=-1)
        self.assertEqual(generated, [])

if __name__ == "__main__":
    unittest.main()
