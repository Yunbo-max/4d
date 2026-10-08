"""Local-only C03 tracked-query producer boundary checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research_math import c03_label_bank as labels


class C03LabelBankTests(unittest.TestCase):
    def test_selection_is_uid_stable_unique_and_sorted(self):
        first = labels._selection("uid-a", 17, 100, 42)
        second = labels._selection("uid-a", 17, 100, 42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), 17)
        self.assertTrue(np.all(first[:-1] < first[1:]))
        self.assertEqual(hashlib.sha256(first.tobytes()).hexdigest(),
                         hashlib.sha256(second.tobytes()).hexdigest())

    def test_selection_rejects_invalid_population_request(self):
        for count in (0, 101):
            with self.assertRaises(ValueError):
                labels._selection("uid-a", count, 100, 42)

    def test_weight_manifest_binds_both_autoencoder_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); model = root / "ActionMesh/autoencoder"
            model.mkdir(parents=True)
            rows = []
            for name, payload in (("config.json", b"config"),
                                  ("model.safetensors", b"weights")):
                path = model / name; path.write_bytes(payload)
                rows.append({"repo": "facebook/ActionMesh",
                    "revision": labels.ACTIONMESH_REVISION,
                    "file": "autoencoder/" + name,
                    "path": "weights/ActionMesh/autoencoder/" + name,
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest()})
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(rows))
            verified = labels._verify_weights(root, manifest)
            self.assertEqual(set(verified), {row["path"] for row in rows})
            (model / "config.json").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "stale"):
                labels._verify_weights(root, manifest)


if __name__ == "__main__":
    unittest.main()
