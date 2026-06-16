"""
Regression test: _resolve_manifest must find manifests via config data dirs,
not only via cwd-relative paths.

Bug history: test_wer used args.manifest raw. In Colab, cwd is
/content/asr_dissertation and manifests live under Drive, so any relative
path like "data/real/test_manifest.json" raised FileNotFoundError.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))
from main import _resolve_manifest


def _cfg(real_dir, sim_dir, comb_dir):
    return SimpleNamespace(data=SimpleNamespace(
        real_data_dir=str(real_dir),
        simulated_data_dir=str(sim_dir),
        combined_data_dir=str(comb_dir),
    ))


class TestManifestResolution(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.real_dir = root / "data" / "real"
        self.sim_dir  = root / "data" / "simulated"
        self.comb_dir = root / "data" / "combined"
        for d in (self.real_dir, self.sim_dir, self.comb_dir):
            d.mkdir(parents=True)
        payload = [{"id": 1, "audio_file": "audio/t.wav", "transcription": "hello"}]
        (self.real_dir / "test_manifest.json").write_text(json.dumps(payload))
        (self.sim_dir  / "test_manifest.json").write_text(json.dumps(payload))
        self.cfg = _cfg(self.real_dir, self.sim_dir, self.comb_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_absolute_path_resolves_directly(self):
        manifest = str(self.real_dir / "test_manifest.json")
        resolved = _resolve_manifest(manifest, self.cfg)
        self.assertEqual(resolved, Path(manifest))

    def test_cwd_relative_path_falls_back_to_config_real(self):
        # Regression: "data/real/test_manifest.json" must resolve via real_data_dir.
        resolved = _resolve_manifest("data/real/test_manifest.json", self.cfg)
        self.assertEqual(resolved, self.real_dir / "test_manifest.json")

    def test_cwd_relative_path_falls_back_to_config_simulated(self):
        resolved = _resolve_manifest("data/simulated/test_manifest.json", self.cfg)
        self.assertEqual(resolved, self.sim_dir / "test_manifest.json")

    def test_missing_manifest_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            _resolve_manifest("data/real/nonexistent.json", self.cfg)

    def test_error_message_shows_requested_and_tried_paths(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            _resolve_manifest("data/real/nonexistent.json", self.cfg)
        msg = str(ctx.exception)
        self.assertIn("nonexistent.json", msg)
        self.assertIn("Requested", msg)
        self.assertIn("Tried", msg)


if __name__ == "__main__":
    unittest.main()
