import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class InstallerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("install_training", Path(__file__).resolve().parents[1] / "scripts/install_training_optimizations.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def fixture(self, root):
        bundle, project = root / "bundle", root / "project"
        (project / "src/agenticwam").mkdir(parents=True)
        (project / "configs").mkdir()
        (project / "train.sh").write_text("existing two-GPU launch")
        (project / "configs/custom.yaml").write_text("coding_v4 and private local paths")
        manifest = {}
        for name in self.module.SOURCE_FILES:
            source = bundle / name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("new source: " + name)
            manifest[name] = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        old = project / "src/agenticwam/trainer.py"
        old.write_text("old trainer")
        (bundle / "MANIFEST.json").write_text(json.dumps(manifest))
        return bundle, project

    def test_verified_install_backs_up_and_preserves_local_launch_and_configuration(self):
        with TemporaryDirectory() as tmp:
            bundle, project = self.fixture(Path(tmp))
            backup = self.module.install(bundle, project)
            self.assertEqual((backup / "src/agenticwam/trainer.py").read_text(), "old trainer")
            for name in self.module.SOURCE_FILES:
                self.assertEqual((project / name).read_bytes(), (bundle / name).read_bytes())
            self.assertEqual((project / "train.sh").read_text(), "existing two-GPU launch")
            self.assertEqual((project / "configs/custom.yaml").read_text(), "coding_v4 and private local paths")

    def test_bad_checksum_leaves_sources_untouched(self):
        with TemporaryDirectory() as tmp:
            bundle, project = self.fixture(Path(tmp))
            (bundle / self.module.SOURCE_FILES[-1]).write_text("tampered")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                self.module.install(bundle, project)
            self.assertEqual((project / "src/agenticwam/trainer.py").read_text(), "old trainer")
            self.assertFalse((project / ".training-throughput-backups").exists())


if __name__ == "__main__":
    unittest.main()
