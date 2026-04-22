import json
import shutil
import unittest
from pathlib import Path

from misinfo_eqa.cli import _latest_run_dir, _resolve_run_arg


class CliTests(unittest.TestCase):
    def test_resolve_run_arg_requires_choice(self):
        with self.assertRaises(ValueError):
            _resolve_run_arg(None, latest=False, runs_dir="runs")

    def test_resolve_run_arg_explicit(self):
        self.assertEqual(
            _resolve_run_arg("runs/demo", latest=False, runs_dir="runs"),
            Path("runs/demo"),
        )

    def test_latest_run_dir(self):
        root = Path.cwd() / "test-workspace" / "cli-runs"
        if root.exists():
            shutil.rmtree(root)
        try:
            older = root / "20240101-000000"
            newer = root / "20240102-000000"
            older.mkdir(parents=True)
            newer.mkdir()
            (older / "marker.json").write_text(json.dumps({"run": "older"}), encoding="utf-8")
            (newer / "marker.json").write_text(json.dumps({"run": "newer"}), encoding="utf-8")
            self.assertEqual(_latest_run_dir(root), newer)
        finally:
            shutil.rmtree(root.parent, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
