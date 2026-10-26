import json
import shutil
import unittest
from pathlib import Path

from misinfo_eqa.evidence import build_evidence_resolver


class EvidenceResolverTests(unittest.TestCase):
    def test_fever_structured_references_without_resolver_are_unavailable(self):
        resolver = build_evidence_resolver({"name": "fever"})
        resolution = resolver.resolve(
            "fever",
            {
                "claim": "Claim",
                "evidence": [['List_of_cities', 1]],
            },
        )
        self.assertIsNotNone(resolution)
        assert resolution is not None
        self.assertEqual(resolution.status, "unavailable")
        self.assertEqual(resolution.reason, "resolver_not_configured")

    def test_fever_structured_references_are_resolved_from_local_index(self):
        root = Path.cwd() / "test-workspace" / "evidence-resolver"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        try:
            index_path = root / "fever_index.json"
            index_path.write_text(
                json.dumps({"List_of_cities": {"1": "A city has over one million residents."}}),
                encoding="utf-8",
            )
            resolver = build_evidence_resolver(
                {
                    "name": "fever",
                    "evidence_resolver": {"type": "fever_local", "path": str(index_path)},
                }
            )
            resolution = resolver.resolve(
                "fever",
                {
                    "claim": "Claim",
                    "evidence": [["List_of_cities", 1]],
                },
            )
            self.assertIsNotNone(resolution)
            assert resolution is not None
            self.assertEqual(resolution.status, "resolved")
            self.assertIn("one million residents", resolution.evidence_text)
        finally:
            shutil.rmtree(root.parent, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
