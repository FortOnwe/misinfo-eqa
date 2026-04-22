import json
import shutil
import unittest
from pathlib import Path

from misinfo_eqa.audit import generate_audit_sheet


class AuditTests(unittest.TestCase):
    def test_generate_audit_sheet(self):
        run = Path.cwd() / "test-workspace" / "audit"
        if run.exists():
            shutil.rmtree(run)
        run.mkdir(parents=True)
        try:
            rows = [
                {"dataset": "a", "id": "1", "label3": "true", "reasons": ["x"], "claim": "c1", "evidence_text": "e1"},
                {"dataset": "a", "id": "2", "label3": "false", "reasons": ["y"], "claim": "c2", "evidence_text": "e2"},
                {"dataset": "b", "id": "3", "label3": "unknown", "reasons": ["z"], "claim": "c3", "evidence_text": "e3"},
            ]
            with (run / "flagged_examples.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            output = generate_audit_sheet(run, per_dataset=1)
            text = output.read_text(encoding="utf-8")
            self.assertIn("audit_verdict", text)
            self.assertIn("relevance_score", text)
            self.assertIn("best_evidence_span", text)
            self.assertIn("a,1,true,x,c1,e1", text)
            self.assertIn("b,3,unknown,z,c3,e3", text)
            self.assertNotIn("a,2,false,y,c2,e2", text)
        finally:
            shutil.rmtree(run.parent, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
