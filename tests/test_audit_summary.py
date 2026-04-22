import csv
import shutil
import unittest
from pathlib import Path

from misinfo_eqa.audit import summarize_audit_sheet, write_audit_summary


class AuditSummaryTests(unittest.TestCase):
    def test_summarize_audit_sheet(self):
        root = Path.cwd() / "test-workspace" / "audit-summary"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        try:
            audit = root / "audit_sheet.csv"
            with audit.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "dataset",
                        "id",
                        "label3",
                        "reasons",
                        "claim",
                        "evidence_text",
                        "relevance_score",
                        "best_evidence_span",
                        "audit_verdict",
                        "notes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "dataset": "a",
                        "id": "1",
                        "label3": "true",
                        "reasons": "low_claim_evidence_relevance",
                        "claim": "c",
                        "evidence_text": "e",
                        "audit_verdict": "evidence_mismatch",
                        "notes": "bad evidence",
                    }
                )
                writer.writerow(
                    {
                        "dataset": "a",
                        "id": "2",
                        "label3": "false",
                        "reasons": "heuristic_label_disagrees:true",
                        "claim": "c",
                        "evidence_text": "e",
                        "audit_verdict": "heuristic_false_positive",
                        "notes": "ok evidence",
                    }
                )

            summary = summarize_audit_sheet(audit)
            self.assertEqual(summary["overall"]["n"], 2)
            self.assertEqual(summary["overall"]["n_issues"], 1)
            self.assertAlmostEqual(summary["overall"]["issue_precision"], 0.5)
            output = write_audit_summary(audit)
            self.assertTrue(output.exists())
            self.assertTrue(output.with_suffix(".json").exists())
        finally:
            shutil.rmtree(root.parent, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
