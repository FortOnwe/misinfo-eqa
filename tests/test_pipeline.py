import csv
import json
import shutil
import unittest
from pathlib import Path

from misinfo_eqa.config import load_config
from misinfo_eqa.pipeline import run_pipeline, scan_config


class PipelineTests(unittest.TestCase):
    def test_end_to_end_local_run(self):
        root = Path.cwd() / "test-workspace" / "pipeline"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        try:
            data_path = root / "demo.csv"
            rows = _fixture_rows()
            with data_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["id", "split", "claim", "evidence", "veracity", "date", "source_url"],
                )
                writer.writeheader()
                writer.writerows(rows)

            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "run_dir: test-workspace/pipeline/runs",
                        "max_examples_per_dataset: 50",
                        "seed: 7",
                        "bootstrap_samples: 20",
                        "datasets:",
                        "  - name: demo",
                        "    source: local",
                        "    path: demo.csv",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)
            scan = scan_config(config)
            self.assertEqual(scan["datasets"]["demo"]["n"], len(rows))

            run_dir = run_pipeline(config)
            self.assertTrue((run_dir / "report.md").exists())
            self.assertTrue((run_dir / "metrics.json").exists())
            self.assertTrue((run_dir / "stressors.json").exists())
            self.assertTrue((run_dir / "risk_flags.json").exists())
            self.assertTrue((run_dir / "plots" / "demo_macro_f1.svg").exists())
            stressors = json.loads((run_dir / "stressors.json").read_text(encoding="utf-8"))
            summary = json.loads((run_dir / "data_summary.json").read_text(encoding="utf-8"))
            evidence = stressors["datasets"]["demo"]["evidence_ablation"]
            duplicates = stressors["datasets"]["demo"]["duplicate_claims"]
            self.assertIn("combined_minus_claim_ci", evidence)
            self.assertIn("n_supported", evidence)
            self.assertIn("n_unsupported", evidence)
            self.assertIn("evidence_status", summary["datasets"]["demo"])
            self.assertGreaterEqual(duplicates["exact_duplicate_rate"], 0.0)
        finally:
            shutil.rmtree(root.parent, ignore_errors=True)

    def test_evidence_stressors_skip_when_only_structured_or_missing(self):
        root = Path.cwd() / "test-workspace" / "pipeline-structured"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        try:
            data_path = root / "structured.csv"
            rows = []
            for idx in range(12):
                rows.append(
                    {
                        "id": str(idx),
                        "split": "train" if idx < 8 else "test",
                        "claim": f"Claim {idx % 3}",
                        "evidence": "c(30098, 301119, NA, NA)",
                        "veracity": "true" if idx % 2 else "false",
                        "date": f"2020-01-{(idx % 28) + 1:02d}",
                        "source_url": "https://example.test",
                    }
                )
            with data_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["id", "split", "claim", "evidence", "veracity", "date", "source_url"],
                )
                writer.writeheader()
                writer.writerows(rows)
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "run_dir: test-workspace/pipeline-structured/runs",
                        "max_examples_per_dataset: 50",
                        "seed: 7",
                        "bootstrap_samples: 20",
                        "datasets:",
                        "  - name: fever",
                        "    source: local",
                        "    path: structured.csv",
                    ]
                ),
                encoding="utf-8",
            )
            run_dir = run_pipeline(load_config(config_path))
            stressors = json.loads((run_dir / "stressors.json").read_text(encoding="utf-8"))
            mismatch = stressors["datasets"]["fever"]["label_rationale_mismatch"]
            evidence = stressors["datasets"]["fever"]["evidence_ablation"]
            duplicates = stressors["datasets"]["fever"]["duplicate_claims"]
            self.assertFalse(mismatch["available"])
            self.assertEqual(mismatch["skip_reason"], "insufficient_resolved_evidence")
            self.assertFalse(evidence["available"])
            self.assertGreater(mismatch["n_unsupported"], 0)
            self.assertGreater(duplicates["exact_duplicate_rate"], 0.0)
        finally:
            shutil.rmtree(root.parent, ignore_errors=True)


def _fixture_rows():
    rows = []
    for idx in range(18):
        label = "true" if idx % 3 == 0 else "false" if idx % 3 == 1 else "unknown"
        split = "train" if idx < 12 else "test"
        if label == "true":
            claim = f"Official health agency confirmed accurate report number {idx}"
            evidence = "Official evidence shows this claim is accurate and verified."
        elif label == "false":
            claim = f"Viral hoax falsely says invented treatment works number {idx}"
            evidence = "Fact checkers debunked this false and misleading claim."
        else:
            claim = f"Reportedly disputed claim remains unverified number {idx}"
            evidence = "Available evidence is mixed and uncertain."
        rows.append(
            {
                "id": str(idx),
                "split": split,
                "claim": claim,
                "evidence": evidence,
                "veracity": label,
                "date": f"2020-01-{(idx % 28) + 1:02d}",
                "source_url": "https://example.test",
            }
        )
    return rows


if __name__ == "__main__":
    unittest.main()
