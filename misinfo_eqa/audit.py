from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any

from .io import read_jsonl


AUDIT_FIELDS = [
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
]


def generate_audit_sheet(
    run_dir: str | Path,
    *,
    per_dataset: int = 25,
    output: str | Path | None = None,
) -> Path:
    run_path = Path(run_dir)
    flagged = read_jsonl(run_path / "flagged_examples.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in flagged:
        grouped.setdefault(str(row.get("dataset", "dataset")), []).append(row)

    output_path = Path(output) if output else run_path / "audit_sheet.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        for dataset in sorted(grouped):
            for row in grouped[dataset][:per_dataset]:
                writer.writerow(
                    {
                        "dataset": dataset,
                        "id": row.get("id", ""),
                        "label3": row.get("label3", ""),
                        "reasons": ";".join(row.get("reasons", [])),
                        "claim": row.get("claim", ""),
                        "evidence_text": row.get("evidence_text", ""),
                        "relevance_score": row.get("relevance_score", ""),
                        "best_evidence_span": row.get("best_evidence_span", ""),
                        "audit_verdict": "",
                        "notes": "",
                    }
                )
    return output_path


def summarize_audit_sheet(path: str | Path) -> dict[str, Any]:
    audit_path = Path(path)
    with audit_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    issue_verdicts = {"evidence_mismatch", "missing_or_weak_evidence", "real_dataset_issue"}
    datasets: dict[str, Any] = {}
    for dataset in sorted({row.get("dataset", "dataset") for row in rows}):
        dataset_rows = [row for row in rows if row.get("dataset") == dataset]
        datasets[dataset] = _summarize_rows(dataset_rows, issue_verdicts)

    summary = {
        "path": str(audit_path),
        "overall": _summarize_rows(rows, issue_verdicts),
        "datasets": datasets,
        "by_reason": _summarize_by_reason(rows, issue_verdicts),
    }
    return summary


def write_audit_summary(
    audit_path: str | Path,
    *,
    output: str | Path | None = None,
) -> Path:
    summary = summarize_audit_sheet(audit_path)
    output_path = Path(output) if output else Path(audit_path).with_name("audit_summary.md")
    output_path.write_text(render_audit_summary(summary), encoding="utf-8")
    output_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def render_audit_summary(summary: dict[str, Any]) -> str:
    lines = ["# MisinfoEQA Audit Summary", ""]
    overall = summary["overall"]
    lines.append(
        f"Audited {overall['n']} examples. Real issue precision: "
        f"{overall['issue_precision']:.1%} ({overall['n_issues']}/{overall['n']})."
    )
    lines.append("")
    lines.append("## By Dataset")
    lines.append("")
    lines.append("| Dataset | Audited | Real Issues | Precision | Verdicts |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for dataset, result in summary["datasets"].items():
        lines.append(
            f"| {dataset} | {result['n']} | {result['n_issues']} | "
            f"{result['issue_precision']:.1%} | {_format_counter(result['verdicts'])} |"
        )
    lines.append("")
    lines.append("## By Flag Reason")
    lines.append("")
    lines.append("| Reason | Audited | Real Issues | Precision |")
    lines.append("| --- | ---: | ---: | ---: |")
    for reason, result in summary["by_reason"].items():
        lines.append(
            f"| {reason} | {result['n']} | {result['n_issues']} | "
            f"{result['issue_precision']:.1%} |"
        )
    lines.append("")
    return "\n".join(lines)


def _summarize_rows(rows: list[dict[str, Any]], issue_verdicts: set[str]) -> dict[str, Any]:
    verdicts = Counter((row.get("audit_verdict") or "").strip() for row in rows)
    n_issues = sum(1 for row in rows if (row.get("audit_verdict") or "").strip() in issue_verdicts)
    n = len(rows)
    return {
        "n": n,
        "n_issues": n_issues,
        "issue_precision": n_issues / n if n else 0.0,
        "verdicts": dict(verdicts),
        "notes_filled": sum(1 for row in rows if (row.get("notes") or "").strip()),
    }


def _summarize_by_reason(
    rows: list[dict[str, Any]],
    issue_verdicts: set[str],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row.get("reasons", ""), []).append(row)
    return {
        reason: _summarize_rows(reason_rows, issue_verdicts)
        for reason, reason_rows in sorted(grouped.items())
    }


def _format_counter(counter: dict[str, int]) -> str:
    return ", ".join(f"{key}:{value}" for key, value in sorted(counter.items()) if key)
