from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from .io import read_json, read_jsonl, write_json


def generate_report(run_dir: str | Path) -> dict[str, str]:
    run_path = Path(run_dir)
    data_summary = read_json(run_path / "data_summary.json")
    metrics = read_json(run_path / "metrics.json")
    stressors = read_json(run_path / "stressors.json")
    risk_flags = read_json(run_path / "risk_flags.json")
    flagged = read_jsonl(run_path / "flagged_examples.jsonl")

    plots_dir = run_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    _write_metric_plots(plots_dir, metrics)

    markdown = _render_markdown(data_summary, metrics, stressors, risk_flags, flagged)
    md_path = run_path / "report.md"
    html_path = run_path / "report.html"
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(_render_html(markdown), encoding="utf-8")

    manifest = {"markdown": str(md_path), "html": str(html_path), "plots": str(plots_dir)}
    write_json(run_path / "report_manifest.json", manifest)
    return manifest


def _render_markdown(
    data_summary: dict[str, Any],
    metrics: dict[str, Any],
    stressors: dict[str, Any],
    risk_flags: dict[str, Any],
    flagged: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# MisinfoEQA Report")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        "This run evaluates misinformation datasets for shortcut, temporal, evidence, ambiguity, "
        "and label-rationale risks. The tool is intended to diagnose evaluation quality, not to "
        "generate new misinformation."
    )
    lines.append("")
    lines.append("## Dataset Coverage")
    lines.append("")
    lines.append(
        "| Dataset | Examples | Label Counts | Natural-Language Evidence | Raw Evidence | Date Coverage |"
    )
    lines.append("| --- | ---: | --- | ---: | ---: | ---: |")
    for dataset, summary in sorted(data_summary["datasets"].items()):
        lines.append(
            f"| {dataset} | {summary['n']} | {_format_counts(summary['labels'])} | "
            f"{summary['evidence_coverage']:.1%} | "
            f"{summary.get('raw_evidence_coverage', summary['evidence_coverage']):.1%} | "
            f"{summary['date_coverage']:.1%} |"
        )
    lines.append("")
    coverage_warnings = [
        (dataset, warning)
        for dataset, summary in sorted(data_summary["datasets"].items())
        for warning in summary.get("warnings", [])
    ]
    if coverage_warnings:
        lines.append("### Coverage Warnings")
        lines.append("")
        for dataset, warning in coverage_warnings:
            lines.append(f"- `{dataset}`: {warning}")
        lines.append("")
    lines.append("## Baseline Metrics")
    lines.append("")
    lines.append("| Dataset | Model | N | Macro-F1 | Balanced Acc. | Accuracy | 95% CI |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
    for dataset, model_scores in sorted(metrics["datasets"].items()):
        for model, score in sorted(model_scores.items()):
            ci = score.get("macro_f1_ci", [0.0, 0.0])
            lines.append(
                f"| {dataset} | {model} | {score['n']} | {score['macro_f1']:.3f} | "
                f"{score['balanced_accuracy']:.3f} | {score['accuracy']:.3f} | "
                f"[{ci[0]:.3f}, {ci[1]:.3f}] |"
            )
    lines.append("")
    lines.append("## Stressor Findings")
    lines.append("")
    for dataset, result in sorted(stressors["datasets"].items()):
        lines.append(f"### {dataset}")
        keyword = result.get("keyword_shortcut", {})
        temporal = result.get("temporal_shift", {})
        ambiguity = result.get("ambiguity_slice", {})
        evidence = result.get("evidence_ablation", {})
        mismatch = result.get("label_rationale_mismatch", {})
        ranking = ambiguity.get("ranking_correlation", {})
        lines.append(
            f"- Keyword shortcut drop: {keyword.get('macro_f1_drop', 0.0):.3f} "
            f"after masking {keyword.get('masked_keyword_count', 0)} high-risk tokens."
        )
        if evidence.get("available", True):
            lines.append(
                f"- Evidence gain: {evidence.get('combined_minus_claim', 0.0):.3f} "
                "macro-F1 for claim-plus-evidence over claim-only."
            )
        else:
            lines.append(
                f"- Evidence ablation: skipped because {evidence.get('reason', 'unavailable')}."
            )
        if temporal.get("available"):
            lines.append(
                f"- Temporal macro-F1 drop: {temporal.get('macro_f1_drop', 0.0):.3f} "
                "from random to date-aware evaluation."
            )
        else:
            lines.append("- Temporal shift: skipped because too few usable dates were available.")
        lines.append(
            f"- Ambiguity slice size: {ambiguity.get('n', 0)}; ranking Spearman: "
            f"{ranking.get('spearman', 1.0):.3f}; Kendall: {ranking.get('kendall', 1.0):.3f}."
        )
        if mismatch.get("available", True):
            lines.append(
                f"- Label-rationale heuristic flags: {mismatch.get('n_flagged', 0)} "
                f"({mismatch.get('rate', 0.0):.1%} of test examples); "
                f"{mismatch.get('examples_stored', mismatch.get('n_flagged', 0))} stored for audit."
            )
        else:
            lines.append(
                f"- Label-rationale heuristic: skipped because {mismatch.get('reason', 'unavailable')}."
            )
        lines.append("")
    lines.append("## Risk Flags")
    lines.append("")
    lines.append("| Dataset | Risk | Severity | Evidence |")
    lines.append("| --- | --- | --- | --- |")
    any_flags = False
    for dataset, flags in sorted(risk_flags["datasets"].items()):
        for flag in flags:
            any_flags = True
            lines.append(
                f"| {dataset} | {flag['risk']} | {flag['severity']} | {flag['evidence']} |"
            )
    if not any_flags:
        lines.append("| all | null_result | info | No configured risk threshold was crossed. |")
    lines.append("")
    lines.append("## Flagged Examples")
    lines.append("")
    grouped = _group_flagged(flagged)
    for dataset, rows in sorted(grouped.items()):
        lines.append(f"### {dataset}")
        lines.append("")
        for row in rows[:5]:
            lines.append(
                f"- `{row['id']}` {', '.join(row['reasons'])}: {row['claim'][:220]}"
            )
        lines.append("")
    if not flagged:
        lines.append("- No examples were flagged by the label-rationale heuristic.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "A strong finding is a stressor that changes a model ranking, exposes a large "
        "slice-specific drop, or identifies a recurring label/evidence issue. A null result is "
        "still useful when confidence intervals are tight and the report shows that common "
        "dataset-quality failure modes were not observed."
    )
    lines.append("")
    return "\n".join(lines)


def _render_html(markdown: str) -> str:
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>MisinfoEQA Report</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:40px auto;"
        "line-height:1.5;color:#1f2937}pre{white-space:pre-wrap;background:#f8fafc;"
        "padding:20px;border:1px solid #e5e7eb}table{border-collapse:collapse}"
        "td,th{border:1px solid #ddd;padding:6px}</style></head><body><pre>"
        + escape(markdown)
        + "</pre></body></html>"
    )


def _write_metric_plots(plots_dir: Path, metrics: dict[str, Any]) -> None:
    for dataset, model_scores in metrics["datasets"].items():
        values = {model: score["macro_f1"] for model, score in model_scores.items()}
        _write_bar_svg(plots_dir / f"{dataset}_macro_f1.svg", values, f"{dataset} Macro-F1")


def _write_bar_svg(path: Path, values: dict[str, float], title: str) -> None:
    width = 760
    height = 320
    margin = 48
    bar_gap = 16
    names = list(values)
    bar_width = max(24, int((width - 2 * margin - bar_gap * max(0, len(names) - 1)) / max(1, len(names))))
    max_value = max([0.05, *values.values()])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img">',
        f'<text x="{margin}" y="28" font-family="Arial" font-size="18" font-weight="700">{escape(title)}</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#334155"/>',
    ]
    for index, (name, value) in enumerate(values.items()):
        x = margin + index * (bar_width + bar_gap)
        bar_height = int((height - 2 * margin) * (value / max_value))
        y = height - margin - bar_height
        parts.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="#2563eb"/>')
        parts.append(f'<text x="{x}" y="{y - 6}" font-family="Arial" font-size="12">{value:.3f}</text>')
        parts.append(
            f'<text x="{x}" y="{height - margin + 18}" font-family="Arial" font-size="11">'
            f"{escape(name[:18])}</text>"
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def _group_flagged(flagged: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in flagged:
        grouped.setdefault(str(row.get("dataset", "dataset")), []).append(row)
    return grouped
