from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import load_config
from .audit import generate_audit_sheet, write_audit_summary
from .pipeline import run_pipeline, scan_config
from .report import generate_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="misinfo-eqa")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Summarize configured datasets.")
    scan_parser.add_argument("--config", required=True, help="Path to YAML or JSON config.")

    run_parser = subparsers.add_parser("run", help="Run baselines, stressors, and report generation.")
    run_parser.add_argument("--config", required=True, help="Path to YAML or JSON config.")

    report_parser = subparsers.add_parser("report", help="Regenerate report for an existing run.")
    report_parser.add_argument("--run", help="Path to a run directory.")
    report_parser.add_argument("--latest", action="store_true", help="Use the most recent run directory.")
    report_parser.add_argument("--runs-dir", default="runs", help="Directory containing run folders.")

    audit_parser = subparsers.add_parser("audit", help="Create a CSV audit sheet from flagged examples.")
    audit_parser.add_argument("--run", help="Path to a run directory.")
    audit_parser.add_argument("--latest", action="store_true", help="Use the most recent run directory.")
    audit_parser.add_argument("--runs-dir", default="runs", help="Directory containing run folders.")
    audit_parser.add_argument("--per-dataset", type=int, default=25, help="Rows to include per dataset.")
    audit_parser.add_argument("--output", help="Optional output CSV path.")

    audit_summary_parser = subparsers.add_parser(
        "audit-summary",
        help="Summarize a completed audit sheet.",
    )
    audit_summary_parser.add_argument("--audit", help="Path to audit_sheet.csv.")
    audit_summary_parser.add_argument("--latest", action="store_true", help="Use audit_sheet.csv in the most recent run.")
    audit_summary_parser.add_argument("--runs-dir", default="runs", help="Directory containing run folders.")
    audit_summary_parser.add_argument("--output", help="Optional output Markdown path.")

    args = parser.parse_args(argv)
    try:
        if args.command == "scan":
            config = load_config(args.config)
            result = scan_config(config)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "run":
            config = load_config(args.config)
            run_dir = run_pipeline(config)
            print(f"Run complete: {run_dir}")
            print(f"Report: {Path(run_dir) / 'report.md'}")
            return 0
        if args.command == "report":
            run_dir = _resolve_run_arg(args.run, latest=args.latest, runs_dir=args.runs_dir)
            manifest = generate_report(run_dir)
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0
        if args.command == "audit":
            run_dir = _resolve_run_arg(args.run, latest=args.latest, runs_dir=args.runs_dir)
            output = generate_audit_sheet(
                run_dir,
                per_dataset=args.per_dataset,
                output=args.output,
            )
            print(f"Audit sheet: {output}")
            return 0
        if args.command == "audit-summary":
            audit_path = _resolve_audit_arg(
                args.audit,
                latest=args.latest,
                runs_dir=args.runs_dir,
            )
            output = write_audit_summary(audit_path, output=args.output)
            print(f"Audit summary: {output}")
            print(f"Audit summary JSON: {Path(output).with_suffix('.json')}")
            return 0
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"misinfo-eqa error: {exc}", file=sys.stderr)
        return 1
    return 1


def _resolve_run_arg(run: str | None, *, latest: bool, runs_dir: str) -> Path:
    if latest:
        if run:
            raise ValueError("Use either --run or --latest, not both")
        return _latest_run_dir(Path(runs_dir))
    if not run:
        raise ValueError("Provide --run <path> or --latest")
    return Path(run)


def _latest_run_dir(runs_dir: Path) -> Path:
    if not runs_dir.exists():
        raise ValueError(f"Runs directory does not exist: {runs_dir}")
    candidates = [path for path in runs_dir.iterdir() if path.is_dir()]
    if not candidates:
        raise ValueError(f"No run directories found in: {runs_dir}")
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def _resolve_audit_arg(audit: str | None, *, latest: bool, runs_dir: str) -> Path:
    if latest:
        if audit:
            raise ValueError("Use either --audit or --latest, not both")
        return _latest_run_dir(Path(runs_dir)) / "audit_sheet.csv"
    if not audit:
        raise ValueError("Provide --audit <path> or --latest")
    return Path(audit)
