# Usage Guide

This guide covers the command-line workflow for MisinfoEQA.

## Install

For local CSV or JSONL demos:

```bat
python -m pip install -e .
```

For the MVP Hugging Face datasets:

```bat
python -m pip install -e ".[full]"
```

If Windows warns that `misinfo-eqa.exe` is not on `PATH`, use:

```bat
python -m misinfo_eqa --help
```

## Commands

### Scan

`scan` loads configured datasets, normalizes a sample, and prints coverage
summaries.

```bat
python -m misinfo_eqa scan --config configs/mvp.yaml
```

Use scan before long runs. It catches missing evidence, missing dates, label
imbalance, and raw evidence that is not natural-language evidence.

### Run

`run` executes baselines, stressors, plots, and report generation.

```bat
python -m misinfo_eqa run --config configs/mvp.yaml
```

The command writes a timestamped directory under `runs/`.

### Report

Regenerate reports from an existing run:

```bat
python -m misinfo_eqa report --latest
python -m misinfo_eqa report --run runs\20260422-040519
```

### Audit

Create a reviewer sheet from flagged examples:

```bat
python -m misinfo_eqa audit --latest --per-dataset 25
```

Summarize a completed audit sheet:

```bat
python -m misinfo_eqa audit-summary --latest
```

## Config Fields

The MVP config uses:

```yaml
run_dir: runs
max_examples_per_dataset: 1000
seed: 42
api_judge: false
bootstrap_samples: 1000
label_space: [true, false, unknown]
datasets:
  - name: fever
    source: huggingface
    hf_name: ComplexDataLab/Misinfo_Datasets
    subset: fever
    splits: [train, validation, test]
```

Local datasets use:

```yaml
datasets:
  - name: my_dataset
    source: local
    path: path/to/file.csv
```

The local loader accepts CSV, JSON, and JSONL files. It recognizes common source
columns such as `claim`, `evidence`, `evidence_text`, `positive_evidence_text`,
`veracity`, `label`, `date`, and `source_url`.

## Interpreting Evidence Coverage

MisinfoEQA distinguishes two evidence notions:

- `raw_evidence_coverage`: any evidence-like field exists.
- `evidence_coverage`: usable natural-language evidence exists.

This distinction is important because some datasets store evidence as structured
references, IDs, or tuple-like fields. Those are useful for retrieval pipelines
but should not be treated as natural-language evidence in a text classifier.

## Interpreting Risk Flags

Risk flags are diagnostics, not final judgments. A high flag rate means the
dataset deserves inspection; manual audit determines whether the flag is a true
dataset issue or a limitation of the heuristic.

Common risk names:

- `evidence_hurts_risk`: adding evidence reduces macro-F1 relative to claim-only.
- `evidence_low_utility_risk`: adding evidence changes little relative to
  claim-only.
- `ambiguity_ranking_risk`: model rankings change on ambiguity slices.
- `label_rationale_mismatch_risk`: label/evidence checks flag many examples.

## Working With Results

Use `report.md` for a human-readable summary and the JSON files for downstream
analysis. The most useful files for portfolio or paper work are:

- `data_summary.json`
- `metrics.json`
- `stressors.json`
- `risk_flags.json`
- `audit_summary.md`
- `audit_summary.json`

