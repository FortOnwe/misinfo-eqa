# Reproducibility

This document records the MVP run used for the paper-style result.

## Environment

- Python: 3.10 or newer
- Platform tested by the user: Windows 10
- Network: required only for Hugging Face dataset download
- Optional dependencies: `datasets`, `pandas`, `pyarrow`, `pyyaml`

Install:

```bat
python -m pip install -e ".[full]"
```

## Data Source

The MVP uses `ComplexDataLab/Misinfo_Datasets` from Hugging Face with these
subsets:

- `fever`
- `climate_fever`
- `pubhealthtab`
- `snopes`

## Canonical Config

Use `configs/mvp.yaml`:

```yaml
max_examples_per_dataset: 1000
seed: 42
api_judge: false
bootstrap_samples: 1000
label_space: [true, false, unknown]
```

## Commands

```bat
python -m misinfo_eqa scan --config configs/mvp.yaml
python -m misinfo_eqa run --config configs/mvp.yaml
python -m misinfo_eqa audit --latest --per-dataset 25
python -m misinfo_eqa audit-summary --latest
```

## Expected Artifacts

The main run directory should contain:

- `report.md`
- `report.html`
- `data_summary.json`
- `metrics.json`
- `stressors.json`
- `risk_flags.json`
- `flagged_examples.jsonl`
- `audit_sheet.csv`
- `audit_summary.md`
- `audit_summary.json`

## Canonical Run Used In The Paper

The local validation run used in the paper was:

```text
runs/20260422-040519
```

Because `runs/` is ignored by Git, new users should reproduce it with the
commands above rather than expecting the artifact directory to be committed.

## Known Non-Reproducibility Sources

- Hugging Face datasets are live repositories, so upstream dataset updates may
  slightly change records or splits.
- Bootstrap confidence intervals depend on `seed`.
- Manual-audit precision depends on reviewer judgment.
- The project currently uses lightweight heuristic and lexical baselines, not
  heavy neural models.

