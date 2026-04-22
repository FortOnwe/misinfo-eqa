# MisinfoEQA

MisinfoEQA is a lightweight Evaluation Quality Assurance harness for
misinformation datasets. It loads public or local datasets, normalizes them into
a common schema, runs simple baselines, creates stress-test slices, and writes a
report showing where model rankings, labels, evidence fields, or dataset
assumptions look fragile.

The project is intentionally about **dataset QA**, not about creating new
misinformation or claiming a new state-of-the-art misinformation detector.

## Author

**Fortune Nwachukwu Onwe**  
MSc Artificial Intelligence and Data Science, University of Hull, United Kingdom  
Institutional email: <f.onwe-2025@hull.ac.uk>  
Personal email: <fortonwe@gmail.com>  
LinkedIn: <https://www.linkedin.com/in/fortune-onwe/>  
GitHub: <https://github.com/FortOnwe>  
Project repository: <https://github.com/FortOnwe/misinfo-eqa>

## What It Tests

MisinfoEQA currently implements five stressors:

- `keyword_shortcut`: masks label-associated tokens and measures performance
  collapse.
- `temporal_shift`: compares random evaluation with date-aware evaluation when
  dates are available.
- `evidence_ablation`: compares claim-only, evidence-only, and combined inputs.
- `ambiguity_slice`: isolates unknown, hedged, mixed, or disputed examples.
- `label_rationale_mismatch`: flags examples where the evidence appears weak or
  inconsistent with the assigned label.

The label-rationale stressor uses sentence/window-level evidence relevance
instead of whole-document lexical overlap. This matters for long fact-checking
articles, where the relevant span may be surrounded by unrelated background.

## Quick Start

Install the optional dependencies for Hugging Face dataset loading:

```bat
python -m pip install -e ".[full]"
```

Run the MVP configuration:

```bat
python -m misinfo_eqa scan --config configs/mvp.yaml
python -m misinfo_eqa run --config configs/mvp.yaml
python -m misinfo_eqa report --latest
```

Create and summarize a manual audit sheet:

```bat
python -m misinfo_eqa audit --latest --per-dataset 25
python -m misinfo_eqa audit-summary --latest
```

If the `misinfo-eqa` console script is not on your Windows `PATH`, use
`python -m misinfo_eqa ...` as shown above.

## MVP Dataset Scope

The default MVP uses the ComplexDataLab Hugging Face collection:

- `fever`
- `climate_fever`
- `pubhealthtab`
- `snopes`

The pipeline also supports local CSV, JSON, and JSONL files. Local records are
normalized into:

```text
id, dataset, split, claim, evidence_text, label3, date, source_url, metadata
```

Accepted `label3` values are `true`, `false`, and `unknown`.

## Outputs

Each run writes a timestamped directory under `runs/`:

- `config.json`
- `normalized_examples.jsonl`
- `data_summary.json`
- `metrics.json`
- `stressors.json`
- `risk_flags.json`
- `flagged_examples.jsonl`
- `audit_sheet.csv` after running `audit`
- `audit_summary.md` and `audit_summary.json` after running `audit-summary`
- `report.md`
- `report.html`
- `plots/*.svg`

`runs/` is ignored by Git so large or sensitive local artifacts are not committed
by accident.

## Validated MVP Result

The current validation run uses `max_examples_per_dataset: 1000`, seed `42`, and
the four MVP datasets above. The strongest result is not a model accuracy claim:
it is an evidence-quality finding.

| Dataset | Natural-language evidence | Raw evidence | Main QA finding |
| --- | ---: | ---: | --- |
| `climate_fever` | 99.8% | 100.0% | Claim+evidence underperformed claim-only by 0.109 macro-F1; manual audit found 76.0% precision among reviewed flags. |
| `fever` | 0.0% | 88.2% | Evidence fields are mostly structured references, not usable natural-language evidence in this normalized source. |
| `pubhealthtab` | 0.0% | 0.0% | Evidence-based stressors are skipped because usable evidence is absent. |
| `snopes` | 100.0% | 100.0% | Long-form evidence produces lower-precision heuristic flags; manual audit precision was 16.0%. |

See [docs/RESULTS.md](docs/RESULTS.md) for the full interpretation.

## Repository Layout

```text
misinfo_eqa/        Python package and CLI implementation
configs/            MVP, smoke, and demo configs
examples/           Tiny local demo dataset
tests/              Unit and smoke tests
docs/               Usage, audit, reproducibility, and result notes
paper/              Paper in Markdown and LaTeX
.github/workflows/  CI test workflow
```

## Development

Run the unit suite:

```bat
python -m unittest discover -s tests
```

Run the dependency-light demo:

```bat
python -m misinfo_eqa run --config configs/demo.yaml
```

The default baselines are intentionally lightweight:

- majority-class classifier
- dependency-free hashed TF-IDF plus softmax logistic regression
- simple heuristic NLI-style cue classifier

This keeps the tool runnable in small environments and makes failures easier to
interpret.

## Documentation

- [docs/USAGE.md](docs/USAGE.md): CLI and config guide.
- [docs/AUDIT_PROTOCOL.md](docs/AUDIT_PROTOCOL.md): manual audit workflow and
  verdict labels.
- [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md): exact reproduction steps.
- [docs/RESULTS.md](docs/RESULTS.md): MVP findings and limitations.
- [paper/misinfoeqa_paper.md](paper/misinfoeqa_paper.md): paper-readable
  Markdown version.
- [paper/misinfoeqa_paper.tex](paper/misinfoeqa_paper.tex): LaTeX version.

## Safety Notes

MisinfoEQA evaluates existing public examples. It does not synthesize new false
claims, retrieve live web pages, or publish transformed misinformation. Stressors
mask, ablate, slice, or audit existing records.

## License

MIT. See [LICENSE](LICENSE).
