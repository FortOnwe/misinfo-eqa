# MisinfoEQA: Stress-Testing Evidence Quality and Evaluation Fragility in Misinformation Benchmarks

Fortune Nwachukwu Onwe  
MSc Artificial Intelligence and Data Science, University of Hull, United Kingdom  
Institutional email: <f.onwe-2025@hull.ac.uk>  
Personal email: <fortonwe@gmail.com>  
LinkedIn: <https://www.linkedin.com/in/fortune-onwe/>  
GitHub: <https://github.com/FortOnwe>  
Project repository: <https://github.com/FortOnwe/misinfo-eqa>  
April 22, 2026

## Abstract

Misinformation datasets are often evaluated as if labels, evidence, and splits
are directly comparable across sources. In practice, evidence fields may contain
natural-language rationales, structured references, long fact-checking articles,
or no usable evidence at all. We present MisinfoEQA, a lightweight evaluation
quality assurance harness that normalizes misinformation datasets, separates raw
evidence availability from natural-language evidence usability, runs simple
interpretable baselines, applies stress tests, and produces manual-audit sheets.
On four subsets from `ComplexDataLab/Misinfo_Datasets` with 1,000 examples per
dataset, MisinfoEQA finds that Climate-FEVER has high natural-language evidence
coverage but claim-plus-evidence underperforms claim-only by 0.109 macro-F1, and
manual audit confirms 76.0% precision among reviewed Climate-FEVER evidence
flags. FEVER has 88.2% raw evidence coverage but 0.0% natural-language evidence
coverage in this normalized source, because the evidence is mostly structured
reference metadata. PubHealthTab has no usable evidence in the sampled records.
Snopes has complete natural-language evidence coverage but low audit precision
for simple evidence heuristics, showing that long-form fact-checking narratives
need richer semantic or article-structure-aware QA. These results support a
simple conclusion: evidence availability is not evidence usability, and
misinformation benchmark reports should include dataset-QA diagnostics alongside
model scores.

## 1. Introduction

Recent work on misinformation evaluation has expanded from static fact-checking
classification to richer concerns such as dataset quality, LLM resilience,
temporal generalization, and contamination resistance. A broad guide to
misinformation detection data and evaluation curated a large unified collection
and highlighted quality concerns such as keyword shortcuts, temporal artifacts,
and feasibility of verification [1, 2]. Adjacent benchmark work such as
MisinfoBench [3] and TripleFact [4] targets LLM resilience and contamination
risks.

MisinfoEQA is complementary. It does not introduce a new LLM benchmark or a new
misinformation generation task. Instead, it asks whether a dataset is safe to
evaluate in the way a practitioner might be tempted to evaluate it. Are labels
mapped consistently? Does the evidence field contain natural-language evidence
or only IDs? Does adding evidence improve or hurt a lightweight classifier? Do
model rankings change on ambiguous slices? Do heuristic rationale flags survive
manual audit?

The contribution is a standalone command-line harness and an MVP validation
study. The tool:

1. normalizes public or local datasets into a shared schema;
2. distinguishes raw evidence coverage from natural-language evidence coverage;
3. runs dependency-light baselines and stressors;
4. generates Markdown/HTML reports, plots, and JSON artifacts;
5. creates audit sheets and summarizes reviewer verdicts.

The empirical claim is deliberately modest but useful: simple QA checks surface
actionable differences between datasets before expensive model evaluation.

## 2. Related Work

FEVER introduced a large-scale claim verification benchmark with claims labeled
as supported, refuted, or not enough information, and evidence annotations for
supported and refuted claims [5]. Climate-FEVER adapted FEVER-style verification
to real-world climate claims and emphasized the subtlety of modeling climate
claims with evidence [6]. The ComplexDataLab misinformation collection unifies
many misinformation datasets and harmonizes labels into true, false, and unknown
categories [1, 2].

Recent benchmark work has broadened the evaluation target. MisinfoBench
evaluates LLM resilience to misinformation across dimensions such as
discernment, resistance, and principled refusal [3]. TripleFact focuses on
benchmark contamination and proposes dynamic and controlled evaluation
components for LLM-driven fake news detection [4]. MisinfoEQA differs by placing
the dataset itself under test. It is designed to run before, or alongside, model
benchmarking.

## 3. System Design

### 3.1 Normalized Schema

Every record is coerced into:

```text
id, dataset, split, claim, evidence_text, label3, date, source_url, metadata
```

`label3` is restricted to `true`, `false`, and `unknown`. Dataset-specific fields
are preserved in `metadata` when they are not mapped into the common schema.

### 3.2 Evidence Typing

MisinfoEQA reports two evidence fields:

- raw evidence coverage: any evidence-like value is present;
- natural-language evidence coverage: the value appears to be usable text.

This distinction prevents structured tuples such as FEVER evidence references
from being treated as ordinary evidence text.

### 3.3 Baselines

The MVP uses lightweight baselines:

- majority-class classifier;
- hashed TF-IDF plus softmax logistic regression on claim text;
- the same classifier on evidence text;
- the same classifier on claim plus evidence;
- a heuristic NLI-style cue classifier.

The TF-IDF classifier is dependency-free so the project can run in small
environments and CI without a heavy ML stack.

### 3.4 Stressors

MisinfoEQA implements five stressors:

- keyword shortcut: mask high-risk label-associated tokens;
- temporal shift: compare random and date-aware evaluation when dates exist;
- evidence ablation: compare claim-only, evidence-only, and combined inputs;
- ambiguity slice: isolate unknown, hedged, mixed, or disputed records;
- label-rationale mismatch: flag missing evidence, low local evidence relevance,
  and guarded label-cue disagreements.

### 3.5 Manual Audit

The harness writes `audit_sheet.csv` from flagged examples. Reviewers mark each
row as `real_dataset_issue`, `evidence_mismatch`, `missing_or_weak_evidence`,
`structured_reference_evidence`, `heuristic_false_positive`,
`baseline_limitation`, or `unclear`. Summary precision treats the first four
labels as real issues.

## 4. Experimental Setup

The MVP evaluates four subsets from `ComplexDataLab/Misinfo_Datasets`:
`fever`, `climate_fever`, `pubhealthtab`, and `snopes`. The canonical run uses
1,000 examples per dataset, seed 42, no API judge, and 1,000 bootstrap samples.

The command sequence is:

```bat
python -m misinfo_eqa scan --config configs/mvp.yaml
python -m misinfo_eqa run --config configs/mvp.yaml
python -m misinfo_eqa audit --latest --per-dataset 25
python -m misinfo_eqa audit-summary --latest
```

The local validation run used for this paper was `runs/20260422-040519`.

## 5. Results

### 5.1 Coverage

| Dataset | Examples | Label counts | Natural-language evidence | Raw evidence | Date coverage |
| --- | ---: | --- | ---: | ---: | ---: |
| `climate_fever` | 1000 | false:183, true:513, unknown:304 | 99.8% | 100.0% | 0.0% |
| `fever` | 1000 | false:195, true:468, unknown:337 | 0.0% | 88.2% | 0.0% |
| `pubhealthtab` | 1000 | false:231, true:536, unknown:233 | 0.0% | 0.0% | 0.0% |
| `snopes` | 1000 | false:595, true:161, unknown:244 | 100.0% | 100.0% | 0.0% |

The strongest coverage finding is that raw evidence coverage can be misleading.
FEVER has raw evidence in 88.2% of sampled records, but those fields are
structured references rather than natural-language evidence.

### 5.2 Baseline Metrics

| Dataset | Majority | Claim TF-IDF | Claim+Evidence TF-IDF | Evidence TF-IDF | Heuristic NLI |
| --- | ---: | ---: | ---: | ---: | ---: |
| `climate_fever` | 0.228 | 0.660 | 0.551 | 0.407 | 0.207 |
| `fever` | 0.217 | 0.374 | 0.367 | 0.217 | 0.168 |
| `pubhealthtab` | 0.214 | 0.391 | 0.391 | 0.214 | 0.125 |
| `snopes` | 0.245 | 0.312 | 0.329 | 0.359 | 0.376 |

Values are macro-F1 on the test portion of the sampled records. The Climate-FEVER
result is the clearest evidence-ablation signal: adding evidence reduces macro-F1
from 0.660 to 0.551. This does not prove that evidence is useless; rather, it
shows that the evidence field and lightweight classifier interact poorly enough
to justify closer QA.

### 5.3 Stressor Findings

| Dataset | Keyword drop | Evidence gain | Ambiguity Spearman | Label-rationale flags |
| --- | ---: | ---: | ---: | ---: |
| `climate_fever` | 0.017 | -0.109 | 0.400 | 185/315 |
| `fever` | 0.007 | skipped | 0.000 | skipped |
| `pubhealthtab` | 0.004 | skipped | 0.000 | skipped |
| `snopes` | 0.013 | 0.016 | 0.900 | 75/301 |

Temporal shift was skipped for all four datasets because date coverage was 0.0%
in the sampled normalized records.

### 5.4 Manual Audit

| Dataset | Audited | Real issues | Precision | Verdict mix |
| --- | ---: | ---: | ---: | --- |
| `climate_fever` | 25 | 19 | 76.0% | evidence_mismatch:8, missing_or_weak_evidence:11, heuristic_false_positive:6 |
| `snopes` | 25 | 4 | 16.0% | real_dataset_issue:2, missing_or_weak_evidence:2, heuristic_false_positive:21 |

The manual audit validates Climate-FEVER as the strongest MVP finding. It also
reveals a limitation: the Snopes flags are mostly false positives because long
fact-checking narratives contain background, quoted claims, and verdict language
that simple lexical heuristics do not reliably isolate.

## 6. Discussion

MisinfoEQA changes the first question from "Which model wins?" to "Is this
dataset being evaluated in a way that matches its fields?" This is useful because
the four MVP datasets fail or stress different assumptions:

- Climate-FEVER has usable-looking evidence, but evidence hurts a lightweight
  classifier and audited flags frequently identify weak or mismatched evidence.
- FEVER requires resolving structured evidence references into sentence text
  before evidence-based evaluation is meaningful.
- PubHealthTab cannot support evidence-ablation or rationale checks from this
  normalized source.
- Snopes contains natural-language evidence, but long-form articles need
  stronger span selection.

The results suggest that evaluation reports should include dataset-QA tables
before presenting model rankings. In particular, evidence coverage should be
reported as natural-language evidence coverage, not just non-empty evidence
fields.

## 7. Limitations

The MVP has several limits:

- It uses 1,000 sampled examples per dataset, not full-dataset sweeps.
- It uses lightweight baselines, not modern neural or API judges.
- Temporal shift is skipped because the sampled records have no usable dates.
- Manual audit covers 50 examples.
- The Snopes heuristic needs better article span extraction or semantic
  validation.
- Results may shift if the upstream Hugging Face collection changes.

## 8. Ethical Considerations

MisinfoEQA is designed to inspect existing public datasets. It does not generate
new misinformation, retrieve live claims, or create adversarial false examples.
Reports may include snippets from existing datasets, so users should handle
outputs with the same care as the source data and avoid presenting flagged claims
without context.

## 9. Conclusion

MisinfoEQA provides a small, reproducible harness for stress-testing
misinformation dataset quality. The MVP shows that evidence fields vary sharply
in usability, that evidence can hurt simple evaluation setups, and that manual
audit is necessary to separate true dataset issues from heuristic artifacts. The
next version should resolve structured evidence references, add stronger
semantic rationale checks, expand manual audit, and compare light QA diagnostics
against stronger LLM or NLI judges.

## References

[1] Camille Thibault, Jacob-Junqi Tian, Gabrielle Peloquin-Skulski, Taylor Lynn
Curtis, James Zhou, Florence Laflamme, Yuxiang Guan, Reihaneh Rabbany,
Jean-Francois Godbout, and Kellin Pelrine. 2025. *A Guide to Misinformation
Detection Data and Evaluation*. arXiv:2411.05060.

[2] Complex Data Lab. 2025. *Misinfo Datasets*. Hugging Face dataset collection.
https://huggingface.co/datasets/ComplexDataLab/Misinfo_Datasets

[3] Ye Yang, Donghe Li, Zuchen Li, Fengyuan Li, Jingyi Liu, Li Sun, and Qingyu
Yang. 2025. *MisinfoBench: A Multi-Dimensional Benchmark for Evaluating LLMs'
Resilience to Misinformation*. Findings of ACL: EMNLP 2025.

[4] Cheng Xu and Nan Yan. 2025. *TripleFact: Defending Data Contamination in the
Evaluation of LLM-driven Fake News Detection*. ACL 2025.

[5] James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal.
2018. *FEVER: a Large-scale Dataset for Fact Extraction and VERification*.
NAACL-HLT 2018.

[6] Thomas Diggelmann, Jordan Boyd-Graber, Jannis Bulian, Massimiliano Ciaramita,
and Markus Leippold. 2020. *CLIMATE-FEVER: A Dataset for Verification of
Real-World Climate Claims*. arXiv:2012.00614.
