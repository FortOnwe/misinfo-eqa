# MVP Results

This page summarizes the validated MVP run:

```text
runs/20260422-040519
```

The run used 1,000 examples per dataset, seed `42`, no API judge, and 1,000
bootstrap samples.

## Coverage

| Dataset | Examples | Label counts | Natural-language evidence | Raw evidence | Date coverage |
| --- | ---: | --- | ---: | ---: | ---: |
| `climate_fever` | 1000 | false:183, true:513, unknown:304 | 99.8% | 100.0% | 0.0% |
| `fever` | 1000 | false:195, true:468, unknown:337 | 0.0% | 88.2% | 0.0% |
| `pubhealthtab` | 1000 | false:231, true:536, unknown:233 | 0.0% | 0.0% | 0.0% |
| `snopes` | 1000 | false:595, true:161, unknown:244 | 100.0% | 100.0% | 0.0% |

The most important coverage finding is that raw evidence is not necessarily
usable natural-language evidence. FEVER has raw structured references in 88.2%
of sampled records, but 0.0% natural-language evidence in this normalized source.

## Baseline Highlights

| Dataset | Best lightweight baseline | Macro-F1 | Main interpretation |
| --- | --- | ---: | --- |
| `climate_fever` | claim-only TF-IDF logistic regression | 0.660 | Claim text is highly predictive; adding evidence hurts. |
| `fever` | claim-only TF-IDF logistic regression | 0.374 | Evidence fields are not natural-language text, so evidence runs are weak. |
| `pubhealthtab` | claim-only TF-IDF logistic regression | 0.391 | Evidence checks are skipped because evidence is absent. |
| `snopes` | evidence-only TF-IDF logistic regression | 0.359 | Evidence helps a little, but long narratives are hard for simple baselines. |

## Stressor Findings

| Dataset | Keyword drop | Evidence gain | Ambiguity Spearman | Label-rationale flags |
| --- | ---: | ---: | ---: | ---: |
| `climate_fever` | 0.017 | -0.109 | 0.400 | 185/315 test examples |
| `fever` | 0.007 | skipped | 0.000 | skipped |
| `pubhealthtab` | 0.004 | skipped | 0.000 | skipped |
| `snopes` | 0.013 | 0.016 | 0.900 | 75/301 test examples |

Evidence gain is `claim_plus_evidence_macro_f1 - claim_only_macro_f1`.

## Risk Flags

| Dataset | Risk | Severity | Evidence |
| --- | --- | --- | --- |
| `climate_fever` | `evidence_hurts_risk` | high | Claim+evidence macro-F1 was 0.109 below claim-only. |
| `climate_fever` | `ambiguity_ranking_risk` | medium | Ambiguity-slice Spearman correlation was 0.400. |
| `climate_fever` | `label_rationale_mismatch_risk` | medium | 58.7% of test examples were heuristic-flagged. |
| `fever` | `ambiguity_ranking_risk` | medium | Ambiguity-slice Spearman correlation was 0.000. |
| `pubhealthtab` | `ambiguity_ranking_risk` | medium | Ambiguity-slice Spearman correlation was 0.000. |
| `snopes` | `evidence_low_utility_risk` | medium | Claim+evidence was within 0.03 macro-F1 of claim-only. |
| `snopes` | `label_rationale_mismatch_risk` | medium | 24.9% of test examples were heuristic-flagged. |

## Manual Audit

Manual audit reviewed 50 examples:

| Dataset | Audited | Real issues | Precision | Verdict mix |
| --- | ---: | ---: | ---: | --- |
| `climate_fever` | 25 | 19 | 76.0% | evidence_mismatch:8, missing_or_weak_evidence:11, heuristic_false_positive:6 |
| `snopes` | 25 | 4 | 16.0% | real_dataset_issue:2, missing_or_weak_evidence:2, heuristic_false_positive:21 |

By reason:

| Reason | Audited | Real issues | Precision |
| --- | ---: | ---: | ---: |
| `low_claim_evidence_relevance` | 33 | 19 | 57.6% |
| `heuristic_label_disagrees:true` | 12 | 2 | 16.7% |
| `heuristic_label_disagrees:false` | 2 | 0 | 0.0% |
| `low_claim_evidence_relevance;heuristic_label_disagrees:true` | 2 | 1 | 50.0% |
| `missing_evidence` | 1 | 1 | 100.0% |

## Interpretation

The MVP result is strongest as a dataset-QA finding:

- Climate-FEVER has many examples where the available evidence appears weak,
  mismatched, or hard to use under the normalized evidence field.
- FEVER cannot be evaluated as claim-plus-natural-language-evidence from this
  source without resolving structured evidence references into sentence text.
- PubHealthTab lacks usable evidence in this normalized source.
- Snopes needs better semantic or article-structure-aware rationale checks
  because simple heuristics mostly over-flag long-form narratives.

## Limitations

- This is a 1,000-example-per-dataset MVP, not a full benchmark sweep.
- The baselines are intentionally light and should not be interpreted as strong
  misinformation detectors.
- The temporal stressor is skipped because date coverage is 0.0% in the sampled
  normalized records.
- Manual audit currently covers 50 flagged examples.

