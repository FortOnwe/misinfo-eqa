# Audit Protocol

Manual audit turns heuristic flags into evidence about dataset quality. The goal
is to measure whether a flag identifies a real issue, not whether the heuristic
was clever.

## Create The Sheet

```bat
python -m misinfo_eqa audit --latest --per-dataset 25
```

The sheet includes:

- dataset and example id
- claim
- label
- evidence text
- flag reasons
- sentence/window-level `relevance_score`
- `best_evidence_span`
- blank reviewer columns

## Verdict Labels

Use one verdict per row:

- `real_dataset_issue`: the example appears mislabeled, internally
  inconsistent, or otherwise broken.
- `evidence_mismatch`: the evidence appears to support a different claim,
  topic, entity, or label than the record.
- `missing_or_weak_evidence`: the evidence is absent or too weak to justify the
  label.
- `structured_reference_evidence`: the field is an ID/reference field rather
  than natural-language evidence.
- `heuristic_false_positive`: the example looks acceptable and the flag is
  mostly a heuristic artifact.
- `baseline_limitation`: the issue is better explained by a weak baseline than
  by dataset quality.
- `unclear`: the reviewer cannot judge quickly.

For summary precision, MisinfoEQA treats `real_dataset_issue`,
`evidence_mismatch`, `missing_or_weak_evidence`, and
`structured_reference_evidence` as real issues.

## Reviewer Notes

Use `reviewer_notes` for the minimum explanation that would let another reviewer
understand the call. Examples:

- "Evidence is about a different entity."
- "Evidence is a citation tuple, not text."
- "Long article contains relevant span; flag is false positive."
- "Claim is broader than evidence supports."

## Suggested Audit Workflow

1. Sort by dataset and reason.
2. Review 25 to 50 examples per dataset for the first pass.
3. Record one verdict per row.
4. Run:

```bat
python -m misinfo_eqa audit-summary --latest
```

5. Treat high-precision reasons as paper findings and low-precision reasons as
   limitations or next-iteration engineering targets.

## Current MVP Audit Result

For the canonical MVP run, 50 examples were audited:

| Dataset | Audited | Real issues | Precision |
| --- | ---: | ---: | ---: |
| `climate_fever` | 25 | 19 | 76.0% |
| `snopes` | 25 | 4 | 16.0% |

The key lesson is that sentence/window-level relevance works well enough to find
Climate-FEVER evidence problems, but long Snopes narratives still need stronger
semantic checks or better retrieval of article verdict spans.

