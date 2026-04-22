from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import random
from typing import Any

from .baselines import HeuristicNliClassifier, MajorityClassifier, TfidfLogRegClassifier
from .io import load_examples, write_json, write_jsonl
from .metrics import (
    bootstrap_delta_ci,
    bootstrap_metric_ci,
    classification_metrics,
    ranking_correlations,
)
from .report import generate_report
from .schema import Example
from .stressors import (
    analyze_label_rationale_mismatches,
    ambiguity_indices,
    mask_keywords,
    top_label_keywords,
)


def scan_config(config: dict[str, Any]) -> dict[str, Any]:
    datasets = config.get("datasets") or []
    max_examples = int(config.get("max_examples_per_dataset") or 1000)
    seed = int(config.get("seed") or 42)
    config_dir = config.get("_config_dir")
    summaries: dict[str, Any] = {}
    for dataset_cfg in datasets:
        examples = load_examples(
            dataset_cfg,
            max_examples=max_examples,
            seed=seed,
            config_dir=config_dir,
        )
        summaries[str(dataset_cfg.get("name"))] = summarize_examples(examples)
    return {"datasets": summaries}


def run_pipeline(config: dict[str, Any]) -> Path:
    seed = int(config.get("seed") or 42)
    max_examples = int(config.get("max_examples_per_dataset") or 1000)
    bootstrap_samples = int(config.get("bootstrap_samples") or 1000)
    run_root = Path(config.get("run_dir") or "runs")
    if not run_root.is_absolute():
        run_root = Path.cwd() / run_root
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    datasets = config.get("datasets") or []
    all_examples: list[Example] = []
    by_dataset: dict[str, list[Example]] = {}
    for dataset_cfg in datasets:
        examples = load_examples(
            dataset_cfg,
            max_examples=max_examples,
            seed=seed,
            config_dir=config.get("_config_dir"),
        )
        name = str(dataset_cfg.get("name") or "dataset")
        by_dataset[name] = examples
        all_examples.extend(examples)

    data_summary = {"datasets": {name: summarize_examples(examples) for name, examples in by_dataset.items()}}
    write_json(run_dir / "config.json", _public_config(config))
    write_json(run_dir / "data_summary.json", data_summary)
    write_jsonl(run_dir / "normalized_examples.jsonl", [example.to_dict() for example in all_examples])

    metrics: dict[str, Any] = {"datasets": {}}
    stressors: dict[str, Any] = {"datasets": {}}
    risk_flags: dict[str, Any] = {"datasets": {}}
    flagged_rows: list[dict[str, Any]] = []

    for name, examples in by_dataset.items():
        dataset_result = evaluate_dataset(
            examples,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
        )
        metrics["datasets"][name] = dataset_result["metrics"]
        stressors["datasets"][name] = dataset_result["stressors"]
        risk_flags["datasets"][name] = dataset_result["risk_flags"]
        flagged_rows.extend(dataset_result["flagged_examples"])

    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "stressors.json", stressors)
    write_json(run_dir / "risk_flags.json", risk_flags)
    write_jsonl(run_dir / "flagged_examples.jsonl", flagged_rows)
    generate_report(run_dir)
    return run_dir


def evaluate_dataset(
    examples: list[Example],
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if not examples:
        return {
            "metrics": {},
            "stressors": {
                "evidence_ablation": {},
                "keyword_shortcut": {"available": False, "reason": "no_examples"},
                "temporal_shift": {"available": False, "reason": "no_examples"},
                "ambiguity_slice": {"n": 0, "rate": 0.0},
                "label_rationale_mismatch": {
                    "available": False,
                    "reason": "no_examples",
                    "n_flagged": 0,
                    "rate": 0.0,
                },
                "coverage": {"test_evidence_coverage": 0.0, "test_date_coverage": 0.0},
            },
            "risk_flags": [
                {
                    "risk": "no_examples",
                    "severity": "high",
                    "evidence": "No examples were available after loading and normalization.",
                }
            ],
            "flagged_examples": [],
        }

    train, test = train_test_split(examples, seed=seed)
    test_evidence_coverage = _coverage(example.evidence_text for example in test)
    test_date_coverage = _coverage(example.date for example in test)
    baseline_bundle = fit_default_baselines(train, seed=seed)
    test_texts = make_text_sets(test)
    y_true = [example.label3 for example in test]

    predictions: dict[str, list[str]] = {}
    metrics: dict[str, Any] = {}
    for model_name, model, text_key in baseline_bundle:
        y_pred = model.predict(test_texts[text_key])
        predictions[model_name] = y_pred
        score = classification_metrics(y_true, y_pred)
        score["macro_f1_ci"] = bootstrap_metric_ci(
            y_true,
            y_pred,
            samples=bootstrap_samples,
            seed=seed,
        )
        metrics[model_name] = score

    stressors = {
        "evidence_ablation": _evidence_ablation(metrics, test_evidence_coverage),
        "keyword_shortcut": _keyword_shortcut(train, test, seed, bootstrap_samples),
        "temporal_shift": _temporal_shift(examples, seed, bootstrap_samples),
        "ambiguity_slice": _ambiguity_slice(test, predictions, y_true, bootstrap_samples, seed),
        "label_rationale_mismatch": {},
        "coverage": {
            "test_evidence_coverage": test_evidence_coverage,
            "test_date_coverage": test_date_coverage,
        },
    }
    if test_evidence_coverage < 0.1:
        flagged_examples = []
        stressors["label_rationale_mismatch"] = {
            "available": False,
            "reason": "low_evidence_coverage",
            "n_flagged": 0,
            "rate": 0.0,
        }
    else:
        mismatch_analysis = analyze_label_rationale_mismatches(test, limit=100)
        flagged_examples = mismatch_analysis["examples"]
        stressors["label_rationale_mismatch"] = {
            "available": True,
            "n_flagged": mismatch_analysis["n_flagged"],
            "n_scanned": mismatch_analysis["n_scanned"],
            "rate": mismatch_analysis["rate"],
            "examples_stored": mismatch_analysis["examples_stored"],
            "examples_truncated": mismatch_analysis["examples_truncated"],
        }
    risk_flags = _risk_flags(metrics, stressors)
    return {
        "metrics": metrics,
        "stressors": stressors,
        "risk_flags": risk_flags,
        "flagged_examples": flagged_examples,
    }


def fit_default_baselines(
    train: list[Example],
    *,
    seed: int,
) -> list[tuple[str, Any, str]]:
    train_texts = make_text_sets(train)
    labels = [example.label3 for example in train]
    baselines: list[tuple[str, Any, str]] = []

    majority = MajorityClassifier().fit(train_texts["claim"], labels)
    baselines.append(("majority", majority, "claim"))

    claim = TfidfLogRegClassifier(name="tfidf_logreg_claim", seed=seed).fit(
        train_texts["claim"], labels
    )
    baselines.append(("tfidf_logreg_claim", claim, "claim"))

    evidence = TfidfLogRegClassifier(name="tfidf_logreg_evidence", seed=seed).fit(
        train_texts["evidence"], labels
    )
    baselines.append(("tfidf_logreg_evidence", evidence, "evidence"))

    combined = TfidfLogRegClassifier(name="tfidf_logreg_claim_evidence", seed=seed).fit(
        train_texts["combined"], labels
    )
    baselines.append(("tfidf_logreg_claim_evidence", combined, "combined"))

    heuristic = HeuristicNliClassifier().fit(train_texts["combined"], labels)
    baselines.append(("heuristic_nli", heuristic, "combined"))
    return baselines


def make_text_sets(examples: list[Example]) -> dict[str, list[str]]:
    return {
        "claim": [example.claim for example in examples],
        "evidence": [example.evidence_text for example in examples],
        "combined": [
            f"Claim: {example.claim}\nEvidence: {example.evidence_text}" for example in examples
        ],
    }


def train_test_split(
    examples: list[Example],
    *,
    seed: int,
    test_fraction: float = 0.3,
) -> tuple[list[Example], list[Example]]:
    if len(examples) < 4:
        return examples, examples

    by_split: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        by_split[example.split.lower()].append(example)
    train = by_split.get("train", [])
    test = by_split.get("test", []) + by_split.get("validation", []) + by_split.get("valid", [])
    if len(train) >= 3 and len(test) >= 2:
        return train, test

    rng = random.Random(seed)
    by_label: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        by_label[example.label3].append(example)

    train_out: list[Example] = []
    test_out: list[Example] = []
    for label_examples in by_label.values():
        shuffled = list(label_examples)
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(shuffled) * test_fraction))) if len(shuffled) > 1 else 0
        test_out.extend(shuffled[:n_test])
        train_out.extend(shuffled[n_test:])

    if not train_out or not test_out:
        shuffled = list(examples)
        rng.shuffle(shuffled)
        cut = max(1, int(len(shuffled) * (1 - test_fraction)))
        train_out = shuffled[:cut]
        test_out = shuffled[cut:] or shuffled[:]
    return train_out, test_out


def summarize_examples(examples: list[Example]) -> dict[str, Any]:
    labels = Counter(example.label3 for example in examples)
    evidence_coverage = _coverage(example.evidence_text for example in examples)
    raw_evidence_coverage = _coverage(
        example.metadata.get("raw_evidence_text", "") for example in examples
    )
    date_coverage = _coverage(example.date for example in examples)
    source_url_coverage = _coverage(example.source_url for example in examples)
    evidence_quality = Counter(
        str(example.metadata.get("evidence_quality", "unknown")) for example in examples
    )
    warnings: list[str] = []
    if evidence_coverage < 0.1:
        warnings.append(
            "low_natural_language_evidence_coverage: evidence-ablation and label-rationale checks may be skipped"
        )
    if raw_evidence_coverage >= 0.1 and evidence_coverage < 0.1:
        warnings.append(
            "structured_reference_evidence: raw evidence exists but is not natural-language evidence"
        )
    if date_coverage < 0.4:
        warnings.append("low_date_coverage: temporal-shift stressor will likely be skipped")

    return {
        "n": len(examples),
        "labels": dict(labels),
        "splits": dict(Counter(example.split for example in examples)),
        "evidence_coverage": evidence_coverage,
        "raw_evidence_coverage": raw_evidence_coverage,
        "evidence_quality": dict(evidence_quality),
        "date_coverage": date_coverage,
        "source_url_coverage": source_url_coverage,
        "warnings": warnings,
    }


def _coverage(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def _evidence_ablation(metrics: dict[str, Any], evidence_coverage: float = 1.0) -> dict[str, Any]:
    claim = metrics.get("tfidf_logreg_claim", {}).get("macro_f1", 0.0)
    evidence = metrics.get("tfidf_logreg_evidence", {}).get("macro_f1", 0.0)
    combined = metrics.get("tfidf_logreg_claim_evidence", {}).get("macro_f1", 0.0)
    if evidence_coverage < 0.1:
        return {
            "available": False,
            "reason": "low_evidence_coverage",
            "evidence_coverage": evidence_coverage,
            "claim_macro_f1": claim,
            "evidence_macro_f1": evidence,
            "combined_macro_f1": combined,
            "combined_minus_claim": 0.0,
            "combined_minus_evidence": 0.0,
        }
    return {
        "available": True,
        "evidence_coverage": evidence_coverage,
        "claim_macro_f1": claim,
        "evidence_macro_f1": evidence,
        "combined_macro_f1": combined,
        "combined_minus_claim": combined - claim,
        "combined_minus_evidence": combined - evidence,
    }


def _keyword_shortcut(
    train: list[Example],
    test: list[Example],
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    keywords_by_label = top_label_keywords(train, top_k=20)
    keywords = {token for tokens in keywords_by_label.values() for token in tokens}
    if not test:
        return {"available": False, "macro_f1_drop": 0.0, "masked_keyword_count": len(keywords)}
    labels = [example.label3 for example in train]
    model = TfidfLogRegClassifier(name="keyword_shortcut_claim", seed=seed).fit(
        [example.claim for example in train], labels
    )
    y_true = [example.label3 for example in test]
    original_pred = model.predict([example.claim for example in test])
    masked_pred = model.predict([mask_keywords(example.claim, keywords) for example in test])
    original = classification_metrics(y_true, original_pred)
    masked = classification_metrics(y_true, masked_pred)
    return {
        "available": True,
        "keywords_by_label": keywords_by_label,
        "masked_keyword_count": len(keywords),
        "original_macro_f1": original["macro_f1"],
        "masked_macro_f1": masked["macro_f1"],
        "macro_f1_drop": original["macro_f1"] - masked["macro_f1"],
        "masked_macro_f1_ci": bootstrap_metric_ci(
            y_true,
            masked_pred,
            samples=bootstrap_samples,
            seed=seed,
        ),
    }


def _temporal_shift(
    examples: list[Example],
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    dated = [example for example in examples if example.date]
    if len(dated) < max(12, int(0.4 * len(examples))):
        return {"available": False, "reason": "too_few_dates", "macro_f1_drop": 0.0}

    random_train, random_test = train_test_split(dated, seed=seed)
    random_score = _single_combined_score(random_train, random_test, seed, bootstrap_samples)

    ordered = sorted(dated, key=lambda example: example.date)
    cut = max(1, int(0.7 * len(ordered)))
    temporal_train = ordered[:cut]
    temporal_test = ordered[cut:] or ordered[-max(1, int(0.3 * len(ordered))) :]
    temporal_score = _single_combined_score(temporal_train, temporal_test, seed, bootstrap_samples)
    return {
        "available": True,
        "random_macro_f1": random_score["macro_f1"],
        "temporal_macro_f1": temporal_score["macro_f1"],
        "macro_f1_drop": random_score["macro_f1"] - temporal_score["macro_f1"],
        "temporal_macro_f1_ci": temporal_score.get("macro_f1_ci", [0.0, 0.0]),
        "n_dated": len(dated),
    }


def _single_combined_score(
    train: list[Example],
    test: list[Example],
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    model = TfidfLogRegClassifier(name="tfidf_logreg_claim_evidence", seed=seed).fit(
        make_text_sets(train)["combined"],
        [example.label3 for example in train],
    )
    texts = make_text_sets(test)["combined"]
    y_true = [example.label3 for example in test]
    y_pred = model.predict(texts)
    score = classification_metrics(y_true, y_pred)
    score["macro_f1_ci"] = bootstrap_metric_ci(y_true, y_pred, samples=bootstrap_samples, seed=seed)
    return score


def _ambiguity_slice(
    test: list[Example],
    predictions: dict[str, list[str]],
    y_true: list[str],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    indices = ambiguity_indices(test)
    if not indices:
        return {
            "n": 0,
            "rate": 0.0,
            "model_scores": {},
            "ranking_correlation": {"spearman": 1.0, "kendall": 1.0},
        }
    model_scores: dict[str, float] = {}
    full_scores: dict[str, float] = {}
    delta_cis: dict[str, list[float]] = {}
    slice_true = [y_true[index] for index in indices]
    for model, y_pred in predictions.items():
        slice_pred = [y_pred[index] for index in indices]
        slice_score = classification_metrics(slice_true, slice_pred)
        full_score = classification_metrics(y_true, y_pred)
        model_scores[model] = slice_score["macro_f1"]
        full_scores[model] = full_score["macro_f1"]
        delta_cis[model] = bootstrap_delta_ci(
            y_true,
            y_pred,
            indices,
            samples=bootstrap_samples,
            seed=seed,
        )
    return {
        "n": len(indices),
        "rate": len(indices) / len(test) if test else 0.0,
        "model_scores": model_scores,
        "delta_ci": delta_cis,
        "ranking_correlation": ranking_correlations(full_scores, model_scores),
    }


def _risk_flags(metrics: dict[str, Any], stressors: dict[str, Any]) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    evidence = stressors["evidence_ablation"]
    keyword = stressors["keyword_shortcut"]
    temporal = stressors["temporal_shift"]
    ambiguity = stressors["ambiguity_slice"]
    mismatch = stressors["label_rationale_mismatch"]
    coverage = stressors.get("coverage", {})
    evidence_coverage = coverage.get("test_evidence_coverage", 0.0)

    claim = evidence.get("claim_macro_f1", 0.0)
    combined = evidence.get("combined_macro_f1", 0.0)
    evidence_gain = combined - claim
    if evidence_coverage >= 0.1 and combined > 0 and evidence_gain <= -0.05:
        flags.append(
            {
                "risk": "evidence_hurts_risk",
                "severity": "high",
                "evidence": f"Claim-plus-evidence macro-F1 ({combined:.3f}) is {abs(evidence_gain):.3f} below claim-only ({claim:.3f}).",
            }
        )
    elif evidence_coverage >= 0.1 and combined > 0 and abs(evidence_gain) < 0.03:
        flags.append(
            {
                "risk": "evidence_low_utility_risk",
                "severity": "medium",
                "evidence": f"Claim-plus-evidence macro-F1 ({combined:.3f}) is within 0.03 of claim-only ({claim:.3f}).",
            }
        )

    if keyword.get("macro_f1_drop", 0.0) >= 0.05:
        flags.append(
            {
                "risk": "shortcut_risk",
                "severity": "high",
                "evidence": f"Masking label-associated tokens dropped macro-F1 by {keyword['macro_f1_drop']:.3f}.",
            }
        )

    if temporal.get("available") and temporal.get("macro_f1_drop", 0.0) >= 0.05:
        flags.append(
            {
                "risk": "temporal_risk",
                "severity": "high",
                "evidence": f"Date-aware evaluation dropped macro-F1 by {temporal['macro_f1_drop']:.3f}.",
            }
        )

    rank = ambiguity.get("ranking_correlation", {})
    if ambiguity.get("n", 0) and rank.get("spearman", 1.0) < 0.7:
        flags.append(
            {
                "risk": "ambiguity_ranking_risk",
                "severity": "medium",
                "evidence": f"Ambiguity-slice Spearman ranking correlation is {rank.get('spearman', 0.0):.3f}.",
            }
        )

    if mismatch.get("available", True) and mismatch.get("rate", 0.0) >= 0.1:
        flags.append(
            {
                "risk": "label_rationale_mismatch_risk",
                "severity": "medium",
                "evidence": f"{mismatch.get('rate', 0.0):.1%} of test examples were heuristic-flagged.",
            }
        )

    if not flags:
        flags.append(
            {
                "risk": "null_result",
                "severity": "info",
                "evidence": "No configured risk threshold was crossed.",
            }
        )
    return flags


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}
