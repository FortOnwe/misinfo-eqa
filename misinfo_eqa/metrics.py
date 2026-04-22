from __future__ import annotations

from collections import Counter
import math
import random
from typing import Any

from .schema import LABELS


def classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: tuple[str, ...] = LABELS,
) -> dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return {
            "n": 0,
            "macro_f1": 0.0,
            "balanced_accuracy": 0.0,
            "accuracy": 0.0,
            "per_label_f1": {label: 0.0 for label in labels},
        }

    per_label_f1: dict[str, float] = {}
    recalls: list[float] = []
    for label in labels:
        tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label_f1[label] = f1
        if any(truth == label for truth in y_true):
            recalls.append(recall)

    accuracy = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred) / len(y_true)
    return {
        "n": len(y_true),
        "macro_f1": sum(per_label_f1.values()) / len(labels),
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else 0.0,
        "accuracy": accuracy,
        "per_label_f1": per_label_f1,
        "label_support": dict(Counter(y_true)),
    }


def bootstrap_metric_ci(
    y_true: list[str],
    y_pred: list[str],
    *,
    metric: str = "macro_f1",
    samples: int = 1000,
    seed: int = 42,
) -> list[float]:
    if not y_true:
        return [0.0, 0.0]
    rng = random.Random(seed)
    values: list[float] = []
    n = len(y_true)
    for _ in range(max(1, samples)):
        indices = [rng.randrange(n) for _ in range(n)]
        sampled_true = [y_true[index] for index in indices]
        sampled_pred = [y_pred[index] for index in indices]
        values.append(float(classification_metrics(sampled_true, sampled_pred)[metric]))
    values.sort()
    lower_idx = min(len(values) - 1, max(0, int(0.025 * len(values))))
    upper_idx = min(len(values) - 1, max(0, int(0.975 * len(values))))
    return [values[lower_idx], values[upper_idx]]


def bootstrap_delta_ci(
    full_true: list[str],
    full_pred: list[str],
    slice_indices: list[int],
    *,
    metric: str = "macro_f1",
    samples: int = 1000,
    seed: int = 42,
) -> list[float]:
    if not full_true or not slice_indices:
        return [0.0, 0.0]
    rng = random.Random(seed)
    values: list[float] = []
    n = len(full_true)
    for _ in range(max(1, samples)):
        full_indices = [rng.randrange(n) for _ in range(n)]
        sampled_full_true = [full_true[index] for index in full_indices]
        sampled_full_pred = [full_pred[index] for index in full_indices]
        sampled_slice_indices = [rng.choice(slice_indices) for _ in range(len(slice_indices))]
        sampled_slice_true = [full_true[index] for index in sampled_slice_indices]
        sampled_slice_pred = [full_pred[index] for index in sampled_slice_indices]
        full_score = float(classification_metrics(sampled_full_true, sampled_full_pred)[metric])
        slice_score = float(classification_metrics(sampled_slice_true, sampled_slice_pred)[metric])
        values.append(slice_score - full_score)
    values.sort()
    lower_idx = min(len(values) - 1, max(0, int(0.025 * len(values))))
    upper_idx = min(len(values) - 1, max(0, int(0.975 * len(values))))
    return [values[lower_idx], values[upper_idx]]


def ranking_correlations(full_scores: dict[str, float], slice_scores: dict[str, float]) -> dict[str, float]:
    common = sorted(set(full_scores) & set(slice_scores))
    if len(common) < 2:
        return {"spearman": 1.0, "kendall": 1.0}
    full_ranks = _ranks({name: full_scores[name] for name in common})
    slice_ranks = _ranks({name: slice_scores[name] for name in common})
    xs = [full_ranks[name] for name in common]
    ys = [slice_ranks[name] for name in common]
    return {"spearman": _pearson(xs, ys), "kendall": _kendall(xs, ys)}


def _ranks(scores: dict[str, float]) -> dict[str, float]:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[str, float] = {}
    for idx, (name, _) in enumerate(ordered, start=1):
        ranks[name] = float(idx)
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def _kendall(xs: list[float], ys: list[float]) -> float:
    concordant = 0
    discordant = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            product = dx * dy
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0
