from __future__ import annotations

from collections import Counter, defaultdict
import math
import random
from typing import Protocol

from .schema import LABELS
from .text import l2_normalize, stable_hash, tokenize


class Classifier(Protocol):
    name: str

    def fit(self, texts: list[str], labels: list[str]) -> "Classifier":
        ...

    def predict(self, texts: list[str]) -> list[str]:
        ...


class MajorityClassifier:
    name = "majority"

    def __init__(self) -> None:
        self.label = "unknown"

    def fit(self, texts: list[str], labels: list[str]) -> "MajorityClassifier":
        counts = Counter(labels)
        self.label = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return [self.label for _ in texts]


class TfidfLogRegClassifier:
    """Small dependency-free TF-IDF plus softmax logistic regression baseline."""

    def __init__(
        self,
        *,
        name: str,
        buckets: int = 4096,
        epochs: int = 24,
        learning_rate: float = 0.35,
        seed: int = 42,
    ) -> None:
        self.name = name
        self.buckets = buckets
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.labels = list(LABELS)
        self.idf = [1.0 for _ in range(buckets)]
        self.weights = [[0.0 for _ in range(buckets)] for _ in self.labels]
        self.bias = [0.0 for _ in self.labels]
        self.fallback = MajorityClassifier()

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfLogRegClassifier":
        self.fallback.fit(texts, labels)
        if len(set(labels)) < 2:
            return self

        raw_features = [self._tf(text) for text in texts]
        doc_freq = [0 for _ in range(self.buckets)]
        for features in raw_features:
            for index in features:
                doc_freq[index] += 1
        n_docs = max(1, len(texts))
        self.idf = [math.log((1 + n_docs) / (1 + df)) + 1 for df in doc_freq]
        features = [self._tfidf_from_tf(tf) for tf in raw_features]

        label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        rng = random.Random(self.seed)
        indices = list(range(len(features)))
        for epoch in range(self.epochs):
            rng.shuffle(indices)
            lr = self.learning_rate / math.sqrt(epoch + 1)
            for row_index in indices:
                x = features[row_index]
                target = label_to_index.get(labels[row_index], label_to_index["unknown"])
                probs = self._predict_proba_one(x)
                for class_index, prob in enumerate(probs):
                    error = prob - (1.0 if class_index == target else 0.0)
                    if error == 0:
                        continue
                    for feature_index, value in x.items():
                        self.weights[class_index][feature_index] -= lr * error * value
                    self.bias[class_index] -= lr * error
        return self

    def predict(self, texts: list[str]) -> list[str]:
        predictions: list[str] = []
        for text in texts:
            features = self._tfidf_from_tf(self._tf(text))
            if not features:
                predictions.append(self.fallback.label)
                continue
            probs = self._predict_proba_one(features)
            best = max(range(len(probs)), key=lambda index: probs[index])
            predictions.append(self.labels[best])
        return predictions

    def _tf(self, text: str) -> dict[int, float]:
        counts: dict[int, float] = defaultdict(float)
        for token in tokenize(text):
            counts[stable_hash(token, self.buckets)] += 1.0
        if not counts:
            return {}
        total = sum(counts.values())
        return {index: count / total for index, count in counts.items()}

    def _tfidf_from_tf(self, tf: dict[int, float]) -> dict[int, float]:
        return l2_normalize({index: value * self.idf[index] for index, value in tf.items()})

    def _predict_proba_one(self, features: dict[int, float]) -> list[float]:
        logits: list[float] = []
        for class_index in range(len(self.labels)):
            score = self.bias[class_index]
            weights = self.weights[class_index]
            for feature_index, value in features.items():
                score += weights[feature_index] * value
            logits.append(score)
        max_logit = max(logits)
        exps = [math.exp(logit - max_logit) for logit in logits]
        total = sum(exps)
        return [value / total for value in exps]


class HeuristicNliClassifier:
    name = "heuristic_nli"

    FALSE_CUES = {
        "false",
        "fake",
        "misleading",
        "incorrect",
        "debunked",
        "refuted",
        "no evidence",
        "not true",
        "hoax",
    }
    TRUE_CUES = {
        "confirmed",
        "accurate",
        "true",
        "verified",
        "evidence shows",
        "according to",
        "official",
    }

    def fit(self, texts: list[str], labels: list[str]) -> "HeuristicNliClassifier":
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return [self.predict_one(text) for text in texts]

    def predict_one(self, text: str) -> str:
        lowered = (text or "").lower()
        false_score = sum(1 for cue in self.FALSE_CUES if cue in lowered)
        true_score = sum(1 for cue in self.TRUE_CUES if cue in lowered)
        if false_score > true_score:
            return "false"
        if true_score > false_score:
            return "true"
        return "unknown"
