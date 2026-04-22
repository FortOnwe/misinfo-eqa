from __future__ import annotations

import hashlib
import math
import re


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]*", re.IGNORECASE)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    tokens = [match.group(0).lower() for match in TOKEN_RE.finditer(text or "")]
    if keep_stopwords:
        return tokens
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def stable_hash(token: str, buckets: int) -> int:
    digest = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(digest[:12], 16) % buckets


def l2_normalize(features: dict[int, float]) -> dict[int, float]:
    norm = math.sqrt(sum(value * value for value in features.values()))
    if norm <= 0:
        return features
    return {key: value / norm for key, value in features.items()}
