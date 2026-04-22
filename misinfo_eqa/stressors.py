from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any

from .baselines import HeuristicNliClassifier
from .schema import Example
from .text import STOPWORDS, TOKEN_RE, tokenize


HEDGE_TERMS = {
    "allegedly",
    "apparently",
    "claimed",
    "could",
    "disputed",
    "may",
    "might",
    "mixed",
    "possibly",
    "reportedly",
    "rumor",
    "rumour",
    "unclear",
    "uncertain",
    "unconfirmed",
    "unknown",
    "unverified",
}

LOW_RELEVANCE_THRESHOLD = 0.18


def top_label_keywords(examples: list[Example], *, top_k: int = 30) -> dict[str, list[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    for example in examples:
        unique_tokens = set(tokenize(example.claim, keep_stopwords=False))
        for token in unique_tokens:
            if token in STOPWORDS or len(token) <= 3:
                continue
            counts[example.label3][token] += 1
            totals[token] += 1

    result: dict[str, list[str]] = {}
    for label, label_counts in counts.items():
        scored: list[tuple[float, str]] = []
        for token, count in label_counts.items():
            other = totals[token] - count
            score = count / (1 + other)
            if count >= 2:
                scored.append((score, token))
        scored.sort(key=lambda item: (-item[0], item[1]))
        result[label] = [token for _, token in scored[:top_k]]
    return result


def mask_keywords(text: str, keywords: set[str]) -> str:
    if not keywords:
        return text

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        return "[MASK]" if token.lower() in keywords else token

    return TOKEN_RE.sub(replace, text or "")


def is_ambiguous(example: Example) -> bool:
    if example.label3 == "unknown":
        return True
    text = f"{example.claim} {example.evidence_text}".lower()
    return any(term in text for term in HEDGE_TERMS)


def ambiguity_indices(examples: list[Example]) -> list[int]:
    return [idx for idx, example in enumerate(examples) if is_ambiguous(example)]


def flag_label_rationale_mismatches(examples: list[Example], *, limit: int = 100) -> list[dict[str, Any]]:
    return analyze_label_rationale_mismatches(examples, limit=limit)["examples"]


def analyze_label_rationale_mismatches(
    examples: list[Example],
    *,
    limit: int = 100,
) -> dict[str, Any]:
    classifier = HeuristicNliClassifier()
    flagged: list[dict[str, Any]] = []
    n_flagged = 0
    for example in examples:
        reasons: list[str] = []
        evidence = example.evidence_text
        if not evidence:
            reasons.append("missing_evidence")
        else:
            relevance = evidence_relevance(example.claim, evidence)
            if relevance["score"] < LOW_RELEVANCE_THRESHOLD:
                reasons.append("low_claim_evidence_relevance")
            heuristic = classifier.predict_one(f"{example.claim} {evidence}")
            if (
                heuristic != "unknown"
                and heuristic != example.label3
                and _should_flag_heuristic_disagreement(evidence, relevance)
            ):
                reasons.append(f"heuristic_label_disagrees:{heuristic}")
        relevance = evidence_relevance(example.claim, evidence) if evidence else {}

        if reasons:
            n_flagged += 1
            if len(flagged) < limit:
                flagged.append(
                    {
                        "id": example.id,
                        "dataset": example.dataset,
                        "label3": example.label3,
                        "claim": example.claim,
                        "evidence_text": evidence[:500],
                        "reasons": reasons,
                        "relevance_score": relevance.get("score", 0.0),
                        "best_evidence_span": relevance.get("best_span", "")[:500],
                        "source_url": example.source_url,
                    }
                )
    n_scanned = len(examples)
    return {
        "examples": flagged,
        "n_flagged": n_flagged,
        "n_scanned": n_scanned,
        "rate": n_flagged / n_scanned if n_scanned else 0.0,
        "examples_stored": len(flagged),
        "examples_truncated": n_flagged > len(flagged),
    }


def token_overlap(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def evidence_relevance(claim: str, evidence: str) -> dict[str, Any]:
    claim_tokens = set(tokenize(claim))
    evidence_tokens = set(tokenize(evidence))
    if not claim_tokens or not evidence_tokens:
        return {
            "score": 0.0,
            "global_jaccard": 0.0,
            "best_claim_coverage": 0.0,
            "best_jaccard": 0.0,
            "best_span": "",
        }

    global_jaccard = len(claim_tokens & evidence_tokens) / len(claim_tokens | evidence_tokens)
    best_score = 0.0
    best_coverage = 0.0
    best_jaccard = 0.0
    best_span = ""

    for span in evidence_spans(evidence):
        span_tokens = set(tokenize(span))
        if not span_tokens:
            continue
        overlap = claim_tokens & span_tokens
        coverage = len(overlap) / len(claim_tokens)
        jaccard = len(overlap) / len(claim_tokens | span_tokens)
        score = max(jaccard, 0.7 * coverage + 0.3 * jaccard)
        if score > best_score:
            best_score = score
            best_coverage = coverage
            best_jaccard = jaccard
            best_span = span

    return {
        "score": max(global_jaccard, best_score),
        "global_jaccard": global_jaccard,
        "best_claim_coverage": best_coverage,
        "best_jaccard": best_jaccard,
        "best_span": best_span,
    }


def evidence_spans(evidence: str) -> list[str]:
    text = (evidence or "").strip()
    if not text:
        return []

    sentence_spans = [
        span.strip()
        for span in re.split(r"(?<=[.!?])\s+", text)
        if span.strip()
    ]
    spans = sentence_spans or [text]

    tokens = text.split()
    if len(tokens) > 80:
        window = 45
        stride = 25
        for start in range(0, len(tokens), stride):
            chunk = tokens[start : start + window]
            if len(chunk) >= 8:
                spans.append(" ".join(chunk))
            if start + window >= len(tokens):
                break
    return spans


def _should_flag_heuristic_disagreement(
    evidence: str,
    relevance: dict[str, Any],
) -> bool:
    if relevance.get("score", 0.0) < LOW_RELEVANCE_THRESHOLD:
        return True
    return len(tokenize(evidence, keep_stopwords=True)) <= 80
