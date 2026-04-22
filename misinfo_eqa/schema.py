from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import re
from typing import Any


LABELS = ("true", "false", "unknown")

NULL_VALUES = {"", "na", "n/a", "none", "null", "nan", "unknown/na"}

TRUE_LABELS = {
    "true",
    "real",
    "mostly true",
    "mostly-true",
    "correct",
    "accurate",
    "supported",
    "supports",
    "entailment",
    "non-rumor",
    "nonrumor",
    "not fake",
}

FALSE_LABELS = {
    "false",
    "fake",
    "pants fire",
    "pants-fire",
    "barely true",
    "barely-true",
    "misinformation",
    "disinformation",
    "refuted",
    "refutes",
    "contradiction",
    "rumor",
    "rumour",
    "incorrect",
    "misleading",
}

UNKNOWN_LABELS = {
    "unknown",
    "unverified",
    "mixed",
    "mixture",
    "half true",
    "half-true",
    "partly false",
    "partly true",
    "uncertain",
    "not enough info",
    "not enough information",
    "nei",
    "neutral",
}

CLAIM_FIELDS = (
    "claim",
    "initial_claim",
    "statement",
    "text",
    "tweet_text",
    "social_media_text",
    "article_title",
    "article_headline",
    "thread_title",
    "title",
    "question",
    "query",
    "Query",
)

EVIDENCE_FIELDS = (
    "positive_evidence_text",
    "evidence",
    "evidence_text",
    "evidence_1",
    "evidence_2",
    "evidence_3",
    "evidence_4",
    "evidence_5",
    "article_content",
    "article",
    "summary",
    "answer",
    "reviewBody",
)

LABEL_FIELDS = (
    "veracity",
    "label3",
    "label",
    "binary_label",
    "tweet_label",
    "three_label",
    "six_label",
    "evidence_label",
    "pred_label",
)

DATE_FIELDS = (
    "date",
    "published_at",
    "created_at",
    "inserted_at",
    "updated_at",
    "scraped_at",
    "meta_data_date",
)

URL_FIELDS = (
    "source_url",
    "fact_check_url",
    "url",
    "link",
    "web",
    "snopes_url",
    "true_url",
    "false_url",
    "link_evidence_1",
)


@dataclass(frozen=True)
class Example:
    id: str
    dataset: str
    split: str
    claim: str
    evidence_text: str
    label3: str
    date: str
    source_url: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_label(value: Any) -> str:
    raw = clean_text(value).lower()
    if raw in NULL_VALUES:
        return "unknown"

    normalized = re.sub(r"[_/]+", " ", raw)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if normalized in TRUE_LABELS:
        return "true"
    if normalized in FALSE_LABELS:
        return "false"
    if normalized in UNKNOWN_LABELS:
        return "unknown"

    if normalized in {"1", "yes"}:
        return "true"
    if normalized in {"0", "no"}:
        return "false"
    if normalized in {"2", "3"}:
        return "unknown"

    if "false" in normalized or "fake" in normalized or "mislead" in normalized:
        return "false"
    if "true" in normalized or "real" in normalized or "support" in normalized:
        return "true"
    return "unknown"


def coerce_example(row: dict[str, Any], dataset: str, index: int) -> Example:
    split = clean_text(row.get("split")) or "unspecified"
    label = normalize_label(first_value(row, LABEL_FIELDS))
    claim = first_value(row, CLAIM_FIELDS)
    raw_evidence = clean_text(join_values(row, EVIDENCE_FIELDS))
    evidence_quality = evidence_text_quality(raw_evidence)
    evidence = raw_evidence if evidence_quality == "natural_language" else ""
    date = parse_date(first_value(row, DATE_FIELDS), row)
    source_url = first_value(row, URL_FIELDS)
    row_id = (
        clean_text(row.get("id"))
        or clean_text(row.get("example_id"))
        or clean_text(row.get("tweet_id"))
        or f"{dataset}-{index}"
    )

    return Example(
        id=row_id,
        dataset=dataset,
        split=split,
        claim=clean_text(claim),
        evidence_text=clean_text(evidence),
        label3=label,
        date=date,
        source_url=clean_text(source_url),
        metadata={
            **{k: v for k, v in row.items() if k not in {"claim", "evidence_text"}},
            "raw_evidence_text": raw_evidence,
            "evidence_quality": evidence_quality,
        },
    )


def first_value(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in row:
            value = clean_text(row.get(key))
            if value and value.lower() not in NULL_VALUES:
                return value
    return ""


def join_values(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key not in row:
            continue
        value = clean_text(row.get(key))
        if not value or value.lower() in NULL_VALUES or value in seen:
            continue
        values.append(value)
        seen.add(value)
    return " ".join(values)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in NULL_VALUES:
        return ""
    return re.sub(r"\s+", " ", text)


def evidence_text_quality(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return "missing"
    lowered = text.lower()
    if _looks_like_r_reference_vector(lowered):
        return "structured_reference"

    alpha_chars = sum(1 for char in text if char.isalpha())
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    if alpha_chars < 25 or len(tokens) < 5:
        return "too_short_or_nonlinguistic"

    identifier_tokens = sum(
        1
        for token in re.findall(r"[A-Za-z0-9_-]+", text)
        if "_" in token or token.isupper() or token.isdigit()
    )
    all_tokens = re.findall(r"[A-Za-z0-9_-]+", text)
    if all_tokens and identifier_tokens / len(all_tokens) > 0.65:
        return "structured_reference"
    return "natural_language"


def _looks_like_r_reference_vector(lowered: str) -> bool:
    stripped = lowered.strip()
    if stripped.startswith("c(") or stripped.startswith("list(c("):
        return True
    if stripped.startswith("list(") and " c(" in stripped:
        return True
    return False


def parse_date(value: Any, row: dict[str, Any] | None = None) -> str:
    text = clean_text(value)
    if not text and row:
        year = clean_text(row.get("year"))
        month = clean_text(row.get("month")) or "1"
        day = clean_text(row.get("day")) or "1"
        if year:
            text = f"{year}-{month}-{day}"
    if not text:
        return ""

    text = text.replace("/", "-")
    candidates = [
        "%Y-%m-%d",
        "%Y-%m",
        "%Y",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in candidates:
        try:
            dt = datetime.strptime(text[:19], fmt)
            return dt.date().isoformat()
        except ValueError:
            continue

    match = re.search(r"(19|20)\d{2}", text)
    if match:
        return f"{match.group(0)}-01-01"
    return ""
