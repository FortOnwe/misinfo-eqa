from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from .schema import EVIDENCE_FIELDS, clean_text


@dataclass(frozen=True)
class EvidenceResolution:
    status: str
    evidence_text: str
    raw_evidence_text: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "status": self.status,
            "evidence_text": self.evidence_text,
            "raw_evidence_text": self.raw_evidence_text,
            "reason": self.reason,
        }


class EvidenceResolver:
    def resolve(self, dataset: str, row: dict[str, Any]) -> EvidenceResolution | None:
        return None


class FeverEvidenceResolver(EvidenceResolver):
    def __init__(self, evidence_index: dict[str, dict[int, str]] | None = None):
        self._evidence_index = evidence_index

    def resolve(self, dataset: str, row: dict[str, Any]) -> EvidenceResolution | None:
        raw_evidence_text = _raw_evidence_text(row)
        if not raw_evidence_text:
            return None
        if "fever" not in dataset.lower():
            return None

        references = _parse_fever_references(row.get("evidence"))
        if not references:
            if _looks_structured(raw_evidence_text):
                return EvidenceResolution(
                    status="structured_only",
                    evidence_text="",
                    raw_evidence_text=raw_evidence_text,
                    reason="structured_reference_unparseable",
                )
            return None

        if not self._evidence_index:
            return EvidenceResolution(
                status="unavailable",
                evidence_text="",
                raw_evidence_text=raw_evidence_text,
                reason="resolver_not_configured",
            )

        resolved_sentences: list[str] = []
        for page, sentence_id in references:
            page_index = self._evidence_index.get(page)
            if not page_index:
                continue
            sentence = clean_text(page_index.get(sentence_id))
            if sentence:
                resolved_sentences.append(sentence)

        if not resolved_sentences:
            return EvidenceResolution(
                status="unavailable",
                evidence_text="",
                raw_evidence_text=raw_evidence_text,
                reason="references_unresolved",
            )

        reason = "resolved_fever_references"
        if len(resolved_sentences) < len(references):
            reason = "partially_resolved_fever_references"
        return EvidenceResolution(
            status="resolved",
            evidence_text=" ".join(dict.fromkeys(resolved_sentences)),
            raw_evidence_text=raw_evidence_text,
            reason=reason,
        )


def build_evidence_resolver(
    dataset_cfg: dict[str, Any],
    *,
    config_dir: str | Path | None = None,
) -> EvidenceResolver:
    resolver_cfg = dataset_cfg.get("evidence_resolver")
    if not isinstance(resolver_cfg, dict):
        if "fever" in str(dataset_cfg.get("name", "")).lower():
            return FeverEvidenceResolver()
        return EvidenceResolver()

    resolver_type = str(resolver_cfg.get("type") or "").lower()
    if resolver_type not in {"fever_local", "fever"}:
        return EvidenceResolver()

    index_path = resolver_cfg.get("path")
    evidence_index: dict[str, dict[int, str]] | None = None
    if index_path:
        path = Path(str(index_path))
        if not path.is_absolute() and config_dir:
            path = Path(config_dir) / path
        evidence_index = _load_fever_index(path)
    return FeverEvidenceResolver(evidence_index)


def _raw_evidence_text(row: dict[str, Any]) -> str:
    values: list[str] = []
    for field in EVIDENCE_FIELDS:
        if field not in row:
            continue
        cleaned = clean_text(row.get(field))
        if cleaned:
            values.append(cleaned)
    return " ".join(dict.fromkeys(values))


def _parse_fever_references(value: Any) -> list[tuple[str, int]]:
    if value is None:
        return []
    references: list[tuple[str, int]] = []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            references.extend(_parse_fever_references(parsed))
        references.extend(_parse_references_from_text(text))
        return _dedupe_references(references)

    if isinstance(value, list):
        if len(value) >= 2 and not isinstance(value[0], (list, tuple, dict)):
            page = clean_text(value[0])
            sentence = _safe_int(value[1])
            if page and sentence is not None:
                return [(page, sentence)]
        for item in value:
            references.extend(_parse_fever_references(item))
        return _dedupe_references(references)

    if isinstance(value, tuple) and len(value) >= 2:
        page = clean_text(value[0])
        sentence = _safe_int(value[1])
        if page and sentence is not None:
            references.append((page, sentence))
        return _dedupe_references(references)

    if isinstance(value, dict):
        page = clean_text(value.get("page") or value.get("title") or value.get("document"))
        sentence = _safe_int(value.get("sentence") or value.get("line") or value.get("sentence_id"))
        if page and sentence is not None:
            references.append((page, sentence))
        return _dedupe_references(references)

    return []


def _parse_references_from_text(text: str) -> list[tuple[str, int]]:
    refs: list[tuple[str, int]] = []
    tuple_pattern = re.compile(
        r"\[\s*['\"](?P<page>[^'\"]+)['\"]\s*,\s*(?P<sent>\d+)\s*[,|\]]"
    )
    for match in tuple_pattern.finditer(text):
        refs.append((match.group("page"), int(match.group("sent"))))
    return refs


def _dedupe_references(references: list[tuple[str, int]]) -> list[tuple[str, int]]:
    seen: set[tuple[str, int]] = set()
    unique: list[tuple[str, int]] = []
    for ref in references:
        if ref in seen:
            continue
        seen.add(ref)
        unique.append(ref)
    return unique


def _safe_int(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _load_fever_index(path: Path) -> dict[str, dict[int, str]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    index: dict[str, dict[int, str]] = {}
    if isinstance(raw, dict):
        for page, sentences in raw.items():
            if not isinstance(sentences, dict):
                continue
            normalized: dict[int, str] = {}
            for sentence_id, sentence in sentences.items():
                sid = _safe_int(sentence_id)
                if sid is None:
                    continue
                normalized[sid] = clean_text(sentence)
            if normalized:
                index[str(page)] = normalized
    return index


def _looks_structured(text: str) -> bool:
    lowered = text.lower()
    if lowered.startswith("c(") or lowered.startswith("list("):
        return True
    return bool(re.search(r"\[\s*['\"][^'\"]+['\"]\s*,\s*\d+", text))
