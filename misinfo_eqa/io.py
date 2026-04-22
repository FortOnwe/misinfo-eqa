from __future__ import annotations

import csv
import json
from pathlib import Path
import random
from typing import Any, Iterable

from .schema import Example, coerce_example


def load_examples(
    dataset_cfg: dict[str, Any],
    *,
    max_examples: int | None,
    seed: int,
    config_dir: str | Path | None = None,
) -> list[Example]:
    source = str(dataset_cfg.get("source", "local")).lower()
    dataset_name = str(dataset_cfg.get("name") or dataset_cfg.get("subset") or "dataset")

    if source in {"local", "csv", "jsonl", "local_csv", "local_jsonl"}:
        rows = list(_load_local_rows(dataset_cfg, config_dir))
    elif source in {"hf", "huggingface"}:
        rows = list(_load_huggingface_rows(dataset_cfg))
    else:
        raise ValueError(f"Unsupported dataset source for {dataset_name}: {source}")

    examples = [
        coerce_example(row, dataset_name, idx)
        for idx, row in enumerate(rows)
        if isinstance(row, dict)
    ]
    examples = [example for example in examples if example.claim and example.label3]

    if max_examples and len(examples) > max_examples:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(examples)), max_examples))
        examples = [examples[index] for index in indices]
    return examples


def write_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_local_rows(
    dataset_cfg: dict[str, Any],
    config_dir: str | Path | None,
) -> Iterable[dict[str, Any]]:
    raw_path = dataset_cfg.get("path")
    if not raw_path:
        raise ValueError(f"Local dataset {dataset_cfg.get('name')} requires a path")
    path = Path(str(raw_path))
    if not path.is_absolute() and config_dir:
        path = Path(config_dir) / path

    files: list[Path]
    if path.is_dir():
        files = sorted(
            file
            for file in path.iterdir()
            if file.suffix.lower() in {".csv", ".jsonl", ".json"}
        )
    else:
        files = [path]

    for file in files:
        suffix = file.suffix.lower()
        if suffix == ".csv":
            yield from _iter_csv(file)
        elif suffix in {".jsonl", ".json"}:
            yield from _iter_json(file)
        else:
            raise ValueError(f"Unsupported local file type: {file}")


def _iter_csv(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield dict(row)


def _iter_json(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        else:
            data = json.load(handle)
            if isinstance(data, list):
                yield from data
            elif isinstance(data, dict) and isinstance(data.get("data"), list):
                yield from data["data"]
            else:
                raise ValueError(f"JSON dataset must be a list or contain data list: {path}")


def _load_huggingface_rows(dataset_cfg: dict[str, Any]) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Hugging Face loading requires optional dependencies. "
            "Run: python -m pip install -e \".[full]\""
        ) from exc

    hf_name = str(dataset_cfg.get("hf_name") or "ComplexDataLab/Misinfo_Datasets")
    subset = dataset_cfg.get("subset") or dataset_cfg.get("name")
    splits = dataset_cfg.get("splits") or ["train", "validation", "test"]
    if isinstance(splits, str):
        splits = [splits]

    loaded_any = False
    errors: list[str] = []
    for split in splits:
        try:
            dataset = load_dataset(hf_name, str(subset), split=str(split))
        except Exception as exc:
            errors.append(f"{split}: {exc}")
            continue
        for row in dataset:
            loaded_any = True
            record = dict(row)
            record.setdefault("split", split)
            yield record
    if not loaded_any:
        detail = "; ".join(errors) if errors else "no rows returned"
        raise RuntimeError(f"No Hugging Face rows loaded for {hf_name}/{subset}: {detail}")
