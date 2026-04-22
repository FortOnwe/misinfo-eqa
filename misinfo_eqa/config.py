from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text)
        except ModuleNotFoundError:
            data = _simple_yaml_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    data["_config_path"] = str(config_path)
    data["_config_dir"] = str(config_path.parent)
    return data


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Load the small YAML subset used by this project's configs.

    This fallback intentionally supports only top-level scalars, inline scalar
    lists, scalar lists, and lists of shallow mappings. Install PyYAML for full
    YAML support.
    """

    result: dict[str, Any] = {}
    current_key: str | None = None
    current_item: dict[str, Any] | None = None

    for raw_line in text.splitlines():
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if indent == 0:
            current_key = None
            current_item = None
            if stripped.endswith(":"):
                key = stripped[:-1].strip()
                result[key] = []
                current_key = key
                continue
            if ":" not in stripped:
                raise ValueError(f"Unsupported config line: {raw_line}")
            key, value = stripped.split(":", 1)
            result[key.strip()] = _parse_scalar(value.strip())
            continue

        if current_key is None:
            raise ValueError(f"Indented line has no parent key: {raw_line}")

        if indent == 2 and stripped.startswith("- "):
            item_text = stripped[2:].strip()
            if ":" in item_text and not item_text.startswith(("'", '"')):
                key, value = item_text.split(":", 1)
                current_item = {key.strip(): _parse_scalar(value.strip())}
                result[current_key].append(current_item)
            else:
                current_item = None
                result[current_key].append(_parse_scalar(item_text))
            continue

        if indent >= 4 and current_item is not None and ":" in stripped:
            key, value = stripped.split(":", 1)
            current_item[key.strip()] = _parse_scalar(value.strip())
            continue

        raise ValueError(f"Unsupported config line: {raw_line}")

    return result


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
