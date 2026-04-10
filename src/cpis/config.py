"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit("Missing dependency 'pyyaml'. Install from environment.yml.") from exc


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Config file must parse to a dict: {p}")
    return data

