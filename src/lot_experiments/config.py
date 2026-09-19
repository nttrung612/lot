"""YAML configuration loading with deterministic resolution."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration cannot be resolved safely."""


def _set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    if not all(parts):
        raise ConfigError(f"Invalid override key: {key!r}")
    cursor = config
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigError(f"Cannot set {key!r}: {part!r} is not a mapping")
        cursor = child
    cursor[parts[-1]] = value


def load_config(
    path: str | Path,
    overrides: Mapping[str, Any] | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load YAML and apply dotted-key overrides.

    String overrides use ``key=value`` and parse values with ``yaml.safe_load``.
    The returned mapping is detached from PyYAML's internal objects and is safe
    to include verbatim in run metadata.
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ConfigError(f"Top-level configuration must be a mapping: {config_path}")
    resolved = copy.deepcopy(loaded)
    if overrides is None:
        return resolved
    if isinstance(overrides, Mapping):
        items = overrides.items()
    else:
        parsed: list[tuple[str, Any]] = []
        for item in overrides:
            if "=" not in item:
                raise ConfigError(f"Override must have key=value form: {item!r}")
            key, raw_value = item.split("=", 1)
            parsed.append((key, yaml.safe_load(raw_value)))
        items = parsed
    for key, value in items:
        _set_dotted(resolved, str(key), value)
    return resolved


def config_json(config: Mapping[str, Any]) -> str:
    """Return the canonical JSON representation stored with every result."""

    return json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)

