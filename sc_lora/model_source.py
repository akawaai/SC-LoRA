from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSourceResolution:
    source: str
    requested: str
    resolved: str


def _read_base_model_name(candidate_path: Path) -> str | None:
    if candidate_path.is_file():
        if candidate_path.name != "adapter_config.json":
            return None
        config_path = candidate_path
    else:
        config_path = candidate_path / "adapter_config.json"

    if not config_path.exists():
        return None

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception:
        return None

    base_model_name = config.get("base_model_name_or_path")
    if isinstance(base_model_name, str) and base_model_name.strip():
        return base_model_name.strip()
    return None


def resolve_model_source(requested: str) -> ModelSourceResolution:
    requested = str(requested).strip()
    if not requested:
        raise ValueError("model source cannot be empty")

    candidate_path = Path(requested)

    if candidate_path.exists():
        base_model_name = _read_base_model_name(candidate_path)
        if base_model_name:
            return ModelSourceResolution(
                source="adapter_config",
                requested=requested,
                resolved=base_model_name,
            )

        if candidate_path.is_dir():
            resolved = str(candidate_path.resolve())
        else:
            resolved = str(candidate_path.resolve().parent)
        return ModelSourceResolution(
            source="local_path",
            requested=requested,
            resolved=resolved,
        )

    return ModelSourceResolution(
        source="huggingface_id",
        requested=requested,
        resolved=requested,
    )