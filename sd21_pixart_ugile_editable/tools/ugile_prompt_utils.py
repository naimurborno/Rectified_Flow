from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def parse_prompts_payload(payload: Any, source: str = "prompt payload") -> list[str]:
    """Accept either a top-level prompt list or a mapping with prompts/prompt keys."""
    if isinstance(payload, dict):
        if "prompts" in payload:
            payload = payload["prompts"]
        elif "prompt" in payload:
            payload = [payload["prompt"]]
        else:
            raise ValueError(f"No prompts/prompts list found in {source}")

    if isinstance(payload, str):
        prompts = [payload]
    elif isinstance(payload, list):
        prompts = payload
    else:
        raise ValueError(f"Unsupported prompt format in {source}: {type(payload).__name__}")

    prompts = [str(prompt) for prompt in prompts if str(prompt).strip()]
    if not prompts:
        raise ValueError(f"No non-empty prompts found in {source}")
    return prompts


def load_prompts_file(prompts_file: str | Path) -> list[str]:
    path = Path(prompts_file)
    if not path.is_file():
        raise FileNotFoundError(f"prompts_file does not exist: {path}")
    payload = yaml.safe_load(path.read_text())
    return parse_prompts_payload(payload, str(path))


def resolve_prompts(cfg: dict, fallback: str = "a photo of a cat") -> list[str]:
    prompts_file = cfg.get("prompts_file")
    if prompts_file:
        return load_prompts_file(prompts_file)
    return parse_prompts_payload(cfg.get("prompts", [fallback]), "config prompts")
