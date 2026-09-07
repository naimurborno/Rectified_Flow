from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "outputs" / "pair_manifest.jsonl"


def tensor_hash(tensor: torch.Tensor) -> str:
    data = tensor.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def stable_config_hash(config: Any) -> str:
    payload = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def append_pair_manifest(
    record: dict[str, Any],
    manifest_path: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> None:
    if manifest_path is None and cfg:
        manifest_path = cfg.get("pair_manifest")
    path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")


def base_manifest_record(
    *,
    cfg: dict[str, Any],
    architecture: str,
    prompt_idx: int,
    prompt: str,
    negative_prompt: str,
    seed: int,
    ugile_enabled: bool,
    scheduler: Any | None = None,
    x_T: torch.Tensor | None = None,
    x_init: torch.Tensor | None = None,
    output_path: str | Path,
    diagnostics: dict[str, Any] | None = None,
    wrapper: Any | None = None,
    sampler: Any | None = None,
    original_xT: torch.Tensor | None = None,
    final_init: torch.Tensor | None = None,
) -> dict[str, Any]:
    scheduler = scheduler or getattr(sampler, "scheduler", None) or getattr(wrapper, "scheduler", None)
    x_T = x_T if x_T is not None else original_xT
    x_init = x_init if x_init is not None else final_init
    if scheduler is None:
        raise ValueError("scheduler is required for a UGILE pair manifest record")
    if x_T is None or x_init is None:
        raise ValueError("x_T/original_xT and x_init/final_init are required")

    flow_cfg = cfg.get("flow", {})
    gen_cfg = cfg.get("generation", {})
    scheduler_config = getattr(scheduler, "config", {})
    prediction_type = getattr(scheduler_config, "prediction_type", None)
    if prediction_type is None and isinstance(scheduler_config, Mapping):
        prediction_type = scheduler_config.get("prediction_type")

    run_id = (
        f"{architecture}|prompt={prompt_idx}|seed={seed}|"
        f"h={gen_cfg.get('height')}|w={gen_cfg.get('width')}|steps={flow_cfg.get('num_steps')}"
    )

    return {
        "run_id": run_id,
        "architecture": architecture,
        "model_id": cfg.get("model_id"),
        "ugile_enabled": bool(ugile_enabled),
        "prompt_idx": int(prompt_idx),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": int(seed),
        "scheduler_class": scheduler.__class__.__name__,
        "scheduler_config_hash": stable_config_hash(scheduler_config),
        "prediction_type": prediction_type,
        "num_steps": int(flow_cfg.get("num_steps", 0)),
        "guidance_scale": float(flow_cfg.get("guidance_scale", 0.0)),
        "height": int(gen_cfg.get("height", 0)),
        "width": int(gen_cfg.get("width", 0)),
        "dtype": str(x_T.dtype),
        "original_xT_hash": tensor_hash(x_T),
        "original_xT_norm": float(x_T.detach().float().norm().item()),
        "final_init_hash": tensor_hash(x_init),
        "final_init_norm": float(x_init.detach().float().norm().item()),
        "output_path": str(output_path),
        "diagnostics": diagnostics or {},
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": tensor_hash(value),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
