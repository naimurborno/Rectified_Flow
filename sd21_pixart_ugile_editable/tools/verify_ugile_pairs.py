#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PAIR_FIELDS = [
    "model_id",
    "prompt",
    "negative_prompt",
    "seed",
    "scheduler_class",
    "scheduler_config_hash",
    "prediction_type",
    "num_steps",
    "guidance_scale",
    "height",
    "width",
    "dtype",
    "original_xT_hash",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify reviewer-fair UGILE OFF/ON manifest pairs.")
    parser.add_argument("--manifest", default=str(ROOT / "outputs" / "pair_manifest.jsonl"))
    parser.add_argument("--architecture", action="append", help="Restrict to one architecture. Repeatable.")
    parser.add_argument("--require-pair", action="store_true", help="Fail if no complete OFF/ON pair exists.")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Manifest not found: {path}")
    records = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
    return records


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def verify_pair(base: dict[str, Any], ugile: dict[str, Any]) -> list[str]:
    errors = []
    label = f"{base.get('architecture')} prompt={base.get('prompt_idx')} seed={base.get('seed')}"
    for field in PAIR_FIELDS:
        if base.get(field) != ugile.get(field):
            errors.append(f"{label}: field {field} differs: OFF={base.get(field)!r} ON={ugile.get(field)!r}")

    if base.get("final_init_hash") != base.get("original_xT_hash"):
        errors.append(f"{label}: OFF path did not use the original x_T unchanged")

    diag = ugile.get("diagnostics") or {}
    method = diag.get("ugile_method") or ugile.get("ugile_method") or "adaptive"
    if method == "threshold":
        for field in ["kappa", "epsilon", "epsilon_cap", "noise_scale", "theta_max", "theta_used"]:
            if not finite_number(diag.get(field)):
                errors.append(f"{label}: threshold ON diagnostics missing finite {field}")
    else:
        for field in [
            "directional_confidence",
            "freedom_score",
            "safe_freedom_score",
            "phase2_angle",
            "epsilon_adaptive",
            "theta_adaptive",
        ]:
            if not finite_number(diag.get(field)):
                errors.append(f"{label}: adaptive ON diagnostics missing finite {field}")

        confidence = float(diag.get("directional_confidence", 0.0) or 0.0)
        if confidence <= 0.0:
            for field in ["phase2_angle", "epsilon_adaptive", "theta_adaptive"]:
                if abs(float(diag.get(field, 0.0) or 0.0)) > 1e-12:
                    errors.append(f"{label}: zero-confidence ON run has nonzero {field}")
            if ugile.get("final_init_hash") != ugile.get("original_xT_hash"):
                errors.append(f"{label}: zero-confidence ON run changed x_T")

    return errors


def main() -> None:
    args = parse_args()
    wanted_arch = set(args.architecture or [])
    records = load_records(Path(args.manifest))
    if wanted_arch:
        records = [record for record in records if record.get("architecture") in wanted_arch]

    grouped = defaultdict(lambda: {"base": [], "ugile": []})
    for record in records:
        key = (record.get("architecture"), record.get("prompt_idx"), record.get("seed"))
        bucket = "ugile" if record.get("ugile_enabled") else "base"
        grouped[key][bucket].append(record)

    pairs = []
    errors = []
    for key, group in grouped.items():
        if not group["base"] or not group["ugile"]:
            continue
        base = group["base"][-1]
        ugile = group["ugile"][-1]
        pairs.append(key)
        errors.extend(verify_pair(base, ugile))

    if args.require_pair and not pairs:
        errors.append("No complete OFF/ON pairs were found in the manifest.")

    if errors:
        for error in errors:
            print(f"[verify] ERROR: {error}")
        raise SystemExit(1)

    print(f"[verify] Verified {len(pairs)} OFF/ON UGILE pair(s).")


if __name__ == "__main__":
    main()
