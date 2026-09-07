#!/usr/bin/env python3
"""Run isolated UGILE smoke tests without editing main configs.

The script writes temporary one-prompt configs, logs, and generated images under
outputs/ugile_smoke/. It leaves the repository's real YAML files untouched.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

ARCHES = {
    "FLUX": {
        "dir": ROOT / "FLUX",
        "config": "config_flux.yaml",
        "base_dir": "outputs/ugile_smoke/FLUX/base",
        "ugile_dir": "outputs/ugile_smoke/FLUX/ugile",
    },
    "SD3": {
        "dir": ROOT / "SD3",
        "config": "config.yaml",
        "base_dir": "outputs/ugile_smoke/SD3/base",
        "ugile_dir": "outputs/ugile_smoke/SD3/ugile",
    },
    "SD3.5": {
        "dir": ROOT / "SD3.5",
        "config": "config.yaml",
        "base_dir": "outputs/ugile_smoke/SD3.5/base",
        "ugile_dir": "outputs/ugile_smoke/SD3.5/ugile",
    },
    "SANA": {
        "dir": ROOT / "SANA",
        "config": "config_sana.yaml",
        "base_dir": "outputs/ugile_smoke/SANA/base",
        "ugile_dir": "outputs/ugile_smoke/SANA/ugile",
    },
    "PIXART": {
        "dir": ROOT / "PIXART",
        "config": "config_pixart_sigma.yaml",
        "base_dir": "outputs/ugile_smoke/PIXART/base",
        "ugile_dir": "outputs/ugile_smoke/PIXART/ugile",
    },
    "SD21": {
        "dir": ROOT / "sd2.1",
        "config": "config_sd21.yaml",
        "base_dir": "outputs/ugile_smoke/SD21/base",
        "ugile_dir": "outputs/ugile_smoke/SD21/ugile",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--prompt",
        default="a photo of a red apple on a wooden table",
        help="Single prompt used for every smoke run.",
    )
    parser.add_argument(
        "--arch",
        choices=sorted(ARCHES),
        action="append",
        help="Run only this architecture. Repeatable. Omit to run all.",
    )
    parser.add_argument(
        "--mode",
        choices=("ugile", "base", "pair"),
        default="pair",
        help="Which smoke run to execute. Default: pair, the reviewer OFF/ON check.",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="Physical GPU id to use for every subprocess, e.g. 1.",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated physical GPU ids to rotate by architecture, e.g. 0,1,2,3; use auto to discover visible GPUs.",
    )
    parser.add_argument(
        "--reset-manifest",
        action="store_true",
        help="Delete the smoke pair manifest before running. Avoid this when launching architectures in parallel.",
    )
    parser.add_argument("--height", type=int, default=None, help="Override smoke image height.")
    parser.add_argument("--width", type=int, default=None, help="Override smoke image width.")
    parser.add_argument("--num-steps", type=int, default=None, help="Override smoke denoising steps.")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Override smoke CFG/guidance scale.")
    parser.add_argument("--max-sequence-length", type=int, default=None, help="Override max text sequence length when supported.")
    return parser.parse_args()


def parse_gpu_list(args: argparse.Namespace) -> list[str]:
    if args.gpu and args.gpus:
        raise SystemExit("Use either --gpu or --gpus, not both.")
    if args.gpu:
        return [args.gpu]
    if args.gpus:
        if args.gpus.strip().lower() == "auto":
            return discover_gpus()
        return [item.strip() for item in args.gpus.split(",") if item.strip()]
    return [""]


def discover_gpus() -> list[str]:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if proc.returncode == 0:
        gpus = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if gpus:
            return gpus

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        count = int(proc.stdout.strip())
    except ValueError:
        count = 0
    if count <= 0:
        raise SystemExit("No CUDA GPUs were discovered. Try passing a known GPU with --gpu N.")
    return [str(idx) for idx in range(count)]


def check_cuda(gpu: str) -> tuple[bool, str]:
    env = os.environ.copy()
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import torch; "
                "print(torch.cuda.is_available()); "
                "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')"
            ),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    ok = proc.returncode == 0 and bool(proc.stdout.splitlines()) and proc.stdout.splitlines()[0] == "True"
    return ok, proc.stdout


def require_usable_gpus(gpus: list[str]) -> list[str]:
    usable = []
    errors = []
    for gpu in gpus:
        ok, output = check_cuda(gpu)
        if ok:
            usable.append(gpu)
        else:
            target = gpu if gpu else "default visible GPU"
            errors.append(f"{target}: {output.strip() or 'no output'}")

    if usable:
        if errors:
            print("[smoke] Skipping unavailable GPU(s):", flush=True)
            for error in errors:
                print(f"[smoke]   {error}", flush=True)
        return usable

    detail = "\n".join(errors)
    raise SystemExit(f"No requested CUDA GPUs are usable in this Python environment.\n{detail}")


def write_temp_config(name: str, meta: dict, args: argparse.Namespace, prompt_file: Path) -> Path:
    seed = args.seed
    src_config = meta["dir"] / meta["config"]
    with src_config.open() as f:
        cfg = yaml.safe_load(f)

    cfg["seed"] = seed
    cfg["seeds"] = [seed]
    prompt_payload = yaml.safe_load(prompt_file.read_text()) or {}
    cfg["prompts"] = prompt_payload.get("prompts", []) if isinstance(prompt_payload, dict) else prompt_payload
    cfg["prompts_file"] = str(prompt_file)
    cfg["prompt_offset"] = 0
    cfg["output"] = f"{name.lower().replace('.', '')}_smoke.png"
    cfg["pair_manifest"] = str(ROOT / "outputs" / "ugile_smoke" / "pair_manifest.jsonl")
    if args.height is not None:
        cfg.setdefault("generation", {})["height"] = args.height
    if args.width is not None:
        cfg.setdefault("generation", {})["width"] = args.width
    if args.num_steps is not None:
        cfg.setdefault("flow", {})["num_steps"] = args.num_steps
    if args.guidance_scale is not None:
        cfg.setdefault("flow", {})["guidance_scale"] = args.guidance_scale
    if args.max_sequence_length is not None:
        cfg["max_sequence_length"] = args.max_sequence_length
    if name == "FLUX":
        visible_gpu_count = _visible_gpu_count_for_config(args)
        if visible_gpu_count is not None and isinstance(cfg.get("max_memory"), dict):
            base_max_memory = cfg["max_memory"]
            gpu_limits = [value for key, value in base_max_memory.items() if str(key).isdigit()]
            fallback_limit = gpu_limits[-1] if gpu_limits else "8GiB"
            cfg["max_memory"] = {
                idx: gpu_limits[idx] if idx < len(gpu_limits) else fallback_limit
                for idx in range(visible_gpu_count)
            }
            if "cpu" in base_max_memory:
                cfg["max_memory"]["cpu"] = base_max_memory["cpu"]
    cfg.setdefault("ugile", {})
    cfg["ugile"]["save_original"] = True
    cfg["ugile"]["original_output_dir"] = str(ROOT / meta["base_dir"])
    cfg["ugile"]["diverse_output_dir"] = str(ROOT / meta["ugile_dir"])

    smoke_root = ROOT / "outputs" / "ugile_smoke" / "_configs"
    smoke_root.mkdir(parents=True, exist_ok=True)
    tmp_config = smoke_root / f"{name.replace('.', '_')}_smoke_config.yaml"
    with tmp_config.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return tmp_config


def _visible_gpu_count_for_config(args: argparse.Namespace) -> int | None:
    gpu_spec = args.gpu or args.gpus
    if not gpu_spec or str(gpu_spec).strip().lower() == "auto":
        return None
    gpu_items = [item.strip() for item in str(gpu_spec).split(",") if item.strip()]
    return len(gpu_items) if gpu_items else None


def run_one(name: str, meta: dict, config: Path, enabled: bool, seed: int, gpu: str) -> None:
    log_dir = ROOT / "outputs" / "ugile_smoke" / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    mode = "ugile" if enabled else "base"
    log_path = log_dir / f"{name.replace('.', '_')}_seed{seed}_{mode}.log"
    cmd = [
        sys.executable,
        "inference.py",
        "--config",
        str(config),
        "--seed",
        str(seed),
        "--ugile-enabled",
        "true" if enabled else "false",
    ]

    env = os.environ.copy()
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    if name == "FLUX" and "expandable_segments" in env.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    gpu_label = gpu if gpu else "default"
    print(f"[smoke] {name} {mode}: gpu={gpu_label} log={log_path}", flush=True)
    with log_path.open("w") as log:
        if gpu:
            log.write(f"CUDA_VISIBLE_DEVICES={gpu}\n")
        if name == "FLUX":
            log.write("PYTORCH_CUDA_ALLOC_CONF cleared for FLUX loader\n")
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=meta["dir"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(f"\n[smoke] returncode={proc.returncode}\n")
    if proc.returncode != 0:
        detail = ""
        if proc.returncode == -9:
            detail = " It was killed by SIGKILL, commonly from CPU/GPU memory pressure."
        raise SystemExit(f"{name} {mode} failed with returncode {proc.returncode}.{detail} Inspect {log_path}")


def verify_manifest(name: str) -> None:
    manifest = ROOT / "outputs" / "ugile_smoke" / "pair_manifest.jsonl"
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "verify_ugile_pairs.py"),
        "--manifest",
        str(manifest),
        "--architecture",
        name,
        "--require-pair",
    ]
    print(f"[smoke] {name}: verifying OFF/ON manifest pair", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"{name} manifest pair verification failed.")


def main() -> None:
    args = parse_args()
    gpus = parse_gpu_list(args)
    gpus = require_usable_gpus(gpus)

    smoke_root = ROOT / "outputs" / "ugile_smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    prompt_file = smoke_root / "one_prompt.yaml"
    with prompt_file.open("w") as f:
        yaml.safe_dump({"prompts": [args.prompt]}, f, sort_keys=False)

    manifest = smoke_root / "pair_manifest.jsonl"
    if args.reset_manifest and manifest.exists():
        manifest.unlink()

    selected = args.arch or ["SD3.5", "SD3", "SANA", "PIXART", "SD21", "FLUX"]
    print(f"[smoke] Root: {smoke_root}", flush=True)
    print(f"[smoke] Seed: {args.seed}", flush=True)
    print(f"[smoke] Prompt file: {prompt_file}", flush=True)
    print(f"[smoke] GPU rotation: {', '.join(gpus) if any(gpus) else 'default'}", flush=True)

    for idx, name in enumerate(selected):
        meta = ARCHES[name]
        gpu = gpus[idx % len(gpus)]
        config = write_temp_config(name, meta, args, prompt_file)
        if args.mode in ("base", "pair"):
            run_one(name, meta, config, enabled=False, seed=args.seed, gpu=gpu)
        if args.mode in ("ugile", "pair"):
            run_one(name, meta, config, enabled=True, seed=args.seed, gpu=gpu)
        if args.mode == "pair":
            verify_manifest(name)

    print("[smoke] All requested smoke tests completed.", flush=True)
    print(f"[smoke] Outputs and logs are under: {smoke_root}", flush=True)


if __name__ == "__main__":
    main()
