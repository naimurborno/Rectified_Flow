"""
run.py
------
Loads a config yaml + its prompts yaml, generates one image per
(prompt, seed) pair using the standard SD3 pipeline, and saves each
image named after its global prompt index, prompt text, and seed.

Usage:
    python run.py                          # uses ./config.yaml
    python run.py --config path/to/cfg.yaml
"""

import re
import argparse
import yaml
from pathlib import Path

from utils import load_config
from pipeline_wrapper import SD3PipelineWrapper


def slugify(text: str, max_len: int = 60) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text[:max_len] if text else "prompt"


def load_prompts(path: str) -> list:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # Accept either a flat list, or a {"prompts": [...]} dict
    if isinstance(data, dict):
        return data.get("prompts", []) or []
    return data or []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config yaml (default: config.yaml)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    prompts_file = cfg.get("prompts_file", "prompts.yaml")
    prompts = load_prompts(prompts_file)

    seeds = cfg.get("seeds") or [cfg.get("seed", 42)]

    # Offset lets multiple parallel runs (e.g. one per GPU, each with a
    # slice of the full prompt list) keep globally consistent filenames.
    offset = cfg.get("prompt_offset", 0)

    out_dir = Path(cfg.get("original_output_dir", "outputs/original"))
    out_dir.mkdir(parents=True, exist_ok=True)

    negative_prompt = cfg.get("negative_prompt", "")

    pipe = SD3PipelineWrapper(cfg, device=cfg.get("device", "cuda"))
    pipe.load()

    print(f"[run] {len(prompts)} prompt(s) x {len(seeds)} seed(s) "
          f"= {len(prompts) * len(seeds)} image(s) | offset={offset}")

    for local_idx, prompt in enumerate(prompts):
        global_idx = offset + local_idx
        slug = slugify(prompt)
        for seed in seeds:
            print(f"[run] prompt {local_idx+1}/{len(prompts)} "
                  f"(global #{global_idx}) | seed {seed} | {prompt}")
            image = pipe.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
            )
            filename = f"{global_idx:03d}_{slug}_seed{seed}.png"
            image.save(out_dir / filename)

    print(f"[run] Done. Images saved to: {out_dir}")


if __name__ == "__main__":
    main()