#!/usr/bin/env python3
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

SAMPLERS = {
    "FLUX": ROOT / "FLUX" / "latent_escape_sampler_flux.py",
    "SD3": ROOT / "SD3" / "latent_escape_sampler.py",
    "SD3.5": ROOT / "SD3.5" / "latent_escape_sampler.py",
    "SANA": ROOT / "SANA" / "latent_escape_sampler_sana.py",
    "PIXART": ROOT / "PIXART" / "latent_escape_sampler_pixart_sigma.py",
    "SD21": ROOT / "sd2.1" / "latent_escape_sampler_sd21.py",
    "SD3.5_EXPERIMENT": ROOT / "sd3.5_experiment" / "latent_escape_sampler.py",
}

CONFIGS = {
    "FLUX": ROOT / "FLUX" / "config_flux.yaml",
    "SD3": ROOT / "SD3" / "config.yaml",
    "SD3.5": ROOT / "SD3.5" / "config.yaml",
    "SANA": ROOT / "SANA" / "config_sana.yaml",
    "PIXART": ROOT / "PIXART" / "config_pixart_sigma.yaml",
    "SD21": ROOT / "sd2.1" / "config_sd21.yaml",
}

INFERENCE = {
    "FLUX": ROOT / "FLUX" / "inference.py",
    "SD3": ROOT / "SD3" / "inference.py",
    "SD3.5": ROOT / "SD3.5" / "inference.py",
    "SANA": ROOT / "SANA" / "inference.py",
    "PIXART": ROOT / "PIXART" / "inference.py",
    "SD21": ROOT / "sd2.1" / "inference.py",
}

REQUIRED_THRESHOLD_KEYS = {
    "num_grad_steps",
    "sigma_lo",
    "sigma_hi",
    "escape_scale",
    "theta_max",
    "walk_steps",
    "branch_noise",
    "J",
    "noise_scale",
    "gamma",
}

REQUIRED_SAMPLER_PATTERNS = [
    "tweedie_potential(",
    "trajectory_covariance_direction(",
    "joint_projector(",
    "geodesic_step(",
    "theta_max=self.theta_max",
    "self.noise_scale",
    "\"ugile_method\": \"threshold\"",
    "build_initialization(",
    "full_forward_pass(",
    "append_pair_manifest(",
    "base_manifest_record(",
]


def fail(message: str) -> None:
    raise SystemExit(f"UGILE threshold consistency check failed: {message}")


def require_file(path: Path) -> str:
    if not path.is_file():
        fail(f"missing required file {path.relative_to(ROOT)}")
    return path.read_text()


def main() -> None:
    smoke = require_file(ROOT / "tools" / "run_ugile_smoke_tests.py")
    if "\"SDXL\"" in smoke or "'SDXL'" in smoke:
        fail("smoke runner still references SDXL")
    if "--faithfulness-guard" in smoke:
        fail("smoke runner still exposes adaptive faithfulness guard")
    require_file(ROOT / "tools" / "verify_ugile_pairs.py")
    require_file(ROOT / "tools" / "ugile_pair_manifest.py")

    for name, path in SAMPLERS.items():
        text = require_file(path)
        for required in REQUIRED_SAMPLER_PATTERNS:
            if required not in text:
                fail(f"{name} sampler missing {required}")
        if "theta_adaptive" in text or "faithfulness_guard" in text:
            fail(f"{name} sampler still contains adaptive-main text")

    for name, path in CONFIGS.items():
        data = yaml.safe_load(require_file(path))
        ugile = data.get("ugile", {})
        if ugile.get("enabled") is None:
            fail(f"{name} config missing ugile.enabled")
        if ugile.get("method") != "threshold":
            fail(f"{name} config must set ugile.method=threshold")
        missing = REQUIRED_THRESHOLD_KEYS.difference(ugile.keys())
        if missing:
            fail(f"{name} threshold config missing keys: {sorted(missing)}")
        if "adaptive_budget" in ugile or "common_discrepancy_space" in ugile:
            fail(f"{name} config still exposes adaptive-main keys")
        if data.get("negative_prompt", None) != "":
            fail(f"{name} negative_prompt must stay empty for fairness")

    sd21 = yaml.safe_load(require_file(CONFIGS["SD21"]))
    if sd21.get("prompts_file") != "prompts_mscoco.yaml":
        fail("SD21 config must default to prompts_mscoco.yaml")

    for name, path in INFERENCE.items():
        if "--ugile-enabled" not in require_file(path):
            fail(f"{name} inference.py missing --ugile-enabled")

    print("UGILE threshold-main consistency check passed.")


if __name__ == "__main__":
    main()
