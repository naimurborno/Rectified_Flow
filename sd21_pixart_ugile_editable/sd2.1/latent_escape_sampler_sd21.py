import hashlib
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from peakback_core import (
    tweedie_potential,
    trajectory_covariance_direction,
    joint_projector,
    geodesic_step,
)
from ugile_pair_manifest import append_pair_manifest, base_manifest_record
from ugile_prompt_utils import load_prompts_file


def latent_hash(latents: torch.Tensor) -> str:
    payload = latents.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def _tensor_norm(latents: torch.Tensor) -> float:
    return float(latents.detach().float().norm().item())


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a_f = a.detach().float().reshape(-1)
    b_f = b.detach().float().reshape(-1)
    denom = a_f.norm() * b_f.norm()
    if not torch.isfinite(denom) or denom <= eps:
        return 0.0
    return float((torch.dot(a_f, b_f) / (denom + eps)).item())


def _stat_float(stats: dict, key: str) -> float:
    value = stats.get(key, 0.0)
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    return float(value)


def _scalar_diag(diag: dict) -> dict:
    out = {}
    for key, value in diag.items():
        if key == "profile_original_latents":
            continue
        if torch.is_tensor(value):
            out[key] = float(value.detach().float().cpu().item()) if value.numel() == 1 else str(tuple(value.shape))
        else:
            out[key] = value
    return out


def _rooted_path(path: str) -> Path:
    out = Path(path)
    return out if out.is_absolute() else ROOT / out


class SD21UGILESampler:
    """Threshold-based UGILE sampler for SD21."""

    def __init__(self, unet, scheduler, cfg: dict, wrapper=None, device: str = "cuda", eps: float = 1e-8, num_grad_steps: int = 50, sigma_lo: float = 0.3, sigma_hi: float = 0.9, escape_scale: float = 1.0, theta_max: float = 0.5, walk_steps: int = 10, branch_noise: float = 0.7, J: int = 1, noise_scale: float = 1.5, gamma: float = 1.2):
        self.unet = unet
        self.scheduler = scheduler
        self.cfg = cfg
        self.wrapper = wrapper
        self.device = device
        self.eps = eps
        ug_cfg = cfg.get("ugile", {})
        self.num_grad_steps = ug_cfg.get("num_grad_steps", num_grad_steps)
        self.sigma_lo = ug_cfg.get("sigma_lo", sigma_lo)
        self.sigma_hi = ug_cfg.get("sigma_hi", sigma_hi)
        self.escape_scale = ug_cfg.get("escape_scale", escape_scale)
        self.theta_max = ug_cfg.get("theta_max", theta_max)
        self.walk_steps = ug_cfg.get("walk_steps", walk_steps)
        self.branch_noise = ug_cfg.get("branch_noise", branch_noise)
        self.J = ug_cfg.get("J", J)
        self.noise_scale = ug_cfg.get("noise_scale", noise_scale)
        self.gamma = ug_cfg.get("gamma", gamma)
        flow_cfg = cfg.get("flow", {})
        self.num_steps = flow_cfg.get("num_steps", 50)
        self.guidance_scale = flow_cfg.get("guidance_scale", 7.5)
        self.do_cfg = self.guidance_scale > 1.0
        gen_cfg = cfg.get("generation", {})
        self.H = gen_cfg.get("height", 512)
        self.W = gen_cfg.get("width", 512)


    def build_initialization(self, x_T, text_embeddings, extra_conditioning=None, seed: int = 0):
        cache = self._forward_pass_with_profiling(x_T, text_embeddings, extra_conditioning)
        n_steps = cache["N"]
        U_vals = torch.tensor(cache["U"], device=x_T.device, dtype=torch.float32)
        U_total = U_vals.sum() + self.eps
        weights = U_vals / U_total if torch.isfinite(U_total) and U_total > self.eps else torch.zeros_like(U_vals)

        semantic_dir = torch.zeros_like(x_T, dtype=torch.float32)
        diffs = []
        for k in range(n_steps):
            diff = (cache["v_cond"][k] - cache["v_uncond"][k]).to(x_T.device).float()
            diffs.append(diff)
            semantic_dir = semantic_dir + weights[k] * diff

        semantic_norm = semantic_dir.norm()
        semantic_unit = semantic_dir / (semantic_norm + self.eps)
        s_flat = semantic_unit.flatten()

        if n_steps > 2:
            d2U = U_vals[:-2] - 2 * U_vals[1:-1] + U_vals[2:]
            kappa = d2U.abs().mean().item()
        else:
            kappa = 1.0
        if not math.isfinite(kappa) or kappa < 0.0:
            kappa = 1.0

        r = x_T.float().norm().item()
        max_eps = r * 0.02
        epsilon = self.noise_scale * (1.0 / (math.sqrt(kappa) + self.eps))
        epsilon = min(epsilon, max_eps)

        covariance_dir = trajectory_covariance_direction(
            diffs=diffs,
            weights=weights,
            semantic_dir=semantic_dir,
            seed=seed,
            shape=x_T.shape,
            device=x_T.device,
            eps=self.eps,
            num_power_iters=3,
        )
        covariance_norm = covariance_dir.float().norm()

        rng = torch.Generator(device=x_T.device)
        rng.manual_seed(seed * 10000 + 1)
        jitter = torch.randn(x_T.shape, generator=rng, dtype=torch.float32, device=x_T.device)
        eta = covariance_dir.float() + 0.1 * jitter
        eta = self._shape_phase2_direction(eta)
        x_flat = x_T.float().flatten()
        eta_flat = joint_projector(eta.float().flatten(), s_flat, x_flat, eps=self.eps)
        eta = eta_flat.view_as(x_T)

        if epsilon <= self.eps or eta.norm() <= self.eps:
            x_phase2 = x_T.float().clone()
        else:
            eta = epsilon * eta / (eta.norm() + self.eps)
            x_phase2 = x_T.float() + eta
            x_phase2 = x_phase2 * (r / (x_phase2.norm() + self.eps))

        phase2_delta = x_phase2 - x_T.float()
        post_phase2_semantic_cos = _cosine(phase2_delta, semantic_unit, self.eps)
        post_phase2_radial_cos = _cosine(phase2_delta, x_T.float(), self.eps)

        rng2 = torch.Generator(device=x_T.device)
        rng2.manual_seed(seed * 10000 + 2)
        tangent = torch.randn(x_phase2.shape, generator=rng2, dtype=torch.float32, device=x_T.device)
        tangent = self._shape_phase3_tangent(tangent)
        x_phase2_flat = x_phase2.float().flatten()
        w_proj = joint_projector(tangent.float().flatten(), s_flat, x_phase2_flat, eps=self.eps)

        if w_proj.norm() <= self.eps:
            x_init = x_phase2
            theta_used = torch.zeros((), device=x_T.device, dtype=torch.float32)
        else:
            x_new_flat, theta_used = geodesic_step(
                x_phase2_flat,
                w_proj,
                r,
                theta_max=self.theta_max,
                eps=self.eps,
            )
            x_init = x_new_flat.view_as(x_phase2)

        phase3_delta = x_init.float() - x_phase2.float()
        post_phase3_semantic_cos = _cosine(phase3_delta, semantic_unit, self.eps)
        post_phase3_radial_cos = _cosine(phase3_delta, x_phase2.float(), self.eps)

        diagnostics = {
            "profile_original_latents": cache["x_N"],
            "ugile_method": "threshold",
            "semantic_norm": float(semantic_norm.detach().cpu().item()) if torch.isfinite(semantic_norm) else 0.0,
            "covariance_norm": float(covariance_norm.detach().cpu().item()) if torch.isfinite(covariance_norm) else 0.0,
            "kappa": float(kappa),
            "epsilon": float(epsilon),
            "epsilon_cap": float(max_eps),
            "noise_scale": float(self.noise_scale),
            "theta_max": float(self.theta_max),
            "theta_used": float(theta_used.detach().cpu().item()) if torch.is_tensor(theta_used) else float(theta_used),
            "post_phase2_semantic_cos": post_phase2_semantic_cos,
            "post_phase2_radial_cos": post_phase2_radial_cos,
            "post_phase3_semantic_cos": post_phase3_semantic_cos,
            "post_phase3_radial_cos": post_phase3_radial_cos,
        }
        return x_init.to(x_T.dtype), diagnostics

    def run(self, x_T, text_embeddings, extra_conditioning=None, seed: int = 0) -> Dict[str, Any]:
        x_init, diagnostics = self.build_initialization(
            x_T,
            text_embeddings,
            extra_conditioning,
            seed=seed,
        )
        final_latents = self.full_forward_pass(x_init, text_embeddings, extra_conditioning)
        return {
            "original_latents": diagnostics["profile_original_latents"],
            "branches": [{
                "branch_idx": 0,
                "theta": diagnostics["theta_used"],
                "cos_x0": _cosine(x_init, x_T, self.eps),
                "cos_xN": _cosine(final_latents, diagnostics["profile_original_latents"], self.eps),
                "latents": final_latents,
                "diagnostics": diagnostics,
            }],
        }

    def _shape_phase2_direction(self, eta: torch.Tensor) -> torch.Tensor:
        return eta

    def _shape_phase3_tangent(self, tangent: torch.Tensor) -> torch.Tensor:
        if tangent.dim() == 4:
            _, _, height, width = tangent.shape
            low = F.interpolate(
                tangent,
                scale_factor=0.25,
                mode="bilinear",
                recompute_scale_factor=False,
                align_corners=False,
            )
            low = F.interpolate(low, size=(height, width), mode="bilinear", align_corners=False)
            tangent = 0.25 * tangent + 0.75 * low
        return tangent


    def _forward_pass_with_profiling(self, x_T, text_embeddings, extra_conditioning):
        self.scheduler.set_timesteps(self.num_steps, device=self.device)
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(0)
        timesteps = self.scheduler.timesteps
        n_steps = len(timesteps)
        cached_x = [None] * (n_steps + 1)
        cached_common_diffs = [None] * n_steps
        cached_v_uncond = [None] * n_steps
        cached_v_cond = [None] * n_steps
        cached_sigma = [None] * n_steps
        cached_t = [None] * n_steps
        cached_U = [None] * n_steps

        x = x_T.clone()
        cached_x[0] = x.float().clone()
        for k, timestep in enumerate(timesteps):
            model_pred, pred_uncond, pred_cond = self._model_forward(
                x,
                timestep,
                text_embeddings,
                extra_conditioning,
                return_split=True,
            )
            sigma_k = self._sigma_at(k, timestep, n_steps)
            cached_common_diffs[k] = (pred_cond - pred_uncond).detach().float().clone()
            cached_v_uncond[k] = pred_uncond.detach().float().clone()
            cached_v_cond[k] = pred_cond.detach().float().clone()
            cached_sigma[k] = sigma_k
            cached_t[k] = timestep
            cached_U[k] = tweedie_potential(pred_cond, pred_uncond, sigma_k).item()
            x = self.scheduler.step(model_pred, timestep, x).prev_sample
            cached_x[k + 1] = x.detach().float().clone()

        return {
            "x": cached_x,
            "common_diffs": cached_common_diffs,
            "v_uncond": cached_v_uncond,
            "v_cond": cached_v_cond,
            "sigma": cached_sigma,
            "t": cached_t,
            "U": cached_U,
            "x_N": x,
            "N": n_steps,
        }

    def _sigma_at(self, k, timestep, n_steps):
        return self._sigma_from_alphas_cumprod(timestep)


    def _sigma_from_alphas_cumprod(self, timestep) -> float:
        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        t_idx = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
        t_idx = max(0, min(t_idx, len(alphas_cumprod) - 1))
        alpha_bar = alphas_cumprod[t_idx].item()
        sigma_t = ((1.0 - alpha_bar) / max(alpha_bar, 1e-8)) ** 0.5
        return max(float(sigma_t), 1e-4)

    def full_forward_pass(self, x_init, text_embeddings, extra_conditioning):
        self.scheduler.set_timesteps(self.num_steps, device=self.device)
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(0)
        x = x_init.clone()
        for timestep in self.scheduler.timesteps:
            model_pred = self._model_forward(x, timestep, text_embeddings, extra_conditioning)
            x = self.scheduler.step(model_pred, timestep, x).prev_sample
        return x

    _full_forward_pass = full_forward_pass

    def _model_forward(self, x, timestep, text_embeddings, extra_conditioning, return_split=False):
        device = next(self.unet.parameters()).device
        dtype = next(self.unet.parameters()).dtype
        latent_input = (torch.cat([x, x]) if self.do_cfg else x).to(device=device, dtype=dtype)
        latent_input = self.scheduler.scale_model_input(latent_input, timestep)
        text_embeddings = text_embeddings.to(device=device, dtype=dtype)
        
        kwargs = dict(sample=latent_input, timestep=timestep, encoder_hidden_states=text_embeddings, return_dict=False)
        if extra_conditioning is not None:
            kwargs["encoder_attention_mask"] = extra_conditioning.to(device=device)
        with torch.no_grad():
            output = self.unet(**kwargs)[0]
            if output.shape[1] == 2 * x.shape[1]:
                output, _ = output.chunk(2, dim=1)
        if self.do_cfg:
            pred_uncond, pred_cond = output.chunk(2)
            model_pred = pred_uncond + self.guidance_scale * (pred_cond - pred_uncond)
            if return_split:
                return model_pred, pred_uncond, pred_cond
            return model_pred
        if return_split:
            return output, output, output
        return output


def run_sd21_ugile(opts: dict):
    from pipeline_wrapper_sd21 import SD21PipelineWrapper

    cfg = opts.get("_cfg", {})
    device = opts["device"]
    seeds = opts.get("seeds") or [opts["seed"]]
    ug_cfg = cfg.get("ugile", {})
    ugile_enabled = ug_cfg.get("enabled", True)

    prompts_file = cfg.get("prompts_file")
    if prompts_file:
        prompts = load_prompts_file(prompts_file)
        print(f"[UGILE-SD21] Loaded {len(prompts)} prompt(s) from {prompts_file}")
    else:
        prompts = opts.get("prompts") or cfg.get("prompts") or [opts["prompt"]]

    prompt_offset = opts.get("prompt_offset", cfg.get("prompt_offset", 0))

    print(f"[UGILE-SD21] Loading model...")
    wrapper = SD21PipelineWrapper(cfg, device=device)
    wrapper.load()

    sampler = SD21UGILESampler(unet=wrapper.unet, scheduler=wrapper.scheduler, cfg=cfg, wrapper=wrapper, device=device)

    base_out = Path(opts["output"])
    diverse_folder = _rooted_path(ug_cfg.get("diverse_output_dir", "outputs/diverse"))
    original_folder = _rooted_path(ug_cfg.get("original_output_dir", "outputs/original"))
    diverse_folder.mkdir(parents=True, exist_ok=True)
    original_folder.mkdir(parents=True, exist_ok=True)
    save_original = ug_cfg.get("save_original", True)

    def _base_path(p_idx, seed):
        return original_folder / (f"{p_idx + 1:06d}_seed{seed}" + base_out.suffix)

    def _branch_path(p_idx, seed, branch_idx):
        return diverse_folder / (f"{p_idx + 1:06d}_seed{seed}" + base_out.suffix)

    records = []
    for local_idx, prompt in enumerate(prompts):
        p_idx = prompt_offset + local_idx
        print(f"\n[UGILE-SD21] Prompt {p_idx + 1} (local {local_idx + 1}/{len(prompts)}): \"{prompt}\"")
        conditioning = wrapper.encode_prompt(prompt, opts["negative_prompt"])
        if not isinstance(conditioning, tuple):
            conditioning = (conditioning, None)
        text_embeddings, extra_conditioning = conditioning

        for seed in seeds:
            print(f"[UGILE-SD21]   seed={seed}")
            x_T = wrapper.get_initial_latents(seed=seed)
            original_hash = latent_hash(x_T)
            print(f"[UGILE-SD21]   ugile_enabled={ugile_enabled} original_latent_hash={original_hash}")

            if ugile_enabled:
                x_init, diagnostics = sampler.build_initialization(
                    x_T,
                    text_embeddings,
                    extra_conditioning,
                    seed=seed,
                )
                profile_original = diagnostics["profile_original_latents"]
            else:
                x_init = x_T
                diagnostics = {}
                profile_original = None

            x_final = sampler.full_forward_pass(x_init, text_embeddings, extra_conditioning)
            final_init_hash = latent_hash(x_init)

            if save_original:
                base_latents = x_final if profile_original is None else profile_original
                base_path = _base_path(p_idx, seed)
                wrapper.decode_latents(base_latents).save(base_path)
                print(f"[UGILE-SD21]   Base  -> {base_path}")
                base_record = base_manifest_record(
                    wrapper=wrapper,
                    sampler=sampler,
                    cfg=cfg,
                    architecture="SD21",
                    prompt_idx=p_idx,
                    prompt=prompt,
                    negative_prompt=opts["negative_prompt"],
                    seed=seed,
                    ugile_enabled=False,
                    original_xT=x_T,
                    final_init=x_T,
                    output_path=base_path,
                )
                append_pair_manifest(base_record, cfg=cfg)
                records.append(base_record)

            if ugile_enabled:
                out_path = _branch_path(p_idx, seed, 0)
                wrapper.decode_latents(x_final).save(out_path)
                print(f"[UGILE-SD21]   Branch 0 -> {out_path}")
                on_record = base_manifest_record(
                    wrapper=wrapper,
                    sampler=sampler,
                    cfg=cfg,
                    architecture="SD21",
                    prompt_idx=p_idx,
                    prompt=prompt,
                    negative_prompt=opts["negative_prompt"],
                    seed=seed,
                    ugile_enabled=True,
                    original_xT=x_T,
                    final_init=x_init,
                    output_path=out_path,
                    diagnostics=_scalar_diag(diagnostics),
                )
                on_record["ugile_method"] = "threshold"
                append_pair_manifest(on_record, cfg=cfg)
                records.append(on_record)

    print(f"\n[UGILE-SD21] Done — {len(records)} manifest record(s)")
    return records
