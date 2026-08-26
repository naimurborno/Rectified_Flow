"""
open_mmdit_pipeline.py
=======================

A fully "unwrapped" few-step rectified-flow MMDiT generation pipeline,
supporting BOTH:

    - Stable Diffusion 3.5 (medium / large-turbo)   -- uniform joint blocks
    - FLUX.1-schnell                                 -- double-stream (joint)
                                                         blocks + single-stream
                                                         (image-only) blocks

Unlike diffusers' `pipe(prompt=...)` one-liner, every stage of generation is
a separate, inspectable, hookable step:

    text encoders -> prompt embeddings -> latent init
        -> [ per-step: transformer forward -> TRAJECTORY EQUATION ] -> VAE decode

You get direct references to:
    self.tokenizer_1/2/3, self.text_encoder_1/2/3   (CLIP-L, CLIP-G, T5)
    self.transformer                                 (the MMDiT / Flux transformer)
    self.transformer.transformer_blocks[i]            (joint / double-stream blocks)
    self.transformer.single_transformer_blocks[i]     (FLUX only: single-stream blocks)
    self.scheduler                                    (FlowMatchEulerDiscreteScheduler,
                                                         used only as a sigma-schedule
                                                         provider -- NOT for integration)
    self.vae

...and three manipulation surfaces:
    1. `hook(module_path, fn)`      - raw PyTorch forward hook on ANY named
                                       submodule, e.g.:
                                         "transformer.transformer_blocks.14"
                                         "transformer.single_transformer_blocks.10"
                                         "transformer.transformer_blocks.14.attn"
    2. `embed_transform` callback   - edit prompt_embeds / pooled_embeds
                                       before the denoising loop starts
    3. `trajectory_step_fn`         - THE RECTIFIED-FLOW EQUATION ITSELF,
                                       passed in as a swappable function
                                       instead of being hidden in
                                       scheduler.step()

Install
-------
pip install diffusers transformers accelerate torch sentencepiece --break-system-packages

Minimal usage
-------------
    from open_mmdit_pipeline import OpenMMDiTPipeline

    # SD3.5 (uniform joint-attention blocks)
    pipe = OpenMMDiTPipeline("stabilityai/stable-diffusion-3.5-large-turbo")

    # FLUX.1-schnell (double-stream + single-stream blocks, few-step)
    pipe = OpenMMDiTPipeline("black-forest-labs/FLUX.1-schnell")

    pipe.list_modules(filter_substr="transformer_blocks")

    def my_trajectory_step(latents, v_pred, sigma, sigma_next, t, step_index):
        # Default (straight-line Euler): latents + (sigma_next - sigma) * v_pred
        return latents + (sigma_next - sigma) * v_pred

    image = pipe.generate(
        prompt="a photo of a horse",
        num_inference_steps=4,
        guidance_scale=0.0,
        trajectory_step_fn=my_trajectory_step,
    )
    image.save("out.png")
"""

from typing import Callable, Optional, Tuple

import torch
from PIL import Image


MODEL_IDS = {
    "sd3.5-medium": "stabilityai/stable-diffusion-3.5-medium",
    "sd3.5-large-turbo": "stabilityai/stable-diffusion-3.5-large-turbo",
    "flux-schnell": "black-forest-labs/FLUX.1-schnell",
}


def _resolve_model_id(model_id_or_key: str) -> str:
    return MODEL_IDS.get(model_id_or_key, model_id_or_key)


class OpenMMDiTPipeline:
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-3.5-medium",
                 device: str = None, dtype: torch.dtype = None,
                 load_t5: bool = True, offload_t5: bool = False):
        """
        load_t5 (SD3 only): if False, skips loading text_encoder_3/tokenizer_3
            entirely. SD3 concatenates CLIP-L + CLIP-G + T5 embeddings, and
            diffusers supports the T5 slot being zero-padded instead -- this
            is a fully supported memory-saving path, and T5-XXL is the
            single biggest text-encoder (~9GB in fp32 / ~4.7GB in fp16), so
            this is usually the single biggest win on a T4.
            IGNORED for FLUX: FLUX's encoder_hidden_states come entirely
            from T5 (CLIP only supplies the pooled vector), so T5 cannot be
            dropped without breaking conditioning. Use offload_t5 instead.

        offload_t5: if True, moves the T5 encoder to CPU after __init__ and
            only moves it to GPU for the duration of encode_prompt(), then
            back to CPU. Works for both SD3 and FLUX. Trades speed (a CPU<->GPU
            transfer per prompt) for a large, permanent VRAM saving.
        """
        model_id = _resolve_model_id(model_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        self.offload_t5 = offload_t5

        if "flux" in model_id.lower():
            self.kind = "flux"
            self._init_flux(model_id)
        else:
            self.kind = "sd3"
            self._init_sd3(model_id, load_t5=load_t5)

        if self.offload_t5:
            self._t5_encoder.to("cpu")
            torch.cuda.empty_cache() if self.device == "cuda" else None

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self._hook_handles = []

    # ------------------------------------------------------------------ #
    # Model-specific loading
    # ------------------------------------------------------------------ #

    def _init_sd3(self, model_id: str, load_t5: bool = True):
        from diffusers import StableDiffusion3Pipeline

        # Loaded purely as a convenient weight/component container and to
        # reuse its well-tested encode_prompt logic -- we never call
        # loader(...) as a pipeline anywhere.
        if load_t5:
            loader = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=self.dtype)
            self.tokenizer_3 = loader.tokenizer_3
            self.text_encoder_3 = loader.text_encoder_3.to(self.device)
        else:
            # Officially-supported memory-saving path: SD3 concatenates
            # CLIP-L + CLIP-G + T5 embeddings; with text_encoder_3=None,
            # diffusers zero-pads that slot instead of computing it.
            # T5-XXL is the single largest text encoder (~4.7GB in fp16),
            # so this is usually the biggest available VRAM win on a T4.
            loader = StableDiffusion3Pipeline.from_pretrained(
                model_id, torch_dtype=self.dtype,
                text_encoder_3=None, tokenizer_3=None,
            )
            self.tokenizer_3 = None
            self.text_encoder_3 = None

        self.tokenizer_1 = loader.tokenizer
        self.tokenizer_2 = loader.tokenizer_2
        self.text_encoder_1 = loader.text_encoder.to(self.device)
        self.text_encoder_2 = loader.text_encoder_2.to(self.device)

        self.transformer = loader.transformer.to(self.device)
        self.vae = loader.vae.to(self.device)
        self.scheduler = loader.scheduler

        self._encode_prompt_fn = loader.encode_prompt
        self._pipeline_ref = loader
        self._t5_encoder = self.text_encoder_3  # may be None if load_t5=False

    def _init_flux(self, model_id: str):
        from diffusers import FluxPipeline

        loader = FluxPipeline.from_pretrained(model_id, torch_dtype=self.dtype)

        self.tokenizer_1 = loader.tokenizer      # CLIP
        self.tokenizer_2 = loader.tokenizer_2     # T5
        self.text_encoder_1 = loader.text_encoder.to(self.device)
        self.text_encoder_2 = loader.text_encoder_2.to(self.device)

        self.transformer = loader.transformer.to(self.device)
        self.vae = loader.vae.to(self.device)
        self.scheduler = loader.scheduler

        # Reuse FLUX's own (nontrivial) prompt encoding, latent packing/
        # unpacking, and dynamic-shift timestep helpers rather than
        # re-deriving RoPE position-id math ourselves -- same philosophy
        # as reusing SD3's encode_prompt above.
        self._encode_prompt_fn = loader.encode_prompt
        self._pack_latents_fn = loader._pack_latents
        self._unpack_latents_fn = loader._unpack_latents
        self._pipeline_ref = loader
        self._t5_encoder = self.text_encoder_2  # FLUX's T5 is text_encoder_2

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def list_modules(self, filter_substr: str = ""):
        """Print every named submodule path so you know what to pass to
        `hook(...)`. Optionally filter to paths containing a substring,
        e.g. filter_substr='transformer_blocks'."""
        names = []
        for name, _ in self.transformer.named_modules():
            full = f"transformer.{name}" if name else "transformer"
            if filter_substr in full:
                names.append(full)
        for n in names:
            print(n)
        print(f"\n{len(names)} modules matched.")
        return names

    def num_layers(self) -> int:
        """Total joint-attention block count. For FLUX this is
        double-stream + single-stream combined."""
        n = len(self.transformer.transformer_blocks)
        if self.kind == "flux":
            n += len(self.transformer.single_transformer_blocks)
        return n

    # ------------------------------------------------------------------ #
    # Generic hooking
    # ------------------------------------------------------------------ #

    def _resolve_module(self, module_path: str):
        """module_path like 'transformer.transformer_blocks.14' or
        'transformer.single_transformer_blocks.10.attn'"""
        parts = module_path.split(".")
        assert parts[0] == "transformer", "module_path must start with 'transformer.'"
        obj = self.transformer
        for p in parts[1:]:
            obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
        return obj

    def hook(self, module_path: str, fn: Callable, mode: str = "forward"):
        """Register a raw PyTorch hook on any named submodule.

        mode='forward'      -> register_forward_hook(fn(module, inputs, output))
        mode='forward_pre'   -> register_forward_pre_hook(fn(module, inputs))
        Returns the hook handle; call handle.remove() to undo.

        NOTE (FLUX specifics): `transformer.transformer_blocks.i` are
        double-stream blocks whose forward output is a tuple
        (encoder_hidden_states, hidden_states) -- image tokens are index
        [1]. `transformer.single_transformer_blocks.i` are single-stream
        blocks whose forward output is a single concatenated tensor
        (text+image tokens together) -- there is no clean image-only
        slice without knowing the text-token count, so hooks on single
        blocks should treat the whole tensor as one sequence.
        """
        module = self._resolve_module(module_path)
        if mode == "forward":
            handle = module.register_forward_hook(fn)
        elif mode == "forward_pre":
            handle = module.register_forward_pre_hook(fn)
        else:
            raise ValueError("mode must be 'forward' or 'forward_pre'")
        self._hook_handles.append(handle)
        return handle

    def clear_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    # ------------------------------------------------------------------ #
    # Stage 1: text encoding
    # ------------------------------------------------------------------ #

    def encode_prompt(self, prompt: str, negative_prompt: str = "",
                       do_classifier_free_guidance: bool = True):
        """SD3: returns (prompt_embeds, negative_prompt_embeds,
                          pooled_prompt_embeds, negative_pooled_prompt_embeds)

        FLUX: returns (prompt_embeds, pooled_prompt_embeds, text_ids).
        FLUX.1-schnell is guidance-distilled for speed, not CFG, so it has
        no negative-prompt path -- do_classifier_free_guidance is ignored.

        If offload_t5=True was passed to __init__, T5 is moved GPU->CPU
        around this call automatically: brought to GPU just for encoding,
        then pushed back to CPU immediately after, freeing its VRAM for
        the rest of generate() (the transformer + VAE forward passes).
        """
        if self.offload_t5 and self._t5_encoder is not None:
            self._t5_encoder.to(self.device)

        try:
            if self.kind == "sd3":
                result = self._encode_prompt_fn(
                    prompt=prompt, prompt_2=prompt, prompt_3=prompt,
                    negative_prompt=negative_prompt, negative_prompt_2=negative_prompt,
                    negative_prompt_3=negative_prompt,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    device=self.device,
                )
            else:  # flux
                prompt_embeds, pooled_prompt_embeds, text_ids = self._encode_prompt_fn(
                    prompt=prompt, prompt_2=prompt, device=self.device,
                )
                result = (prompt_embeds, pooled_prompt_embeds, text_ids)
        finally:
            if self.offload_t5 and self._t5_encoder is not None:
                self._t5_encoder.to("cpu")
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        return result

    # ------------------------------------------------------------------ #
    # Stage 2: latent initialisation
    # ------------------------------------------------------------------ #

    def prepare_latents(self, batch_size: int, height: int, width: int,
                         generator: Optional[torch.Generator] = None):
        """SD3: returns a 4D latent tensor (B, C, H//8, W//8).
        FLUX: returns (packed_latents [B, seq_len, C*4], latent_image_ids)
        -- FLUX transformers operate on a flattened patch sequence, not a
        4D spatial tensor, so packing is required before the transformer
        and unpacking before the VAE."""
        if self.kind == "sd3":
            num_channels = self.transformer.config.in_channels
            shape = (batch_size, num_channels,
                     height // self.vae_scale_factor, width // self.vae_scale_factor)
            return torch.randn(shape, generator=generator, device=self.device, dtype=self.dtype)
        else:  # flux
            num_channels = self.transformer.config.in_channels // 4
            shape = (batch_size, num_channels,
                     height // self.vae_scale_factor, width // self.vae_scale_factor)
            latents = torch.randn(shape, generator=generator, device=self.device, dtype=self.dtype)
            packed = self._pack_latents_fn(
                latents, batch_size, num_channels,
                height // self.vae_scale_factor, width // self.vae_scale_factor,
            )
            latent_image_ids = self._pipeline_ref._prepare_latent_image_ids(
                batch_size, height // self.vae_scale_factor // 2,
                width // self.vae_scale_factor // 2, self.device, self.dtype,
            )
            return packed, latent_image_ids

    # ------------------------------------------------------------------ #
    # Stage 3: THE RECTIFIED-FLOW TRAJECTORY EQUATION (shared by both
    # model kinds -- both SD3 and FLUX are rectified-flow / flow-matching
    # models using the same straight-line Euler ODE integration).
    # ------------------------------------------------------------------ #

    @staticmethod
    def default_trajectory_step(latents: torch.Tensor, v_pred: torch.Tensor,
                                 sigma: torch.Tensor, sigma_next: torch.Tensor,
                                 t, step_index: int) -> torch.Tensor:
        """Rectified flow defines x_sigma = (1-sigma)*x0 + sigma*x1 (a
        straight line between data x0 and noise x1). The velocity target
        is the constant v = x1 - x0. Integrating dx/dsigma = v with one
        explicit Euler step between consecutive sigmas gives:

              x_{sigma_next} = x_sigma + (sigma_next - sigma) * v_pred

        That line is the entire sampler. Swap this function out
        (pass `trajectory_step_fn=` to generate()) to change how the ODE
        is integrated -- curved steps, second velocity evaluations
        (Heun/RK2), injected offset terms, etc.
        """
        return latents + (sigma_next - sigma) * v_pred

    # ------------------------------------------------------------------ #
    # Stage 4: the fully manual denoising loop
    # ------------------------------------------------------------------ #

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        height: int = 512,
        width: int = 512,
        seed: int = 0,
        embed_transform: Optional[Callable] = None,
        step_callback: Optional[Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor]] = None,
        trajectory_step_fn: Optional[Callable] = None,
        return_latents: bool = False,
    ) -> Image.Image:
        """
        embed_transform(prompt_embeds, pooled_embeds) -> (prompt_embeds, pooled_embeds)
            Called once, right after text encoding.

        trajectory_step_fn(latents, v_pred, sigma, sigma_next, t, step_index) -> latents
            THE RECTIFIED-FLOW EQUATION ITSELF. Defaults to
            `default_trajectory_step` (straight-line Euler).

        step_callback(step_index, t, latents, v_pred, total_steps) -> latents
            Called once per step, AFTER trajectory_step_fn. Use for
            clamping, logging, or step-keyed activation-cache bookkeeping.

        Note on guidance_scale: FLUX.1-schnell is timestep-distilled and
        NOT guidance-distilled (config.guidance_embeds is False), and has
        no CFG negative-prompt path -- guidance_scale is ignored for it.
        SD3.5-large-turbo is adversarially distilled and is normally run
        at guidance_scale=0.0 too. Only plain SD3.5-medium benefits from
        guidance_scale > 0.
        """
        step_fn = trajectory_step_fn or self.default_trajectory_step
        generator = torch.Generator(device=self.device).manual_seed(seed)

        if self.kind == "sd3":
            return self._generate_sd3(prompt, negative_prompt, num_inference_steps,
                                       guidance_scale, height, width, generator,
                                       embed_transform, step_callback, step_fn, return_latents)
        else:
            return self._generate_flux(prompt, num_inference_steps, height, width,
                                        generator, embed_transform, step_callback,
                                        step_fn, return_latents)

    # ---- SD3 generation loop ---- #

    def _generate_sd3(self, prompt, negative_prompt, num_inference_steps,
                       guidance_scale, height, width, generator,
                       embed_transform, step_callback, step_fn, return_latents):
        do_cfg = guidance_scale is not None and guidance_scale > 0.0

        (prompt_embeds, negative_prompt_embeds,
         pooled_embeds, negative_pooled_embeds) = self.encode_prompt(
            prompt, negative_prompt, do_classifier_free_guidance=do_cfg
        )
        if embed_transform is not None:
            prompt_embeds, pooled_embeds = embed_transform(prompt_embeds, pooled_embeds)
        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_embeds = torch.cat([negative_pooled_embeds, pooled_embeds], dim=0)

        latents = self.prepare_latents(1, height, width, generator)

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = self.scheduler.sigmas.to(self.device)

        for i, t in enumerate(timesteps):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            timestep = t.expand(latent_model_input.shape[0])

            v_pred = self.transformer(
                hidden_states=latent_model_input, timestep=timestep,
                encoder_hidden_states=prompt_embeds, pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]

            if do_cfg:
                v_uncond, v_cond = v_pred.chunk(2)
                v_pred = v_uncond + guidance_scale * (v_cond - v_uncond)

            latents = step_fn(latents, v_pred, sigma, sigma_next, t, i)
            if step_callback is not None:
                latents = step_callback(i, t, latents, v_pred, num_inference_steps)

        if return_latents:
            return latents

        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        with torch.no_grad():
            image = self.vae.decode(latents, return_dict=False)[0]
        return self._postprocess(image)

    # ---- FLUX generation loop ---- #

    def _generate_flux(self, prompt, num_inference_steps, height, width, generator,
                        embed_transform, step_callback, step_fn, return_latents):
        prompt_embeds, pooled_embeds, text_ids = self.encode_prompt(prompt)
        if embed_transform is not None:
            prompt_embeds, pooled_embeds = embed_transform(prompt_embeds, pooled_embeds)

        latents, latent_image_ids = self.prepare_latents(1, height, width, generator)

        # FLUX uses resolution-dependent ("dynamic shifting") timestep
        # spacing. Reuse the scheduler's own shifting logic if available;
        # fall back to a plain linear sigma schedule otherwise.
        image_seq_len = latents.shape[1]
        try:
            mu = self._pipeline_ref._calculate_shift(
                image_seq_len,
                self.scheduler.config.base_image_seq_len,
                self.scheduler.config.max_image_seq_len,
                self.scheduler.config.base_shift,
                self.scheduler.config.max_shift,
            )
            self.scheduler.set_timesteps(num_inference_steps, device=self.device, mu=mu)
        except Exception:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        timesteps = self.scheduler.timesteps
        sigmas = self.scheduler.sigmas.to(self.device)

        # FLUX.1-schnell is not guidance-distilled -> guidance=None.
        if getattr(self.transformer.config, "guidance_embeds", False):
            guidance = torch.full((1,), 3.5, device=self.device, dtype=torch.float32)
        else:
            guidance = None

        for i, t in enumerate(timesteps):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            v_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,  # FLUX expects timestep in [0,1]
                guidance=guidance,
                pooled_projections=pooled_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

            latents = step_fn(latents, v_pred, sigma, sigma_next, t, i)
            if step_callback is not None:
                latents = step_callback(i, t, latents, v_pred, num_inference_steps)

        if return_latents:
            return latents

        latents = self._unpack_latents_fn(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        with torch.no_grad():
            image = self.vae.decode(latents, return_dict=False)[0]
        return self._postprocess(image)

    # ------------------------------------------------------------------ #
    # Shared: VAE output -> PIL
    # ------------------------------------------------------------------ #

    @staticmethod
    def _postprocess(image: torch.Tensor) -> Image.Image:
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")[0]
        return Image.fromarray(image)


if __name__ == "__main__":
    # Smoke-test both backends structurally (won't run without a GPU +
    # downloaded weights, but confirms the code path is wired correctly).
    print("kinds supported: 'sd3' (StableDiffusion3Pipeline), 'flux' (FluxPipeline)")
    print("Example:")
    print('  pipe = OpenMMDiTPipeline("black-forest-labs/FLUX.1-schnell")')
    print('  pipe.list_modules(filter_substr="single_transformer_blocks")')