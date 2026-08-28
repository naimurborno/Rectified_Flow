"""
color_eval.py
-------------
Phase 0 featurization + per-prompt metrics.

Usage:
    # check folder mapping first, no GPU:
    python color_eval.py --run-dir outputs/original --out results/sd3 --dry-run

    # full run:
    python color_eval.py --run-dir outputs/original --out results/sd3

Writes:
    results/sd3/per_image.csv   -- one row per (prompt, seed)
    results/sd3/metrics.csv     -- one row per prompt
    results/sd3/cache.npz       -- histograms and CLIP embeddings (for gates.py)
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2lab

# ── frozen config: never change these between runs you intend to compare ──
N_SEEDS       = 64
L_BINS, A_BINS, B_BINS = 4, 8, 8
N_BINS        = L_BINS * A_BINS * B_BINS          # 256
SMOOTH_SIGMA  = (0.0, 0.7, 0.7)                   # (L, a*, b*) bin-space
MASK_THRESH   = 0.5
MASK_FALLBACK = 0.3
MIN_AREA      = 0.02
DD_K          = 1.1                               # Diverse-Diffusion (2310.12583)
CLIPSEG_ID    = "CIDAS/clipseg-rd64-refined"
CLIP_ID       = "openai/clip-vit-large-patch14"

# index-aligned to prompts_phase0.yaml
# 0-27  variable-color  (manufactured artifacts)
# 28-39 canonical-color (negative control)
NOUNS = [
    "car", "truck", "minivan", "bicycle", "scooter",        # 0-4
    "gown", "jersey", "sweatshirt", "kimono", "shoe",       # 5-9
    "necktie", "socks", "backpack", "handbag", "umbrella",  # 10-14
    "sofa", "chair", "lamp", "pillow", "curtain",           # 15-19
    "mug", "teapot", "bottle", "vase",                      # 20-23
    "towel", "yarn", "mitten", "tricycle",                  # 24-27
    "zebra", "panda", "dog", "tiger",                       # 28-31
    "ladybug", "butterfly", "fish",                         # 32-34
    "lemon", "bananas", "strawberry", "broccoli", "pineapple",  # 35-39
]
N_VARIABLE = 28


# ───────────────────────────────── metrics ────────────────────────────────
def rke2(K: np.ndarray) -> float:
    """Order-2 Vendi (RKE).

    VS_2 = n^2 / ||K||_F^2   (exact, no eigendecomposition needed).
    Range [1, n]: 1 when all samples identical, n when mutually orthogonal.
    Finite-sample convergence holds for any kernel at order 2.
    """
    n = K.shape[0]
    return float(n * n / np.sum(K * K))


def lab_hist(lab: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Smoothed, L1-normalised 4×8×8 CIELAB histogram over masked pixels.

    Bin edges are frozen (not data-dependent) so histograms are comparable
    across all prompts, seeds, and model configs.
    """
    L = np.clip(lab[..., 0][mask], 0.0,   99.999)
    a = np.clip(lab[..., 1][mask], -100.0, 99.999)
    b = np.clip(lab[..., 2][mask], -100.0, 99.999)

    li = np.clip((L / 25.0).astype(np.int64),            0, L_BINS - 1)
    ai = np.clip(((a + 100.0) / 25.0).astype(np.int64),  0, A_BINS - 1)
    bi = np.clip(((b + 100.0) / 25.0).astype(np.int64),  0, B_BINS - 1)

    flat = li * (A_BINS * B_BINS) + ai * B_BINS + bi
    h = np.bincount(flat, minlength=N_BINS).astype(np.float64)

    # Smooth only in (a*, b*) -- L has only 4 bins and represents scene
    # lighting, not colorimetric diversity.
    h = gaussian_filter(h.reshape(L_BINS, A_BINS, B_BINS),
                        sigma=SMOOTH_SIGMA, mode="nearest").ravel()
    s = h.sum()
    return h / s if s > 0 else np.full(N_BINS, 1.0 / N_BINS)


def bhattacharyya_gram(H: np.ndarray) -> np.ndarray:
    """Bhattacharyya affinity kernel: K_ij = sum_b sqrt(h_i[b] * h_j[b]).

    Equivalent to cosine similarity between sqrt(h_i) and sqrt(h_j).
    PSD by construction (inner product in R^256).
    Unit diagonal for free since ||sqrt(h)||^2 = sum(h) = 1.
    No bandwidth parameter -- nothing to tune.
    """
    S = np.sqrt(H)
    return S @ S.T


def dd_label(rgb: np.ndarray, mask: np.ndarray | None, k: float = DD_K) -> str:
    """Diverse-Diffusion (arXiv:2310.12583) per-image colour label.

    Mean RGB over masked pixels -> dominant channel by ratio k.
    """
    px = rgb[mask] if mask is not None else rgb.reshape(-1, 3)
    if px.size == 0:
        return "None"
    r, g, b_ = px.mean(axis=0)
    if r > k * g and r > k * b_:   return "R"
    if g > k * r and g > k * b_:   return "G"
    if b_ > k * r and b_ > k * g:  return "B"
    return "None"


def hue_circvar(thetas: np.ndarray) -> float:
    """Circular variance of per-image chroma-weighted mean hues.

    1 - |mean resultant|  in [0, 1].  Kernel-free diversity check.
    """
    z = np.exp(1j * thetas).mean()
    return float(1.0 - np.abs(z))


def rke_subsample(K: np.ndarray, m: int = 32, reps: int = 50,
                  seed: int = 0):
    """Subsample-based RKE stability check at n=m.

    IMPORTANT: never compare RKE values computed at different n -- the
    statistic is hard-capped at n.  This is for within-config CI only.
    """
    rng = np.random.default_rng(seed)
    n = K.shape[0]
    if n <= m:
        return float("nan"), float("nan")
    vals = []
    for _ in range(reps):
        idx = rng.choice(n, size=m, replace=False)
        vals.append(rke2(K[np.ix_(idx, idx)]))
    return float(np.mean(vals)), float(np.std(vals))


# ────────────────────────────────── models ──────────────────────────────────
class Featurizer:
    """Loads CLIPSeg (masking) and CLIP ViT-L/14 (diversity embedding)."""

    def __init__(self, device: str = "cuda", batch: int = 16):
        from transformers import (CLIPModel, CLIPProcessor,
                                  CLIPSegForImageSegmentation, CLIPSegProcessor)
        self.device = device
        self.batch  = batch
        self.seg_proc  = CLIPSegProcessor.from_pretrained(CLIPSEG_ID)
        self.seg       = CLIPSegForImageSegmentation.from_pretrained(
            CLIPSEG_ID).to(device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(CLIP_ID)
        self.clip      = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()

    @torch.no_grad()
    def masks(self, imgs: list, noun: str, hw: tuple) -> np.ndarray:
        """Sigmoid CLIPSeg logits, upsampled to native resolution."""
        out = []
        for i in range(0, len(imgs), self.batch):
            chunk = imgs[i:i + self.batch]
            inp = self.seg_proc(text=[noun] * len(chunk), images=chunk,
                                padding=True, return_tensors="pt").to(self.device)
            logits = self.seg(**inp).logits
            if logits.ndim == 2:
                logits = logits[None]
            p = torch.sigmoid(logits)[:, None]       # (B,1,352,352)
            p = F.interpolate(p, size=hw, mode="bilinear", align_corners=False)
            out.append(p[:, 0].float().cpu().numpy())
        return np.concatenate(out, axis=0)

    @torch.no_grad()
    def clip_feats(self, imgs: list) -> np.ndarray:
        """L2-normalised CLIP ViT-L/14 embeddings of the FULL unmasked image."""
        out = []
        for i in range(0, len(imgs), self.batch):
            inp = self.clip_proc(images=imgs[i:i + self.batch],
                                return_tensors="pt").to(self.device)
            vision_out = self.clip.vision_model(pixel_values=inp["pixel_values"])
            pooled = vision_out.pooler_output if hasattr(vision_out, "pooler_output") else vision_out[1]
            f = self.clip.visual_projection(pooled)   # (B, 768) projected embedding
            out.append((f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy())
        return np.concatenate(out, axis=0)


# ──────────────────────────────────── main ────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Phase 0 featurization: Color-RKE, CLIP-RKE, and allies.")
    ap.add_argument("--run-dir",  required=True,
                    help="root of the generation run "
                         "(one sub-folder per prompt, seed*.png inside)")
    ap.add_argument("--out",      required=True,
                    help="output directory for CSVs and cache.npz")
    ap.add_argument("--prompts",  default="prompts_phase0.yaml")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch",    type=int, default=16)
    ap.add_argument("--n-seeds",  type=int, default=N_SEEDS)
    ap.add_argument("--dry-run",  action="store_true",
                    help="resolve folder -> prompt mapping and exit; "
                         "no GPU, no models loaded")
    args = ap.parse_args()

    from discovery import resolve_run
    records, problems = resolve_run(
        args.run_dir, args.prompts, NOUNS, N_VARIABLE, args.n_seeds)
    if args.dry_run:
        raise SystemExit(0 if not problems else 1)
    if not records:
        raise SystemExit("nothing to featurize")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fz = Featurizer(args.device, args.batch)
    per_image_rows, metric_rows, cache, dropped = [], [], {}, []

    for rec in records:
        idx  = rec["idx"]
        name = rec["dir"]
        noun = rec["noun"]
        cls  = rec["cls"]
        seed_pairs = rec["seeds"]          # [(seed_int, Path), ...]
        seeds = [s for s, _ in seed_pairs]
        imgs  = [Image.open(p).convert("RGB") for _, p in seed_pairs]

        hw     = (imgs[0].height, imgs[0].width)
        probs  = fz.masks(imgs, noun, hw)
        clip_e = fz.clip_feats(imgs)

        H, thetas, chroma_vals, fracs = [], [], [], []
        labels_masked, labels_full, valid_flags = [], [], []

        for j, (s, img) in enumerate(zip(seeds, imgs)):
            rgb = np.asarray(img, dtype=np.float64) / 255.0
            lab = rgb2lab(rgb)                   # sRGB (D65) -> CIELAB

            m = probs[j] >= MASK_THRESH
            if m.mean() < MIN_AREA:
                m = probs[j] >= MASK_FALLBACK   # frozen fallback
            frac = float(m.mean())
            ok   = frac >= MIN_AREA
            if not ok:
                m = np.ones(hw, dtype=bool)     # full-frame fallback

            a_m = lab[..., 1][m]
            b_m = lab[..., 2][m]
            c   = float(np.sqrt(a_m**2 + b_m**2).mean())
            th  = float(np.arctan2(b_m.mean(), a_m.mean()))

            H.append(lab_hist(lab, m))
            thetas.append(th)
            chroma_vals.append(c)
            fracs.append(frac)
            valid_flags.append(ok)
            labels_masked.append(dd_label(rgb, m))
            labels_full.append(dd_label(rgb, None))

            per_image_rows.append(dict(
                prompt_idx=idx, prompt_dir=name, noun=noun, cls=cls,
                seed=s, mask_frac=round(frac, 5), valid=int(ok),
                mean_chroma=round(c, 4), hue_theta=round(th, 5),
                dd_masked=labels_masked[-1], dd_full=labels_full[-1],
            ))

        n_valid = int(sum(valid_flags))
        if len(seeds) < args.n_seeds or n_valid < args.n_seeds:
            dropped.append((idx, name, len(seeds), n_valid))

        H      = np.stack(H)
        K_col  = bhattacharyya_gram(H)
        K_clip = clip_e @ clip_e.T              # cosine Gram, PSD, unit diag
        thetas_arr  = np.asarray(thetas)
        chroma_arr  = np.asarray(chroma_vals)
        c32_m, c32_s = rke_subsample(K_col)

        mask_suspect = not (0.05 <= float(np.median(fracs)) <= 0.90)
        metric_rows.append(dict(
            prompt_idx=idx, prompt_dir=name, noun=noun, cls=cls,
            n=len(seeds), n_valid=n_valid,
            color_rke=round(rke2(K_col),   4),
            clip_rke =round(rke2(K_clip),  4),
            mean_chroma=round(float(chroma_arr.mean()),  4),
            chroma_std =round(float(chroma_arr.std()),   4),
            hue_circvar=round(hue_circvar(thetas_arr),   4),
            dd_masked  =len(set(labels_masked)),
            dd_full    =len(set(labels_full)),
            mask_frac_median=round(float(np.median(fracs)), 5),
            n_invalid  =len(valid_flags) - n_valid,
            color_rke_n32_mean=round(c32_m, 4),
            color_rke_n32_std =round(c32_s, 4),
        ))
        cache[f"hist_{idx:03d}"]  = H
        cache[f"clip_{idx:03d}"]  = clip_e

        print(f"[{idx:03d}] {noun:<11s} {cls:<9s} "
              f"color-RKE {metric_rows[-1]['color_rke']:7.3f}  "
              f"CLIP-RKE {metric_rows[-1]['clip_rke']:6.3f}  "
              f"C* {metric_rows[-1]['mean_chroma']:6.2f}  "
              f"mask {metric_rows[-1]['mask_frac_median']:.3f}"
              + ("  <-- MASK SUSPECT" if mask_suspect else ""))

    # ── write outputs ──
    for rows, fn in ((per_image_rows, "per_image.csv"),
                     (metric_rows,    "metrics.csv")):
        with open(out_dir / fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    np.savez_compressed(out_dir / "cache.npz", **cache)

    if dropped:
        print("\nIncomplete prompts (drop from ALL configs -- RKE is capped "
              "at n, so comparing unequal n is invalid):")
        for idx, nm, nf, nv in dropped:
            print(f"  {idx:03d} {nm}  found={nf}  valid_masks={nv}")

    print(f"\nwrote {out_dir}/metrics.csv  ({len(metric_rows)} prompts)")


if __name__ == "__main__":
    main()
