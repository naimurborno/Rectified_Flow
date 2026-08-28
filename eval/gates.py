"""
gates.py
--------
Phase 0 go/no-go gates.

Usage:
    # Gate A + B1 (single config -- run this first):
    python gates.py --a results/sd3/metrics.csv

    # add Gate B2 (needs a second config):
    python gates.py --a results/sd3/metrics.csv --b results/sd3_4step/metrics.csv

Gate order matters. A validates that the metric measures colour at all.
B1 validates that it is not redundant with what the field already reports.
B2 is the headline claim. Do not interpret B2 before A and B1 pass.
"""

import argparse
import csv

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

FLOAT_COLS = ("color_rke", "clip_rke", "mean_chroma", "chroma_std",
              "hue_circvar", "mask_frac_median")
INT_COLS   = ("prompt_idx", "n", "n_valid", "dd_masked", "dd_full", "n_invalid")

MASK_LO, MASK_HI = 0.05, 0.90
RULE = "=" * 70


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty metrics file: {path}")
    for r in rows:
        for k in FLOAT_COLS:
            if k in r:
                r[k] = float(r[k])
        for k in INT_COLS:
            if k in r:
                r[k] = int(r[k])
    return {r["prompt_idx"]: r for r in rows}


def col(rows, idxs, key):
    return np.array([rows[i][key] for i in idxs], dtype=float)


def boot_spearman(x, y, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(reps):
        s = rng.integers(0, n, n)
        if len(np.unique(x[s])) < 3 or len(np.unique(y[s])) < 3:
            continue
        vals.append(spearmanr(x[s], y[s]).statistic)
    if not vals:
        return float("nan"), float("nan")
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def pct_drop(logratio):
    """log(A/B) -> percent drop from A to B."""
    return 100.0 * (1.0 - np.exp(-logratio))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True,
                    help="reference config metrics.csv (e.g. many-step)")
    ap.add_argument("--b",
                    help="intervention config metrics.csv (e.g. few-step)")
    ap.add_argument("--rho-max", type=float, default=0.6,
                    help="Gate B1 threshold on Spearman rho (default 0.6)")
    args = ap.parse_args()

    A = load(args.a)
    keep = sorted(i for i, r in A.items() if r["n_valid"] == r["n"])
    if len(keep) < len(A):
        print(f"note: excluding {len(A) - len(keep)} prompt(s) with invalid "
              f"masks or short seed counts\n")
    if not keep:
        raise SystemExit("no complete prompts survive; fix masking first")

    var = [i for i in keep if A[i]["cls"] == "variable"]
    can = [i for i in keep if A[i]["cls"] == "canonical"]

    # ── mask guard ──────────────────────────────────────────────────────
    # Every gate below reads masked pixels. If the mask is wrong the numbers
    # are meaningless, and no p-value will tell you that.
    bad = [i for i in keep
           if not MASK_LO <= A[i]["mask_frac_median"] <= MASK_HI]
    print(RULE)
    print(f"MASK GUARD   {len(bad)} suspect prompt(s) of {len(keep)}")
    for i in bad:
        print(f"   {i:03d} {A[i]['noun']:<11s} "
              f"median mask frac = {A[i]['mask_frac_median']:.4f}")
    if not bad:
        print("   all median mask fractions within "
              f"[{MASK_LO:.2f}, {MASK_HI:.2f}]")

    # ── Gate A: does Color-RKE actually measure colour? ─────────────────
    if not var or not can:
        raise SystemExit("Gate A needs both variable and canonical prompts")
    cv, cc = col(A, var, "color_rke"), col(A, can, "color_rke")
    res = mannwhitneyu(cv, cc, alternative="greater")
    U, p_a = res.statistic, res.pvalue
    rb = 2.0 * U / (len(cv) * len(cc)) - 1.0
    pass_a = (p_a < 0.05) and (rb > 0.5)

    print("\n" + RULE)
    print(f"GATE A   variable > canonical Color-RKE   "
          f"(n={len(cv)} vs {len(cc)})")
    print(f"   median Color-RKE  variable   {np.median(cv):8.3f}")
    print(f"   median Color-RKE  canonical  {np.median(cc):8.3f}")
    print(f"   Mann-Whitney U={U:.1f}   p={p_a:.3e}   "
          f"rank-biserial={rb:+.3f}")
    print(f"   ->  {'PASS' if pass_a else 'FAIL'}")
    print(f"   corroboration   hue-circvar  "
          f"{np.median(col(A, var, 'hue_circvar')):.3f} vs "
          f"{np.median(col(A, can, 'hue_circvar')):.3f}")
    print(f"   prior art       DD-masked    "
          f"{np.median(col(A, var, 'dd_masked')):.1f} vs "
          f"{np.median(col(A, can, 'dd_masked')):.1f}   (ceiling 4)")
    print(f"   n=32 stability  mean sd across prompts "
          f"{np.mean([float(A[i]['color_rke_n32_std']) for i in keep]):.3f}")

    # ── Gate B1: redundant with CLIP-RKE? ───────────────────────────────
    x, y = col(A, keep, "color_rke"), col(A, keep, "clip_rke")
    sp = spearmanr(x, y)
    lo, hi = boot_spearman(x, y)
    pass_b1 = hi < args.rho_max

    print("\n" + RULE)
    print(f"GATE B1  Color-RKE vs CLIP-RKE across {len(keep)} prompts")
    print(f"   Spearman rho = {sp.statistic:+.3f}   "
          f"95% CI [{lo:+.3f}, {hi:+.3f}]   p={sp.pvalue:.3e}")
    print(f"   ->  {'PASS' if pass_b1 else 'FAIL'}   "
          f"(want upper CI < {args.rho_max})")

    if not args.b:
        print("\n" + RULE)
        print("GATE B2  skipped -- pass --b with the intervention config.")
        print("   Needs the SAME checkpoint with step count changed and CFG")
        print("   held fixed. A different model family confounds step count")
        print("   with weights and with CFG, and the result cannot be")
        print("   attributed to distillation.")
        return

    # ── Gate B2: does colour collapse more than CLIP diversity? ─────────
    B = load(args.b)
    pair = [i for i in var if i in B and B[i]["n_valid"] == B[i]["n"]]
    if len(pair) < 6:
        raise SystemExit(f"only {len(pair)} paired variable prompts; "
                         f"too few for Wilcoxon")

    # Log-ratios, not raw deltas: Color-RKE and CLIP-RKE live on different
    # scales, so subtracting one from the other is only meaningful in log space.
    rc = np.log(col(A, pair, "color_rke") / col(B, pair, "color_rke"))
    rl = np.log(col(A, pair, "clip_rke")  / col(B, pair, "clip_rke"))
    d  = rc - rl
    wres = wilcoxon(d, alternative="greater")
    nz = int(np.count_nonzero(d))
    rbc = 2.0 * wres.statistic / (nz * (nz + 1) / 2) - 1.0 if nz else float("nan")
    pass_b2 = wres.pvalue < 0.05

    print("\n" + RULE)
    print(f"GATE B2  colour collapses more than CLIP diversity   "
          f"({len(pair)} variable prompts, paired)")
    print(f"   median log-ratio  Color-RKE  {np.median(rc):+.4f}   "
          f"({pct_drop(np.median(rc)):+.1f}% A->B)")
    print(f"   median log-ratio  CLIP-RKE   {np.median(rl):+.4f}   "
          f"({pct_drop(np.median(rl)):+.1f}% A->B)")
    print(f"   Wilcoxon W={wres.statistic:.1f}   p={wres.pvalue:.3e}   "
          f"rank-biserial={rbc:+.3f}")
    print(f"   ->  {'PASS' if pass_b2 else 'FAIL'}")
    print(f"   corroboration  mean chroma C* "
          f"{np.mean(col(A, pair, 'mean_chroma')):.2f} -> "
          f"{np.mean(col(B, pair, 'mean_chroma')):.2f}   (oversaturation)")
    print(f"   corroboration  hue-circvar   "
          f"{np.median(col(A, pair, 'hue_circvar')):.3f} -> "
          f"{np.median(col(B, pair, 'hue_circvar')):.3f}")
    print(f"   prior art      DD-masked     "
          f"{np.median(col(A, pair, 'dd_masked')):.1f} -> "
          f"{np.median(col(B, pair, 'dd_masked')):.1f}   (ceiling 4)")

    # canonical prompts are excluded from B2 above -- no colour headroom to
    # lose -- but reporting them is a useful null check.
    canp = [i for i in can if i in B and B[i]["n_valid"] == B[i]["n"]]
    if canp:
        rcc = np.log(col(A, canp, "color_rke") / col(B, canp, "color_rke"))
        print(f"   null check     canonical Color-RKE log-ratio "
              f"{np.median(rcc):+.4f}  (expect near 0)")

    print("\n" + RULE)
    print(f"SUMMARY   A={'PASS' if pass_a else 'FAIL'}   "
          f"B1={'PASS' if pass_b1 else 'FAIL'}   "
          f"B2={'PASS' if pass_b2 else 'FAIL'}")


if __name__ == "__main__":
    main()
