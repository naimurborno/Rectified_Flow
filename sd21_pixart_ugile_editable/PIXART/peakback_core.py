"""
peakback_core.py
----------------
Pure-tensor math for PeakBack, kept separate from model/scheduler code so it
can be unit-tested without loading SD3 (see test_peakback_core.py).

Implements, with direct references to the derivations in the method section:
  - common_x0_discrepancy   : shared predicted-clean-sample discrepancy space
  - ugile_potential         : U_k = ||d_k^x0||
  - tweedie_potential       : legacy PeakBack potential
  - joint_projector         : orthogonal to semantic and radial axes
  - geodesic_step           : exact-norm move, no renormalization
  - quantile_threshold        : Eq. 7  (u*)
  - leverage_surrogate        : Eq. 9  (zero-cost finite-difference leverage score)
"""

import math
import torch


def tweedie_potential(v_cond: torch.Tensor, v_uncond: torch.Tensor, sigma: float) -> torch.Tensor:
    """Eq. 5: U_k = sigma_k * ||v_cond - v_uncond||_2 (free — no extra forward pass)."""
    return sigma * (v_cond.float() - v_uncond.float()).norm()


def common_x0_discrepancy(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Shared predicted-clean-sample discrepancy: d_k^x0 = -sigma_k (f_c - f_u)."""
    sigma_t = torch.as_tensor(
        sigma,
        dtype=torch.float32,
        device=pred_cond.device,
    )
    return -sigma_t * (
        pred_cond.float() - pred_uncond.float()
    )


def ugile_potential(common_diff: torch.Tensor) -> torch.Tensor:
    """UGILE potential in common predicted-x0 discrepancy space: U_k = ||d_k^x0||_2."""
    return common_diff.float().norm()


def _safe_unit(vec: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norm = vec.norm()
    if not torch.isfinite(norm) or norm <= eps:
        return torch.zeros_like(vec)
    return vec / norm


def joint_projector(w: torch.Tensor, s: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Project flattened vector w onto the orthogonal complement of span(s, x).

    The implementation builds a rank-aware orthonormal basis using a stable
    Gram-Schmidt/reprojection pass, so nearly collinear semantic/radial axes
    do not require a fragile 2x2 solve.
    """
    out_dtype = w.dtype
    w = w.reshape(-1).double()
    s = s.reshape(-1).double()
    x = x.reshape(-1).double()

    basis = []
    s_hat = _safe_unit(s, eps)
    if s_hat.norm() > 0:
        basis.append(s_hat)

    x_orth = x.clone()
    for b in basis:
        x_orth = x_orth - torch.dot(x_orth, b) * b
    x_hat = _safe_unit(x_orth)
    if x_hat.norm() > 0:
        basis.append(x_hat)

    w = w.clone()
    for _ in range(2):
        for b in basis:
            w = w - torch.dot(w, b) * b
    return w.to(dtype=out_dtype)


def adaptive_escape_budget(
    common_diffs,
    weights,
    semantic_dir,
    covariance_dir,
    trajectory_states,
    x_T,
    eps=1e-8,
):
    """
    Compute the online trajectory-adaptive UGILE budget.

    Returns lambda1, orthogonal_energy, directional_confidence, freedom_score,
    safe_freedom_score, trajectory_step_scale, phase2_angle,
    epsilon_adaptive, and theta_adaptive. All reductions are FP32 and no
    covariance matrix is materialized.
    """
    device = x_T.device
    x_T_f = x_T.float()
    shape = x_T.shape

    clean_diffs = [
        d.reshape(-1).to(device=device, dtype=torch.float32)
        for d in common_diffs
        if d is not None
    ]
    if not clean_diffs:
        clean_diffs = [torch.zeros_like(x_T_f).reshape(-1)]

    weights = torch.as_tensor(weights, device=device, dtype=torch.float32).flatten()
    if weights.numel() != len(clean_diffs):
        weights = torch.ones(len(clean_diffs), device=device, dtype=torch.float32)
    weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
    weight_sum = weights.sum()

    zero = torch.zeros((), device=device, dtype=torch.float32)

    def zero_budget():
        return {
            "lambda1": zero,
            "orthogonal_energy": zero,
            "directional_confidence": zero,
            "freedom_score": zero,
            "safe_freedom_score": zero,
            "trajectory_step_scale": zero,
            "phase2_angle": zero,
            "epsilon_adaptive": zero,
            "theta_adaptive": zero,
        }

    if not torch.isfinite(weight_sum) or weight_sum <= eps:
        return zero_budget()

    weights = weights / weight_sum

    semantic_flat = semantic_dir.reshape(-1).to(device=device, dtype=torch.float32)
    semantic_norm = semantic_flat.norm()
    if not torch.isfinite(semantic_norm) or semantic_norm <= eps:
        return zero_budget()
    s_hat = semantic_flat / (semantic_norm + eps)

    v1 = covariance_dir.reshape(-1).to(device=device, dtype=torch.float32)
    v1 = v1 - torch.dot(v1, s_hat) * s_hat
    v1_norm = v1.norm()
    if not torch.isfinite(v1_norm) or v1_norm <= eps:
        return zero_budget()
    v1 = v1 / (v1_norm + eps)

    mu = torch.zeros_like(clean_diffs[0])
    for d_k, w_k in zip(clean_diffs, weights):
        mu = mu + w_k * d_k
    mu_norm = mu.norm()
    if not torch.isfinite(mu_norm):
        return zero_budget()

    lambda1 = torch.zeros((), device=device, dtype=torch.float32)
    orthogonal_energy = torch.zeros((), device=device, dtype=torch.float32)
    for d_k, w_k in zip(clean_diffs, weights):
        centered = d_k - mu
        centered_perp = centered - torch.dot(centered, s_hat) * s_hat
        rayleigh = torch.dot(centered_perp, v1)
        lambda1 = lambda1 + w_k * rayleigh.square()
        orthogonal_energy = orthogonal_energy + w_k * centered_perp.square().sum()

    if (
        not torch.isfinite(lambda1)
        or not torch.isfinite(orthogonal_energy)
        or orthogonal_energy <= eps
    ):
        return zero_budget()

    A = torch.clamp(lambda1 / (orthogonal_energy + eps), 0.0, 1.0)
    F_score = torch.clamp(lambda1 / (lambda1 + mu_norm.square() + eps), 0.0, 1.0)
    G = torch.sqrt(torch.clamp(A * F_score, 0.0, 1.0))

    if not torch.isfinite(G) or G <= eps:
        return zero_budget()

    states = [
        s.to(device=device, dtype=torch.float32)
        for s in trajectory_states
        if s is not None
    ]
    if len(states) < 2:
        return zero_budget()

    traj_sq = torch.zeros((), device=device, dtype=torch.float32)
    usable_steps = min(len(clean_diffs), len(states) - 1, weights.numel())
    for k in range(usable_steps):
        step = (states[k + 1] - states[k]).reshape(-1)
        step_norm_sq = step.square().sum()
        if not torch.isfinite(step_norm_sq):
            return zero_budget()
        traj_sq = traj_sq + weights[k] * step_norm_sq

    if not torch.isfinite(traj_sq) or traj_sq < 0:
        return zero_budget()

    trajectory_step_scale = torch.sqrt(torch.clamp(traj_sq, min=0.0))
    r = x_T_f.norm()
    if not torch.isfinite(r) or r <= eps:
        return zero_budget()

    phase2_angle = G * torch.atan2(trajectory_step_scale, r + eps)
    epsilon_adaptive = r * torch.tan(phase2_angle)
    theta_adaptive = torch.sqrt(A) * torch.atan2(torch.sqrt(lambda1), mu_norm + eps)

    if not all(bool(torch.isfinite(v).item()) for v in [phase2_angle, epsilon_adaptive, theta_adaptive]):
        return zero_budget()

    return {
        "lambda1": lambda1,
        "orthogonal_energy": orthogonal_energy,
        "directional_confidence": A,
        "freedom_score": F_score,
        "safe_freedom_score": G,
        "trajectory_step_scale": trajectory_step_scale,
        "phase2_angle": phase2_angle,
        "epsilon_adaptive": epsilon_adaptive,
        "theta_adaptive": theta_adaptive,
    }


def trajectory_covariance_direction(
    diffs,
    weights,
    semantic_dir,
    seed,
    shape,
    device,
    eps=1e-8,
    num_power_iters=3,
):
    """Dominant trajectory-covariance direction without materializing C."""
    device = torch.device(device)
    semantic_flat = semantic_dir.reshape(-1).to(device=device, dtype=torch.float32)
    s_hat = _safe_unit(semantic_flat, eps)

    clean_diffs = [
        d.reshape(-1).to(device=device, dtype=torch.float32)
        for d in diffs
        if d is not None
    ]
    if not clean_diffs:
        clean_diffs = [torch.zeros(int(torch.tensor(shape).prod().item()), device=device)]

    weights = torch.as_tensor(weights, device=device, dtype=torch.float32).flatten()
    if weights.numel() != len(clean_diffs):
        weights = torch.ones(len(clean_diffs), device=device, dtype=torch.float32)
    weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
    weight_sum = weights.sum()

    generator = torch.Generator(device=device)
    generator.manual_seed(seed * 10000 + 0)

    def fallback():
        vec = torch.randn(shape, generator=generator, dtype=torch.float32, device=device).flatten()
        if s_hat.norm() > 0:
            vec = vec - torch.dot(vec, s_hat) * s_hat
        return _safe_unit(vec, eps).view(shape)

    if weight_sum <= eps:
        return fallback()

    weights = weights / weight_sum
    mu = torch.zeros_like(clean_diffs[0])
    for d_k, w_k in zip(clean_diffs, weights):
        mu = mu + w_k * d_k

    v = torch.randn(shape, generator=generator, dtype=torch.float32, device=device).flatten()
    if s_hat.norm() > 0:
        v = v - torch.dot(v, s_hat) * s_hat
    v = _safe_unit(v, eps)

    for _ in range(num_power_iters):
        v_next = torch.zeros_like(v)
        for d_k, w_k in zip(clean_diffs, weights):
            dc = d_k - mu
            v_next = v_next + w_k * dc * torch.dot(dc, v)
        if s_hat.norm() > 0:
            v_next = v_next - torch.dot(v_next, s_hat) * s_hat
        if not torch.isfinite(v_next).all() or v_next.norm() <= eps:
            return fallback()
        v = _safe_unit(v_next, eps)

    if not torch.isfinite(v).all() or v.norm() <= eps:
        return fallback()
    return v.view(shape)


def geodesic_step(x: torch.Tensor, w: torch.Tensor, r: float, theta_max: float = None, eps: float = 1e-12):
    """
    Eq. 11. Exact great-circle move on the sphere of radius r, in the plane
    spanned by x and w (w must already be orthogonal to x, e.g. the output
    of joint_projector). Returns (x_new, theta_used).

    No renormalization step exists or is needed: ||x_new|| == r exactly,
    for any theta (Proposition 2).
    """
    x = x.float()
    w = w.float()
    w_norm = w.norm() + eps
    theta = w_norm / r
    if theta_max is not None:
        theta = torch.clamp(theta, max=theta_max)
    w_hat = w / w_norm
    x_new = x * torch.cos(theta) + r * w_hat * torch.sin(theta)
    return x_new, theta


def coupling_error_bound(theta: float, r: float, v_norm: float) -> float:
    """Eq. 12 upper bound: |delta_sigma'| <= (theta^2 / 2) * (r / ||v||)."""
    return 0.5 * (theta ** 2) * (r / v_norm)


def quantile_threshold(values, q: float) -> float:
    """Eq. 7: Quantile_{1-q}({U_k}) via simple sorted-index lookup (no numpy dependency)."""
    vals = sorted(float(v) for v in values)
    if not vals:
        return float("inf")
    idx = round((1.0 - q) * (len(vals) - 1))
    idx = max(0, min(idx, len(vals) - 1))
    return vals[idx]


def leverage_surrogate(U_seq, sigma_seq, eps: float = 1e-8):
    """
    Eq. 9: free finite-difference leverage score, using only the cached
    {U_k, sigma_k} from the forward profiling pass — zero extra network calls.
    Returns a dict {index: score} over interior indices.
    """
    n = len(U_seq)
    scores = {}
    for k in range(1, n - 1):
        dU = abs(U_seq[k + 1] - U_seq[k - 1])
        dsig = abs(sigma_seq[k + 1] - sigma_seq[k - 1]) + eps
        scores[k] = U_seq[k] * (dU / dsig)
    return scores


def select_top_peaks(scores: dict, top_j: int, min_sep: int):
    """Greedy top-J selection by score, enforcing a minimum index separation."""
    ordered = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    selected = []
    for k in ordered:
        if all(abs(k - s) >= min_sep for s in selected):
            selected.append(k)
        if len(selected) >= top_j:
            break
    return sorted(selected)
