"""The Yin-Yang classification dataset (Kriener et al., 2022).

Points are drawn uniformly from a disc and labelled ``yin`` (0), ``yang`` (1) or
``dot`` (2) by the classic yin-yang construction: a big circle of radius ``r_big``
split by two small circles (the "eyes") of radius ``r_small``.  The task is a
small, non-linear three-way problem that logic-gate networks handle well on a CPU,
which is why arXiv:2607.09399 uses it as its smallest benchmark.

Each point is described by four features ``(x, y, 1 - x, 1 - y)``; supplying the
complements is the standard encoding that makes the two "eyes" separable.
"""

from __future__ import annotations

import math

import torch

R_BIG = 0.5
R_SMALL = 0.1


def _which_class(x: float, y: float) -> int:
    # Coordinates live in [0, 1]^2 with the big circle centred at (0.5, 0.5).
    # The two eyes sit at the left and right thirds of the horizontal midline.
    left_x, right_x = 0.5 * R_BIG, 1.5 * R_BIG
    mid_y = R_BIG
    d_left = math.hypot(x - left_x, y - mid_y)
    d_right = math.hypot(x - right_x, y - mid_y)

    if d_left < R_SMALL or d_right < R_SMALL:
        return 2  # inside an eye -> dot

    crit_small_dot = d_right <= R_SMALL
    crit_left_band = R_SMALL < d_left <= 0.5 * R_BIG
    crit_upper = y > mid_y and d_right > 0.5 * R_BIG
    is_yin = crit_small_dot or crit_left_band or crit_upper
    return 0 if is_yin else 1


def make_yin_yang(n: int, *, seed: int, r_big: float = R_BIG) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``n`` labelled yin-yang points.

    Args:
        n: Number of points to return.
        seed: RNG seed for reproducibility.
        r_big: Radius of the enclosing circle.

    Returns:
        ``(features, labels)`` where ``features`` is ``(n, 4)`` float32 in ``[0, 1]``
        and ``labels`` is ``(n,)`` int64 in ``{0, 1, 2}``.
    """
    gen = torch.Generator().manual_seed(seed)
    cx = cy = r_big
    feats = torch.empty(n, 4, dtype=torch.float32)
    labels = torch.empty(n, dtype=torch.long)
    filled = 0
    while filled < n:
        batch = torch.rand(4 * (n - filled), 2, generator=gen) * (2 * r_big)
        for px, py in batch.tolist():
            if math.hypot(px - cx, py - cy) > r_big:
                continue
            feats[filled] = torch.tensor([px, py, 2 * r_big - px, 2 * r_big - py])
            labels[filled] = _which_class(px, py)
            filled += 1
            if filled == n:
                break
    # Normalise features to [0, 1] regardless of r_big.
    feats /= 2 * r_big
    return feats, labels
