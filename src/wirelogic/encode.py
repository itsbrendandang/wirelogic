"""Thermometer encoding of real features into binary input wires.

Logic gates consume bits, so continuous features are binarised before they reach
the first layer.  A thermometer (unary) code turns a value ``v in [0, 1]`` into
``bits`` monotonic indicators ``1[v > t_j]`` for evenly spaced thresholds
``t_j``.  Unlike a plain binary code it is order-preserving - nearby values share
most of their bits - which is what a logic network can exploit.

Crucially the encoded inputs are exactly ``{0, 1}``, so a hardened circuit sees
the same inputs as the soft model and the discretisation is lossless at the input
boundary.
"""

from __future__ import annotations

import torch


def thermometer_thresholds(bits: int) -> torch.Tensor:
    """Return ``bits`` thresholds evenly spaced in the open interval ``(0, 1)``."""
    if bits < 1:
        raise ValueError("need at least one threshold bit")
    return (torch.arange(bits, dtype=torch.float32) + 1.0) / (bits + 1.0)


def thermometer_encode(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Encode features ``(N, F)`` in ``[0, 1]`` into bits ``(N, F * bits)``.

    Args:
        x: Feature matrix with values in ``[0, 1]``.
        bits: Number of threshold bits per feature.

    Returns:
        A float32 tensor of ``0.0``/``1.0`` values, feature-major: the first
        ``bits`` columns encode feature 0, the next ``bits`` feature 1, and so on.
    """
    thr = thermometer_thresholds(bits).to(x.device)
    encoded = (x.unsqueeze(-1) > thr).to(torch.float32)  # (N, F, bits)
    return encoded.reshape(x.shape[0], -1)
