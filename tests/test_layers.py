"""Straight-through selection and layer plumbing."""

from __future__ import annotations

import torch

from wirelogic.layers import GroupSum, LogicLayer, _straight_through_argmax


def test_straight_through_forward_is_hard_backward_is_soft() -> None:
    torch.manual_seed(0)
    logits = torch.randn(3, 4, requires_grad=True)
    values = torch.randn(3, 4)
    probs = torch.softmax(logits, dim=-1)
    out = _straight_through_argmax(probs, values, dim=-1)

    # Forward value equals the arg-max pick, exactly.
    hard = values.gather(-1, probs.argmax(-1, keepdim=True)).squeeze(-1)
    assert torch.equal(out, hard)

    # Gradient is that of the soft mixture, so it is generally non-zero on all
    # candidates (a hard arg-max would give zero gradient almost everywhere).
    out.sum().backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_layer_shapes_both_modes() -> None:
    x = torch.rand(8, 12)
    for learnable in (True, False):
        layer = LogicLayer(12, 20, learnable_connections=learnable, candidates=5)
        out = layer(x)
        assert out.shape == (8, 20)
        assert layer.resolved_gate_ids().shape == (20,)
        assert layer.resolved_wiring().shape == (20, 2)


def test_groupsum_divides_into_classes() -> None:
    head = GroupSum(num_wires=6, num_classes=3, tau=1.0)
    x = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 1.0]])
    out = head(x)
    assert out.shape == (1, 3)
    assert torch.allclose(out, torch.tensor([[2.0, 0.0, 2.0]]))
