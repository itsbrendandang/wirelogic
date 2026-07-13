"""Hardening must reproduce the soft model exactly in the discrete limit."""

from __future__ import annotations

import torch

from wirelogic.gates import hard_gate
from wirelogic.harden import HardLayer, harden
from wirelogic.model import LogicGateNetwork, ModelConfig


def _reference_bits(circuit_layers: list[HardLayer], x: torch.Tensor) -> torch.Tensor:
    """Naive Python evaluation of a hard circuit, used to check the vector path."""
    h = x.to(torch.long)
    for layer in circuit_layers:
        g = layer.gate_ids.tolist()
        w = layer.wiring.tolist()
        out = torch.zeros(h.shape[0], len(g), dtype=torch.long)
        for j in range(len(g)):
            a = h[:, w[j][0]]
            b = h[:, w[j][1]]
            out[:, j] = hard_gate(torch.full_like(a, g[j]), a, b).to(torch.long)
        h = out
    return h


def _random_binary(n: int, d: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return (torch.rand(n, d, generator=gen) > 0.5).to(torch.float32)


def test_vectorized_hard_forward_matches_reference() -> None:
    cfg = ModelConfig(in_features=16, hidden=(24, 24), out_wires=12, num_classes=3)
    model = LogicGateNetwork(cfg, generator=torch.Generator().manual_seed(1))
    circuit = harden(model)
    x = _random_binary(20, 16, seed=7)
    fast = circuit.forward_bits(x).to(torch.long)
    slow = _reference_bits(circuit.layers, x)
    assert torch.equal(fast, slow)


def _make_one_hot(logits: torch.Tensor, choice: torch.Tensor) -> None:
    """In place: turn ``logits`` into a hard one-hot at ``choice`` along dim -1."""
    logits.zero_()
    logits.scatter_(-1, choice.unsqueeze(-1), 30.0)  # softmax(30, 0, ...) == 1.0 in fp32


def test_one_hot_soft_model_matches_hard_circuit() -> None:
    # When every gate-type and connection distribution is exactly one-hot, the soft
    # mixture is discrete: on binary inputs each layer emits exact {0,1} values, so
    # the differentiable model and the hardened circuit must agree bit for bit.
    cfg = ModelConfig(in_features=16, hidden=(24, 24), out_wires=12, num_classes=3, candidates=6)
    gen = torch.Generator().manual_seed(2)
    model = LogicGateNetwork(cfg, generator=gen)
    with torch.no_grad():
        for layer in model.logic_layers():
            g, a = layer.gate_logits.shape
            _make_one_hot(layer.gate_logits, torch.randint(0, a, (g,), generator=gen))
            gc, pc, k = layer.conn_logits.shape
            _make_one_hot(layer.conn_logits, torch.randint(0, k, (gc, pc), generator=gen))

    x = _random_binary(64, 16, seed=3)
    soft_pred = model(x).argmax(dim=-1)
    hard_pred = harden(model).predict(x)
    assert torch.equal(soft_pred, hard_pred)


def test_hard_circuit_gate_count() -> None:
    cfg = ModelConfig(in_features=16, hidden=(24, 24), out_wires=12, num_classes=3)
    model = LogicGateNetwork(cfg, generator=torch.Generator().manual_seed(0))
    assert harden(model).num_gates() == model.num_gates() == 24 + 24 + 12
