"""The soft relaxations must equal the hard truth tables on binary inputs."""

from __future__ import annotations

import itertools

import torch

from wirelogic.gates import GATE_NAMES, NUM_GATES, hard_gate, soft_gates


def test_soft_matches_hard_on_binary_inputs() -> None:
    combos = torch.tensor(list(itertools.product([0.0, 1.0], repeat=2)))
    a, b = combos[:, 0], combos[:, 1]
    soft = soft_gates(a, b)  # (4, 16)
    for gate in range(NUM_GATES):
        gid = torch.full((4,), gate, dtype=torch.long)
        hard = hard_gate(gid, a, b).to(torch.float32)
        assert torch.allclose(soft[:, gate], hard), f"mismatch for {GATE_NAMES[gate]}"


def test_known_gate_semantics() -> None:
    a = torch.tensor([0.0, 0.0, 1.0, 1.0])
    b = torch.tensor([0.0, 1.0, 0.0, 1.0])
    soft = soft_gates(a, b)
    # gate 1 = AND, 6 = XOR, 7 = OR, 14 = NAND
    assert torch.allclose(soft[:, 1], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert torch.allclose(soft[:, 6], torch.tensor([0.0, 1.0, 1.0, 0.0]))
    assert torch.allclose(soft[:, 7], torch.tensor([0.0, 1.0, 1.0, 1.0]))
    assert torch.allclose(soft[:, 14], torch.tensor([1.0, 1.0, 1.0, 0.0]))


def test_soft_gates_stay_in_unit_interval() -> None:
    torch.manual_seed(0)
    a = torch.rand(500)
    b = torch.rand(500)
    soft = soft_gates(a, b)
    assert soft.min() >= -1e-6
    assert soft.max() <= 1 + 1e-6
