"""Yin-Yang generation and thermometer encoding."""

from __future__ import annotations

import torch

from wirelogic.data import make_yin_yang
from wirelogic.encode import thermometer_encode, thermometer_thresholds


def test_yin_yang_shapes_range_and_determinism() -> None:
    x1, y1 = make_yin_yang(500, seed=0)
    x2, y2 = make_yin_yang(500, seed=0)
    assert x1.shape == (500, 4)
    assert y1.shape == (500,)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)  # reproducible
    assert x1.min() >= 0.0 and x1.max() <= 1.0
    assert set(y1.tolist()) <= {0, 1, 2}
    # complements: x[...,2] == 1 - x[...,0], x[...,3] == 1 - x[...,1]
    assert torch.allclose(x1[:, 2], 1.0 - x1[:, 0], atol=1e-6)
    assert torch.allclose(x1[:, 3], 1.0 - x1[:, 1], atol=1e-6)
    # all three classes should appear
    assert len(set(y1.tolist())) == 3


def test_thermometer_is_binary_and_monotone() -> None:
    x = torch.tensor([[0.0, 0.5, 1.0]])
    enc = thermometer_encode(x, bits=4)
    assert enc.shape == (1, 12)
    assert set(enc.reshape(-1).tolist()) <= {0.0, 1.0}
    # a larger value fires at least as many bits as a smaller one
    counts = enc.reshape(3, 4).sum(dim=-1)
    assert counts[0] <= counts[1] <= counts[2]


def test_thresholds_strictly_increasing_in_unit_interval() -> None:
    thr = thermometer_thresholds(8)
    assert thr.shape == (8,)
    assert (thr[1:] > thr[:-1]).all()
    assert thr.min() > 0.0 and thr.max() < 1.0
