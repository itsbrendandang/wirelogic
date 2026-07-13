"""Discretise a trained network into a concrete, integer-only logic circuit.

Once training is done each gate has a single best gate type (arg-max over the
gate-type distribution) and a single best source per input pin (arg-max over the
connection distribution).  Freezing those choices turns the soft network into an
ordinary boolean circuit that runs with integer table look-ups and no floating
point.  Because the encoded inputs are already binary, the circuit reproduces the
soft model's arg-max predictions up to the small gap left by an unconverged
distribution - which the training script measures explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .gates import hard_gate
from .model import LogicGateNetwork


@dataclass(slots=True)
class HardLayer:
    """A frozen logic layer: one gate id and two source wires per gate."""

    gate_ids: torch.Tensor  # (G,) int64
    wiring: torch.Tensor  # (G, 2) int64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate on binary inputs (N, in_features) -> (N, G), all ``uint8``."""
        a = x[:, self.wiring[:, 0]]
        b = x[:, self.wiring[:, 1]]
        return hard_gate(self.gate_ids.expand_as(a), a, b)


@dataclass(slots=True)
class HardCircuit:
    """A fully discretised logic-gate classifier."""

    layers: list[HardLayer]
    num_classes: int

    def forward_bits(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final layer's output bits (N, out_wires) as ``uint8``."""
        h = x.to(torch.uint8)
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def class_counts(self, x: torch.Tensor) -> torch.Tensor:
        """Per-class fired-gate counts (N, num_classes) via GroupSum grouping."""
        bits = self.forward_bits(x).to(torch.long)
        return bits.reshape(bits.shape[0], self.num_classes, -1).sum(dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predicted class per row (N,)."""
        return self.class_counts(x).argmax(dim=-1)

    def num_gates(self) -> int:
        return sum(layer.gate_ids.numel() for layer in self.layers)


@torch.no_grad()
def harden(model: LogicGateNetwork) -> HardCircuit:
    """Freeze a trained :class:`LogicGateNetwork` into a :class:`HardCircuit`."""
    layers = [
        HardLayer(gate_ids=layer.resolved_gate_ids(), wiring=layer.resolved_wiring())
        for layer in model.logic_layers()
    ]
    return HardCircuit(layers=layers, num_classes=model.cfg.num_classes)
