"""The full logic-gate network: a stack of :class:`LogicLayer` plus a GroupSum head."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import torch
from torch import nn

from .layers import GroupSum, LogicLayer


@dataclass(slots=True)
class ModelConfig:
    """Architecture of a logic-gate network.

    Attributes:
        in_features: Number of binary input wires (after input encoding).
        hidden: Width (gate count) of each hidden logic layer.
        out_wires: Number of wires in the final logic layer; must be divisible by
            ``num_classes``.
        num_classes: Number of output classes.
        learnable_connections: Learn the wiring (paper) vs. fix it randomly.
        candidates: Per-pin candidate count for learnable connections (``None`` =
            all input wires).
        trim_constant_gates: Drop constant-0/1 gate types from training.
        tau: GroupSum temperature.
    """

    in_features: int
    hidden: tuple[int, ...] = field(default=(120, 120))
    out_wires: int = 120
    num_classes: int = 3
    learnable_connections: bool = True
    candidates: int | None = None
    trim_constant_gates: bool = True
    tau: float = 10.0


class LogicGateNetwork(nn.Module):
    """A differentiable network of logic gates ending in a GroupSum classifier."""

    def __init__(self, cfg: ModelConfig, generator: torch.Generator | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        widths = [*cfg.hidden, cfg.out_wires]
        layers: list[LogicLayer] = []
        prev = cfg.in_features
        for width in widths:
            layers.append(
                LogicLayer(
                    prev,
                    width,
                    learnable_connections=cfg.learnable_connections,
                    candidates=cfg.candidates,
                    trim_constant_gates=cfg.trim_constant_gates,
                    generator=generator,
                )
            )
            prev = width
        self.layers = nn.ModuleList(layers)
        self.head = GroupSum(cfg.out_wires, cfg.num_classes, tau=cfg.tau)

    def logic_layers(self) -> list[LogicLayer]:
        """The logic layers as a typed list (``nn.ModuleList`` erases the type)."""
        return [cast(LogicLayer, m) for m in self.layers]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Class logits (N, num_classes) from binary inputs (N, in_features)."""
        for layer in self.logic_layers():
            x = layer(x)
        return cast(torch.Tensor, self.head(x))

    def num_gates(self) -> int:
        """Total logic gates in the network."""
        return sum(layer.num_gates for layer in self.logic_layers())
