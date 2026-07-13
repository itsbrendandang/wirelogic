"""Logic-gate layers with fixed or learnable connections, and a GroupSum head.

A :class:`LogicLayer` holds ``num_gates`` two-input gates.  Two things are learned
per gate:

* **the gate type** - a distribution over the sixteen boolean gates, mixed softly
  during training and taken as the arg-max once the network is hardened;
* **the connections** - which two wires of the previous layer feed the gate.

Fixed-connection layers (Petersen et al., 2022) wire each input pin to a single
random source and never move it.  The learnable-connection layer - the
contribution of arXiv:2607.09399 - instead keeps a distribution over a *candidate
set* of sources per pin and learns which source has the highest merit, using a
straight-through estimator so the forward pass commits to one wire while gradients
still flow to every candidate.  This lets a much smaller network match a large
fixed-connection one, because gates are no longer stuck with unlucky wiring.
"""

from __future__ import annotations

import torch
from torch import nn

from .gates import CONSTANT_GATE_IDS, NUM_GATES, soft_gates


def _straight_through_argmax(probs: torch.Tensor, values: torch.Tensor, dim: int) -> torch.Tensor:
    """Select ``values`` at ``argmax(probs)`` with a soft backward pass.

    Forward output equals the hard pick ``values[argmax]``; the gradient is that
    of the soft mixture ``sum(probs * values)``.  This is the standard
    straight-through estimator applied to a categorical selection.
    """
    probs = torch.broadcast_to(probs, values.shape)
    soft = (probs * values).sum(dim=dim)
    index = probs.argmax(dim=dim, keepdim=True)
    hard = values.gather(dim, index).squeeze(dim)
    return soft + (hard - soft).detach()


class LogicLayer(nn.Module):
    """One layer of two-input logic gates.

    Args:
        in_features: Number of wires arriving from the previous layer.
        num_gates: Number of gates (and therefore output wires) in this layer.
        learnable_connections: If ``True``, learn one source per pin from a
            candidate set; if ``False``, wire each pin to a fixed random source.
        candidates: Size of the per-pin candidate set when connections are
            learnable.  ``None`` means "use every input wire" (full optimisation);
            an integer ``k`` samples ``k`` candidates per pin (partial
            optimisation).  Ignored for fixed connections.
        trim_constant_gates: Drop the constant-0 and constant-1 gate types from
            the trainable set, as recommended by the paper for stability.
        generator: Optional RNG for reproducible wiring/candidate sampling.
    """

    def __init__(
        self,
        in_features: int,
        num_gates: int,
        *,
        learnable_connections: bool = True,
        candidates: int | None = None,
        trim_constant_gates: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if in_features < 2:
            raise ValueError("a logic layer needs at least two input wires")
        self.in_features = in_features
        self.num_gates = num_gates
        self.learnable_connections = learnable_connections

        active = [g for g in range(NUM_GATES) if g not in CONSTANT_GATE_IDS]
        self.gate_ids: torch.Tensor
        self.register_buffer(
            "gate_ids",
            torch.tensor(active if trim_constant_gates else list(range(NUM_GATES))),
            persistent=True,
        )
        # Gate-type logits, one row of scores over the active gate set per gate.
        self.gate_logits = nn.Parameter(torch.zeros(num_gates, self.gate_ids.numel()))

        if learnable_connections:
            k = in_features if candidates is None else min(candidates, in_features)
            # Candidate source indices per (gate, pin), and a learnable score over
            # them.  Sampling with replacement is fine: duplicate candidates simply
            # share probability mass.
            cand = torch.randint(0, in_features, (num_gates, 2, k), generator=generator)
            self.candidates: torch.Tensor
            self.register_buffer("candidates", cand, persistent=True)
            self.conn_logits = nn.Parameter(torch.zeros(num_gates, 2, k))
        else:
            wiring = torch.randint(0, in_features, (num_gates, 2), generator=generator)
            self.wiring: torch.Tensor
            self.register_buffer("wiring", wiring, persistent=True)

    def _gather_inputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the two per-gate input activations ``(a, b)`` of shape (N, G)."""
        if self.learnable_connections:
            # x: (N, in_features) -> candidate values (N, G, 2, K)
            flat = self.candidates.reshape(-1)  # (G*2*K,)
            gathered = x[:, flat].reshape(x.shape[0], self.num_gates, 2, self.candidates.shape[-1])
            probs = torch.softmax(self.conn_logits, dim=-1)  # (G, 2, K)
            selected = _straight_through_argmax(probs, gathered, dim=-1)  # (N, G, 2)
            return selected[..., 0], selected[..., 1]
        a = x[:, self.wiring[:, 0]]
        b = x[:, self.wiring[:, 1]]
        return a, b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map previous-layer activations (N, in_features) to (N, num_gates)."""
        a, b = self._gather_inputs(x)
        all_gate_out = soft_gates(a, b)  # (N, G, 16)
        active_out = all_gate_out[..., self.gate_ids]  # (N, G, A)
        gate_probs = torch.softmax(self.gate_logits, dim=-1)  # (G, A)
        return (active_out * gate_probs).sum(dim=-1)  # (N, G)

    @torch.no_grad()
    def resolved_gate_ids(self) -> torch.Tensor:
        """The concrete boolean gate chosen for each gate (arg-max), shape (G,)."""
        choice = self.gate_logits.argmax(dim=-1)
        return self.gate_ids[choice]

    @torch.no_grad()
    def resolved_wiring(self) -> torch.Tensor:
        """The concrete (source_a, source_b) index per gate, shape (G, 2)."""
        if self.learnable_connections:
            pick = self.conn_logits.argmax(dim=-1, keepdim=True)  # (G, 2, 1)
            return self.candidates.gather(-1, pick).squeeze(-1)  # (G, 2)
        return self.wiring


class GroupSum(nn.Module):
    """Aggregate output wires into class logits by summing equal-sized groups.

    The ``num_wires`` incoming wires are split into ``num_classes`` contiguous
    groups; each group's activations are summed and divided by ``tau``.  With soft
    inputs this yields class scores for a cross-entropy loss; with hard ``{0,1}``
    inputs it counts how many output gates fired for each class.
    """

    def __init__(self, num_wires: int, num_classes: int, tau: float = 10.0) -> None:
        super().__init__()
        if num_wires % num_classes != 0:
            raise ValueError(
                f"num_wires ({num_wires}) must be divisible by num_classes ({num_classes})"
            )
        self.num_classes = num_classes
        self.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grouped = x.reshape(x.shape[0], self.num_classes, -1)
        return grouped.sum(dim=-1) / self.tau
