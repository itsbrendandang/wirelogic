"""wirelogic: fully-trainable differentiable logic-gate networks.

A compact, CPU-friendly reference implementation of the learnable-connection logic
gate networks from Mommen et al., "Fully Trainable Deep Differentiable Logic Gate
Networks and Lookup Table Networks" (arXiv:2607.09399, 2026).
"""

from __future__ import annotations

from .harden import HardCircuit, harden
from .model import LogicGateNetwork, ModelConfig

__all__ = ["HardCircuit", "LogicGateNetwork", "ModelConfig", "harden"]
__version__ = "0.1.0"
