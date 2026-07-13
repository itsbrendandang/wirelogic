"""The sixteen two-input boolean functions, in soft and hard form.

A logic-gate network treats each wire as carrying a probability that the bit is
``1``.  For two *independent* input probabilities ``a, b in [0, 1]`` every one of
the sixteen boolean gates has an exact probabilistic relaxation - the expected
value of the gate output when the inputs are independent Bernoulli variables.
Those relaxations are differentiable in ``a`` and ``b``, which is what lets a
network of gates be trained by gradient descent (Petersen et al., 2022).

Gates are indexed ``0..15`` so that the familiar gates land on familiar numbers
(``1`` = AND, ``6`` = XOR, ``7`` = OR, ``14`` = NAND).  For gate ``i`` the output
on inputs ``(a, b)`` is bit ``3 - (2a + b)`` of ``i``.  Reading each row below
under the column order ``(a,b) = 00, 01, 10, 11``::

    idx  name         truth table (00,01,10,11)   soft form
    0    FALSE        0 0 0 0                      0
    1    AND          0 0 0 1                      a * b
    2    A_AND_NOT_B  0 0 1 0                      a - a * b
    3    A            0 0 1 1                      a
    4    NOT_A_AND_B  0 1 0 0                      b - a * b
    5    B            0 1 0 1                      b
    6    XOR          0 1 1 0                      a + b - 2 a b
    7    OR           0 1 1 1                      a + b - a b
    8    NOR          1 0 0 0                      1 - (a + b - a b)
    9    XNOR         1 0 0 1                      1 - (a + b - 2 a b)
    10   NOT_B        1 0 1 0                      1 - b
    11   A_OR_NOT_B   1 0 1 1                      1 - b + a b
    12   NOT_A        1 1 0 0                      1 - a
    13   NOT_A_OR_B   1 1 0 1                      1 - a + a b
    14   NAND         1 1 1 0                      1 - a b
    15   TRUE         1 1 1 1                      1

The soft forms are used during training; the hard truth tables are used once the
network is discretised into a concrete circuit (see :mod:`wirelogic.harden`).
"""

from __future__ import annotations

import torch

NUM_GATES = 16

GATE_NAMES: tuple[str, ...] = (
    "FALSE",
    "AND",
    "A_AND_NOT_B",
    "A",
    "NOT_A_AND_B",
    "B",
    "XOR",
    "OR",
    "NOR",
    "XNOR",
    "NOT_B",
    "A_OR_NOT_B",
    "NOT_A",
    "NOT_A_OR_B",
    "NAND",
    "TRUE",
)

# Gate types whose output ignores both inputs (constant 0 / constant 1).  The
# paper trims these during training because a gate that collapses to a constant
# carries no signal yet can still win the softmax and stall learning.
CONSTANT_GATE_IDS: tuple[int, ...] = (0, 15)


def soft_gates(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Stack the soft relaxations of all sixteen gates along a new last axis.

    Args:
        a: Left input probabilities, any shape ``S``.
        b: Right input probabilities, broadcastable to ``a``.

    Returns:
        Tensor of shape ``(*S, 16)`` where ``[..., i]`` is gate ``i`` applied to
        ``(a, b)``.  Values lie in ``[0, 1]`` whenever the inputs do.
    """
    ab = a * b
    zeros = torch.zeros_like(a * b)
    ones = zeros + 1.0
    return torch.stack(
        [
            zeros,  # 0  FALSE
            ab,  # 1  AND
            a - ab,  # 2  A AND NOT B
            a.expand_as(ab),  # 3  A
            b - ab,  # 4  NOT A AND B
            b.expand_as(ab),  # 5  B
            a + b - 2.0 * ab,  # 6  XOR
            a + b - ab,  # 7  OR
            1.0 - (a + b - ab),  # 8  NOR
            1.0 - (a + b - 2.0 * ab),  # 9  XNOR
            1.0 - b,  # 10 NOT B
            1.0 - b + ab,  # 11 A OR NOT B
            1.0 - a,  # 12 NOT A
            1.0 - a + ab,  # 13 NOT A OR B
            1.0 - ab,  # 14 NAND
            ones,  # 15 TRUE
        ],
        dim=-1,
    )


# Truth table of every gate, shape (16, 4).  Column ``k`` is bit ``k`` of the
# gate index, so row ``i`` read as a 4-bit little-endian integer is exactly ``i``.
# The input pair ``(a, b)`` maps to column ``idx = 3 - (2*a + b)``: this places
# ``(1, 1) -> 0`` and ``(0, 0) -> 3`` so that gate ``1`` is AND, ``7`` is OR,
# ``6`` is XOR, and so on, matching the soft relaxations above.
_TRUTH_TABLE: torch.Tensor = torch.tensor(
    [[(i >> k) & 1 for k in range(4)] for i in range(NUM_GATES)],
    dtype=torch.uint8,
)


def input_column(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Map binary inputs ``(a, b)`` to their truth-table column index."""
    return 3 - ((a.to(torch.long) << 1) | b.to(torch.long))


def hard_gate(gate_id: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Evaluate concrete boolean gates on binary inputs.

    Args:
        gate_id: Integer gate indices, shape ``S``.
        a: Left inputs in ``{0, 1}``, shape broadcastable to ``S``.
        b: Right inputs in ``{0, 1}``, shape broadcastable to ``S``.

    Returns:
        The gate outputs in ``{0, 1}`` as a ``uint8`` tensor of shape ``S``.
    """
    table = _TRUTH_TABLE.to(gate_id.device)
    col = input_column(a, b)
    return table[gate_id.to(torch.long), col]
