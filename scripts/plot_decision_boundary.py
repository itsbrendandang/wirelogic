"""Train a learnable-connection network, harden it, and plot what it learned.

Produces ``docs/decision_boundary.png``: the decision regions of the *hardened*
integer-only circuit across the input square, with the held-out test points drawn
on top.  Because the plotted regions come from the discrete circuit (not the soft
model), the picture is exactly what the exported logic gates compute.

Run with::

    uv run --extra viz python scripts/plot_decision_boundary.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from wirelogic import ModelConfig, harden
from wirelogic.data import make_yin_yang
from wirelogic.encode import thermometer_encode
from wirelogic.train import Dataset, build_dataset, evaluate, train_model

BITS = 16
CLASS_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c")  # yin, yang, dot
CLASS_NAMES = ("yin", "yang", "dot")


def _grid_predictions(circuit, n: int) -> torch.Tensor:
    """Predict the hardened circuit's class over an ``n x n`` grid of the square."""
    axis = torch.linspace(0.0, 1.0, n)
    gx, gy = torch.meshgrid(axis, axis, indexing="xy")
    xs = gx.reshape(-1)
    ys = gy.reshape(-1)
    feats = torch.stack([xs, ys, 1.0 - xs, 1.0 - ys], dim=1)
    bits = thermometer_encode(feats, BITS)
    return circuit.predict(bits).reshape(n, n)


def main() -> None:
    data: Dataset = build_dataset(bits=BITS, n_train=2000, n_test=1000, seed=0)
    cfg = ModelConfig(
        in_features=data.in_features,
        hidden=(64, 64),
        out_wires=96,
        num_classes=3,
        learnable_connections=True,
        candidates=8,
    )
    model = train_model(cfg, data, epochs=400, lr=0.05, seed=0, verbose=False)
    circuit = harden(model)
    metrics = evaluate(model, data)

    grid = _grid_predictions(circuit, n=300)
    x_test, y_test = make_yin_yang(600, seed=99)

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(CLASS_COLORS)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    ax.imshow(
        grid.numpy(),
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap=cmap,
        vmin=0,
        vmax=2,
        alpha=0.35,
        interpolation="nearest",
    )
    for c in range(3):
        pts = x_test[y_test == c]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=8,
            color=CLASS_COLORS[c],
            edgecolor="white",
            linewidth=0.3,
            label=CLASS_NAMES[c],
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Hardened logic circuit on Yin-Yang\n"
        f"{int(metrics['gates'])} gates, learnable connections, "
        f"test accuracy {metrics['hard_test']:.1%}",
        fontsize=11,
    )
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "docs" / "decision_boundary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}  (hard test acc {metrics['hard_test']:.4f})")


if __name__ == "__main__":
    main()
