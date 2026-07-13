"""Train a logic-gate network on Yin-Yang and report soft vs. hardened accuracy.

Run directly::

    wirelogic --epochs 300 --hidden 120,120

or compare fixed against learnable connections at an equal gate budget::

    wirelogic --compare
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import nn

from .data import make_yin_yang
from .encode import thermometer_encode
from .harden import harden
from .model import LogicGateNetwork, ModelConfig


@dataclass(slots=True)
class Dataset:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    in_features: int


def build_dataset(*, bits: int, n_train: int, n_test: int, seed: int) -> Dataset:
    x_tr, y_tr = make_yin_yang(n_train, seed=seed)
    x_te, y_te = make_yin_yang(n_test, seed=seed + 1)
    xb_tr = thermometer_encode(x_tr, bits)
    xb_te = thermometer_encode(x_te, bits)
    return Dataset(xb_tr, y_tr, xb_te, y_te, in_features=xb_tr.shape[1])


@torch.no_grad()
def soft_accuracy(model: LogicGateNetwork, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    pred = model(x).argmax(dim=-1)
    return float((pred == y).float().mean().item())


def train_model(
    cfg: ModelConfig,
    data: Dataset,
    *,
    epochs: int,
    lr: float,
    seed: int,
    log_every: int = 50,
    verbose: bool = True,
) -> LogicGateNetwork:
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = LogicGateNetwork(cfg, generator=gen)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(data.x_train)
        loss = loss_fn(logits, data.y_train)
        loss.backward()
        opt.step()
        if verbose and (epoch % log_every == 0 or epoch == 1):
            acc = soft_accuracy(model, data.x_test, data.y_test)
            print(f"  epoch {epoch:>4}  loss {loss.item():.4f}  test(soft) {acc:.4f}")
    return model


def evaluate(model: LogicGateNetwork, data: Dataset) -> dict[str, float]:
    soft_train = soft_accuracy(model, data.x_train, data.y_train)
    soft_test = soft_accuracy(model, data.x_test, data.y_test)
    circuit = harden(model)
    hard_train = (circuit.predict(data.x_train) == data.y_train).float().mean().item()
    hard_test = (circuit.predict(data.x_test) == data.y_test).float().mean().item()
    return {
        "soft_train": soft_train,
        "soft_test": soft_test,
        "hard_train": hard_train,
        "hard_test": hard_test,
        "gates": float(model.num_gates()),
    }


def _report(tag: str, metrics: dict[str, float]) -> None:
    print(
        f"[{tag}] gates={int(metrics['gates'])}  "
        f"soft_test={metrics['soft_test']:.4f}  hard_test={metrics['hard_test']:.4f}  "
        f"(train soft={metrics['soft_train']:.4f} hard={metrics['hard_train']:.4f})"
    )


def _parse_hidden(text: str) -> tuple[int, ...]:
    return tuple(int(p) for p in text.split(",") if p)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--hidden", type=_parse_hidden, default=(120, 120))
    parser.add_argument("--out-wires", type=int, default=120)
    parser.add_argument("--bits", type=int, default=16, help="thermometer bits per feature")
    parser.add_argument("--candidates", type=int, default=8, help="learnable-conn candidates/pin")
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--connections",
        choices=("learnable", "fixed"),
        default="learnable",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="train fixed and learnable connections at equal gate budget and compare",
    )
    args = parser.parse_args(argv)

    data = build_dataset(bits=args.bits, n_train=args.n_train, n_test=args.n_test, seed=args.seed)
    print(
        f"Yin-Yang: {args.n_train} train / {args.n_test} test, "
        f"{data.in_features} binary inputs ({args.bits} bits x 4 features)"
    )

    def make_cfg(learnable: bool) -> ModelConfig:
        return ModelConfig(
            in_features=data.in_features,
            hidden=args.hidden,
            out_wires=args.out_wires,
            num_classes=3,
            learnable_connections=learnable,
            candidates=args.candidates,
        )

    if args.compare:
        for learnable in (False, True):
            tag = "learnable" if learnable else "fixed"
            print(f"\n== training {tag} connections ==")
            model = train_model(
                make_cfg(learnable), data, epochs=args.epochs, lr=args.lr, seed=args.seed
            )
            _report(tag, evaluate(model, data))
        return

    learnable = args.connections == "learnable"
    print(f"\n== training {args.connections} connections ==")
    model = train_model(make_cfg(learnable), data, epochs=args.epochs, lr=args.lr, seed=args.seed)
    _report(args.connections, evaluate(model, data))


if __name__ == "__main__":
    main()
