# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-13

### Added

- Differentiable and hard forms of all sixteen two-input boolean gates (`gates.py`).
- `LogicLayer` supporting both fixed random connections and learnable connections with straight-through selection over a candidate set, plus a `GroupSum` classification head (`layers.py`).
- Thermometer input encoding so real features become binary wires and hardening stays lossless at the input boundary (`encode.py`).
- Yin-Yang dataset generator (`data.py`).
- `LogicGateNetwork` and `ModelConfig` (`model.py`).
- `harden` and `HardCircuit` for discretisation into an integer-only boolean circuit with a fast forward pass (`harden.py`).
- `wirelogic` training CLI with a `--compare` experiment for fixed versus learnable connections (`train.py`).
- Test suite covering gate semantics, straight-through gradients, and bit-exact hardening; clean `ruff` and strict `mypy`.

### Measured

- Yin-Yang, 224 gates, 400 epochs, seed 0: fixed connections 87.7% test, learnable connections 94.0% test; hardened circuit matches the soft model exactly in both cases.
