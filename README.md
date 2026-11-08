# The Price of Hiding the Route

Code for *The Price of Hiding the Route: Lower Bounds and Mechanisms for Routing Privacy in
Mixture-of-Experts Inference.*

Mixture-of-Experts (MoE) models route each token to a content-dependent subset of experts, and **which
experts execute is observable** — through batch interference, cache/timing side channels, or network
timing. This repository measures how much that routing trace leaks about a prompt, proves how expensive
it is to hide, and implements and benchmarks the mechanisms that hide it.

## What this repository provides

- **Leakage measurement.** How well an observer of the executed experts recovers sensitive attributes
  (nationality, sensitive content, medical vs. general, code vs. general) from real prompts, on
  open-weight MoE models (OLMoE, Granite) and small from-scratch proxies, under full, noisy,
  single-layer, identity-free, and batch-aggregated observation.
- **Lower bounds (numerical checks).** The coverage bound, the request-level bound, batch dilution, the
  injected-request corollary, and the reduction-to-partition theorem, verified against exact
  enumeration and Monte-Carlo.
- **Mechanisms and cost.** Physically executed decoy experts and static grouping, measured in vLLM on
  real GPUs (OLMoE 64e, Granite 32e, Qwen3-30B 128e, Ling-mini 256e), including single-GPU and
  two-GPU expert-parallel deployments.
- **Design-space study.** Whether any partition — chosen post hoc, learned during training, or enforced
  by batch load alignment — hides a strongly routed attribute below full-coverage cost.
- **Analysis pipeline for a real side channel.** Given a recovered-expert-set trace from a published
  channel, measure its fidelity, run the attribute attack, and compare undefended vs. full coverage
  (see `CHANNEL_HANDOFF.md`).

## Repository layout

```
01–20      from-scratch MoE training, leakage, toy-model mechanisms and cost
13,15,19   attribute leakage on open-weight models and real prompts
21,22,26   vLLM / CUDA cost measurements (single-GPU and expert-parallel)
23,24,27   weaker observers, batch dilution, injected requests
31–37      channel-fidelity sweep, partition theorem/optimization/training,
           batch load alignment, the real-channel pipeline, a Transformer attacker
common/    shared MoE model, attribute attackers, open-weight loaders
cluster/   Slurm scripts and the vLLM decoy plugin for GPU runs
```

## Setup

```bash
pip install torch transformers        # CPU torch is enough for most scripts; GPU scripts need vLLM
# public-domain text for the from-scratch models:
curl -s -o pride.txt  https://www.gutenberg.org/files/1342/1342-0.txt
curl -s -o origin.txt https://www.gutenberg.org/files/2009/2009-0.txt
```

The real-prompt experiments additionally need the public datasets (WildChat, ChatDoctor, CodeAlpaca,
Dolly) placed in `data/`; they are not redistributed here. Prompt text is never printed or logged.

## Quick start

```bash
python3 01_train_models.py all                         # train the small proxy models
python3 14_theory_checks.py                            # numerical checks of the bounds
python3 19_realistic_tasks.py allenai/OLMoE-1B-7B-0924 # attribute leakage on real prompts
```

GPU cost runs use `cluster/` on Slurm; see the header of each `2x_*.py` script for its invocation.

## Scope

This repository contains the **defense and the analysis**, not an attack: there is no microarchitectural
extraction code. Leakage is measured on routing the scripts record from models we run ourselves, and
the real-channel study is driven by degrading that recorded routing to a chosen fidelity. To plug in a
real published side channel (e.g. MoEcho, GateBleed) reproduced separately, feed its output to
`36_channel_pipeline.py` using the schema in `CHANNEL_HANDOFF.md`.

## License

Released for research use.
