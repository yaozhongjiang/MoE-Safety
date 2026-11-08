# MoE Routing Privacy — Experimental Code

This is the experimental code behind "Privatizing the Observable Route: Decoupling
Privacy from Computation in Mixture-of-Experts Inference." It was reconstructed and
re-verified in a fresh environment (the original session's scratch files were not
persisted), so please read the **Reproducibility status** section before treating any
printed number as a reproduction guarantee.

## Setup

```
pip install torch requests          # CPU-only torch is sufficient
python3 -c "import urllib.request; urllib.request.urlretrieve('https://www.gutenberg.org/files/1342/1342-0.txt','pride.txt')"
python3 -c "import urllib.request; urllib.request.urlretrieve('https://www.gutenberg.org/files/2009/2009-0.txt','origin.txt')"
```
(or simply `curl -s -o pride.txt https://www.gutenberg.org/files/1342/1342-0.txt` and
the equivalent for `origin.txt`, Darwin's *On the Origin of Species*, ID 2009).

Then, from this directory (`pip install transformers` is additionally needed for scripts 12, 13 and 15):

```
python3 01_train_models.py all      # trains all four checkpoints into checkpoints/
                                     # (or one at a time: moe_8e / moe_32e / moe_64e / moe_mixed)
python3 02_dp_composition.py                                # Sec. 4.1
python3 03_block_sharing.py                                 # Sec. 4.2 (single window; see 10 d)
python3 04_dummy_routing_and_attacks.py                     # Sec. 6.2-6.3, Table 2
python3 05_decode_phase_and_grouping.py                     # Sec. 6.4, Table 5
python3 06_margin_dependence.py                             # Sec. 5 (2 tasks, 1 seed; superseded by 09)
python3 07_bucket_vs_decoy_and_physical_cost.py             # Sec. 6.4 toy CPU cost (3.0-7.7x)
python3 08_ek_scaling.py                                    # Sec. 7.5 (1 seed; superseded by 10 a)
N_PER_CLASS=150 python3 09_attribute_tasks_all.py           # Table 1, Table 6, RQ1/RQ3 (5 seeds)
N_PER_CLASS=150 python3 10_overhead_scaling_and_zero_loss.py   # Table 4; 32e grouping; Delta L; block sharing
python3 11_noise_geometries.py                              # Sec. 4.2-4.3, Sec. 5 utility cost
python3 12_gpt2_proxy_reconstruction.py                     # Sec. 4.1 GPT-2 fixed-router proxy
python3 13_open_weight_moe.py ibm-granite/granite-3.0-1b-a400m-base        # Table 7 (CPU, fp32)
DTYPE=bf16 python3 13_open_weight_moe.py allenai/OLMoE-1B-7B-0924           # Table 7 (CPU, bf16, ~14 GB)
python3 14_theory_checks.py                                 # Appendix A numerical checks
python3 15_open_weight_attributes.py                        # Table 8, OLMoE (~50 min CPU)
SIGMA_GRID=0.3,0.35,0.4,0.45,0.6,0.7,0.75,0.8,0.9 python3 15_open_weight_attributes.py ibm-granite/granite-3.0-1b-a400m-base   # Table 8, Granite
python3 16_gpu_overhead.py allenai/OLMoE-1B-7B-0924        # Table 3, GPU cost (Apple MPS)
python3 16_gpu_overhead.py ibm-granite/granite-3.0-1b-a400m-base
python3 17_partial_coverage.py allenai/OLMoE-1B-7B-0924     # Sec. 7.7 partial-coverage designs, Table 9 (routing on MPS, cached)
python3 17_partial_coverage.py ibm-granite/granite-3.0-1b-a400m-base
python3 18_grouping_design.py allenai/OLMoE-1B-7B-0924      # Sec. 7.7 grouping partitions (uses 17's cached routes)
python3 18_grouping_design.py ibm-granite/granite-3.0-1b-a400m-base
python3 19_realistic_tasks.py allenai/OLMoE-1B-7B-0924      # Table 8: real prompts / sensitive attributes (needs data/, see below)
python3 19_realistic_tasks.py ibm-granite/granite-3.0-1b-a400m-base
python3 23_degraded_observation.py allenai/OLMoE-1B-7B-0924   # Table 'weaker observers' (offline, uses 19's cached routes)
python3 23_degraded_observation.py ibm-granite/granite-3.0-1b-a400m-base
python3 24_batch_dilution_theory.py allenai/OLMoE-1B-7B-0924   # Theorem 6 (batch dilution bound) vs measured (needs 23's results)
python3 25_decode_leakage.py allenai/OLMoE-1B-7B-0924         # decode-phase leakage table (MPS generation, cached)
python3 27_batch_injection.py allenai/OLMoE-1B-7B-0924        # Corollary 3: batch observer with c injected requests (uses 19's cached routes)
python3 27_batch_injection.py ibm-granite/granite-3.0-1b-a400m-base
python3 25_decode_leakage.py ibm-granite/granite-3.0-1b-a400m-base
python3 20_make_figures.py                                  # Figure 3 -> ../paper/src/fig_coverage.pdf
```

### vLLM / CUDA cost (Table 3)

`21_vllm_cuda_overhead.py` runs on an NVIDIA GPU with vLLM (tested: vLLM 0.30.0, PyTorch 2.13 + CUDA 13.0,
driver 580, single H200 and A100-80GB). It wraps `FusedMoERouter.select_experts` so decoy experts are
appended to each token's expert list with weight 0 and executed by the fused MoE kernel (CUDA graphs on).
Padding tokens (expert id -1) get decoy id -1. One process per configuration:

```
python3 21_vllm_cuda_overhead.py allenai/OLMoE-1B-7B-0924 56 out.json    # model, n_dummy, output
```

`cluster/` holds the Slurm scripts used (`run_vllm.sbatch`, with `SPECS="model:n_dummy ..."`,
`--account=YOUR_ACCOUNT`, `--constraint=h200` or `a100_80g`; `dl.sbatch` downloads the models on a compute node).
If the system `nvcc` is old, the script disables the FlashInfer sampler and uses the CUDA 13 `nvcc`
from the `nvidia-cuda-nvcc` wheel. Raw results: `results/vllm/21_*.json`; job logs: `logs/vllm/`.

Script 19 needs four public dataset files in `data/` (not redistributed here; ~320 MB):

```
python3 -c "
from huggingface_hub import hf_hub_download as d; import shutil
for r,f,o in [('allenai/WildChat-1M','data/train-00000-of-00014.parquet','wildchat_00000.parquet'),
              ('lavita/ChatDoctor-HealthCareMagic-100k','data/train-00000-of-00001-5e7cb295b9cff0bf.parquet','chatdoctor.parquet'),
              ('sahil2801/CodeAlpaca-20k','code_alpaca_20k.json','code_alpaca_20k.json'),
              ('databricks/databricks-dolly-15k','databricks-dolly-15k.jsonl','dolly15k.jsonl')]:
    shutil.copy(d(r,f,repo_type='dataset'),'data/'+o)"
```
Licenses: WildChat-1M ODC-BY, CodeAlpaca CC-BY-4.0, Dolly CC-BY-SA-3.0, ChatDoctor per its repository. Script 19
never prints or logs prompt text.

Logs of the run whose numbers are in the paper are in `logs/`, machine-readable results in
`results/`. The trained checkpoints are distributed separately (`moe_experiments_checkpoints_v2.zip`,
unzip next to this directory); with them, scripts 02-11 and 14 reproduce the logged numbers exactly on
CPU. Scripts 03-11 need the checkpoints (or `01_train_models.py` first); 13 and 15 download the
open-weight models from Hugging Face (13.8 GB and 5.5 GB).

Section, table and appendix numbers below refer to the 13-page version of the paper; the current
11-page version reorganizes sections (see the paper source for current numbering)
(`paper/src/main.tex` / `sp_paper_source_v2.zip`).

## File-to-paper mapping

| Script | Paper location | What it produces |
|---|---|---|
| `common/moe_model.py` | Sec. 3, App. B | The shared small MoE Transformer (4 layers, top-$k$ routing that determines computation) and training loop. |
| `common/attr_utils.py` | (shared) | Window sampling, the four attribute tasks, observable-trace features (clean / noisy / decoy / grouping), MLP attacker. |
| `common/open_moe.py` | Sec. 7.7 | Hook-based adapter for OLMoE / Granite-MoE (transformers 5.5): real router top-k, physically executed decoys, functional router noise. |
| `01_train_models.py` | App. B | The four checkpoints (8/32/64-expert single-genre, two-genre), 90/10 split, held-out loss. |
| `02_dp_composition.py` | Sec. 4.1 | zCDP composition (total $\epsilon$ vs. context length) and required $\sigma$. |
| `03_block_sharing.py` | Sec. 4.2 | Block-shared routing, one window (the paper uses the 50-window version, `10 d`). |
| `04_dummy_routing_and_attacks.py` | Sec. 6.2-6.3, Table 2 | Naive / prefix-keyed / whole-input-hash decoys, repeated-query and prefix-comparison attacks, $\Delta L=0$. |
| `05_decode_phase_and_grouping.py` | Sec. 6.4, Table 5 | Decode-phase fresh-nonce vulnerability, static grouping, turns-vs-precision curve. |
| `06_margin_dependence.py` | Sec. 5 | Original 2-task, 1-seed noise experiment (superseded by 09). |
| `07_bucket_vs_decoy_and_physical_cost.py` | Sec. 6.4 | Toy CPU wall-clock cost of physically enforced grouping on the 8-expert model (median of 20 runs). |
| `17_partial_coverage.py` | Sec. 7.7, Table 9 | Uniform vs. mimic vs. balanced (water-filling) decoys at equal overhead on OLMoE / Granite, against the mean-pool and a position-aware DeepSets attacker; 5 seeds. |
| `18_grouping_design.py` | Sec. 7.7 | Static-grouping partitions (index, keyed random, co-activation clustering): overhead and leakage under both attackers. |
| `19_realistic_tasks.py` | Sec. 7.7, Table 8 | User nationality and sensitive content (WildChat, user-disjoint split), medical and code requests vs. general (ChatDoctor / CodeAlpaca / Dolly), on OLMoE and Granite; noise, decoys, grouping, full coverage; mean-pool and DeepSets attackers; 5 seeds. |
| `20_make_figures.py` | Figure 3 | Attacker accuracy vs. overhead and vs. decoy coverage (32e, 64e, OLMoE, Granite). |
| `21_vllm_cuda_overhead.py` | Sec. 5, Table 3 | vLLM cost of physically executed decoys on NVIDIA H200 / A100 (prefill throughput, decode step latency at batch 1-128, output check). |
| `22_vllm_expert_coverage.py` | Sec. 5, Table 3 | Share of experts touched by real routing per decode step inside vLLM (eager mode), batch 1-256; used for Qwen3-30B-A3B (E=128). `cluster/run_qwen3.sbatch` runs it plus script 21 for n_dummy 0/8/24/56/120, with a free-GPU-memory guard. |
| `cluster/run_ling.sbatch` | Sec. 5, Table 3 | Scripts 22 and 21 (n_dummy 0/8/24/56/120/248) on Ling-mini-base-2.0 (256 experts, top-8, group-limited routing as in DeepSeek-V3). Its config class is not in transformers/vLLM; `cluster/moe_decoy_plugin_pkg/moe_decoy_plugin/bailing_config.py` provides a local one (no `trust_remote_code`) and drops the checkpoint's multi-token-prediction layer, which vLLM's loader does not skip. |
| `23_degraded_observation.py` | Sec. 7 RQ2, weaker-observer table | Real-prompt attacks re-run offline on recorded routing under weaker observers: noisy (50% misses, 2% false positives), one layer, identity-free load, and batch-aggregated load with B = 4/16/64 users; none / uniform decoys / full coverage; 5 seeds. |
| `24_batch_dilution_theory.py` | App. A, Theorem 6 | Distribution-free batch-dilution bound (1+p_B)/2, p_B = max pmf of Bin(B-1, 1/2), vs. the measured attacks of script 23 (all within the bound), and exact Bayes accuracy when the attribute determines the trace (equals the bound: tightness). Also prints the earlier Gaussian-LDA estimate for comparison. |
| `33_partition_optimization.py` | Sec. "Optimizing the partition" | Per-layer partitions (index / random / co-activation / discriminability-optimized) swept over group count G; attacker accuracy vs. execution overhead. Negative result: no partition hides strongly-routed attributes (medical/code 96-100%) below full-coverage overhead; co-activation only lowers overhead at equal leakage. `logs/33_direct_optimizer_check.log`: an aggressive direct swap optimizer still leaves the test attacker at 99.7-100%. |
| `34_train_partition_aware.py` | Sec. "Optimizing the partition" | Training-time partition-aware routing: 32-expert top-4 MoE trained from scratch on two genres with a gradient-reversal adversary making the per-group soft routing genre-invariant; sweep penalty lambda, 3 seeds. Negative result: genre leakage falls 85->62% only as LM loss rises 1.44->2.05 (+42%) and never reaches chance. |
| `36_channel_pipeline.py` (+ `test_36_channel_pipeline.py`) | real side-channel analysis | Ingests a MoEcho/GateBleed channel's recovered expert-set traces (schema in `CHANNEL_HANDOFF.md`), measures the channel's fidelity (recall/fp) and places it on the script-31 grid, runs the attribute attack on the recovered traces, and compares undefended vs. full coverage. `synth` mode generates placeholder traces from real routing to validate the pipeline before the channel exists. The microarchitectural extraction is NOT part of this repo. |
| `37_transformer_attacker.py` | Sec. 7, attacker protocol | Position-aware Transformer attacker over the per-token routing sequence; matches mean-pool/DeepSets undefended and falls to exactly 50.0% under full coverage (both models). Shows "-> chance" is not an artifact of a weak attacker. |
| `35_batch_load_alignment.py` | Sec. 5 / discussion | Capacity padding against a per-expert load observer on homogeneous batches: cost = max/mean load - 1. Negative result: cost is 1.6-5x and does NOT amortize with batch size, because MoE load is systematically skewed (OLMoE max/mean ~4.9 even at B=64). Cheaper than full coverage but not free. |
| `32_partition_reduction.py` | App. A, Thm. "Reduction to a partition" | Verifies that repeated observation collapses a mechanism onto its induced partition: fresh-nonce decoy -> identity (no privacy), grouping -> group-label partition; T-sample distinguisher advantage matches the partition prediction (0 within class, 0.5 across). |
| `31_channel_fidelity.py` | Sec. 7 RQ2, "Any expert-set channel" | Attribute leakage vs. side-channel fidelity: observation = real executed set seen with recall r and false-positive rate f, swept over a grid, undefended vs. full coverage. Undefended leakage is insensitive to fidelity; full coverage is chance at every (r,f). Subsumes MoEcho / GateBleed without reproducing their extraction (empirical companion to the coverage theorem). |
| `27_batch_injection.py` | App. A, Corollary 3; Sec. 7 RQ2 | Batch-load observer when the attacker injects c of the B-1 other requests and subtracts their load (B = 16, 64; c up to B-1); measured accuracy vs. the bound (1+p_{B-c})/2. All 72 measurements within the bound. |
| `26_vllm_ep_overhead.py` | Sec. 5 | vLLM cost of decoys under expert parallelism (TP=2 + `enable_expert_parallel`), same measurements as script 21. Decoys are injected in every vLLM worker by the general plugin in `cluster/moe_decoy_plugin_pkg` (`pip install -e`; `MOE_DECOY_ND` sets n_dummy). `cluster/run_ep.sbatch` runs n_dummy 0/24/120 on Qwen3-30B-A3B-Base with 2 GPUs. |
| `25_decode_leakage.py` | Sec. 7 RQ5, decode table | Routing of 32 greedily generated tokens per real prompt; attribute attacks on the decode trace under none / decoys / static grouping g=E/4, E/2 / full coverage. |
| `16_gpu_overhead.py` | Sec. 6.4, Table 3 | GPU latency of physically executed decoys on OLMoE / Granite (fused zero-weight decoys, separate decoy GEMMs, dense all-expert execution) for prefill and batched decode, plus the share of experts touched by real routing per decode batch. |
| `08_ek_scaling.py` | Sec. 7.5 | Original 1-seed 32e/64e comparison (superseded by 10 a). |
| `09_attribute_tasks_all.py` | Sec. 5, 7.4-7.6, Tables 1 and 6 | Four attribute tasks: noise, decoys $n_{dummy}\in\{3,6\}$, grouping $g\in\{2,4,8\}$, 5 seeds. |
| `10_overhead_scaling_and_zero_loss.py` | Sec. 4.2, 7.3-7.6, Table 4 | (a) overhead sweep on 32e/64e; (b) 32e grouping $g\in\{4,8,16\}$; (c) $\Delta L=0$ with executed decoys; (d) block sharing over 50 windows. |
| `11_noise_geometries.py` | Sec. 4.2-4.3, Sec. 5 | LM cost of noise, AR(1) noise, targeted noise and the repeated-query instability attack. |
| `12_gpt2_proxy_reconstruction.py` | Sec. 4.1, RQ1 | GPT-2-small fixed-random-router token decoder and 20% trace corruption. |
| `13_open_weight_moe.py` | Sec. 7.7, Table 7 | $\Delta L=0$, repeated-query, prefix-comparison and cross-turn experiments on OLMoE and Granite. |
| `14_theory_checks.py` | App. A | Lemma 2's closed-form TV vs. enumeration and the PRF sampler; Proposition 1 vs. Table 5; Proposition 2 (uniform decoys minimax-optimal) against random non-uniform mechanisms; Theorem 4 (coverage lower bound) on random variable-size mechanisms, plus the implied minimum cost for OLMoE/Qwen3/DeepSeek-V3. |
| `15_open_weight_attributes.py` | Sec. 7.7, Table 8 | Attribute inference on OLMoE / Granite: calibrated router noise, decoys, grouping, overhead sweep; 5 seeds. |

## Reproducibility status

Every number in the current paper was produced by the scripts above in the 2026-10-01 run and can be
traced to a log (table below). Small-model training is deterministic on CPU given the seeds in
`01_train_models.py` (the 32e/64e final training losses also match the pre-rerun records exactly).
Earlier drafts of the paper contained numbers from an unlogged development session; those were
replaced, and where they disagreed qualitatively (margin ordering on open-weight models, the 64-expert
$8\times$ reading, the GPT-2 proxy accuracy) the paper text was revised rather than the numbers kept.
## Changes made in the 2026-10-01 re-run

* `01_train_models.py` now trains on the first 90% of each corpus and reports held-out loss on the last
  10%, as Appendix B describes (the previous version trained on the full text, so script 03's
  "validation" window had been seen in training). Final training-batch losses for 32e/64e are
  bit-identical to Appendix B's (2.203 / 2.398).
* `07_bucket_vs_decoy_and_physical_cost.py` times with one warm-up pass and reports the median of 20 runs
  (the mean of 5 cold runs was not stable across invocations).
* Scripts 09-12 were added. The attribute attacks now use 5 seeds x 150 windows/class (`N_PER_CLASS`);
  with the old 50 windows/class the seed-to-seed std was 3-9 pt, enough to produce spurious orderings.

* Script 16 measures physical-execution cost on the Apple M5 Max GPU (MPS). MPS has no fused MoE
  kernel and its run-to-run timings vary by roughly 10-30%; CUDA / vLLM numbers are still to be measured.
* Scripts 13-15 were added for the open-weight models and the Appendix A checks. Open-weight runs use
  CPU: on Apple MPS, clean-vs-clean loss differed by up to 1e-3 between runs, which breaks the bitwise
  $\Delta L$ comparison and the determinism assumption (A4).

## Paper-number provenance (13-page version)

| Paper location | Source |
|---|---|
| Sec 4.1 composition | `logs/02.log` |
| Sec 4.1 GPT-2 proxy (96.3 / 23.0 / 74.5 / 82.9) | `logs/12.log` |
| Sec 4.2 block sharing (69.6%, 125.7%) | `logs/10_d.log` |
| Sec 4.2 AR(1); RQ1 per-position decoder 93.5% | `logs/11_b.log` |
| Sec 4.3 targeted noise | `logs/11_c.log` |
| Table 1 (noise and decoy columns), Sec 5 utility cost | `logs/09_*_n150.log`, `logs/11_a.log` |
| Sec 6.2-6.3, Table 2 | `logs/04_dummy_routing_and_attacks.log` |
| Sec 6.4 decode agreement 1.3%, Table 5 | `logs/05_decode_phase_and_grouping.log` |
| Table 3 (vLLM cost on H200/A100/H100) | `results/vllm/21_*.json`, `results/vllm/22_*.json`, `logs/vllm/` |
| Table 3 last row (experts touched); Apple-GPU comparison | `logs/16_olmoe.log`, `logs/16_granite.log`, `results/16_*.json` |
| Sec 6.4 toy CPU cost 3.0-7.7x | `logs/07_timing.log` |
| Table 4, RQ1 32e/64e, RQ4 $E/k$ | `logs/10_a_n150.log` |
| Table 6, RQ3 | `logs/09_entity_n150.log` |
| 32e grouping g=4/8/16 (RQ5) | `logs/10_b_n150.log` |
| $\Delta L = 0$ at 8/32/64 experts (RQ2) | `logs/10_c.log` |
| App. B losses | `logs/train_*.log` |
| Table 7 | `logs/13_olmoe.log`, `logs/13_granite.log` |
| Sec 7.7 book-task attribute text | `logs/15_olmoe.log`, `logs/15_granite.log`, `results/15_*.json` |
| Table 8 (real prompts) | `logs/19_olmoe.log`, `logs/19_granite.log`, `results/19_OLMoE-1B-7B-0924.json`, `results/19_granite-3.0-1b-a400m-base.json` |
| Weaker-observer table (RQ2) | `logs/23_OLMoE.log`, `logs/23_grani.log`, `results/23_*.json` |
| Decode-phase table (RQ5) | `logs/25_olmoe.log`, `logs/25_granite.log`, `results/25_*.json` |
| Theorem 6 check | `logs/24.log`, `results/24_*.json` |
| Corollary 3 (injected requests) | `logs/27_olmoe.log`, `logs/27_granite.log`, `results/27_*.json` |
| Partition-reduction theorem | `logs/32.log` |
| Partition optimization (post-hoc) | `logs/33_olmoe.log`, `logs/33_granite.log`, `results/33_*.json`, `logs/33_direct_optimizer_check.log` |
| Partition-aware training | `logs/34.log`, `results/34_partition_aware.json` |
| Batch load alignment | `logs/35_olmoe.log`, `logs/35_granite.log`, `results/35_*.json` |
| Channel-fidelity sweep | `logs/31_olmoe.log`, `logs/31_granite.log`, `results/31_*.json` |
| Expert-parallel cost | `results/vllm/26_*.json`, `logs/vllm/` |
| 256-expert cost (Ling) | `results/vllm/21_*Ling*`, `results/vllm/22_*Ling*`, `logs/vllm/moe-ling-cost-*.out` |
| App. A numerical claims (Theorems 4-5, Props.) | `logs/14.log` |
| Table 9, partial-coverage designs | `logs/17_olmoe.log`, `logs/17_granite.log`, `results/17_*.json` |
| Co-activation grouping | `logs/18_OLMoE.log`, `logs/18_grani.log`, `results/18_*.json` |

## Controlled real GPU counter experiment on RTX 3070 / WSL2

The new `real_channel/` artifact provides a one-command, NVTX-assisted adaptation of MoEcho's GPU
performance-counter channel, physical full-coverage execution, independent trace recovery, and three
adaptive attribute attackers. It uses ModelScope for model/data and `docker.1ms.run` for the CUDA image.
Run `bash run.sh --doctor-only` and then `bash run.sh --preset smoke` from that directory.
See [the experiment guide](real_channel/README.zh-CN.md) and
[the Windows/WSL2 setup guide](real_channel/WSL_SETUP.zh-CN.md).
This controlled profiling experiment is explicitly annotation-assisted; GateBleed's Intel AMX
power-gating channel is not available on an RTX 3070. CPU checks and source validation are included;
actual GPU/WSL acceptance is performed by the target-machine doctor and smoke stages.
