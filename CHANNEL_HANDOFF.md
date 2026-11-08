# Real expert-set side channel: analysis handoff

**New controlled GPU artifact:** `real_channel/run.sh` implements an NVTX-assisted MoEcho GPU
performance-counter adaptation for Granite-3.0 on one 8GB GPU, including physical full coverage,
independent counter recovery, and adaptive attribute analysis. See `real_channel/README.zh-CN.md`
and `real_channel/WSL_SETUP.zh-CN.md`. This is controlled profiler access with static public schedule
annotations, not an uninstrumented co-tenant attack. GateBleed requires Intel AMX and is not an RTX
GPU channel. The older handoff below remains relevant to independently acquired uninstrumented traces.

The microarchitectural **extraction** (recovering which experts a co-located victim executed, via cache /
page-fault / GPU-counter / timing, as in MoEcho and GateBleed) is done with the channel's own tool on
your own hardware. Everything *around* it — fidelity measurement, the attribute attack on the recovered
traces, and the undefended-vs-full-coverage comparison — is already built in `36_channel_pipeline.py`.
You produce one file per victim configuration; the pipeline produces the paper's numbers.

## What you run (the only new piece)

1. Start the victim: your own vLLM serving the attribute-labeled prompts, in two configurations:
   - `defense=none`  — stock model.
   - `defense=full`  — full coverage (every expert executed; `MOE_DECOY_ND = E-k`, the mechanism in
     `21_vllm_cuda_overhead.py` / the `cluster/moe_decoy_plugin_pkg` plugin).
   Log, per request, the **true** per-layer executed expert set (you have this: it is the victim's own
   routing) so fidelity can be measured.
2. Run the MoEcho/GateBleed probe against the victim and record, per request, the **recovered** per-layer
   expert set.
3. Write one JSON per configuration in the schema below.

## Schema (`*_none.json`, `*_full.json`)

```json
{
  "model": "allenai/OLMoE-1B-7B-0924",
  "defense": "none",                     // "none" | "full"
  "E": 64, "K": 8, "L": 16,
  "channel": "moecho-cache",             // free text, for the paper
  "requests": [
    {
      "id": "req-00001",                 // unique; pairs none/full, and splits
      "split": "test",                   // optional: "train" | "test" (user-disjoint)
      "labels": {"medical": 1},          // ground-truth attribute(s) — you have these
      "recovered": [[3,7,12], [ ... ], ...],   // length L: per-layer recovered executed experts
      "truth":     [[3,7],    [ ... ], ...]    // length L: per-layer TRUE executed experts (victim log)
                                               //   omit / null for the full-coverage file
    }
  ]
}
```
Per-layer entries are the **set** of experts seen executed over the request (MoEcho granularity). If the
channel is per token, aggregate to the per-request set before writing.

## What you run to get the numbers

```bash
# Validate the pipeline now, with synthetic traces at recall=0.7, fp=0.05 (no channel needed):
python3 36_channel_pipeline.py synth allenai/OLMoE-1B-7B-0924 0.7 0.05 /tmp/ch
python3 36_channel_pipeline.py run /tmp/ch_none.json /tmp/ch_full.json

# Then, with your real files:
python3 36_channel_pipeline.py run real_none.json real_full.json
```

Output (printed and written to `results/36_channel_pipeline.json`):
- **channel fidelity**: measured recall r and false-positive rate f (recovered vs. truth), and the
  undefended leakage that `31_channel_fidelity.py`'s offline grid predicts at that (r, f) — the real
  channel becomes a point on that curve, closing the "is the offline degradation realistic?" question.
- **attack accuracy** per attribute, on the recovered traces, undefended vs. full coverage.
- the headline table: *undefended → full coverage* for each attribute (expected: high → chance).

The synthetic run already reproduces the expected shape: recovered-trace leakage (code 99%, medical 92%,
sensitive 75%) collapses to 50–52% under full coverage.

## Ethics / responsible disclosure (required for S&P)

- Victim and attacker are **your own** processes on **your own / authorized** hardware; no third party,
  no real user data beyond the public datasets already used (ChatDoctor / CodeAlpaca / Dolly; WildChat
  under its license, no prompt text logged).
- The channels (MoEcho, GateBleed) are **already publicly disclosed**; you reproduce them only to
  evaluate a mitigation. State this in an Ethics section.
- If reproduction reveals a **new**, undisclosed channel detail, follow coordinated disclosure (notify
  the vendor / original authors) before publishing it.
- Obtain IRB / ethics-board sign-off per your institution; a controlled, single-tenant, self-measurement
  study with no third-party data is typically exempt, but confirm.
