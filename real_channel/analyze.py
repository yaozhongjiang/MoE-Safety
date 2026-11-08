"""Adaptive attribute attacks on measured counter-derived sets, with held-out tests."""
from pathlib import Path
import statistics

import numpy as np
import torch
from torch import nn
from scipy.stats import binomtest

from util import write_json


def validate_pair(none, full):
    for key in ("E", "K", "L", "positions", "original_layer_ids", "model"):
        if none[key] != full[key]:
            raise ValueError(f"paired metadata mismatch: {key}")
    a, b = none["requests"], full["requests"]
    if len(a) != len(b) or not a:
        raise ValueError("empty or unpaired experiment")
    seen, group_split = set(), {}
    for x, y in zip(a, b):
        for key in ("id", "labels", "split", "group", "window_hash"):
            if x[key] != y[key]:
                raise ValueError(f"paired request mismatch: {key}")
        if x["id"] in seen:
            raise ValueError("duplicate request ID")
        seen.add(x["id"])
        if x["split"] not in ("train", "test"):
            raise ValueError("unknown split")
        for group in (x["group"], x["window_hash"]):
            if group in group_split and group_split[group] != x["split"]:
                raise ValueError("train/test prompt or user overlap")
            group_split[group] = x["split"]
        for r in (x, y):
            for name in ("recovered_tokens", "truth_tokens"):
                if len(r[name]) != len(none["positions"]):
                    raise ValueError("token count mismatch")
                for token in r[name]:
                    if len(token) != none["L"]:
                        raise ValueError("layer count mismatch")
                    for experts in token:
                        if len(set(experts)) != len(experts) or any(type(e) is not int or not 0 <= e < none["E"] for e in experts):
                            raise ValueError("invalid expert IDs")
        if any(s != list(range(none["E"])) for token in y["truth_tokens"] for s in token):
            raise ValueError("full coverage truth is incomplete")
    tasks = sorted({task for r in a for task in r["labels"]})
    for task in tasks:
        for split in ("train", "test"):
            labels = [r["labels"][task] for r in a if r["split"] == split and task in r["labels"]]
            if any(type(v) is not int or v not in (0, 1) for v in labels) or labels.count(0) != labels.count(1) or labels.count(0) < 2:
                raise ValueError(f"{task}/{split}: expected balanced binary labels")


def fidelity(data):
    tp = fp = fn = tn = 0
    for r in data["requests"]:
        for observed, truth in zip(r["recovered_tokens"], r["truth_tokens"]):
            for observed_set, truth_set in zip(observed, truth):
                a, b = set(observed_set), set(truth_set)
                tp += len(a & b)
                fp += len(a - b)
                fn += len(b - a)
                tn += data["E"] - len(a | b)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, recall=tp / (tp + fn) if tp + fn else None,
                precision=tp / (tp + fp) if tp + fp else None,
                false_positive_rate=fp / (fp + tn) if fp + tn else None,
                note="Full coverage 的 FPR 为 null：没有负例专家" if fp + tn == 0 else "token/layer/expert micro-average")


def features(data):
    array = np.zeros((len(data["requests"]), len(data["positions"]), data["L"], data["E"]), dtype=np.float32)
    for i, r in enumerate(data["requests"]):
        for t, token in enumerate(r["recovered_tokens"]):
            for layer, experts in enumerate(token):
                array[i, t, layer, experts] = 1
    return torch.from_numpy(array.reshape(len(array), array.shape[1], -1))


class Attacker(nn.Module):
    def __init__(self, dimension, steps, kind):
        super().__init__()
        self.kind = kind
        self.encode = nn.Sequential(nn.Linear(dimension, 32), nn.ReLU())
        if kind == "transformer":
            self.position = nn.Parameter(torch.randn(1, steps, 32) * .02)
            self.sequence = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(32, 4, 64, dropout=0, batch_first=True), num_layers=2,
                enable_nested_tensor=False,
            )
        self.head = nn.Linear(32, 2)

    def forward(self, x):
        if self.kind == "mean_mlp":
            value = self.encode(x.mean(dim=1))
        elif self.kind == "deepsets":
            value = self.encode(x).mean(dim=1)
        else:
            value = self.sequence(self.encode(x) + self.position).mean(dim=1)
        return self.head(value)


def train_predict(x, train, test, y, kind, seed, epochs, shuffle=False):
    torch.manual_seed(seed)
    model = Attacker(x.shape[-1], x.shape[1], kind)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.001)
    labels = y[train].clone()
    if shuffle:
        labels = labels[torch.randperm(len(labels))]
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x[train]), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return model(x[test]).argmax(-1).numpy()


def score(predictions, truth, groups):
    hits = np.asarray(predictions) == np.asarray(truth)[None, :]
    accuracies = hits.mean(axis=1).tolist()
    # Average correctness over seeds BEFORE resampling held-out groups.
    # Never treat 5 predictions of the same example as 5 independent observations.
    group_ids = sorted(set(groups))
    by_group = [np.flatnonzero(np.asarray(groups) == group) for group in group_ids]
    rng, values = np.random.default_rng(42), []
    mean_hits = hits.mean(axis=0)
    for _ in range(1000):
        idx = np.concatenate([by_group[j] for j in rng.integers(0, len(by_group), len(by_group))])
        values.append(float(mean_hits[idx].mean()))
    return dict(accuracy=statistics.mean(accuracies), std=statistics.pstdev(accuracies),
                per_seed=accuracies, heldout_group_bootstrap_ci95=np.quantile(values, [.025, .975]).tolist(),
                n_test=len(truth), n_groups=len(group_ids),
                first_seed_binomial_p=binomtest(int(hits[0].sum()), len(truth), .5).pvalue
                if len(group_ids) == len(truth) else None)


def analyze(none, full, cfg, run):
    validate_pair(none, full)
    torch.set_num_threads(4)
    result = dict(channel=none["channel"], annotation_assisted=True, synthetic=False,
                  positions=none["positions"], original_layer_ids=none["original_layer_ids"],
                  fidelity={"none": fidelity(none), "full": fidelity(full)}, tasks={})
    for task in sorted({t for r in none["requests"] for t in r["labels"]}):
        selected = [(i, r) for i, r in enumerate(none["requests"]) if task in r["labels"]]
        train = [i for i, r in selected if r["split"] == "train"]
        test = [i for i, r in selected if r["split"] == "test"]
        y = torch.tensor([r["labels"].get(task, 0) for r in none["requests"]])
        truth = y[test].numpy()
        groups = [none["requests"][i]["group"] for i in test]
        task_result = dict(n_train=len(train), n_test=len(test), majority_baseline=.5, attacks={})
        for kind in ("mean_mlp", "deepsets", "transformer"):
            values = {}
            for defense, data in (("none", none), ("full", full)):
                x = features(data)
                predictions = [train_predict(x, train, test, y, kind, seed, cfg.epochs) for seed in range(cfg.seeds)]
                values[defense] = score(predictions, truth, groups)
                # Diagnostics are reported, never used to select hyperparameters or discard runs.
                if defense == "none":
                    shuffled = [train_predict(x, train, test, y, kind, seed, cfg.epochs, shuffle=True)
                                for seed in range(cfg.seeds)]
                    values["train_label_shuffle_control"] = score(shuffled, truth, groups)
            task_result["attacks"][kind] = values
        result["tasks"][task] = task_result
        print(f"[analysis] {task}: adaptive mean-MLP, DeepSets, Transformer finished", flush=True)
    write_json(Path(run) / "results.json", result)
    write_report(result, run)
    return result


def write_report(result, run):
    lines = ["# Controlled MoEcho GPU-counter experiment", "",
             "Real Nsight Compute counters, with public static NVTX schedule assistance. "
             "This is not a GateBleed/AMX reproduction, a TLB attack, or an uninstrumented co-tenant attack.", "",
             f"Measured original layers (zero-based): {result['original_layer_ids']}; prompt token positions: {result['positions']}.", "",
             "Only the expert execution-set observation is evaluated; other channels may remain.", "",
             "| Task | Adaptive attacker | Undefended | Full coverage |", "|---|---|---:|---:|"]
    tex = ["% Measured GPU-counter results; NVTX-assisted controlled profiling, not GateBleed.",
           "\\begin{tabular}{llrr}", "\\toprule", "Task & Attacker & Undefended & Full coverage \\\\ ", "\\midrule"]
    for task, val in result["tasks"].items():
        for kind, values in val["attacks"].items():
            a, b = values["none"], values["full"]
            lines.append(f"| {task} | {kind} | {100*a['accuracy']:.2f}% ± {100*a['std']:.2f} | {100*b['accuracy']:.2f}% ± {100*b['std']:.2f} |")
            tex.append(f"{task} & {kind.replace('_', ' ')} & {100*a['accuracy']:.2f} & {100*b['accuracy']:.2f} \\\\ ")
    tex.extend(["\\bottomrule", "\\end{tabular}"])
    lines.extend(["", "Fidelity and held-out group bootstrap intervals are in results.json. "
                  "Full-coverage FPR is undefined and stored as null. "
                  "Latency in validation_gpu.json is measured outside the profiler, using the eager adapter.", ""])
    Path(run, "report.md").write_text("\n".join(lines), encoding="utf-8")
    Path(run, "table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tasks = list(result["tasks"])
    x = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(max(5, 2 * len(tasks)), 3.6))
    for offset, defense, label in ((-.17, "none", "Undefended"), (.17, "full", "Full coverage")):
        values = [result["tasks"][t]["attacks"]["mean_mlp"][defense] for t in tasks]
        ax.bar(x + offset, [100 * v["accuracy"] for v in values], .32,
               yerr=[100 * v["std"] for v in values], label=label, capsize=3)
    ax.axhline(50, color="gray", linestyle="--", label="Chance")
    ax.set(xticks=x, xticklabels=tasks, ylim=(0, 105), ylabel="Held-out accuracy (%)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(Path(run, "attack_accuracy.pdf"))
    fig.savefig(Path(run, "attack_accuracy.png"), dpi=180)
    plt.close(fig)
