"""
Figure 3 of the paper: named-entity attacker accuracy versus decoy overhead (a) and
versus the share of the non-real expert pool covered by decoys (b), for the 32/64-
expert trained models and the two open-weight models. Reads results/10a_n150.json
and results/15_*.json; writes ../paper/src/fig_coverage.pdf.
"""
import json, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = [1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 8.0]
a = json.load(open("results/10a_n150.json"))
series = []  # (label, E, k, accuracies)
series.append(("32 experts (ours)", 32, 4, [st.mean(a["checkpoints/moe_32e.pt"][str(r)]) * 100 for r in R]))
series.append(("64 experts (ours)", 64, 4, [st.mean(a["checkpoints/moe_64e.pt"][str(r)]) * 100 for r in R]))
for name, tag, E, k in [("OLMoE-1B-7B", "OLMoE-1B-7B-0924", 64, 8), ("Granite-1B-A400M", "granite-3.0-1b-a400m-base", 32, 8)]:
    d = json.load(open(f"results/15_{tag}.json"))["tasks"]["entity"]
    series.append((name, E, k, [st.mean([d[s][f"ratio{r}"] for s in d]) * 100 for r in R]))

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots 1-4 (light)
MARKERS = ["o", "s", "^", "D"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6})
fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.0), sharey=True, constrained_layout=True)
for (lab, E, k, acc), c, mk in zip(series, COLORS, MARKERS):
    nd = [max(0, min(round(r * k) - k, E - k)) for r in R]
    cov = [n / (E - k) for n in nd]
    pts_b = sorted(dict(zip(cov, acc)).items())  # Granite reaches full coverage from 4.5x on; the repeated x=1 points are all 50%
    for ax, xs, ys in [(axes[0], R, acc), (axes[1], [p[0] for p in pts_b], [p[1] for p in pts_b])]:
        ax.plot(xs, ys, color=c, lw=1.2, marker=mk, ms=3.2, mec="white", mew=0.5, label=lab)
for ax in axes:
    ax.axhline(50, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_xscale("log", base=2); axes[0].set_xticks([1, 2, 4, 8]); axes[0].set_xticklabels(["1", "2", "4", "8"])
axes[0].set_xlabel("(a) overhead $m/k$"); axes[1].set_xlabel("(b) decoy coverage")
axes[1].set_xticks([0, 0.5, 1]); axes[1].set_xticklabels(["0", "0.5", "1"])
axes[0].set_ylabel("named-entity acc. (%)")
axes[0].text(1.02, 50.8, "chance", color=MUTED, fontsize=6)
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=5.8, frameon=False, handlelength=1.8, labelcolor=INK)
out = os.path.join(os.path.dirname(__file__), "..", "paper", "src", "fig_coverage.pdf")
fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200)
print("wrote", out)
