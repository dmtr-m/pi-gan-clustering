"""Training curves (eval reward, policy entropy) for the P2 lambda sweep, from logs/zn/P2_*.log.
    .venv/bin/python experiments/zn_curves.py
"""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAMS = ["0", "0.125", "0.25", "0.5", "1", "2"]
rx = re.compile(r"^\[KS\]\s+(\d+)/\d+\s.*?eval=(-?[\d.]+).*?H=([\d.]+)/")
def curve(n):
    ep, ev, h = [], [], []
    for l in open(f"logs/zn/{n}.log"):
        m = rx.match(l)
        if m:
            ep.append(int(m.group(1))); ev.append(float(m.group(2))); h.append(float(m.group(3)))
    return ep, ev, h

fig, axes = plt.subplots(1, 4, figsize=(24, 5.2))
cm = plt.get_cmap("viridis")
for i, pre in enumerate((0, 5)):
    name = "no warm start" if pre == 0 else "5-epoch MST warm start"
    for k, l in enumerate(LAMS):
        ep, ev, h = curve(f"P2_l{l}_pre{pre}")
        c = cm(k / (len(LAMS) - 1))
        axes[2 * i].plot(ep, ev, color=c, lw=1.8, label=f"λ={l}")
        axes[2 * i + 1].plot(ep, h, color=c, lw=1.8, label=f"λ={l}")
    axes[2 * i].set_title(f"{name}: eval reward R", fontsize=14)
    axes[2 * i + 1].set_title(f"{name}: policy entropy H", fontsize=14)
    axes[2 * i].axhline(0, color="0.6", lw=0.6)
    axes[2 * i + 1].set_ylim(-0.02, 0.72)
    axes[2 * i + 1].axhline(0.693, color="0.6", lw=0.6, ls="--")
for ax in axes:
    ax.set_xlabel("RL epoch", fontsize=13); ax.tick_params(labelsize=12); ax.grid(alpha=0.3)
axes[0].set_ylabel("eval reward R (each run on its own λ)", fontsize=13)
axes[1].set_ylabel("H (dashed: ln 2 = 0.693, uniform)", fontsize=13)
axes[1].legend(fontsize=12, ncol=2, title="λ", title_fontsize=12)
fig.suptitle("P2 λ sweep training curves — zeta_correct, final_sum, k=4 K=2, seed 0, 100 RL epochs", fontsize=16)
fig.tight_layout()
fig.savefig("figures/zn_P2_curves_row.png", dpi=75)
print("wrote figures/zn_P2_curves_row.png")
