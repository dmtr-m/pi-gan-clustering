"""Annotated A-Z contact sheets for the logs/zn night queue (one panel per run).

    PYTHONPATH=src .venv/bin/python experiments/zn_sheets.py

Each panel is the greedy policy's fragment (A, Z) yield per collision (5000 collisions, both
spectator sides), titled with the run's setup and its eval numbers.  Policy fragment counters
are cached in outputs/zn/<run>/az_counter.pkl.  All panels share one log colour scale.
"""
import glob, os, pickle, sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from baseline_grid import BINS, counter_to_map, stats  # noqa: E402
from target_reference import load_target  # noqa: E402
from zn_summary import ev, fr  # noqa: E402

A_MAX, Z_MAX, N_EV = 132, 60, 5000


def counter_for(n):
    f = Path(f"outputs/zn/{n}/az_counter.pkl")
    if f.exists():
        return pickle.load(open(f, "rb"))
    from clustering.split_prediction.dataset import NucleonDataset, collate_fn
    from clustering.split_prediction.model import SplitPredictionModel
    from clustering.split_prediction.trainer import k_level_forward
    from torch.utils.data import DataLoader
    torch.set_num_threads(2)
    run = f.parent
    cfg = yaml.safe_load(open(run / "resolved_config.yaml"))["config"]
    m = SplitPredictionModel(input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"],
                             n_iters=cfg["n_iters"], n_clusters=cfg["n_clusters"])
    m.load_state_dict(torch.load(run / "dm_model.pt", map_location="cpu")); m.eval()
    c = Counter()
    with torch.no_grad():
        for side in ("SpectatorsLeft", "SpectatorsRight"):
            ds = NucleonDataset("data/xecs_hse.parquet", particle_type=side, n_events=N_EV)
            for b in DataLoader(ds, batch_size=128, shuffle=False, collate_fn=collate_fn):
                x, mask = b["x"], b["mask"]
                isp = x[..., 7] == 1
                for lm in k_level_forward(m, x, mask, cfg["split_k"], 2)["leaf_masks"].values():
                    for a, z in zip(lm.sum(1).tolist(), (isp & lm).sum(1).tolist()):
                        if a > 0:
                            c[(a, z)] += 1
    pickle.dump(c, open(f, "wb"))
    return c


def describe(n):
    cfg = yaml.safe_load(open(f"outputs/zn/{n}/resolved_config.yaml"))["config"]
    pre = cfg["pretrain_epochs"]
    start = "no warm start" if pre == 0 else f"{pre}-epoch MST warm start"
    b = "B/A" if cfg.get("b_per_nucleon") else "B"
    shape = cfg["b_shape"]
    bs = "" if shape == "none" else f", {shape} sharpening" + (f" κ={cfg['b_kappa']:g}" if shape == "cone" else "")
    l1 = f"λ={cfg['bwm_weight']:g} on {b}{bs}"
    l2 = f"{start} · seed {cfg['seed']} · {cfg['split_epochs']} RL epochs"
    return l1, l2


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    names = sorted(os.path.basename(f)[:-5] for f in glob.glob("logs/zn/P*.done"))
    with Pool(4) as p:
        counters = dict(zip(names, p.map(counter_for, names)))
    n_coll = 5000.0
    ref = load_target()
    g = np.array([ref[(ref[:, 0] >= lo) & (ref[:, 0] <= hi), 2].sum() for lo, hi in BINS])
    tmap = np.zeros((Z_MAX + 1, A_MAX + 1))
    for A, Z, y in ref:
        if 2 <= A <= A_MAX and 0 <= Z <= Z_MAX:
            tmap[int(Z), int(A)] += y
    maps = {n: counter_to_map(c, n_coll, A_MAX, Z_MAX) for n, c in counters.items()}
    pos = np.concatenate([tmap.ravel()] + [m.ravel() for m in maps.values()]); pos = pos[pos > 0]
    norm = LogNorm(vmin=max(pos.min(), 1e-5), vmax=pos.max())

    def panel(ax, n, fs=14):
        m = maps[n]
        mesh = ax.pcolormesh(np.arange(A_MAX + 2) - .5, np.arange(Z_MAX + 2) - .5,
                             np.ma.masked_where(m <= 0, m), norm=norm, cmap="viridis")
        e, f, s = ev(n), fr(n), stats(counters[n], n_coll, g)
        l1, l2 = describe(n)
        ax.set_title(f"{l1}\n{l2}", fontsize=fs, loc="left")
        ax.text(0.03, 0.97,
                f"eval R {e['R']:.2f}  (no-split {e['nosplit']:.2f}, gain {e['R']-e['nosplit']:+.2f})\n"
                f"unsplit {e['unsplit']:.0f}%  ·  {s['frags']:.2f} frags/collision\n"
                f"free nucleons {f.get('free', float('nan')):.1f}  ·  A–Z RMS {s['rms']:.2f}",
                transform=ax.transAxes, va="top", fontsize=fs - 1, color="white",
                bbox=dict(fc="black", alpha=0.55, ec="none", pad=3))
        ax.set_xlim(0, A_MAX); ax.set_ylim(0, Z_MAX)
        ax.grid(color="white", lw=.3, alpha=.35); ax.tick_params(labelsize=12)
        return mesh

    def wide(title, rows, out, cell=(5.4, 5.0), fs=12):
        """rows: list of equal-length lists of run names; no target row (setup goes in the title)."""
        nrows, ncols = len(rows), len(rows[0])
        fig, axes = plt.subplots(nrows, ncols, figsize=(cell[0] * ncols, cell[1] * nrows), squeeze=False)
        mesh = None
        for i, row in enumerate(rows):
            for j, n in enumerate(row):
                ax = axes[i][j]
                mesh = panel(ax, n, fs)
                ax.set_xlabel("$A$", fontsize=fs + 1); ax.set_ylabel("$Z$", fontsize=fs + 1)
                ax.tick_params(labelsize=fs - 1)
        fig.suptitle(title, fontsize=fs + 8, y=0.995)
        fig.tight_layout(rect=(0, 0, 0.965, 0.95))
        cax = fig.add_axes([0.972, 0.1, 0.008, 0.8])
        fig.colorbar(mesh, cax=cax).set_label("yield per collision (greedy policy)", fontsize=fs + 1)
        cax.tick_params(labelsize=fs - 1)
        fig.savefig(out, dpi=75, bbox_inches="tight"); plt.close(fig); print("wrote", out)

    def sheet(title, setup, rows, ncols, out):
        """rows: list of lists of run names; a Target cell plus setup text come first."""
        nrows = len(rows) + 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 6.4 * nrows), squeeze=False)
        mesh = None
        ax = axes[0][0]
        mesh = ax.pcolormesh(np.arange(A_MAX + 2) - .5, np.arange(Z_MAX + 2) - .5,
                             np.ma.masked_where(tmap <= 0, tmap), norm=norm, cmap="viridis")
        ax.set_title("Target (reference yields)", fontsize=14, loc="left")
        ax.set_xlim(0, A_MAX); ax.set_ylim(0, Z_MAX); ax.grid(color="white", lw=.3, alpha=.35)
        ax.tick_params(labelsize=12)
        for j in range(1, ncols):
            axes[0][j].axis("off")
        axes[0][1].text(0, 0.98, setup, va="top", fontsize=14, transform=axes[0][1].transAxes,
                        linespacing=1.5)
        for i, row in enumerate(rows, 1):
            for j in range(ncols):
                if j < len(row):
                    mesh = panel(axes[i][j], row[j])
                else:
                    axes[i][j].axis("off")
        for i in range(nrows):
            for j in range(ncols):
                if axes[i][j].axison:
                    axes[i][j].set_xlabel("$A$", fontsize=14); axes[i][j].set_ylabel("$Z$", fontsize=14)
        fig.suptitle(title, fontsize=20, y=0.995)
        fig.tight_layout(rect=(0, 0, 0.93, 0.985))
        cax = fig.add_axes([0.94, 0.08, 0.012, 0.84])
        fig.colorbar(mesh, cax=cax).set_label("yield per collision (greedy policy)", fontsize=14)
        cax.tick_params(labelsize=12)
        fig.savefig(out, dpi=75, bbox_inches="tight"); plt.close(fig); print("wrote", out)

    common = ("Setup common to all runs:\nzeta_correct reward (full QMD + λ·(−B_BWM)), final_sum,\n"
              "k=4 levels, K=2 slots, lr 3e-4, clip 1.0, learned critic,\n"
              "trained on SpectatorsLeft events 0–1999; A–Z map from 5000\ncollisions (both sides), greedy policy.")
    P = lambda s: [n for n in names if n.startswith(s)]
    sheet("P1 — does the MST pre-train matter? (λ=0.5, 200 RL epochs, rows: pre-train epochs, cols: seeds)",
          common + "\nPre-train lr 1e-3 (default), MST d_cut 2.0 fm.",
          [[f"P1_pre{p}_s{s}" for s in (0, 1, 2)] for p in (0, 1, 5, 10)], 3, "figures/zn_sheet_P1.png")
    lams = [0, 0.125, 0.25, 0.5, 1, 2]
    wide("P2 — λ sweep on B (zeta_correct, final_sum, k=4 K=2, seed 0, 100 RL epochs).  "
         "Top row: no warm start.  Bottom row: 5-epoch MST warm start (lr 1e-3).",
         [[f"P2_l{l:g}_pre0" for l in lams], [f"P2_l{l:g}_pre5" for l in lams]],
         "figures/zn_sheet_P2_wide.png")
    for part, ls in (("a", lams[:3]), ("b", lams[3:])):
        sheet(f"P2{part} — λ sweep on B (seed 0, 100 RL epochs; left: no warm start, right: 5-epoch warm start)",
              common + "\nPre-train lr 1e-3 (default).",
              [[f"P2_l{l:g}_pre0", f"P2_l{l:g}_pre5"] for l in ls], 2, f"figures/zn_sheet_P2{part}.png")
    sheet("λ sweep on B, no warm start (zeta_correct, seed 0, 100 RL epochs)",
          common + "\nNo MST pre-train. Configs differ only in λ.",
          [[f"P2_l{l:g}_pre0" for l in lams[:3]], [f"P2_l{l:g}_pre0" for l in lams[3:]]],
          3, "figures/zn_lambda_sweep_nows.png")
    wide("λ sweep on B, no warm start — zeta_correct, final_sum, k=4 K=2, seed 0, 100 RL epochs "
         "(configs differ only in λ)", [[f"P2_l{l:g}_pre0" for l in lams]],
         "figures/zn_lambda_sweep_nows_row.png", cell=(5.4, 5.6))
    sheet("λ sweep on B, 5-epoch MST warm start (zeta_correct, seed 0, 100 RL epochs)",
          common + "\nPre-train lr 1e-3 (default). Configs differ only in λ.",
          [[f"P2_l{l:g}_pre5" for l in lams[:3]], [f"P2_l{l:g}_pre5" for l in lams[3:]]],
          3, "figures/zn_lambda_sweep_ws.png")
    sheet("P3a — cone-sharpened B (seed 0, 100 RL epochs; left: no warm start, right: 5-epoch warm start)",
          common + "\nκ=38 unless stated; ε=0.5.",
          [[f"P3_cone_l{l:g}_pre0", f"P3_cone_l{l:g}_pre5"] for l in (0.25, 0.5, 1)] +
          [["P3_cone_k100_l0.5_pre0", "P3_cone_k100_l0.5_pre5"]], 2, "figures/zn_sheet_P3a.png")
    sheet("P3b — exp-sharpened B (seed 0, 100 RL epochs; left: no warm start, right: 5-epoch warm start)",
          common + "\nτ=25.", [[f"P3_exp_l{l:g}_pre0", f"P3_exp_l{l:g}_pre5"] for l in (0.25, 0.5, 1)],
          2, "figures/zn_sheet_P3b.png")
    sheet("P4 — B/A in place of B (seed 0, 100 RL epochs; left: no warm start, right: 5-epoch warm start)",
          common + "\nNOTE: eval R is on the B/A reward itself,\nnot comparable to the other sheets.",
          [[f"P4_perA_l{l:g}_pre0", f"P4_perA_l{l:g}_pre5"] for l in (0.1, 0.5, 2, 8)], 2, "figures/zn_sheet_P4.png")


if __name__ == "__main__":
    main()
