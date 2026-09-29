"""(A, Z) fragment-yield maps for a trained split policy, next to the classical baselines.

    PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python experiments/az_policy.py \\
        --run-dir outputs/2026-09-29/17-40-17 --n-events 5000

Same renderer, target and stat box as experiments/baseline_grid.py, so the figure is
directly comparable with figures/az_*.png.  Panels: Target, the policy's greedy
(argmax) partition, the policy sampled as during REINFORCE, and MST d=2.0 / d=1.5 /
MSTp d=3.0,p=150 on the same events.

One collision = two events (one per spectator side), and the target is a yield per
collision, so both sides are used: --n-events events per side -> n-events collisions.
The policy was trained on SpectatorsLeft only; the right side is out of distribution
for it, and is included only so the yields are per collision like the other figures.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from baseline_grid import BINS, print_bands, print_table, render_grid  # noqa: E402
from target_reference import load_target  # noqa: E402

from clustering.split_prediction.dataset import NucleonDataset, collate_fn  # noqa: E402
from clustering.split_prediction.model import SplitPredictionModel  # noqa: E402
from clustering.split_prediction.mst import mst_clusters  # noqa: E402
from clustering.split_prediction.trainer import k_level_forward  # noqa: E402

A_MAX, Z_MAX = 132, 60     # render_grid's defaults; checked against the data below


def add_fragments(counter: Counter, x: torch.Tensor, leaf_masks) -> None:
    """Count every non-empty leaf, per event, as one (A, Z) fragment."""
    isp = (x[..., 7] == 1)
    for lm in leaf_masks.values():
        A = lm.sum(dim=1)
        Z = (isp & lm).sum(dim=1)
        for a, z in zip(A.tolist(), Z.tolist()):
            if a > 0:
                counter[(a, z)] += 1


def labels_to_leaves(labels: torch.Tensor, mask: torch.Tensor):
    return {c: (labels == c) & mask for c in range(int(labels.max().item()) + 1)
            if ((labels == c) & mask).any()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-events", type=int, default=5000, help="events per spectator side")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--data", default="data/xecs_hse.parquet")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Path(args.run_dir)
    cfg = yaml.safe_load(open(run / "resolved_config.yaml"))["config"]
    model = SplitPredictionModel(input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"],
                                 n_iters=cfg["n_iters"], n_clusters=cfg["n_clusters"])
    model.load_state_dict(torch.load(run / "dm_model.pt", map_location="cpu"))
    model.eval()
    k = cfg["split_k"]

    names = ["policy", "policy sampled", "MST d=2.0", "MST d=1.5", "MSTp d=3.0/p=150"]
    counters: Dict[str, Counter] = {n: Counter() for n in names}
    n_events = 0
    sides = ("SpectatorsLeft", "SpectatorsRight")
    loaders = []
    for side in sides:
        ds = NucleonDataset(args.data, particle_type=side, n_events=args.n_events)
        loaders.append((side, DataLoader(ds, batch_size=args.batch, shuffle=False,
                                         collate_fn=collate_fn)))
    total_batches = sum(len(l) for _, l in loaders)
    print(f"run {run} (trained on SpectatorsLeft, first {cfg['n_events']} events)")
    print(f"{args.n_events} events per side x 2 sides in {total_batches} batches of "
          f"{args.batch}; one collision = two events")

    with torch.no_grad(), tqdm(total=total_batches, desc="A-Z", unit="batch",
                               mininterval=2.0, dynamic_ncols=True) as bar:
        for side, loader in loaders:
            for batch in loader:
                x, mask = batch["x"], batch["mask"]
                n_events += x.shape[0]
                add_fragments(counters["policy"], x,
                              k_level_forward(model, x, mask, k, 2)["leaf_masks"])
                model.train()
                add_fragments(counters["policy sampled"], x,
                              k_level_forward(model, x, mask, k, 2)["leaf_masks"])
                model.eval()
                add_fragments(counters["MST d=2.0"], x, labels_to_leaves(
                    mst_clusters(x, mask, d_cut=2.0, metric="coord"), mask))
                add_fragments(counters["MST d=1.5"], x, labels_to_leaves(
                    mst_clusters(x, mask, d_cut=1.5, metric="coord"), mask))
                add_fragments(counters["MSTp d=3.0/p=150"], x, labels_to_leaves(
                    mst_clusters(x, mask, d_cut=3.0, p_cut=150.0, metric="mstp"), mask))
                bar.update(1)

    n_coll = n_events / 2.0
    # render_grid crops silently to A<=A_MAX, Z<=Z_MAX: check the data fits.
    amax = max(a for c in counters.values() for (a, _z) in c)
    zmax = max(z for c in counters.values() for (_a, z) in c)
    print(f"\n{n_events} events = {n_coll:.0f} collisions; largest fragment A={amax}, Z={zmax} "
          f"(plot limits A<={A_MAX}, Z<={Z_MAX}{'  -- CROPPED!' if amax > A_MAX or zmax > Z_MAX else ''})")

    ref = load_target()
    g = np.array([ref[(ref[:, 0] >= lo) & (ref[:, 0] <= hi), 2].sum() for lo, hi in BINS])
    layout: List[Tuple[int, int, str]] = [
        (0, 1, "Policy (greedy, argmax)"), (0, 2, "Policy (sampled)"),
        (1, 0, "MST d=2.0"), (1, 1, "MST d=1.5"), (1, 2, "MSTp d=3.0, p=150")]
    panels = [(i, j, t, counters[n]) for (i, j, t), n in zip(layout, names)]
    out = args.out or f"figures/az_policy_{run.name}.png"
    render_grid(panels, ref, n_coll, g, out,
                f"Fragment (A, Z) yield per collision — {n_coll:.0f} collisions — {run.name}",
                n_rows=2, n_cols=3, a_max=A_MAX, z_max=Z_MAX)
    print_table(panels, ref, n_coll, g)
    print_bands(panels, ref)


if __name__ == "__main__":
    main()
