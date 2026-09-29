"""Evaluate a trained split policy on the terminal reward, against reference partitions.

    PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python experiments/eval_final_sum.py \\
        --run-dir outputs/2026-09-29/17-40-17 --n-events 5000

Loads ``dm_model.pt`` and its ``resolved_config.yaml`` from a run directory and scores
the *greedy* (argmax) policy's final partition with R = -sum_leaves E(leaf) / N on the
first ``--n-events`` events, using the energy the run trained on.  Reference
partitions are scored on the same events, so the comparison is paired:

  policy         the trained model, eval mode (argmax assignments), k levels
  policy sampled the same model in train mode, i.e. sampling assignments — the
                 stochastic policy that REINFORCE actually optimised
  all together   no split
  MST d=2.0 / 1.5, MSTp d=3.0/p=150, singletons

The dataset orders events by id, so events [0, train_n_events) are the ones the model
trained on ("seen") and the rest are unseen.  Nucleon order within an event is
shuffled by NucleonDataset without a seed, so this is reproducible only up to that.
Per-event results are written to <run-dir>/eval_final_sum_<n>.csv.
"""
import argparse
import math
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from clustering.physics import saca_qmd_minus_b_energy, zeta_correct_energy
from clustering.split_prediction.dataset import NucleonDataset, collate_fn
from clustering.split_prediction.model import SplitPredictionModel
from clustering.split_prediction.mst import mst_clusters
from clustering.split_prediction.trainer import compute_final_sum_reward, k_level_forward


def leaves(labels: torch.Tensor, mask: torch.Tensor):
    return {c: (labels == c) & mask for c in range(int(labels.max().item()) + 1)
            if ((labels == c) & mask).any()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-events", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--data", default="data/xecs_hse.parquet")
    ap.add_argument("--lam", type=float, default=None,
                    help="score with this bwm_weight instead of the run's own (the CSV name "
                         "then carries it), e.g. to score one model at several lambdas")
    args = ap.parse_args()

    run = Path(args.run_dir)
    cfg = yaml.safe_load(open(run / "resolved_config.yaml"))["config"]
    rt = cfg["reward_type"]
    lam = cfg["bwm_weight"] if args.lam is None else args.lam
    if rt == "saca_qmd_minus_b":
        energy = partial(saca_qmd_minus_b_energy, bwm_weight=lam, binding=cfg["bwm_form"])
    elif rt == "zeta_correct":
        energy = partial(zeta_correct_energy, bwm_weight=lam,
                         spin_factor=cfg["zeta_spin_factor"], yukawa=cfg["zeta_yukawa"])
    else:
        raise SystemExit(f"eval_final_sum supports saca_qmd_minus_b / zeta_correct, not {rt}")
    k, n_train = cfg["split_k"], cfg["n_events"]

    model = SplitPredictionModel(input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"],
                                 n_iters=cfg["n_iters"], n_clusters=cfg["n_clusters"])
    model.load_state_dict(torch.load(run / "dm_model.pt", map_location="cpu"))
    model.eval()

    ds = NucleonDataset(args.data, particle_type=cfg["particle_type"], n_events=args.n_events)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
    print(f"run {run}: {rt}, lambda={lam}, mode={cfg.get('reward_mode', 'node_diff')}, "
          f"trained on the first {n_train} events, k={k}, K={cfg['n_clusters']}")
    print(f"evaluating {len(ds)} events in {len(loader)} batches of {args.batch} "
          f"(events with index < {n_train} are 'seen')")

    names = ["policy", "policy sampled", "all together", "MST d=2.0", "MST d=1.5", "MSTp d=3.0/p=150", "singletons"]
    R = {n: [] for n in names}
    frags = {n: [] for n in names}
    largest = {n: [] for n in names}
    n_nuc, unsplit = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", unit="batch", mininterval=2.0, dynamic_ncols=True):
            x, mask = batch["x"], batch["mask"]
            N = x.shape[1]
            out = k_level_forward(model, x, mask, k, min_fragment_size=2)
            model.train()                       # sampling, as during REINFORCE
            out_s = k_level_forward(model, x, mask, k, min_fragment_size=2)
            model.eval()
            parts = {
                "policy": out["leaf_masks"],
                "policy sampled": out_s["leaf_masks"],
                "all together": {(): mask},
                "MST d=2.0": leaves(mst_clusters(x, mask, d_cut=2.0, metric="coord"), mask),
                "MST d=1.5": leaves(mst_clusters(x, mask, d_cut=1.5, metric="coord"), mask),
                "MSTp d=3.0/p=150": leaves(
                    mst_clusters(x, mask, d_cut=3.0, p_cut=150.0, metric="mstp"), mask),
                "singletons": {j: mask & (torch.arange(N)[None, :] == j) for j in range(N)},
            }
            for n, lv in parts.items():
                R[n].append(compute_final_sum_reward(x, mask, lv, 7, 2, energy_fn=energy))
                sizes = torch.stack([m.sum(dim=1) for m in lv.values()])        # (leaves, B)
                frags[n].append((sizes >= 2).sum(dim=0).float())
                largest[n].append(sizes.max(dim=0).values.float())
            n_nuc.append(mask.sum(dim=1).float())
            sizes = torch.stack([m.sum(dim=1) for m in parts["policy"].values()])
            unsplit.append(((sizes > 0).sum(dim=0) <= 1).float())

    R = {n: torch.cat(v).numpy() for n, v in R.items()}
    frags = {n: torch.cat(v).numpy() for n, v in frags.items()}
    largest = {n: torch.cat(v).numpy() for n, v in largest.items()}
    n_nuc, unsplit = torch.cat(n_nuc).numpy(), torch.cat(unsplit).numpy()
    seen = np.arange(len(n_nuc)) < n_train

    def sem(a):
        return float(np.std(a, ddof=1) / math.sqrt(len(a))) if len(a) > 1 else float("nan")

    for label, sel in (("ALL", np.ones_like(seen)), ("SEEN (trained on)", seen),
                       ("UNSEEN", ~seen)):
        if sel.sum() < 2:
            continue
        print(f"\n=== {label}: {int(sel.sum())} events ===")
        print(f"{'partition':<20}{'R mean':>9}{'± sem':>7}{'policy - this':>15}{'± sem':>7}"
              f"{'frags':>7}{'largest':>9}")
        for n in names:
            d = R["policy"][sel] - R[n][sel]
            dtxt = f"{d.mean():15.2f}{sem(d):7.2f}" if n != "policy" else f"{'':>22}"
            print(f"{n:<20}{R[n][sel].mean():9.2f}{sem(R[n][sel]):7.2f}{dtxt}"
                  f"{frags[n][sel].mean():7.2f}{largest[n][sel].mean():9.1f}")
        print(f"policy leaves events unsplit: {100 * unsplit[sel].mean():.1f}%   "
              f"mean nucleons/event: {n_nuc[sel].mean():.1f}")

    tag = "" if args.lam is None else f"_lam{args.lam:g}"
    out = run / f"eval_final_sum_{len(n_nuc)}{tag}.csv"
    with open(out, "w") as f:
        f.write("idx,seen,n_nucleons,unsplit," + ",".join(f"R[{n}]" for n in names)
                + ",frags[policy],largest[policy]\n")
        for i in range(len(n_nuc)):
            f.write(f"{i},{int(seen[i])},{int(n_nuc[i])},{int(unsplit[i])},"
                    + ",".join(f"{R[n][i]:.4f}" for n in names)
                    + f",{int(frags['policy'][i])},{int(largest['policy'][i])}\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
