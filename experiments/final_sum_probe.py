"""Score the terminal reward R = -sum_leaves E(leaf) / N on hand-built partitions.

    PYTHONPATH=src .venv/bin/python experiments/final_sum_probe.py --n-events 100

Higher R is better (R = minus the SACA-style energy per parent nucleon).  The
partitions run from no split to full atomisation, with MST at several cutoffs and
noised copies of an MST partition, so the reward can be read along one axis:

  all together    one leaf = the whole event
  MST d=...       coordinate MST at several cutoffs (more fragments as d falls)
  MSTp            d=3.0, p=150 MeV/c, the best classical clusterizer we have
  singletons      every nucleon alone (E = 0 for each, so R = 0)
  noised MST      MST d=2.0 with a fraction f of nucleons moved to a random other
                  fragment label — how fast does R degrade as the partition rots
  random K-way    uniform random assignment, the floor

Columns give mean R, its standard error over events, the mean number of fragments
with >= 2 nucleons, and the rank.  The random and noised rows use a seeded
generator, so they are reproducible.
"""
import argparse
from functools import partial
from typing import Dict, List

import torch

from clustering.physics import saca_qmd_minus_b_energy, zeta_correct_energy
from clustering.split_prediction.dataset import NucleonDataset
from clustering.split_prediction.mst import mst_clusters
from clustering.split_prediction.trainer import compute_final_sum_reward


def leaves_from_labels(labels: torch.Tensor, mask: torch.Tensor) -> Dict[int, torch.Tensor]:
    return {c: (labels == c) & mask for c in range(int(labels.max().item()) + 1)
            if ((labels == c) & mask).any()}


def noised(labels: torch.Tensor, mask: torch.Tensor, frac: float,
           gen: torch.Generator) -> torch.Tensor:
    """Move a fraction ``frac`` of real nucleons to a uniformly random existing label."""
    k = int(labels.max().item()) + 1
    flip = (torch.rand(labels.shape, generator=gen) < frac) & mask
    new = torch.randint(0, k, labels.shape, generator=gen)
    return torch.where(flip, new, labels)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-events", type=int, default=100)
    ap.add_argument("--data", default="data/xecs_hse.parquet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = NucleonDataset(args.data, particle_type="SpectatorsLeft")
    n = min(args.n_events, len(ds))
    events = [ds[i] for i in range(n)]
    events = [e[0] if isinstance(e, (tuple, list)) else e for e in events]

    energies = {
        "saca_qmd_minus_b  l=0": partial(saca_qmd_minus_b_energy, bwm_weight=0.0),
        "saca_qmd_minus_b  l=0.5": partial(saca_qmd_minus_b_energy, bwm_weight=0.5),
        "saca_qmd_minus_b  l=1": partial(saca_qmd_minus_b_energy, bwm_weight=1.0),
        "zeta_correct      l=0": partial(zeta_correct_energy, bwm_weight=0.0),
        "zeta_correct      l=1": partial(zeta_correct_energy, bwm_weight=1.0),
    }
    R: Dict[str, Dict[str, List[torch.Tensor]]] = {e: {} for e in energies}
    frags: Dict[str, List[torch.Tensor]] = {}
    gen = torch.Generator().manual_seed(args.seed)

    for lo in range(0, n, 16):
        chunk = events[lo:lo + 16]
        N = max(t.shape[0] for t in chunk)
        x = torch.zeros(len(chunk), N, chunk[0].shape[1])
        mask = torch.zeros(len(chunk), N, dtype=torch.bool)
        for i, t in enumerate(chunk):
            x[i, : t.shape[0]] = t
            mask[i, : t.shape[0]] = True

        mst2 = mst_clusters(x, mask, d_cut=2.0, metric="coord")
        parts: Dict[str, Dict[int, torch.Tensor]] = {
            "all together": {0: mask},
            "MST d=3.0": leaves_from_labels(mst_clusters(x, mask, d_cut=3.0, metric="coord"), mask),
            "MST d=2.0": leaves_from_labels(mst2, mask),
            "MST d=1.5": leaves_from_labels(mst_clusters(x, mask, d_cut=1.5, metric="coord"), mask),
            "MSTp d=3.0/p=150": leaves_from_labels(
                mst_clusters(x, mask, d_cut=3.0, p_cut=150.0, metric="mstp"), mask),
            "singletons": {j: mask & (torch.arange(N)[None, :] == j) for j in range(N)},
        }
        for f in (0.05, 0.15, 0.30):
            parts[f"MST d=2.0 + {int(f*100)}% noise"] = leaves_from_labels(
                noised(mst2, mask, f, gen), mask)
        for K in (2, 6):
            parts[f"random {K}-way"] = leaves_from_labels(
                torch.randint(0, K, (len(chunk), N), generator=gen), mask)

        for name, leaves in parts.items():
            frags.setdefault(name, []).append(
                sum((lm.sum(dim=1) >= 2).float() for lm in leaves.values()))
            for en, fn in energies.items():
                R[en].setdefault(name, []).append(
                    compute_final_sum_reward(x, mask, leaves, energy_fn=fn))

    print(f"{n} events, {args.data}.  R = -sum E(leaf) / N  [MeV/nucleon]; higher is better\n")
    names = list(next(iter(R.values())).keys())
    for en in energies:
        print(en)
        rows = []
        for name in names:
            r = torch.cat(R[en][name])
            rows.append((name, float(r.mean()), float(r.std() / len(r) ** 0.5),
                         float(torch.cat(frags[name]).mean())))
        order = {nm: i + 1 for i, (nm, *_rest) in
                 enumerate(sorted(rows, key=lambda t: -t[1]))}
        print(f"  {'partition':<26}{'frags':>7}{'R mean':>10}{'± sem':>8}{'rank':>6}")
        for name, m, se, fr in rows:
            print(f"  {name:<26}{fr:7.2f}{m:10.2f}{se:8.2f}{order[name]:6d}")
        print()


if __name__ == "__main__":
    main()
