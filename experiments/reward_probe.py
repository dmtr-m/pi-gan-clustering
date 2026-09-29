"""What the QMD - B reward pays for a partition, before spending a run on it.

    .venv/bin/python experiments/reward_probe.py --n-events 200

The split reward is  q = (E_parent - sum E_leaf) / N_parent  with
E = V - lambda*B, so writing it out per parent nucleon,

    q*N = V_cut + lambda * dB,
    V_cut = V_parent - sum V_leaf        (the bonds the split breaks)
    dB    = sum B_leaf - B_parent        (binding gained by reshaping the mass)

Both pieces are measured here on real events for four partitions, so the two
pressures can be read off separately instead of inferred from a training curve:

  no-split   the trivial partition.  q = 0 identically, and it is the thing to
             beat: any partition scoring below it means the reward's preferred
             answer is "do not split at all".
  MST        coordinate MST at d_cut = 2.0 — the warm start the policy is
             pretrained on.
  MSTp       d_cut = 3.0, p_cut = 150 MeV/c — the best classical clusterizer we
             have (mass RMS 0.207, BASELINES.md), i.e. the closest available
             stand-in for the answer we want the policy to find.
  random     a uniform random 6-way assignment — noise, as the floor.

The point of the lambda scan is that dB and V_cut have opposite signs for a good
partition, so lambda sets which one wins.  On the SACA annealer raising lambda
collapsed multiplicity (3.73 -> 2.20 fragments as lambda went 0.25 -> 1.5); this
says whether the same knob does the same thing to the reward landscape.
"""
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch

from clustering.physics import (
    bethe_weizsacker,
    qmd_minus_b_energy,
    saca_qmd_minus_b_energy,
    total_potential_energy,
    weizsacker_formula,
    zeta_correct_energy,
)
from clustering.split_prediction.dataset import NucleonDataset
from clustering.split_prediction.mst import mst_clusters

LAMBDAS = [0.0, 0.25, 0.5, 1.0, 1.5]
ZETA_LAMBDAS = [0.0, 0.5, 1.0, 1.5, 2.0, 4.0]   # zeta_correct needed lambda >~ 1.5 in SACA
N_RANDOM = 6


def binding(A: torch.Tensor, Z: torch.Tensor, form: str) -> torch.Tensor:
    """B(A, Z) [MeV], zero below A = 2 — the convention qmd_minus_b_energy uses."""
    B = torch.zeros_like(A)
    big = A >= 2
    if big.any():
        B[big] = (bethe_weizsacker(A[big], Z[big], modified=True) if form == "bwm"
                  else weizsacker_formula(A[big], Z[big]))
    return B


def leaf_masks_from_labels(labels: torch.Tensor, mask: torch.Tensor) -> List[torch.Tensor]:
    """One (B, N) bool mask per label id, over the whole batch."""
    out = []
    for c in range(int(labels.max().item()) + 1):
        m = (labels == c) & mask
        if m.any():
            out.append(m)
    return out


def score(x: torch.Tensor, mask: torch.Tensor,
          leaves: List[torch.Tensor], zeta_kw: Dict) -> Dict[str, torch.Tensor]:
    """V_cut, dB (both mass formulas) and the fragment count, per event."""
    A_p = mask.sum(dim=1).float()
    Z_p = ((x[..., 7] == 1) & mask).sum(dim=1).float()
    V_cut = total_potential_energy(x, mask)
    E_cut = saca_qmd_minus_b_energy(x, mask, bwm_weight=0.0)
    Z_cut = zeta_correct_energy(x, mask, bwm_weight=0.0, **zeta_kw)
    dB = {f: -binding(A_p, Z_p, f) for f in ("bwm", "bw")}
    n_frag = torch.zeros_like(A_p)

    for lm in leaves:
        A = lm.sum(dim=1).float()
        Z = ((x[..., 7] == 1) & lm).sum(dim=1).float()
        V_cut = V_cut - total_potential_energy(x, lm)
        E_cut = E_cut - saca_qmd_minus_b_energy(x, lm, bwm_weight=0.0)
        Z_cut = Z_cut - zeta_correct_energy(x, lm, bwm_weight=0.0, **zeta_kw)
        for f in dB:
            dB[f] = dB[f] + binding(A, Z, f)
        n_frag = n_frag + (A >= 2).float()

    return dict(V_cut=V_cut, E_cut=E_cut, Z_cut=Z_cut, dB_bwm=dB["bwm"], dB_bw=dB["bw"],
                n_frag=n_frag, n_parent=A_p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-events", type=int, default=200)
    ap.add_argument("--data", default="data/xecs_hse.parquet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zeta-spin", type=float, default=0.5)
    ap.add_argument("--zeta-yukawa", default="folded")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ds = NucleonDataset(args.data, particle_type="SpectatorsLeft")
    n = min(args.n_events, len(ds))
    events = [ds[i] for i in range(n)]
    events = [e[0] if isinstance(e, (tuple, list)) else e for e in events]

    zeta_kw = dict(spin_factor=args.zeta_spin, yukawa=args.zeta_yukawa)
    acc: Dict[str, List[Dict[str, torch.Tensor]]] = {}
    batch = 16
    for lo in range(0, n, batch):
        chunk = events[lo:lo + batch]
        N = max(t.shape[0] for t in chunk)
        x = torch.zeros(len(chunk), N, chunk[0].shape[1])
        mask = torch.zeros(len(chunk), N, dtype=torch.bool)
        for i, t in enumerate(chunk):
            x[i, : t.shape[0]] = t
            mask[i, : t.shape[0]] = True

        parts: Dict[str, List[torch.Tensor]] = {
            "no-split": [mask],
            "MST d=2.0": leaf_masks_from_labels(
                mst_clusters(x, mask, d_cut=2.0, metric="coord"), mask),
            "MSTp d=3.0/p=150": leaf_masks_from_labels(
                mst_clusters(x, mask, d_cut=3.0, p_cut=150.0, metric="mstp"), mask),
        }
        rnd = torch.randint(0, N_RANDOM, (len(chunk), N))
        parts[f"random {N_RANDOM}-way"] = [(rnd == c) & mask for c in range(N_RANDOM)]

        for name, leaves in parts.items():
            acc.setdefault(name, []).append(score(x, mask, leaves, zeta_kw))

    print(f"{n} events, {args.data}\n")
    hdr = (f"{'partition':<20}{'frags':>7}{'V_cut':>10}"
           + "".join(f"{'dB_' + f:>9}" for f in ("bwm", "bw")))
    lam_hdr = "".join(f"{'q*N l=' + f'{l:g}':>12}" for l in LAMBDAS)
    print("BWM binding")
    print(hdr + lam_hdr)
    rows = {}
    for name, chunks in acc.items():
        agg = {k: torch.cat([c[k] for c in chunks]) for k in chunks[0]}
        rows[name] = agg
        line = (f"{name:<20}{agg['n_frag'].mean():7.2f}{agg['V_cut'].mean():10.1f}"
                f"{agg['dB_bwm'].mean():9.1f}{agg['dB_bw'].mean():9.1f}")
        line += "".join(f"{(agg['V_cut'] + l * agg['dB_bwm']).mean():12.1f}"
                        for l in LAMBDAS)
        print(line)
    print("\nplain BW binding (what the weizsacker_qmd reward uses)")
    print(hdr + lam_hdr)
    for name, agg in rows.items():
        line = (f"{name:<20}{agg['n_frag'].mean():7.2f}{agg['V_cut'].mean():10.1f}"
                f"{agg['dB_bwm'].mean():9.1f}{agg['dB_bw'].mean():9.1f}")
        line += "".join(f"{(agg['V_cut'] + l * agg['dB_bw']).mean():12.1f}"
                        for l in LAMBDAS)
        print(line)

    # The same table on the baselines' own QMD energy — a saturating Skyrme plus
    # the rest-frame kinetic term — instead of physics.py's potential.  This is
    # the energy the SACA annealer minimizes, so it is the one the ported
    # objective is supposed to be built on.
    print("\nBWM binding, on the baselines' QMD energy (qmd_energy.cluster_energy)")
    print(hdr.replace("V_cut", "E_cut") + lam_hdr)
    for name, agg in rows.items():
        line = (f"{name:<20}{agg['n_frag'].mean():7.2f}{agg['E_cut'].mean():10.1f}"
                f"{agg['dB_bwm'].mean():9.1f}{agg['dB_bw'].mean():9.1f}")
        line += "".join(f"{(agg['E_cut'] + l * agg['dB_bwm']).mean():12.1f}"
                        for l in LAMBDAS)
        print(line)

    # zeta_correct: the reward reward_type="zeta_correct" pays.  Its lambda multiplies
    # bwd = -B_BWM, so q*N = Z_cut + lambda * dB_bwm, same algebra as above.
    print(f"\nzeta_correct (spin={args.zeta_spin:g}, yukawa={args.zeta_yukawa}), BWM binding")
    print(hdr.replace("V_cut", "Z_cut") + "".join(f"{'q*N l=' + f'{l:g}':>12}" for l in ZETA_LAMBDAS))
    for name, agg in rows.items():
        line = (f"{name:<20}{agg['n_frag'].mean():7.2f}{agg['Z_cut'].mean():10.1f}"
                f"{agg['dB_bwm'].mean():9.1f}{agg['dB_bw'].mean():9.1f}")
        line += "".join(f"{(agg['Z_cut'] + l * agg['dB_bwm']).mean():12.1f}"
                        for l in ZETA_LAMBDAS)
        print(line)

    print("\nq*N is per parent nucleus, MeV; no-split is 0 by construction.")
    print("A partition scores above no-split only where its q*N > 0.")


if __name__ == "__main__":
    main()
