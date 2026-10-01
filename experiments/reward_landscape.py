"""Reward landscape: what the final-sum reward pays for hand-built partitions of real events.

    PYTHONUNBUFFERED=1 .venv/bin/python experiments/reward_landscape.py \
        --n-events 200 --out outputs/reward_landscape > landscape.log 2>&1

The reward under test is the RL policy's terminal reward (``compute_final_sum_reward``
with ``saca_qmd_minus_b_energy``):

    R(lambda) = - sum_leaves (T + V - lambda * B_BWM)(leaf) / N_parent,  E_QMD = T (kinetic) + V (potential)

with singletons (A = 1) contributing exactly 0.  Higher R is better; R > 0 means the
partition is bound.  E_QMD and B_BWM are computed once per leaf, so every lambda comes
from the same pass:  R(lambda) = (-sum E + lambda * sum B) / N.

Partitions are a registry (``PARTITIONS``): name -> f(x, mask, gen) -> labels (B, N),
where label < 0 means "free nucleon".  Adding an experiment is adding one function.

  all_together     every nucleon in one cluster
  all_separate     every nucleon alone (R = 0 by construction — the reference level)
  rand_K{2..6}     uniform random labels in 0..K-1; label 0 = free nucleons, labels
                   1..K-1 = clusters (so K=2 is "one random cluster + free nucleons")
  mst_d{...}       coordinate MST at several cutoffs (fm)

Per partition and lambda the table gives mean R and its standard error over events,
plus the per-nucleon components T/N, V/N, B/N (so R = -T/N - V/N + lambda B/N) and the fragment structure (fragments with A >= 2, free nucleons, % in fragments,
largest fragment).  Per-event values are saved to ``--out``/per_event.npz.
"""
import argparse
import json
import os
from functools import partial
from typing import Callable, Dict, List

import numpy as np
import torch

from clustering.baselines.coalescence import coalescence_clusters
from clustering.baselines.qmd_energy import cluster_energy
from clustering.baselines.saca import SacaParams, _SacaEvent, _anneal, saca_clusters
from clustering.physics import bethe_weizsacker
from clustering.split_prediction.dataset import NucleonDataset
from clustering.split_prediction.mst import mst_clusters

TYPE_INDEX = 7
LAMBDAS = (0.0, 0.5, 1.0)
MST_CUTS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)


# ─── partition registry ──────────────────────────────────────────────────────

def all_together(x, mask, gen):
    return torch.where(mask, 0, -1)


def all_separate(x, mask, gen):
    return torch.full(mask.shape, -1)


def random_labels(K: int):
    def f(x, mask, gen):
        lab = torch.randint(0, K, mask.shape, generator=gen)
        return torch.where(mask & (lab > 0), lab, -1)        # label 0 -> free nucleons
    return f


def mst(d_cut: float):
    def f(x, mask, gen):
        return mst_clusters(x, mask, d_cut=d_cut, metric="coord")
    return f


def noised_mst(d_cut: float, frac: float):
    """MST labels with a fraction of nucleons moved to the label of a uniformly chosen
    other nucleon (so large fragments attract more mistakes) — a policy that is mostly
    right.  Free nucleons that get moved join a fragment; fragment members can be lost."""
    def f(x, mask, gen):
        lab = mst_clusters(x, mask, d_cut=d_cut, metric="coord")
        donor = lab.gather(1, torch.randint(0, lab.shape[1], lab.shape, generator=gen))
        flip = (torch.rand(lab.shape, generator=gen) < frac) & mask
        return torch.where(flip, donor, lab)
    return f


def mst_metric(metric: str, d_cut: float = 2.0, p_cut: float = 150.0):
    def f(x, mask, gen):
        return mst_clusters(x, mask, d_cut=d_cut, p_cut=p_cut, metric=metric)
    return f


def coalescence(cut_set: str, selection: str):
    def f(x, mask, gen):
        return torch.as_tensor(
            coalescence_clusters(x[0], cut_set=cut_set, selection=selection).labels)[None]
    return f


def saca(energy_model: str, lam: float, **anneal):
    """SACA seeded from coordinate MST d=2.0 (as experiments/qmd_minus_b.py).  With
    energy_model='qmd_minus_b' the annealer minimises sum_f (E_QMD - lam*B), i.e. exactly
    the negative of this reward — so its output is the reward's own optimiser."""
    params = SacaParams(e_cut_light=0.0, energy_model=energy_model, bwm_weight=lam, **anneal)
    def f(x, mask, gen):
        rng = np.random.default_rng(int(torch.randint(0, 2 ** 31, (1,), generator=gen)))
        lab = saca_clusters(x[0], params, d_cut=2.0, metric="coord", rng=rng).labels
        return torch.as_tensor(lab)[None]
    return f


def saca_whole(lam: float, start: str, alpha: float, trials: int, t_max: float = 2.0,
               pass2: bool = True):
    """Unrestricted annealer on this reward: no stable/unstable split, no admissibility cut.

    ``saca_clusters`` only anneals MST fragments it classifies as unstable, so the big
    bound residue is never touched and the schedule hardly matters.  Here pass 1 (release /
    transfer, ``allow_loss=True``) runs on the *whole* event from ``start`` ('all' = one
    cluster, 'mst2' = MST d=2.0 fragments) minimising sum_f (E_QMD - lam*B), the negative of
    the reward, and returns the best configuration visited.  ``pass2`` adds SACA's second
    pass (absorb free nucleons / merge clusters).  Leaves are scored as they come out."""
    params = SacaParams(e_cut_light=0.0, energy_model="qmd_minus_b", bwm_weight=lam,
                        alpha=alpha, trials_per_nucleon=trials, t_max=t_max)
    def f(x, mask, gen):
        rng = np.random.default_rng(int(torch.randint(0, 2 ** 31, (1,), generator=gen)))
        ev = _SacaEvent(x[0], TYPE_INDEX, asymmetry=params.asymmetry, e_0_asy=params.e_0_asy,
                        gamma_asy=params.gamma_asy, energy_model=params.energy_model,
                        bwm_weight=params.bwm_weight, mix_alpha=params.mix_alpha,
                        mix_per_nucleon=params.mix_per_nucleon, spin_factor=params.spin_factor,
                        yukawa=params.yukawa)
        n = x.shape[1]
        if start == "all":
            cl = {0: list(range(n))}
        else:
            lab = mst_clusters(x, mask, d_cut=2.0, metric="coord")[0].numpy()
            cl = {int(c): list(np.where(lab == c)[0]) for c in np.unique(lab)}
        cl, _, _ = _anneal(ev, cl, params, rng, allow_loss=True)
        if pass2 and len(cl) > 1:
            cl, _, _ = _anneal(ev, cl, params, rng, allow_loss=False)
        lab = np.full(n, -1, dtype=np.int64)
        for c, m in enumerate(cl.values()):
            lab[m] = c
        return torch.as_tensor(lab)[None]
    return f


PARTITIONS: Dict[str, Callable] = {
    "all_together": all_together,
    "all_separate": all_separate,
    **{f"rand_K{K}": random_labels(K) for K in (2, 3, 4, 5, 6)},
    **{f"mst_d{d:g}": mst(d) for d in MST_CUTS},
    "mstp_d3_p150": mst_metric("mstp", 3.0, 150.0),
    "mom_p90": mst_metric("momentum", p_cut=90.0),
    "coal_M2_ebind": coalescence("M2", "ebind"),
    "saca_qmd": saca("qmd", 0.0),
    "saca_qmd-B_l0.5": saca("qmd_minus_b", 0.5),
    "saca_qmd-B_l1": saca("qmd_minus_b", 1.0),
    # longer anneals: ~35 T-steps x 4 trials/nucleon is the default (alpha=0.9)
    **{f"saca_qmd-B_l{lam:g}_{tag}": saca("qmd_minus_b", lam, alpha=a, trials_per_nucleon=tr)
       for lam in (0.0, 0.5, 1.0)
       for tag, a, tr in (("a98", 0.98, 4), ("a98t16", 0.98, 16), ("a995t32", 0.995, 32))},
    "whole_all_l0.5_T20": saca_whole(0.5, "all", 0.9, 4, t_max=20.0),
    **{f"whole_{st}_l{lam:g}_{tag}": saca_whole(lam, st, a, tr)
       for st in ("all", "mst2") for lam in (0.0, 0.5, 1.0)
       for tag, a, tr in (("base", 0.9, 4), ("long", 0.98, 8), ("vlong", 0.995, 16))},
    **{f"mst_d2_noise{int(100 * q)}": noised_mst(2.0, q) for q in (0.02, 0.05, 0.1, 0.2, 0.4)},
}


# ─── scoring ─────────────────────────────────────────────────────────────────

def score_event(xb: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """E_QMD and B sums over leaves with A >= 2, plus fragment structure, for one event."""
    n = len(labels)                          # one unpadded event
    T = V = B = 0.0
    sizes: List[int] = []
    for c in np.unique(labels[labels >= 0]):
        sel = labels == c
        A = int(sel.sum())
        if A < 2:
            continue
        sizes.append(A)
        # same call and defaults as saca_qmd_minus_b_energy; E_QMD = T + V
        terms = cluster_energy(xb[sel].astype(np.float64))
        T += terms.kinetic
        V += terms.total - terms.kinetic
        Z = int((xb[sel][:, TYPE_INDEX] == 1).sum())
        B += float(bethe_weizsacker(torch.tensor([float(A)]), torch.tensor([float(Z)]), modified=True))
    in_frag = sum(sizes)
    return dict(T=T, V=V, E=T + V, B=B, N=n, frags=len(sizes), in_frag=in_frag,
                free=n - in_frag, amax=max(sizes, default=0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-events", type=int, default=200)
    ap.add_argument("--data", default="data/xecs_hse.parquet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", help="subset of partition names")
    ap.add_argument("--out", default="outputs/reward_landscape")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    np.random.seed(args.seed)    # NucleonDataset shuffles nucleon order with the global numpy RNG
    ds = NucleonDataset(args.data, particle_type="SpectatorsLeft")
    n = min(args.n_events, len(ds))
    events = [ds[i] for i in range(n)]
    events = [e[0] if isinstance(e, (tuple, list)) else e for e in events]
    names = args.only or list(PARTITIONS)
    gen = torch.Generator().manual_seed(args.seed)

    rec: Dict[str, Dict[str, List[float]]] = {nm: {} for nm in names}
    for i, t in enumerate(events):
        x = t[None].float()
        mask = torch.ones(1, t.shape[0], dtype=torch.bool)
        for nm in names:
            lab = PARTITIONS[nm](x, mask, gen)[0].numpy()
            for k, v in score_event(t.numpy(), lab).items():
                rec[nm].setdefault(k, []).append(v)
        if (i + 1) % 20 == 0:
            print(f"event {i + 1}/{n}", flush=True)

    # per-event arrays + summary
    np.savez(os.path.join(args.out, "per_event.npz"),
             **{f"{nm}/{k}": np.array(v) for nm, d in rec.items() for k, v in d.items()})
    summary = {}
    for nm in names:
        d = {k: np.array(v) for k, v in rec[nm].items()}
        s = dict(frags=d["frags"].mean(), free=d["free"].mean(),
                 pct_in_frag=100 * d["in_frag"].sum() / d["N"].sum(), amax=d["amax"].mean())
        for lam in LAMBDAS:
            R = (-d["E"] + lam * d["B"]) / d["N"]
            s[f"R{lam:g}"] = float(R.mean())
            s[f"sem{lam:g}"] = float(R.std(ddof=1) / len(R) ** 0.5)
        s["T_per_N"] = float((d["T"] / d["N"]).mean())
        s["V_per_N"] = float((d["V"] / d["N"]).mean())
        s["B_per_N"] = float((d["B"] / d["N"]).mean())
        summary[nm] = {k: float(v) for k, v in s.items()}
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)

    print(f"\n{n} events, seed {args.seed}.  R = (-sum E_QMD + lambda sum B_BWM)/N [MeV/nucleon]; "
          f"mean ± sem over events\n")
    hdr = f"{'partition':<14}{'frags':>7}{'free':>7}{'%frag':>7}{'Amax':>6}{'T/N':>7}{'V/N':>7}{'B/N':>7}"
    hdr += "".join(f"{'R l=' + format(l, 'g'):>16}" for l in LAMBDAS)
    print(hdr)
    for nm, s in summary.items():
        row = f"{nm:<14}{s['frags']:7.2f}{s['free']:7.1f}{s['pct_in_frag']:7.1f}{s['amax']:6.1f}"
        row += f"{s['T_per_N']:7.2f}{s['V_per_N']:7.2f}{s['B_per_N']:7.2f}"
        row += "".join(f"{s[f'R{l:g}']:9.2f} ±{s[f'sem{l:g}']:5.2f}" for l in LAMBDAS)
        print(row)


if __name__ == "__main__":
    main()
