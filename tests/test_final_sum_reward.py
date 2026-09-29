"""Check the terminal ("final sum") reward and the trainer mode that uses it.

Run from the repo root:  PYTHONPATH=src:tests .venv/bin/python tests/test_final_sum_reward.py

`compute_final_sum_reward` is R = -sum_leaves U(leaf) / N_parent.  Assertions:

1. it equals the hand-computed -sum U / N on a real partition;
2. it differs from the node-difference reward only by the per-event constant
   U(parent)/N:  R = q - U(parent)/N.  (So partitions rank identically within an
   event; the modes differ in credit assignment, not in what is preferred.)
3. an unsplit event scores -U(parent)/N, not 0 and not "invalid";
4. KSplitTrainer(reward_mode="final_sum") runs a real optimizer step on real
   events, with and without a critic, and the parameters actually move;
5. KSplitTrainer(reward_mode="node_diff") still runs, and a bad mode is rejected.
"""
from functools import partial

import torch

from clustering.physics import zeta_correct_energy
from clustering.split_prediction.model import SplitPredictionModel, SplitValueCritic
from clustering.split_prediction.trainer import (
    KSplitTrainer, compute_final_sum_reward, compute_k_level_reward,
)
from test_qmd_minus_b import _batch, _sub

TOL = 1e-3


def _trainer(mode, critic=True, k=2):
    torch.manual_seed(0)
    model = SplitPredictionModel(n_clusters=3)
    crit = SplitValueCritic(input_dim=8, hidden_dim=16) if critic else None
    params = list(model.parameters()) + (list(crit.parameters()) if crit else [])
    return KSplitTrainer(model, None, optimizer=torch.optim.AdamW(params, lr=1e-3),
                         scheduler=None, k=k, critic=crit, reward_type="zeta_correct",
                         bwm_weight=0.5, reward_mode=mode)


def main() -> bool:
    ok = True
    x, mask = _batch(6)
    fn = partial(zeta_correct_energy, bwm_weight=0.5)

    parent = _sub(mask, 20)
    l0, l1 = torch.zeros_like(parent), torch.zeros_like(parent)
    for i in range(x.shape[0]):
        ix = torch.nonzero(parent[i]).flatten()
        l0[i, ix[:12]] = True
        l1[i, ix[12:20]] = True
    leaves = {(0,): l0, (1,): l1}

    n = parent.sum(dim=1).float()          # some events have fewer than 20 nucleons
    R = compute_final_sum_reward(x, parent, leaves, energy_fn=fn)
    want = -(fn(x, l0, 7) + fn(x, l1, 7)) / n
    hit = float((R - want).abs().max()) < TOL
    ok &= hit
    print(f"1. R == -sum U(leaf)/N : max |diff| = {float((R - want).abs().max()):.2e} "
          f"-> {'PASS' if hit else 'FAIL'}")

    q = compute_k_level_reward(x, parent, leaves, energy_fn=fn)
    d = float((R - (q - fn(x, parent, 7) / n)).abs().max())
    hit = d < TOL
    ok &= hit
    print(f"2. R == q - U(parent)/N : max |diff| = {d:.2e} -> {'PASS' if hit else 'FAIL'}")

    R0 = compute_final_sum_reward(x, parent, {(): parent}, energy_fn=fn)
    d = float((R0 + fn(x, parent, 7) / n).abs().max())
    hit = d < TOL and float(R0.abs().max()) > 1.0
    ok &= hit
    print(f"3. unsplit event: R == -U(parent)/N (nonzero) : max |diff| = {d:.2e}, "
          f"R = {[round(float(v), 2) for v in R0]} -> {'PASS' if hit else 'FAIL'}")

    for critic in (True, False):
        t = _trainer("final_sum", critic=critic)
        before = [p.detach().clone() for p in t.model.parameters()]
        loss, r, vl, gn, st = t._step(x, mask)
        moved = any(not torch.equal(a, b) for a, b in zip(before, t.model.parameters()))
        hit = (r is not None and torch.isfinite(r).all().item() and moved
               and st["n_valid_items"] == x.shape[0] and gn > 0)
        ok &= hit
        print(f"4. final_sum step (critic={critic}): loss={loss:.3f} R_mean={float(r.mean()):.2f} "
              f"grad_norm={gn:.3f} params moved={moved} -> {'PASS' if hit else 'FAIL'}")

    t = _trainer("node_diff")
    loss, r, vl, gn, st = t._step(x, mask)
    hit = r is not None and st["ent_n"] == st["wn_sum"]
    try:
        _trainer("bogus")
        hit = False
    except ValueError:
        pass
    ok &= hit
    print(f"5. node_diff still runs; bad mode rejected -> {'PASS' if hit else 'FAIL'}")
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
