"""Check the `zeta_correct` RL reward against the SACA energy it is ported from.

Run from the repo root:  PYTHONPATH=src:tests .venv/bin/python tests/test_zeta_correct_reward.py

`zeta_correct_energy` (physics.py) carries `SacaParams(energy_model="zeta_correct")`
into the RL reward.  Assertions:

1. at lambda = 0 it is `full_cluster_energy(...).total`, per masked set;
2. at lambda > 0 it is that plus `lambda * bwd`, with `bwd = -bwm_binding(A, Z)`
   taken from saca.py's own (independent, cached) BWM — so the torch B used here
   and the numpy B the annealer uses are the same number;
3. A < 2 scores 0 (no B, no pairs), matching the annealer;
4. the trainer accepts `reward_type="zeta_correct"`, and its split reward on a
   real batch is finite and equals (E_parent - sum E_leaf) / N computed by hand.
"""
import numpy as np
import torch

from clustering.baselines.qmd_full import full_cluster_energy
from clustering.baselines.saca import bwm_binding
from clustering.physics import zeta_correct_energy
from clustering.split_prediction.trainer import compute_k_level_reward
from test_qmd_minus_b import _batch, _sub

TOL = 1e-3


def main() -> bool:
    ok = True
    x, mask = _batch(6)
    xn = x.double().numpy()

    worst0 = worst1 = 0.0
    for keep in (2, 3, 4, 12, 40):
        m = _sub(mask, keep)
        for kw in ({"spin_factor": 0.5, "yukawa": "folded"},
                   {"spin_factor": 1.0, "yukawa": "off"}):
            e0 = zeta_correct_energy(x, m, bwm_weight=0.0, **kw)
            for lam in (0.0, 0.5, 1.5):
                got = zeta_correct_energy(x, m, bwm_weight=lam, **kw)
                for i in range(x.shape[0]):
                    sub = xn[i][m[i].numpy()]
                    A = sub.shape[0]
                    Z = int((sub[:, 7] == 1).sum())
                    want = full_cluster_energy(sub, **kw).total
                    if lam == 0.0:
                        worst0 = max(worst0, abs(float(got[i]) - want))
                    else:
                        want += lam * (-bwm_binding(A, Z))
                        worst1 = max(worst1, abs(float(got[i]) - want))
    hit = worst0 < TOL
    ok &= hit
    print(f"1. lambda=0 == full_cluster_energy.total : max |diff| = {worst0:.2e} MeV "
          f"-> {'PASS' if hit else 'FAIL'}")
    hit = worst1 < TOL
    ok &= hit
    print(f"2. E == total + lambda*(-B_BWM) (saca.bwm_binding) : max |diff| = {worst1:.2e} MeV "
          f"-> {'PASS' if hit else 'FAIL'}")

    e1 = zeta_correct_energy(x, _sub(mask, 1), bwm_weight=1.0)
    hit = float(e1.abs().max()) < TOL
    ok &= hit
    print(f"3. A = 1 -> E = 0 : max |E| = {float(e1.abs().max()):.2e} -> {'PASS' if hit else 'FAIL'}")

    # 4 — split reward on a real batch: parent = 20 nucleons, two 10-nucleon leaves.
    parent = _sub(mask, 20)
    idx = [torch.nonzero(parent[i]).flatten() for i in range(x.shape[0])]
    l0, l1 = torch.zeros_like(parent), torch.zeros_like(parent)
    for i, ix in enumerate(idx):
        l0[i, ix[:10]] = True
        l1[i, ix[10:20]] = True
    from functools import partial
    fn = partial(zeta_correct_energy, bwm_weight=0.5)
    q = compute_k_level_reward(x, parent, {(0,): l0, (1,): l1}, energy_fn=fn)
    want = (fn(x, parent, 7) - fn(x, l0, 7) - fn(x, l1, 7)) / 20.0
    hit = bool(torch.isfinite(q).all()) and float((q - want).abs().max()) < TOL
    ok &= hit
    print(f"4. compute_k_level_reward finite and == (E_p - sum E_leaf)/N : "
          f"q = {[round(float(v), 2) for v in q]} -> {'PASS' if hit else 'FAIL'}")

    # 5 — the trainer builds with the new reward_type, and rejects a bad Yukawa mode.
    from clustering.split_prediction.model import SplitPredictionModel
    from clustering.split_prediction.trainer import KSplitTrainer
    model = SplitPredictionModel(n_clusters=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    t = KSplitTrainer(model, None, optimizer=opt, scheduler=None, k=2, reward_type="zeta_correct",
                      bwm_weight=0.5, zeta_spin_factor=0.5, zeta_yukawa="folded")
    a = t.energy_fn(x, _sub(mask, 12))
    b = zeta_correct_energy(x, _sub(mask, 12), bwm_weight=0.5)
    hit = float((a - b).abs().max()) < TOL
    try:
        KSplitTrainer(model, None, optimizer=opt, scheduler=None, k=2, reward_type="zeta_correct",
                      zeta_yukawa="bogus")
        hit = False
    except ValueError:
        pass
    ok &= hit
    print(f"5. KSplitTrainer(reward_type='zeta_correct').energy_fn == zeta_correct_energy, "
          f"bad yukawa rejected -> {'PASS' if hit else 'FAIL'}")

    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
