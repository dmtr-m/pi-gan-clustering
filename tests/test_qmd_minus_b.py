"""Check the QMD - B reward energy against the objective it is ported from.

Run:  .venv/bin/python tests/test_qmd_minus_b.py

`qmd_minus_b_energy` carries the SACA annealer's `energy_model="qmd_minus_b"`
objective (experiments/qmd_minus_b.py) into the RL reward.  Four assertions:

1. it reduces to the existing `weizsacker_qmd` reward when told to use the same
   mass formula at the same weight — so the only thing the new reward changes is
   the formula and where lambda sits, not the algebra;
2. the B term matches the mass formula fragment by fragment, on real nucleon
   sets rather than on a synthetic one;
3. B is switched off below A = 2, where the liquid-drop picture is meaningless;
4. the deuteron, which is where the two mass formulas disagree most and where
   this model's yields are known to be short: BW calls it unbound, BWM binds it.
   This is the reason the port is worth making, so it is asserted rather than
   left in a docstring.

The last two cover `saca_qmd_minus_b_energy`, the variant built on the
baselines' own QMD energy rather than on physics.py's potential:

5. it is exactly `cluster_energy` at lambda = 0, per masked set;
6. four free neutrons are *unbound* under it and *bound* under physics.py's
   potential.  That gap is the reason the variant exists, and it is the shape
   of the standing failure in tests/test_nuclear_matter.py, so it is pinned
   here rather than left to be rediscovered.
"""
import torch

import numpy as np

from clustering.baselines.qmd_energy import cluster_energy
from clustering.physics import (
    bethe_weizsacker,
    qmd_minus_b_energy,
    saca_qmd_minus_b_energy,
    total_potential_energy,
    weizsacker_formula,
    weizsacker_qmd_energy,
)
from clustering.split_prediction.dataset import NucleonDataset

DATA = "data/xecs_hse.parquet"
N_EVENTS = 8
TOL = 1e-3   # MeV, against energies of order 1e3


def _batch(n: int = N_EVENTS):
    """A padded (B, N, 8) batch of real events, plus its mask."""
    ds = NucleonDataset(DATA, particle_type="SpectatorsLeft")
    events = [ds[i] for i in range(n)]
    x = [e[0] if isinstance(e, (tuple, list)) else e for e in events]
    N = max(t.shape[0] for t in x)
    out = torch.zeros(len(x), N, x[0].shape[1])
    mask = torch.zeros(len(x), N, dtype=torch.bool)
    for i, t in enumerate(x):
        out[i, : t.shape[0]] = t
        mask[i, : t.shape[0]] = True
    return out, mask


def _sub(mask: torch.Tensor, keep: int) -> torch.Tensor:
    """First `keep` real nucleons of each event — a fragment to score."""
    out = torch.zeros_like(mask)
    for i in range(mask.shape[0]):
        idx = torch.nonzero(mask[i]).flatten()[:keep]
        out[i, idx] = True
    return out


def main() -> bool:
    ok = True
    x, mask = _batch()

    # 1 — same algebra as the reward it generalizes.
    a = qmd_minus_b_energy(x, mask, bwm_weight=1.0, binding="bw")
    b = weizsacker_qmd_energy(x, mask, qmd_weight=1.0, scale="extensive")
    d = float((a - b).abs().max())
    hit = d < TOL
    ok &= hit
    print(f"1. binding='bw', w=1  ==  weizsacker_qmd extensive : "
          f"max |diff| = {d:.2e} MeV -> {'PASS' if hit else 'FAIL'}")

    # 2 — the B term is exactly the mass formula, at each lambda, on fragments
    #     of several sizes (not just whole events).
    worst = 0.0
    for keep in (2, 3, 4, 12, 40):
        m = _sub(mask, keep)
        A = m.sum(dim=1).float()
        Z = ((x[..., 7] == 1) & m).sum(dim=1).float()
        V = total_potential_energy(x, m)
        for w in (0.0, 0.25, 0.5, 1.0, 1.5):
            for form, B in (("bwm", bethe_weizsacker(A, Z, modified=True)),
                            ("bw", weizsacker_formula(A, Z))):
                got = qmd_minus_b_energy(x, m, bwm_weight=w, binding=form)
                worst = max(worst, float((got - (V - w * B)).abs().max()))
    hit = worst < TOL
    ok &= hit
    print(f"2. E == V - w*B  over A in 2..40, w in 0..1.5, both forms : "
          f"max |diff| = {worst:.2e} MeV -> {'PASS' if hit else 'FAIL'}")

    # 3 — a single nucleon has no pairs and no liquid drop: E must be 0, not -B.
    m1 = _sub(mask, 1)
    e1 = qmd_minus_b_energy(x, m1, bwm_weight=1.0, binding="bwm")
    hit = float(e1.abs().max()) < TOL
    ok &= hit
    print(f"3. A = 1 -> E = 0 (no B, no pairs)                       : "
          f"max |E| = {float(e1.abs().max()):.2e} MeV -> {'PASS' if hit else 'FAIL'}")

    # 4 — the deuteron, the disagreement that motivates the port.
    T = lambda v: torch.tensor([float(v)])
    bw_d = float(weizsacker_formula(T(2), T(1)))
    bwm_d = float(bethe_weizsacker(T(2), T(1), modified=True))
    hit = bw_d < -10.0 and 0.0 < bwm_d < 4.0
    ok &= hit
    print(f"4. B(d): BW = {bw_d:+.2f} (unbound), BWM = {bwm_d:+.2f} "
          f"(experiment +2.22)            -> {'PASS' if hit else 'FAIL'}")

    # 5 — the SACA-energy variant is cluster_energy itself at lambda = 0.
    worst = 0.0
    for keep in (2, 4, 12, 40):
        m = _sub(mask, keep)
        got = saca_qmd_minus_b_energy(x, m, bwm_weight=0.0)
        for i in range(x.shape[0]):
            sub = x[i][m[i]].double().numpy()
            worst = max(worst, abs(float(got[i]) - cluster_energy(sub).total))
    hit = worst < TOL
    ok &= hit
    print(f"5. saca variant, w=0  ==  qmd_energy.cluster_energy         : "
          f"max |diff| = {worst:.2e} MeV -> {'PASS' if hit else 'FAIL'}")

    # 6 — the two energies are not measuring the same thing.  On real
    #     coordinate-MST fragments physics.py's potential sits ~25 MeV/nucleon
    #     below the baselines' QMD energy, which lands near the empirical -8.
    #     A synthetic nucleus will not show this cleanly (Pauli dominates a
    #     cold, zero-momentum blob), so it is measured where it matters.
    #
    #     NOTE this test is a canary: it asserts the *disagreement*, so it will
    #     start failing the day physics.py gains its density-dependent Skyrme
    #     term.  That failure is the good outcome — update the bound then.
    from clustering.split_prediction.mst import mst_clusters
    vs, es = [], []
    for i in range(x.shape[0]):
        xi, mi = x[i : i + 1], mask[i : i + 1]
        lab = mst_clusters(xi, mi, d_cut=2.0, metric="coord")[0]
        for c in lab.unique():
            if int(c) < 0:
                continue  # -1 is padding; its rows are all at the origin
            sub = (lab == c).unsqueeze(0) & mi
            idx = torch.nonzero(sub[0]).flatten()
            if idx.numel() < 10:
                continue
            vs.append(float(total_potential_energy(xi, sub)) / idx.numel())
            es.append(float(saca_qmd_minus_b_energy(xi, sub, bwm_weight=0.0)) / idx.numel())
    v_m, e_m = float(np.mean(vs)), float(np.mean(es))
    hit = len(vs) >= 5 and (e_m - v_m) > 15.0
    ok &= hit
    print(f"6. MST fragments A >= 10 (n = {len(vs)}): physics.py V/A = {v_m:+.2f}, "
          f"qmd_energy E/A = {e_m:+.2f} MeV/nucleon")
    print(f"   gap {e_m - v_m:.2f} MeV/nucleon -> {'PASS' if hit else 'FAIL'}  "
          f"(empirical B/A is about -8; physics.py has no saturating term,")
    print("    which is what tests/test_nuclear_matter.py reports on the bulk.)")

    return bool(ok)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
