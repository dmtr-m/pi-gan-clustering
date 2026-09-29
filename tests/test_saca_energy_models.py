"""The energy models added to SACA for experiments/baselines_comparison.py.

Run:  .venv/bin/python tests/test_saca_energy_models.py
"""
import numpy as np
import torch

from clustering.baselines.saca import SacaParams, _SacaEvent, bwm_binding, saca_clusters
from clustering.physics import total_potential_energy


def approx(a, b, rel=1e-6, abs_=1e-6):
    return abs(a - b) <= max(abs_, rel * abs(b))


def _event(n: int = 40, seed: int = 0) -> torch.Tensor:
    """Columns as in the parquet: px py pz E x y z type (GeV/c, GeV, fm)."""
    g = torch.Generator().manual_seed(seed)
    p = 0.15 * torch.randn(n, 3, generator=g)
    r = 3.0 * torch.randn(n, 3, generator=g)
    typ = (torch.rand(n, generator=g) < 0.42).float()
    E = torch.sqrt(0.938 ** 2 + (p ** 2).sum(1))
    return torch.cat([p, E[:, None], r, typ[:, None]], dim=1)


def test_submatrix_v_equals_total_potential_energy():
    ev = _event()
    se = _SacaEvent(ev, energy_model="physics_v")
    rng = np.random.default_rng(1)
    for _ in range(20):
        idx = rng.choice(len(ev), size=int(rng.integers(2, 25)), replace=False)
        ref = total_potential_energy(ev[idx].unsqueeze(0),
                                     torch.ones(1, len(idx), dtype=torch.bool)).item()
        assert approx(se.physics_v(idx), ref, 1e-4, 1e-3)


def test_mix_endpoints_and_scales():
    ev = _event()
    idx = np.arange(12)
    Z = int((ev[idx, 7] == 1).sum())
    plain = _SacaEvent(ev)
    e = plain.fragment_energy(idx)
    b = bwm_binding(12, Z)
    mix1 = _SacaEvent(ev, energy_model="mix", mix_alpha=1.0)
    mix0 = _SacaEvent(ev, energy_model="mix", mix_alpha=0.0)
    mixh = _SacaEvent(ev, energy_model="mix", mix_alpha=0.5)
    inten = _SacaEvent(ev, energy_model="mix", mix_alpha=0.5, mix_per_nucleon=True)
    assert approx(mix1.objective_energy(idx), e)
    assert approx(mix0.objective_energy(idx), -b)
    assert approx(mixh.objective_energy(idx), 0.5 * e - 0.5 * b)
    # both variants test the cut on the same per-nucleon number
    assert approx(inten.zeta_objective(idx), mixh.zeta_objective(idx))
    assert approx(inten.objective_energy(idx), mixh.objective_energy(idx) / 12)


def test_singletons_are_free():
    ev = _event()
    for kw in ({"energy_model": "physics_v"}, {"energy_model": "physics_v_minus_b"},
               {"energy_model": "mix", "mix_alpha": 0.3},
               {"energy_model": "mix", "mix_alpha": 0.3, "mix_per_nucleon": True}):
        assert _SacaEvent(ev, **kw).objective_energy(np.array([3])) == 0.0


def test_mix_alpha1_objective_cut_reproduces_plain_saca():
    ev = _event(30, seed=3)
    plain = SacaParams(e_cut=-4.0, e_cut_light=0.0)
    mix = SacaParams(energy_model="mix", mix_alpha=1.0, e_cut_model="objective",
                     e_cut=-4.0, e_cut_light=0.0)
    a = saca_clusters(ev, plain, d_cut=2.0, metric="coord", rng=np.random.default_rng(0))
    b = saca_clusters(ev, mix, d_cut=2.0, metric="coord", rng=np.random.default_rng(0))
    assert (a.labels == b.labels).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
