"""The full-QMD ζ (qmd_full.py) and the ``zeta_correct`` SACA energy.

Run:  .venv/bin/python tests/test_qmd_full.py

The ``test_*`` functions assert what can be derived by hand.  ``main`` then prints
the nuclear-matter acceptance scan from tests/test_nuclear_matter.py against this
energy — *informational, not asserted*, because it fails: see the "Known
limitation" section of qmd_full's docstring.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_qmd_energy import make_nucleus  # noqa: E402

from clustering.baselines import qmd_full as Q  # noqa: E402
from clustering.baselines.qmd_full import full_cluster_energy  # noqa: E402
from clustering.baselines.saca import (SacaParams, _SacaEvent, bwm_binding,  # noqa: E402
                                       saca_clusters)

M_P, M_N = 0.938272, 0.939565


def approx(a, b, rel=1e-9, abs_=1e-9):
    return abs(a - b) <= max(abs_, rel * abs(b))


def _pair(types, r, p=((0, 0, 0), (0, 0, 0))):
    """(2, 8) array: two nucleons on the x axis, ``r`` fm apart."""
    m = [M_P if t > 0 else M_N for t in types]
    rows = []
    for k in range(2):
        pk = np.asarray(p[k], float)
        rows.append([*pk, np.sqrt(pk @ pk + m[k] ** 2), r * k, 0.0, 0.0, types[k]])
    return np.asarray(rows)


def test_empty_and_singleton_are_zero():
    assert full_cluster_energy(np.zeros((0, 8))).total == 0.0
    assert full_cluster_energy(_pair((1, -1), 2.0)[:1]).total == 0.0


def test_pair_at_rest_matches_the_formulas_by_hand():
    r = 2.0
    g = (Q.ALPHA / np.pi) ** 1.5 * np.exp(-Q.ALPHA * r ** 2)
    pauli_pref = Q.V_0_PAULI * (Q.HBARC / (Q.Q_0 * Q.P_0)) ** 3

    pp = full_cluster_energy(_pair((1, 1), r), yukawa="point")
    assert approx(pp.kinetic, 0.0)
    assert approx(pp.skyrme2, Q.T_1 * g)                      # one pair
    assert approx(pp.skyrme3, 2 * Q.T_GAMMA / (Q.GAMMA + 1) * g ** Q.GAMMA)  # one term per nucleon
    assert approx(pp.yukawa, Q.V_0_YUK * np.exp(-r / Q.GAMMA_Y) / r)
    assert approx(pp.coulomb, Q.E_SQ / r)
    assert approx(pp.pauli, pauli_pref * np.exp(-r ** 2 / (2 * Q.Q_0 ** 2)))

    pn = full_cluster_energy(_pair((1, -1), r))
    assert approx(pn.coulomb, 0.0)     # neutron carries no charge
    assert approx(pn.pauli, 0.0)       # Pauli needs the same isospin


def _brute_folded_yukawa(r, n=500):
    """∫ d³u G(u) V₀ e^{−|r−u|/γ}/|r−u| by 2-D trapezoid in cylindrical (ρ, z)."""
    s2 = 1.0 / (2.0 * Q.ALPHA)
    s = np.sqrt(s2)
    z = np.linspace(-9 * s, 9 * s, n)
    rho = np.linspace(0.0, 9 * s, n)
    Z, R = np.meshgrid(z, rho, indexing="ij")
    G = (2 * np.pi * s2) ** -1.5 * np.exp(-(Z ** 2 + R ** 2) / (2 * s2))
    d = np.sqrt(R ** 2 + (r - Z) ** 2)
    f = G * Q.V_0_YUK * np.exp(-d / Q.GAMMA_Y) / np.maximum(d, 1e-9) * 2 * np.pi * R
    return np.trapezoid(np.trapezoid(f, rho, axis=1), z)


def test_folded_yukawa_matches_direct_convolution():
    for r in (0.3, 1.0, 2.0, 4.0, 8.0):
        got = float(Q.yukawa_pair(np.array([r]), "folded")[0])
        assert approx(got, _brute_folded_yukawa(r), 5e-3, 1e-4), (r, got)


def test_folded_yukawa_preserves_the_volume_integral():
    r = np.linspace(1e-4, 80.0, 200001)
    integral = np.trapezoid(4 * np.pi * r ** 2 * Q.yukawa_pair(r, "folded"), r)
    assert approx(integral, 4 * np.pi * Q.V_0_YUK * Q.GAMMA_Y ** 2, 1e-4)


def test_folded_yukawa_is_finite_at_short_range_and_safe_at_long():
    v = Q.yukawa_pair(np.array([1e-6, 1e-3, 0.3, 30.0, 500.0, 5e4, np.inf]), "folded")
    assert np.isfinite(v).all() and v[-1] == 0.0
    assert abs(v[0]) < 10.0                              # point form: ~ -8.5e7
    assert v[3] < 0 and abs(v[4]) < 1e-12                # decays, no overflow branch flip
    # the two branches of the implementation agree where they meet (a = -20)
    r_meet = (1.0 / Q.GAMMA_Y) * (1 / (2 * Q.ALPHA)) + 20.0 * np.sqrt(2.0) * np.sqrt(1 / (2 * Q.ALPHA))
    lo, hi = Q.yukawa_pair(np.array([r_meet - 1e-3, r_meet + 1e-3]), "folded")
    assert approx(lo, hi, 1e-2, 1e-300)


def test_total_is_the_sum_of_terms():
    t = full_cluster_energy(make_nucleus(16, 8, 0.16))
    assert approx(t.total, t.kinetic + t.skyrme2 + t.skyrme3 + t.yukawa
                  + t.coulomb + t.pauli)


def test_spin_factor_scales_only_pauli():
    x = make_nucleus(16, 8, 0.16)
    a = full_cluster_energy(x, spin_factor=1.0)
    b = full_cluster_energy(x, spin_factor=0.5)
    assert approx(b.pauli, 0.5 * a.pauli)
    assert approx(b.total - b.pauli, a.total - a.pauli)


def test_kinetic_is_sqrt_p2_plus_m2_minus_m():
    # Far apart, so the potential is zero and only T is left.
    p = 0.1   # GeV/c, back to back
    t = full_cluster_energy(_pair((-1, -1), 1000.0, p=((p, 0, 0), (-p, 0, 0))), boost=False)
    assert approx(t.kinetic, 2 * (np.sqrt(p ** 2 + M_N ** 2) - M_N) * 1000.0, 1e-9)
    # +m², not −m²: the latter is not even real for p < m.
    assert t.kinetic > 0


def test_kinetic_term_is_exactly_boost_invariant():
    # A genuine Lorentz boost of a far-apart pair (potential zero, so only T is
    # left).  Adding a constant to p is *not* a boost — see test_qmd_energy.
    p, beta = 0.1, 0.76
    gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
    base = _pair((-1, -1), 1000.0, p=((p, 0, 0), (-p, 0, 0)))
    moved = base.copy()
    moved[:, 0] = gamma * (base[:, 0] + beta * base[:, 3])
    moved[:, 3] = gamma * (base[:, 3] + beta * base[:, 0])
    assert approx(full_cluster_energy(moved).kinetic, full_cluster_energy(base).kinetic, 1e-6)


def test_lorentz_boost_leaves_internal_energy_unchanged():
    # Same 0.5 MeV/A tolerance as test_qmd_energy.  The kinetic term is exact; the
    # residual (~0.15 MeV/A) is in the positions, whose boosted equal-time
    # convention is only approximate, and Yukawa's 1/r makes it the most sensitive.
    rest = full_cluster_energy(make_nucleus(40, 20, 0.16, seed=3)).total / 40
    boosted = full_cluster_energy(make_nucleus(40, 20, 0.16, seed=3, beam_beta=0.76)).total / 40
    assert abs(boosted - rest) < 0.5, (rest, boosted)


def test_stored_energy_column_is_ignored():
    x = make_nucleus(16, 8, 0.16)
    y = x.copy()
    y[:, 3] += 0.02        # the parquet's fE is off-shell by ~20 MeV
    assert approx(full_cluster_energy(y).total, full_cluster_energy(x).total, 1e-12, 1e-9)


def _event(n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = 0.15 * torch.randn(n, 3, generator=g)
    r = 3.0 * torch.randn(n, 3, generator=g)
    typ = torch.where(torch.rand(n, generator=g) < 0.42, 1.0, -1.0)
    E = torch.sqrt(0.938 ** 2 + (p ** 2).sum(1))
    return torch.cat([p, E[:, None], r, typ[:, None]], dim=1)


def test_zeta_correct_is_zeta_plus_lambda_bwd():
    ev = _event()
    idx = np.arange(14)
    A, Z = len(idx), int((ev[idx, 7] == 1).sum())
    for lam in (0.0, 0.5, 1.5):
        se = _SacaEvent(ev, energy_model="zeta_correct", bwm_weight=lam)
        zeta = full_cluster_energy(ev.numpy().astype(np.float64)[idx]).total
        bwd = -bwm_binding(A, Z)
        assert bwd < 0                           # a bound nuclide's Weizsäcker energy
        assert approx(se.objective_energy(idx), zeta + lam * bwd, 1e-9, 1e-9)
    # a free nucleon costs nothing, whatever lambda is
    assert _SacaEvent(ev, energy_model="zeta_correct").objective_energy(np.array([3])) == 0.0


def test_zeta_uses_the_full_energy_only_for_zeta_correct():
    ev = _event()
    idx = np.arange(14)
    old = _SacaEvent(ev, energy_model="qmd").fragment_energy(idx)
    new = _SacaEvent(ev, energy_model="zeta_correct").fragment_energy(idx)
    assert not approx(old, new, 1e-3)


def test_saca_runs_end_to_end_with_zeta_correct():
    ev = _event(30)
    res = saca_clusters(ev, SacaParams(energy_model="zeta_correct", bwm_weight=1.0,
                                       e_cut_model="objective", e_cut=-6.0,
                                       t_max=5.0, trials_per_nucleon=1),
                        rng=np.random.default_rng(0))
    assert res.labels.shape == (30,)
    assert np.isfinite(res.e_final) and res.e_final <= res.e_initial + 1e-6


def test_final_state_terms_add_up_to_the_objective():
    ev = _event(30)
    for lam in (0.0, 1.5):
        res = saca_clusters(ev, SacaParams(energy_model="zeta_correct", bwm_weight=lam,
                                           e_cut_model="objective", e_cut=8.0,
                                           t_max=5.0, trials_per_nucleon=1),
                            rng=np.random.default_rng(0))
        assert np.isfinite([res.t_sum, res.v_sum, res.bwd_sum]).all()
        assert approx(res.t_sum + res.v_sum + lam * res.bwd_sum, res.e_final, 1e-9, 1e-6)
        assert res.t_sum >= 0 and res.bwd_sum <= 0    # kinetic >= 0, -B <= 0; 0 if all free


def main() -> None:
    print("Cold nuclei at rho_0 = 0.16 fm^-3 (5 seeds), full-QMD zeta per nucleon [MeV]")
    print(f"{'':<8}{'T':>7}{'Sk2':>7}{'Sk3':>7}{'Yuk':>8}{'Coul':>7}{'Pauli':>7}{'zeta':>8}{'empirical':>11}")
    for name, A, Z, emp in (("O-16", 16, 8, -7.98), ("Ca-40", 40, 20, -8.55),
                            ("Sn-119", 119, 50, -8.50)):
        ts = [full_cluster_energy(make_nucleus(A, Z, 0.16, seed=s)) for s in range(5)]
        m = {k: np.mean([getattr(t, k) for t in ts]) / A
             for k in ("kinetic", "skyrme2", "skyrme3", "yukawa", "coulomb", "pauli", "total")}
        print(f"{name:<8}{m['kinetic']:7.2f}{m['skyrme2']:7.2f}{m['skyrme3']:7.2f}"
              f"{m['yukawa']:8.2f}{m['coulomb']:7.2f}{m['pauli']:7.2f}{m['total']:8.2f}{emp:11.2f}")

    print("\nDensity scan, A=40 Z=20 (a saturating energy has a minimum near rho_0):")
    scan = {}
    for rho in (0.04, 0.08, 0.12, 0.16, 0.24, 0.32, 0.48, 0.80):
        scan[rho] = np.mean([full_cluster_energy(make_nucleus(40, 20, rho, seed=s)).total
                             for s in range(5)]) / 40
        print(f"  rho = {rho:4.2f}   zeta = {scan[rho]:8.2f} MeV/A")
    saturates = scan[0.80] > scan[0.16] and scan[0.32] > scan[0.16]
    print(f"\n  saturation: {'PASS' if saturates else 'FAIL (no minimum, compression always pays)'}"
          "  [informational — see qmd_full docstring]")


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL  {name}  {e}")
    print()
    main()
    raise SystemExit(1 if fails else 0)
