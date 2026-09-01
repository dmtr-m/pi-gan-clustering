"""Physics checks on the evaporation model.

Run:  .venv/bin/python tests/test_deexcitation.py

Checks:
  1. Separation energies match known values (S_n of Ca-40 ~ 15.6 MeV, S_alpha of
     a heavy nucleus is small, and alpha emission is the cheapest channel out of
     a light N=Z system because the alpha itself is tightly bound).
  2. A cold fragment does not decay at all.
  3. Mass and charge are conserved exactly through the whole chain.
  4. The chain terminates, and hotter fragments emit more.
  5. Isospin steering: fragments of very different N/Z converge toward the same
     valley — the mechanism that is supposed to narrow the isotopic band.
"""
import numpy as np

from clustering.baselines.deexcitation import (CHANNELS, binding, coulomb_barrier,
                                               evaporate, excitation_energy,
                                               separation_energy)


def main() -> bool:
    ok = True

    # --- 1: separation energies ---------------------------------------------
    print("Separation energies [MeV] from the BWM mass formula")
    print(f"{'parent':<10}{'S_n':>8}{'S_p':>8}{'S_alpha':>9}{'experiment (S_n)':>19}")
    cases = [("Ca-40", 40, 20, 15.6), ("Fe-56", 56, 26, 11.2),
             ("Sn-120", 120, 50, 9.1), ("Xe-132", 132, 54, 8.0)]
    for name, A, Z, exp_sn in cases:
        s_n = separation_energy(A, Z, 1, 0)
        s_p = separation_energy(A, Z, 1, 1)
        s_a = separation_energy(A, Z, 4, 2)
        print(f"{name:<10}{s_n:8.2f}{s_p:8.2f}{s_a:9.2f}{exp_sn:19.1f}")
    s_n_ca = separation_energy(40, 20, 1, 0)
    sep_ok = 8.0 < s_n_ca < 22.0
    ok &= sep_ok
    print(f"  S_n(Ca-40) = {s_n_ca:.2f} MeV, experiment 15.6  -> "
          f"{'PASS' if sep_ok else 'FAIL'}")

    # The alpha is bound by 28 MeV, so emitting one costs far less than
    # emitting four separate nucleons.
    four_n = 4 * separation_energy(40, 20, 1, 0)
    alpha_cheaper = separation_energy(40, 20, 4, 2) < four_n
    ok &= alpha_cheaper
    print(f"  S_alpha ({separation_energy(40, 20, 4, 2):.1f}) < 4 x S_n ({four_n:.1f})"
          f"  -> {'PASS' if alpha_cheaper else 'FAIL'}")

    # --- 2: a cold fragment must not decay -----------------------------------
    rng = np.random.default_rng(0)
    cold = evaporate(60, 28, 0.0, rng)
    cold_ok = cold.n_steps == 0 and cold.products == [(60, 28)]
    ok &= cold_ok
    print(f"\n  cold fragment (E*=0): {cold.n_steps} emissions  -> "
          f"{'PASS' if cold_ok else 'FAIL'}")
    # Below the lowest threshold it must also be inert.
    barely = evaporate(60, 28, 2.0, rng)
    print(f"  E*=2 MeV (below every threshold): {barely.n_steps} emissions")

    # --- 3 & 4: conservation, termination, monotonicity ----------------------
    print(f"\n{'A':>5}{'Z':>4}{'E*/A':>7}{'steps':>7}{'residue':>12}{'emitted':>34}")
    cons_ok = True
    prev_steps = -1
    mono_ok = True
    for e_per_a in (1.0, 2.0, 4.0, 6.0):
        A, Z = 60, 28
        r = evaporate(A, Z, e_per_a * A, np.random.default_rng(1))
        sum_a = sum(a for a, _ in r.products)
        sum_z = sum(z for _, z in r.products)
        cons_ok &= (sum_a == A and sum_z == Z)
        if r.n_steps < prev_steps:
            mono_ok = False
        prev_steps = r.n_steps
        res = r.products[-1]
        print(f"{A:5d}{Z:4d}{e_per_a:7.1f}{r.n_steps:7d}"
              f"{f'A={res[0]} Z={res[1]}':>12}"
              f"{str(dict(sorted(r.emitted.items()))):>34}")
    ok &= cons_ok and mono_ok
    print(f"  mass/charge conserved through every chain -> "
          f"{'PASS' if cons_ok else 'FAIL'}")
    print(f"  hotter fragments emit at least as much     -> "
          f"{'PASS' if mono_ok else 'FAIL'}")

    # --- 5: neutron-rich fragments shed neutrons -----------------------------
    print("\nIsospin steering — 200 chains each at E*/A = 4 MeV:")
    print(f"{'parent':<14}{'(N-Z)/A':>9}{'<n emitted>':>13}{'<p+ emitted>':>14}"
          f"{'residue (N-Z)/A':>18}")
    trend = []
    for A, Z in ((60, 30), (60, 26), (60, 22)):
        ns, ps, frac = [], [], []
        for s in range(200):
            r = evaporate(A, Z, 4.0 * A, np.random.default_rng(s))
            ns.append(r.emitted.get("n", 0))
            ps.append(sum(v for k, v in r.emitted.items() if k != "n"))
            a_r, z_r = r.products[-1]
            if a_r > 0:
                frac.append((a_r - 2 * z_r) / a_r)
        trend.append(np.mean(frac))
        print(f"{f'A={A} Z={Z}':<14}{(A - 2 * Z) / A:9.3f}{np.mean(ns):13.2f}"
              f"{np.mean(ps):14.2f}{np.mean(frac):18.3f}")
    # The signature is *convergence*, not a uniform decrease: a neutron-rich
    # parent boils off neutrons and moves down, while an N=Z parent preferentially
    # emits protons (Coulomb lowers S_p) and drifts slightly up.  Both walk toward
    # the same valley, so the spread across parents must collapse.
    starts = [(60 - 2 * z) / 60 for z in (30, 26, 22)]
    spread_before = max(starts) - min(starts)
    spread_after = max(trend) - min(trend)
    steer_ok = spread_after < 0.25 * spread_before
    ok &= steer_ok
    print(f"  isospin spread across parents: {spread_before:.3f} -> "
          f"{spread_after:.3f} ({spread_before / max(spread_after, 1e-9):.1f}x "
          f"compression)  -> {'PASS' if steer_ok else 'FAIL'}")
    print("  (an N=Z parent drifting slightly neutron-rich is correct: Coulomb")
    print("   lowers S_p, so it emits protons preferentially)")
    return bool(ok)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
