"""Acceptance test for the paper-faithful QMD cluster energy.

Run:  .venv/bin/python tests/test_qmd_energy.py

Checks, in order:

  1. **The interaction density is normalized.**  rho_int in the bulk must equal
     the density it was built from, or the Skyrme coefficients do not transfer.
  2. **Nuclear matter saturates**, with the minimum at rho_0 and the right depth.
     The analytic value for the soft EoS is
         T/A + alpha/2 + beta/(gamma+1) = 21.8 - 178.0 + 139.8 = -17.0 MeV/A.
     This is the property the repo's own potential lacks — see
     tests/test_nuclear_matter.py, which fails on purpose.
  3. **A boost changes nothing.**  The same nucleus Lorentz-boosted to the beam
     velocity — momenta, energies and the length contraction together — must
     have the same internal energy.

Bulk matter is probed with a large sphere and only its core counted: the
wave-packet width is sqrt(L) ~ 2.9 fm at L = 8.66 fm^2, comparable to the radius
of a small nucleus, so an A=80 droplet is all surface and says nothing about
saturation.
"""
import numpy as np

from clustering.baselines.qmd_energy import (L_GAUSS, RHO_0, SKYRME_EOS,
                                             cluster_energy, fermi_momentum,
                                             interaction_density, zeta)

M_P, M_N = 0.938272, 0.939565
GEV_TO_MEV = 1000.0
P_FERMI = fermi_momentum(RHO_0)   # 267 MeV/c


def in_ball(n, radius, rng):
    u = rng.random(n) ** (1.0 / 3.0)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d * (u * radius)[:, None]


def make_nucleus(A, Z, rho, p_fermi=P_FERMI, seed=0, beam_beta=0.0):
    """(A, 8): px py pz E x y z type.  GeV/c and fm, as the dataset stores.

    ``beam_beta`` applies a genuine Lorentz boost along z — momenta transform as
    p_z' = gamma (p_z + beta E), energies as E' = gamma (E + beta p_z), and the
    length contracts as r_z' = r_z / gamma.  Adding a constant to p_z instead
    would NOT be a boost: it shifts every nucleon by the same amount regardless
    of its energy, which changes the internal momentum distribution and so
    changes the internal energy for real.
    """
    rng = np.random.default_rng(seed)
    R = (3.0 * A / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
    r = in_ball(A, R, rng)
    p = in_ball(A, p_fermi, rng)
    types = np.concatenate([np.ones(Z), -np.ones(A - Z)])
    m = np.where(types > 0, M_P, M_N)
    E = np.sqrt((p ** 2).sum(1) + m ** 2)

    if beam_beta:
        gamma = 1.0 / np.sqrt(1.0 - beam_beta ** 2)
        pz, e = p[:, 2].copy(), E.copy()
        p[:, 2] = gamma * (pz + beam_beta * e)
        E = gamma * (e + beam_beta * pz)
        r[:, 2] = r[:, 2] / gamma
    return np.column_stack([p, E, r, types])


def bulk_energy_per_nucleon(rho, A=3000, eos="soft", L=L_GAUSS, seed=0):
    """E/A of uniform symmetric matter at density ``rho``, core nucleons only."""
    rng = np.random.default_rng(seed)
    R = (3.0 * A / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
    r = in_ball(A, R, rng)
    # p_F must scale with the density, or the kinetic term stays flat and the
    # saturation minimum comes out at 1.5 rho_0 instead of 1.0.
    p = in_ball(A, fermi_momentum(rho), rng)
    core = np.linalg.norm(r, axis=1) < 0.4 * R

    alpha, beta, gam = SKYRME_EOS[eos]
    u = interaction_density(r, L) / RHO_0
    e_sky = 0.5 * alpha * u + beta / (gam + 1.0) * u ** gam
    m = 0.5 * (M_P + M_N)
    t = (np.sqrt((p ** 2).sum(1) + m ** 2) - m) * GEV_TO_MEV
    return float((e_sky[core] + t[core]).mean()), float(t[core].mean())


def main() -> bool:
    ok = True
    alpha, beta, gam = SKYRME_EOS["soft"]
    m = 0.5 * (M_P + M_N)
    t_per_a = 0.6 * (P_FERMI ** 2 / (2 * m)) * GEV_TO_MEV   # Fermi gas, 3/5 p_F^2/2m
    analytic = t_per_a + 0.5 * alpha + beta / (gam + 1.0)
    print(f"soft EoS: alpha={alpha} MeV, beta={beta} MeV, gamma={gam:.4f}, "
          f"rho_0={RHO_0} fm^-3, L={L_GAUSS} fm^2")
    print(f"p_F(rho_0) = {P_FERMI * GEV_TO_MEV:.1f} MeV/c, T/A = {t_per_a:.1f} MeV")
    print(f"analytic saturation: T/A + alpha/2 + beta/(gamma+1) = {analytic:.2f} MeV/A\n")

    # --- 1: density normalization -------------------------------------------
    rng = np.random.default_rng(0)
    R = (3.0 * 3000 / (4.0 * np.pi * RHO_0)) ** (1.0 / 3.0)
    r = in_ball(3000, R, rng)
    core = np.linalg.norm(r, axis=1) < 0.4 * R
    rho_core = float(interaction_density(r, L_GAUSS)[core].mean())
    dens_ok = abs(rho_core / RHO_0 - 1.0) < 0.10
    ok &= dens_ok
    print(f"  density    : rho_int(core) = {rho_core:.4f} vs rho_0 = {RHO_0} "
          f"({rho_core / RHO_0:.3f} x)  -> {'PASS' if dens_ok else 'FAIL'}\n")

    # --- 2: saturation ------------------------------------------------------
    print("Symmetric nuclear matter (bulk, core of a 3000-nucleon sphere):")
    print(f"{'rho/rho_0':>10}{'E/A':>10}{'T/A':>8}")
    scan = {}
    for ratio in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
        e, t = bulk_energy_per_nucleon(ratio * RHO_0)
        scan[ratio] = e
        print(f"{ratio:10.2f}{e:10.2f}{t:8.2f}")

    at_min = min(scan, key=scan.get)
    saturates = 0.9 <= at_min <= 1.25 and scan[3.0] > scan[1.0]
    depth_ok = abs(scan[1.0] - analytic) < 4.0
    ok &= saturates and depth_ok
    print(f"\n  saturation : minimum at rho/rho_0 = {at_min:.2f}, "
          f"E/A(3 rho_0) = {scan[3.0]:.1f} > E/A(rho_0) = {scan[1.0]:.1f}  -> "
          f"{'PASS' if saturates else 'FAIL'}")
    print(f"  depth      : E/A(rho_0) = {scan[1.0]:.2f} vs analytic {analytic:.2f} "
          f"-> {'PASS' if depth_ok else 'FAIL'}")

    # --- 3: finite nuclei ---------------------------------------------------
    print("\nCold nuclei at rho_0, Coulomb on (finite-size, so shallower than bulk):")
    print(f"{'nucleus':<10}{'zeta [MeV/A]':>14}{'empirical B/A':>16}")
    for name, A, Z, emp in (("O-16", 16, 8, -7.98), ("Ca-40", 40, 20, -8.55),
                            ("Sn-120", 120, 50, -8.51)):
        z = float(np.mean([zeta(make_nucleus(A, Z, RHO_0, seed=s)) for s in range(5)]))
        print(f"{name:<10}{z:14.2f}{emp:16.2f}")

    # --- 4: boost invariance ------------------------------------------------
    # beta = 0.76 is this dataset's spectator velocity (~1.1 GeV/c per nucleon).
    at_rest = zeta(make_nucleus(40, 20, RHO_0, seed=0))
    boosted = zeta(make_nucleus(40, 20, RHO_0, seed=0, beam_beta=0.76))
    frame_ok = abs(at_rest - boosted) < 0.5
    ok &= frame_ok
    print(f"\n  frame      : at rest {at_rest:.2f} vs boosted to beta = 0.76 "
          f"{boosted:.2f} MeV/A  -> {'PASS' if frame_ok else 'FAIL'}")
    unboosted = zeta(make_nucleus(40, 20, RHO_0, seed=0, beam_beta=0.76), boost=False)
    print(f"  (same fragment scored without the boost correction: "
          f"{unboosted:.2f} MeV/A — the error the correction removes)")
    return bool(ok)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
