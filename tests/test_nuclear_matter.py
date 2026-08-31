"""Acceptance test for the QMD potential: does it bind real nuclei correctly?

Run:  .venv/bin/python tests/test_nuclear_matter.py

Any change to the potential in ``physics.py`` has to be judged here first. The
SACA baseline minimizes this energy directly, so an error in it is not a small
bias — the annealer finds and exploits whatever the potential rewards.

Two properties are checked, against a cold synthetic nucleus: nucleons uniform
in a sphere of radius R = (3A / 4*pi*rho)^(1/3), momenta uniform in a Fermi
sphere of radius p_F = 260 MeV/c.

  1. **Depth.** At rho_0 = 0.16 fm^-3 the energy per nucleon should land near
     the empirical B/A, about -8 MeV.
  2. **Saturation.** zeta(rho) must have a *minimum* near rho_0. A potential
     with attraction alone falls monotonically with density, so compression
     always pays and no fragment has a preferred size or density.

Measured before any fix (two-body Skyrme + Yukawa + Coulomb + Pauli, with the
density-dependent Skyrme term declared in physics.py but never used):

    Ca-40 at rho_0        zeta = -13.90 MeV/A   vs  -8.55 empirical
    density scan  0.04 -> 0.80 fm^-3:  +13.15 -> -127.92, no minimum

Harness sanity check: T/A comes out at 21.7 MeV, matching the Fermi-gas value
3/5 * p_F^2 / 2m = 21.6 MeV at p_F = 260 MeV/c.
"""
import numpy as np
import torch

from clustering.physics import pairwise_potential_matrix

M_P, M_N = 0.938272, 0.939565   # GeV/c^2
P_FERMI = 0.260                 # GeV/c
RHO_0 = 0.16                    # fm^-3
EMPIRICAL_CA40 = -8.55          # MeV/nucleon


def make_nucleus(A, Z, rho, p_fermi=P_FERMI, seed=0):
    """Cold nucleus as an (A, 8) tensor: px py pz E x y z type.

    Momenta in GeV/c and coordinates in fm, matching the dataset's convention.
    ``E`` is put exactly on shell here, unlike the parquet's ``fE`` which is
    off-shell by 18-21 MeV because it carries the source's binding.
    """
    rng = np.random.default_rng(seed)
    R = (3.0 * A / (4.0 * np.pi * rho)) ** (1.0 / 3.0)

    def in_ball(radius):
        # r ~ U^(1/3) * radius gives a uniform density inside the ball
        u = rng.random(A) ** (1.0 / 3.0)
        d = rng.normal(size=(A, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return d * (u * radius)[:, None]

    r = in_ball(R)
    p = in_ball(p_fermi)
    types = np.concatenate([np.ones(Z), -np.ones(A - Z)])
    m = np.where(types > 0, M_P, M_N)
    E = np.sqrt((p ** 2).sum(1) + m ** 2)
    x = np.column_stack([p, E, r, types])
    return torch.tensor(x, dtype=torch.float32), m


def zeta(x, m):
    """SACA energy per nucleon [MeV]: kinetic in the fragment frame + pair potential.

    Returns (zeta, T/A, V/A).
    """
    A = x.shape[0]
    mask = torch.ones(1, A, dtype=torch.bool)
    # The matrix carries both (i,j) and (j,i); halving gives the sum over i < j.
    V = float(pairwise_potential_matrix(x.unsqueeze(0), mask)[0].sum()) / 2.0
    p = x[:, :3].numpy().astype(np.float64)
    p_rel = p - p.sum(0) / A
    T = float((np.sqrt((p_rel ** 2).sum(1) + m ** 2) - m).sum()) * 1000.0  # GeV -> MeV
    return (T + V) / A, T / A, V / A


def mean_zeta(A, Z, rho, n_seeds=5):
    return np.array([zeta(*make_nucleus(A, Z, rho, seed=s))
                     for s in range(n_seeds)]).mean(0)


def main() -> bool:
    print(f"Cold nuclei at rho_0 = {RHO_0} fm^-3, averaged over 5 seeds")
    print(f"{'nucleus':<14}{'zeta [MeV/A]':>14}{'T/A':>9}{'V/A':>10}{'empirical B/A':>16}")
    for name, A, Z, emp in (("alpha (4,2)", 4, 2, -7.07), ("O-16", 16, 8, -7.98),
                            ("Ca-40", 40, 20, -8.55), ("Sn-119", 119, 50, -8.50)):
        z, t, v = mean_zeta(A, Z, RHO_0)
        print(f"{name:<14}{z:14.2f}{t:9.2f}{v:10.2f}{emp:16.2f}")

    print("\nDensity scan, A=40 Z=20 (a saturating potential has a minimum near rho_0):")
    print(f"{'rho [1/fm^3]':>13}{'R [fm]':>9}{'zeta':>10}{'T/A':>9}{'V/A':>10}")
    scan = {}
    for rho in (0.04, 0.08, 0.12, 0.16, 0.24, 0.32, 0.48, 0.80):
        z, t, v = mean_zeta(40, 20, rho)
        scan[rho] = z
        R = (3.0 * 40 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
        print(f"{rho:13.2f}{R:9.2f}{z:10.2f}{t:9.2f}{v:10.2f}")

    depth_ok = abs(scan[RHO_0] - EMPIRICAL_CA40) < 3.0
    saturates = scan[0.80] > scan[RHO_0] and scan[0.32] > scan[RHO_0]
    print(f"\n  depth      : Ca-40 zeta = {scan[RHO_0]:.2f} MeV/A vs "
          f"{EMPIRICAL_CA40} empirical  -> {'PASS' if depth_ok else 'FAIL'}")
    print(f"  saturation : zeta(0.16) = {scan[0.16]:.1f}, zeta(0.32) = {scan[0.32]:.1f}, "
          f"zeta(0.80) = {scan[0.80]:.1f}  -> "
          f"{'PASS' if saturates else 'FAIL (no minimum, compression always pays)'}")
    return depth_ok and saturates


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
