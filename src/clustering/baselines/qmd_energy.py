"""The QMD cluster energy the SACA lineage actually minimizes.

Kept separate from ``clustering.physics``: that module's potential is what the
RL reward uses, and changing it would silently move every trained result. This
one is written to the papers in ``papers/`` and is used by the baselines.

What the papers specify
-----------------------
FRIGA states that SACA's binding energy uses **Skyrme + Yukawa + Coulomb only**
— no Pauli term, no momentum-dependent interaction. The Skyrme part is *not* a
pairwise sum. Folding Aichelin's delta interaction with the Gaussian wave
packets gives a single-particle potential in the *interaction density*

    U(rho) = alpha (rho/rho_0) + beta (rho/rho_0)^gamma

    rho_i = (pi L)^(-3/2) SUM_{j != i} exp( -(r_i - r_j)^2 / L )

and the corresponding energy per nucleon carries the factors 1/2 and
1/(gamma+1) that come from not double-counting the pair and from integrating
the density dependence:

    E_Skyrme = SUM_i [ (alpha/2)(rho_i/rho_0) + (beta/(gamma+1))(rho_i/rho_0)^gamma ]

That is the term this repo was missing. ``physics._calculate_skyrme_potential``
implements only the attractive two-body piece, so nothing opposed compression
and the SACA annealer drove fragments to arbitrarily high density. The
density-dependent term is repulsive (beta > 0) and, with gamma > 1, grows faster
than the attraction — it is what makes nuclear matter saturate.

The two coefficients are jointly fixed by the saturation point, which is the
test in ``tests/test_qmd_energy.py``: for the soft EoS at rho_0,

    T/A + alpha/2 + beta/(gamma+1) = 22 - 178 + 139.9 = -16 MeV/nucleon

Frames
------
Densities and the potential are evaluated with the fragment boosted to **its own
rest frame** (FRIGA: a cluster's density is computed "from the sole nucleons
composing it, as if it were isolated"). This matters here: the spectator source
moves at beta ~ 0.76, so lab longitudinal separations are contracted by
gamma ~ 1.54 and using them would inflate the density by the same factor.

The kinetic term follows the papers' own expression, with momenta taken relative
to the fragment's mean momentum:

    T = SUM_a [ sqrt( (p_a - P/N)^2 + m_a^2 ) - m_a ]

It never reads the dataset's stored energy, which is off-shell by 18-21 MeV.
"""
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# ─── Parameters, from the papers ──────────────────────────────────────────────
# Skyrme (alpha [MeV], beta [MeV], gamma) — FRIGA Table I / Aichelin's QMD.
# "soft" is the EoS the SACA figures were produced with.
SKYRME_EOS: Dict[str, Tuple[float, float, float]] = {
    "hard": (-124.0, 70.5, 2.0),        # K = 380 MeV
    "soft": (-356.0, 303.0, 7.0 / 6.0),  # K = 200 MeV  <- SACA's choice
    "soft_mdi": (-390.1, 320.3, 1.14),   # PHQMD's SM row (its delta term omitted)
}

# rho_0 travels with the Skyrme set and must not be mixed across codes.  The
# alpha/beta/gamma above are BQMD's, whose native rho_0 is 0.15 fm^-3.  Pairing
# them with PHQMD's 0.168 shifts the saturation depth (-13.9 instead of -16.5),
# and going the other way is worse: PHQMD's own alpha/beta/gamma
# (-390.1/320.3/1.14) are the *momentum-dependent* set, and used without their
# MDI term they do not saturate at all — measured minimum at 1.5 rho_0, E/A =
# -21.6 MeV/A.  See papers/notes/yukawa_and_parameters.md.
RHO_0 = 0.15         # fm^-3, BQMD
# Wave-packet width.  The papers quote three values for three codes — PHQMD
# 2.16, BQMD 4.33, IQMD 8.66 fm^2 — a factor-of-2 trap flagged in
# papers/notes/saca_puri_aichelin.md.  Bulk saturation is insensitive to the
# choice (the minimum sits at rho_0 for all three), but finite nuclei are not:
# sqrt(L) is the smearing length, and at 8.66 it is 2.9 fm against a 3.9 fm
# radius for Ca-40, so a light nucleus becomes all surface.  Measured zeta for
# Ca-40: -4.18 at L = 2.16, -1.89 at 4.33, +2.51 at 8.66.  NOTE this leaves the
# set mixed: alpha/beta/gamma and rho_0 are BQMD's, whose own L is 4.33, while
# 2.16 is PHQMD's.  It is kept because bulk saturation is insensitive to L (the
# minimum sits at rho_0 either way) and finite nuclei are much better at 2.16 —
# but it is a deviation, and PHQMD's density normalisation factor C that would
# accompany its L is not implemented.
L_GAUSS = 2.16       # fm^2, PHQMD

V_0_YUKAWA = -85.1   # MeV fm
MU_YUKAWA = 1.0      # fm  (BQMD quotes 1.5, IQMD 0.4; this is the repo's value)
HBARC = 197.3269804  # MeV fm
E_SQ = 1.4399644     # MeV fm

# FRIGA's asymmetry term.  SACA's energy is Skyrme + Yukawa + Coulomb, which is
# isospin-blind apart from Coulomb — and Coulomb only penalizes protons, so
# nothing pulls a fragment back toward the valley.  FRIGA adds
#
#     B_asy = E_0 ((rho_n - rho_p) / rho_B)^2 (<rho_B> / rho_0)^gamma
#
# "in contradistinction to SACA in which only the first term is used", and
# reports that it narrows mass distributions toward N = Z and is "a key
# ingredient".  For a uniformly mixed cluster (rho_n - rho_p)/rho_B = (N - Z)/A,
# so this reduces to E_0 (N - Z)^2 / A — the Weizsacker symmetry energy, and
# E_0 = 23.3 MeV is within 0.5 MeV of that formula's a_sym.  It vanishes
# identically for N = Z, so it does not touch symmetric nuclear matter or the
# saturation test.
E_0_ASY = 23.3       # MeV
GAMMA_ASY = 1.0      # FRIGA's default; it scans 0.5 / 1 / 1.5
M_P, M_N = 0.938272, 0.939565   # GeV/c^2
GEV_TO_MEV = 1000.0


def fermi_momentum(rho: float = RHO_0) -> float:
    """Fermi momentum [GeV/c] of symmetric nuclear matter at density ``rho``.

    ``k_F = (3 pi^2 rho / 2)^(1/3)``, the 2 being the spin-isospin degeneracy 4
    over 2.  Gives 267 MeV/c at rho_0, matching the literature's ~265.  The
    kinetic energy must scale with density like this or the saturation minimum
    lands in the wrong place — held fixed, it sits at 1.5 rho_0 instead of 1.0.
    """
    return HBARC * (1.5 * np.pi ** 2 * rho) ** (1.0 / 3.0) / GEV_TO_MEV


def interaction_density(r: np.ndarray, L: float = L_GAUSS) -> np.ndarray:
    """Interaction density at each nucleon [fm^-3], FRIGA Eq. (4).

    ``rho_i = (pi L)^(-3/2) SUM_{j != i} exp(-(r_i - r_j)^2 / L)``.  The
    prefactor normalizes the Gaussian to unit integral, so for uniform matter of
    density rho this returns rho in the bulk — which is what makes the Skyrme
    coefficients transferable.  The self term is excluded.
    """
    d2 = ((r[:, None, :] - r[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return (np.pi * L) ** (-1.5) * np.exp(-d2 / L).sum(1)


@dataclass
class EnergyTerms:
    """Breakdown in MeV, so a wrong term can be found without guessing."""
    kinetic: float = 0.0
    skyrme: float = 0.0
    yukawa: float = 0.0
    coulomb: float = 0.0
    asymmetry: float = 0.0

    @property
    def total(self) -> float:
        return (self.kinetic + self.skyrme + self.yukawa + self.coulomb
                + self.asymmetry)


def boost_to_rest_frame(p: np.ndarray, E: np.ndarray, r: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Boost a fragment's nucleons into the fragment rest frame.

    ``p`` (n,3) GeV/c, ``E`` (n,) GeV, ``r`` (n,3) fm.  Equal-time convention
    (t = 0 in the lab), so the component of r along beta picks up a factor gamma
    — the lab sees the fragment length-contracted, and the rest frame is where
    its density is the physical one.
    """
    beta = p.sum(0) / max(E.sum(), 1e-12)
    b2 = float(beta @ beta)
    if b2 < 1e-16:
        return p.copy(), r.copy()
    b2 = min(b2, 1.0 - 1e-12)
    gamma = 1.0 / np.sqrt(1.0 - b2)
    bhat = beta / np.sqrt(b2)
    bmag = np.sqrt(b2)

    r_par = (r @ bhat)[:, None] * bhat
    r_out = (r - r_par) + gamma * r_par

    p_par = (p @ bhat)[:, None] * bhat
    p_out = (p - p_par) + (gamma * ((p @ bhat) - bmag * E))[:, None] * bhat
    return p_out, r_out


def cluster_energy(nucleons: np.ndarray, *, eos: str = "soft",
                   rho_0: float = RHO_0, L: float = L_GAUSS,
                   coulomb: bool = True, yukawa: bool = False,
                   asymmetry: bool = False, e_0_asy: float = E_0_ASY,
                   gamma_asy: float = GAMMA_ASY,
                   boost: bool = True) -> EnergyTerms:
    """Energy [MeV] of one fragment.  ``nucleons`` is (n, 8): p(3) E x(3) type.

    Sign convention: negative means bound.

    ``yukawa`` defaults to **off**, and for BQMD's parameters that is the
    self-consistent choice rather than an omission.  BQMD compensates the Yukawa
    inside the Skyrme coupling precisely so that binding stays Yukawa-independent
    — Hartnack et al. (Eur. Phys. J. A 1, 151), verbatim: "In order to keep the
    nuclear equation of state and the binding energy independent of the Yukawa
    interactions and to keep the binding energy at its experimental value, the
    coupling constant t1 of the Skyrme-type two body interaction is modified".
    So the tabulated alpha = -356 MeV is already post-compensation, and adding an
    attractive Yukawa on top of it *double-counts*.  No source in ``papers/``
    prints a Yukawa coefficient anyway (all four defer to Aichelin, Phys. Rep.
    202 (1991), which we do not have), so the term stays available and off.
    """
    n = nucleons.shape[0]
    terms = EnergyTerms()
    if n == 0:
        return terms

    p = nucleons[:, 0:3].astype(np.float64)
    E = nucleons[:, 3].astype(np.float64)
    r = nucleons[:, 4:7].astype(np.float64)
    is_p = nucleons[:, 7] > 0
    m = np.where(is_p, M_P, M_N)

    if boost:
        p, r = boost_to_rest_frame(p, E, r)

    # Kinetic energy in the fragment frame, per the papers' zeta.
    p_rel = p - p.sum(0) / n
    terms.kinetic = float((np.sqrt((p_rel ** 2).sum(1) + m ** 2) - m).sum()) * GEV_TO_MEV
    if n == 1:
        return terms   # a free nucleon has no interaction partner

    d2 = ((r[:, None, :] - r[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)          # excludes the self term everywhere

    # Skyrme: per nucleon in the interaction density, not per pair.
    alpha, beta, gam = SKYRME_EOS[eos]
    rho = (np.pi * L) ** (-1.5) * np.exp(-d2 / L).sum(1)   # = interaction_density(r, L)
    u = rho / rho_0
    terms.skyrme = float((0.5 * alpha * u + beta / (gam + 1.0) * u ** gam).sum())

    if asymmetry:
        # Cluster-level isospin asymmetry, times the mean density factor.
        #
        # Evaluating (rho_n - rho_p)/rho_B per nucleon from the discrete Gaussian
        # sums does NOT work: each nucleon's neighbourhood has a fluctuating n/p
        # balance, the square keeps that shot noise, and it never cancels — the
        # naive form returns +87.55 MeV for an N = Z Ca-40, where the term must
        # be identically zero.  FRIGA writes the density factor as <rho_B>, with
        # averaging brackets, so these are smooth densities; the cluster-level
        # ratio (N - Z)/A is their discrete counterpart.
        #
        # This reduces to E_0 (N - Z)^2 / A at normal density — the Weizsacker
        # symmetry energy, whose a_sym = 23.2 MeV is within 0.5 MeV of FRIGA's
        # E_0 = 23.3.  That agreement is the check that the reading is right.
        n_p = int(is_p.sum())
        n_n = n - n_p
        rho_mean = float(rho.mean())
        terms.asymmetry = (e_0_asy * ((n_n - n_p) / n) ** 2 * n
                           * (rho_mean / rho_0) ** gamma_asy)

    d = np.sqrt(d2)
    # Yukawa and Coulomb are genuine pair sums; 0.5 * full matrix = sum over i<j.
    if yukawa:
        terms.yukawa = 0.5 * float((V_0_YUKAWA * np.exp(-d / MU_YUKAWA) / d).sum())
    if coulomb:
        pp = np.outer(is_p, is_p).astype(np.float64)
        np.fill_diagonal(pp, 0.0)
        terms.coulomb = 0.5 * float((E_SQ * pp / d).sum())
    return terms


def zeta(nucleons: np.ndarray, **kw) -> float:
    """Energy per nucleon [MeV] — the quantity SACA's admissibility cut tests."""
    n = nucleons.shape[0]
    if n == 0:
        return 0.0
    return cluster_energy(nucleons, **kw).total / n


_GS_CACHE: Dict[Tuple[int, int], float] = {}


def ground_state_zeta(A: int, Z: int, seeds: int = 3) -> float:
    """Energy per nucleon [MeV] of the (A, Z) ground state *in this model*.

    Built by constructing a cold nucleus — uniform sphere, Fermi sphere at the
    density's own p_F — and taking the lowest energy over a density scan.

    Why this exists: excitation energy must be measured against the ground state
    of the same Hamiltonian.  Referencing it to the BWM mass formula instead
    charges this model's under-binding to E*.  Measured gap, our ground state
    minus BWM's: +2.73 MeV/nucleon at A=16 falling to +0.64 at A=80, mean +1.41.
    """
    key = (int(A), int(Z))
    hit = _GS_CACHE.get(key)
    if hit is not None:
        return hit
    if A < 2:
        _GS_CACHE[key] = 0.0
        return 0.0

    best = float("inf")
    for rho_f in (0.7, 0.85, 1.0, 1.2, 1.5):
        rho = rho_f * RHO_0
        radius = (3.0 * A / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
        p_f = fermi_momentum(rho)
        vals = []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)

            def in_ball(scale):
                u = rng.random(A) ** (1.0 / 3.0)
                d = rng.normal(size=(A, 3))
                d /= np.linalg.norm(d, axis=1, keepdims=True)
                return d * (u * scale)[:, None]

            r = in_ball(radius)
            p = in_ball(p_f)
            types = np.concatenate([np.ones(Z), -np.ones(A - Z)])
            m = np.where(types > 0, M_P, M_N)
            E = np.sqrt((p ** 2).sum(1) + m ** 2)
            x = np.column_stack([p, E, r, types])
            vals.append(cluster_energy(x).total / A)
        best = min(best, float(np.mean(vals)))
    _GS_CACHE[key] = best
    return best
