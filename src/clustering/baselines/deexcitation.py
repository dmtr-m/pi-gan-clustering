"""Secondary de-excitation: statistical evaporation of hot primary fragments.

**Provenance, stated up front.** The papers in ``papers/`` establish that this
stage is needed and what it does — FRIGA runs GEMINI++ at ~1000 fm/c and reports
that secondary decay raises Z < 5 yields at the expense of the large fragments,
contributing up to 50% of the total alpha yield — but none of them prints the
decay equations. GEMINI++ is a separate code we do not have. So this module
implements the **standard Weisskopf statistical evaporation** that GEMINI is
built on. That is textbook nuclear physics rather than something invented here,
but it is *not* transcribed from the four papers, unlike everything in
``qmd_energy.py``. Treat its outputs accordingly.

Why the pipeline needs it
-------------------------
A clusterizer returns *primary* fragments: which nucleons are travelling
together. Those fragments are hot. Measured on this dataset with MSTp 3.0/150,
primaries sit **3-5 MeV/nucleon above their ground state**, so an A=60 fragment
must shed roughly 28 nucleons before it is cold. What a detector sees — and what
the generator's reference distribution contains — is the cold survivor.

Three separate discrepancies in `NOTES.md` and `BAND_WIDTH.md` point here:
the A = 3-4 deficit (evaporation *creates* light fragments), the heavy-residue
excess (evaporation moves mass down), and the too-wide isotopic band
(a neutron-rich fragment preferentially boils off neutrons, walking toward the
valley — something partitioning provably cannot do).

The model
---------
A compound nucleus ``(A, Z)`` with excitation ``E*`` emits particle ``v`` with
Weisskopf decay width

    Gamma_v  ~  (2 s_v + 1) mu_v A_d^(2/3) INT_0^Emax  eps * rho_d(Emax - eps) d eps

    Emax = E* - S_v - V_v,      rho(U) ~ exp(2 sqrt(a U)),      a = A/8 MeV^-1

with ``S_v`` the separation energy from the mass formula, ``V_v`` the Coulomb
barrier, and ``rho`` the Fermi-gas level density. A channel is open only when
``Emax > 0``. One emission is sampled in proportion to its width, the kinetic
energy ``eps`` is sampled from the same integrand, and the daughter carries
``E* - S_v - V_v - eps`` into the next step. The chain stops when no channel is
open — the fragment is then cold.

Separation energies come from ``physics.bethe_weizsacker`` (the BWM form, which
is verified against the paper's own Fe-56 value), so the same mass formula that
defines the ground state defines the decay thresholds.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from clustering.baselines.qmd_energy import ground_state_zeta
from clustering.physics import bethe_weizsacker

# Emitted species: (name, A, Z, 2s+1).  GEMINI's light-particle channels.
CHANNELS: Sequence[Tuple[str, int, int, int]] = (
    ("n", 1, 0, 2),
    ("p", 1, 1, 2),
    ("d", 2, 1, 3),
    ("t", 3, 1, 2),
    ("He3", 3, 2, 2),
    ("alpha", 4, 2, 1),
)

E_SQ = 1.4399644     # MeV fm
R_0 = 1.2            # fm, radius parameter in the Coulomb barrier
BARRIER_FACTOR = 0.7  # empirical barrier reduction; charged emission is not
                      # classically blocked, it tunnels.  GEMINI treats this
                      # properly; 0.7 is the usual crude stand-in.
LEVEL_DENSITY_DIV = 8.0   # a = A / 8 MeV^-1, the standard evaporation value
_N_GRID = 64              # quadrature points for the width integral

# Experimental binding energies of the emitted light particles [MeV].  The mass
# formula is poor for A <= 4 — it gives 16.7 MeV for the alpha against the true
# 28.3, which would make alpha emission look ~12 MeV more expensive than it is
# and suppress the channel that matters most.  GEMINI likewise uses experimental
# masses.  The parent and residue keep the mass formula, so thresholds stay
# consistent with the ground state that defines E*.
LIGHT_BINDING: Dict[Tuple[int, int], float] = {
    (1, 0): 0.0, (1, 1): 0.0,
    (2, 1): 2.2246,     # d
    (3, 1): 8.4818,     # t
    (3, 2): 7.7180,     # He-3
    (4, 2): 28.2957,    # alpha
}

_B_CACHE: Dict[Tuple[int, int], float] = {}


def binding(A: int, Z: int) -> float:
    """Ground-state binding energy [MeV], positive.  0 for a single nucleon."""
    if A < 2 or Z < 0 or Z > A:
        return 0.0
    key = (A, Z)
    if key in LIGHT_BINDING:
        return LIGHT_BINDING[key]
    hit = _B_CACHE.get(key)
    if hit is None:
        hit = float(bethe_weizsacker(torch.tensor([float(A)]),
                                     torch.tensor([float(Z)])).item())
        _B_CACHE[key] = hit
    return hit


def separation_energy(A: int, Z: int, a_v: int, z_v: int) -> float:
    """Energy to remove particle (a_v, z_v) from (A, Z) [MeV].

    ``S = B(parent) - B(daughter) - B(particle)`` — the binding you have to pay
    back to break the parent into those two pieces.  Getting this backwards
    makes every channel open at any excitation, and a barely-warm fragment
    evaporates itself down to nothing.
    """
    return binding(A, Z) - binding(A - a_v, Z - z_v) - binding(a_v, z_v)


def coulomb_barrier(A_d: int, Z_d: int, a_v: int, z_v: int,
                    barrier_factor: float = BARRIER_FACTOR) -> float:
    """Coulomb barrier seen by a charged emitted particle [MeV].  0 for neutrons.

    ``barrier_factor`` scales the classical barrier.  1.0 is the full classical
    value; below that stands in for tunnelling, which lets charged particles out
    more easily than classical mechanics allows.  It is the knob that sets how
    *selectively* the evaporation sheds neutrons: a stiffer barrier suppresses
    p / d / alpha emission, so the matter thrown away gets more neutron-rich.
    """
    if z_v == 0 or Z_d <= 0:
        return 0.0
    r = R_0 * (A_d ** (1.0 / 3.0) + a_v ** (1.0 / 3.0))
    return barrier_factor * Z_d * z_v * E_SQ / r


def _width_and_spectrum(A: int, Z: int, e_star: float, a_v: int, z_v: int,
                        g_v: int, barrier_factor: float = BARRIER_FACTOR
                        ) -> Tuple[float, np.ndarray, np.ndarray]:
    """(width, eps grid, unnormalized spectrum) for one channel."""
    A_d, Z_d = A - a_v, Z - z_v
    if A_d < 1 or Z_d < 0 or Z_d > A_d:
        return 0.0, np.empty(0), np.empty(0)
    e_max = (e_star - separation_energy(A, Z, a_v, z_v)
             - coulomb_barrier(A_d, Z_d, a_v, z_v, barrier_factor))
    if e_max <= 1e-9:
        return 0.0, np.empty(0), np.empty(0)

    eps = np.linspace(1e-6, e_max, _N_GRID)
    a_d = A_d / LEVEL_DENSITY_DIV
    u = np.maximum(e_max - eps, 0.0)
    # exp(2 sqrt(a U)) overflows for large U; work relative to its own maximum.
    expo = 2.0 * np.sqrt(a_d * u)
    integrand = eps * np.exp(expo - expo.max())
    mu = a_v * A_d / float(a_v + A_d)     # reduced mass in nucleon-mass units
    width = g_v * mu * (A_d ** (2.0 / 3.0)) * float(np.trapezoid(integrand, eps))
    # Carry the shift out so widths of different channels stay comparable.
    return width * float(np.exp(expo.max() - 60.0)), eps, integrand


@dataclass
class DecayResult:
    """Products of one fragment's decay chain."""
    products: List[Tuple[int, int]] = field(default_factory=list)  # (A, Z) incl. residue
    emitted: Dict[str, int] = field(default_factory=dict)
    n_steps: int = 0
    e_star_initial: float = 0.0
    e_star_final: float = 0.0


def evaporate(A: int, Z: int, e_star: float, rng: np.random.Generator,
              max_steps: int = 400,
              barrier_factor: float = BARRIER_FACTOR) -> DecayResult:
    """Run the evaporation chain on one hot fragment.

    Returns every product, the residue included.  ``A`` strictly decreases on
    each emission, so this always terminates; ``max_steps`` is a guard, not the
    normal exit.
    """
    res = DecayResult(e_star_initial=float(e_star))
    if A < 2 or e_star <= 0.0:
        res.products.append((A, Z))
        res.e_star_final = max(0.0, float(e_star))
        return res

    for _ in range(max_steps):
        widths, grids, spectra, chans = [], [], [], []
        for name, a_v, z_v, g_v in CHANNELS:
            w, eps, spec = _width_and_spectrum(A, Z, e_star, a_v, z_v, g_v,
                                               barrier_factor)
            if w > 0.0:
                widths.append(w); grids.append(eps); spectra.append(spec)
                chans.append((name, a_v, z_v))
        if not widths:
            break                      # cold: nothing can be emitted any more

        w = np.array(widths)
        k = int(rng.choice(len(w), p=w / w.sum()))
        name, a_v, z_v = chans[k]
        eps_grid, spec = grids[k], spectra[k]
        cdf = np.cumsum(spec); cdf /= cdf[-1]
        eps = float(np.interp(rng.random(), cdf, eps_grid))

        A_d, Z_d = A - a_v, Z - z_v
        e_star = (e_star - separation_energy(A, Z, a_v, z_v)
                  - coulomb_barrier(A_d, Z_d, a_v, z_v, barrier_factor) - eps)
        A, Z = A_d, Z_d
        res.products.append((a_v, z_v))
        res.emitted[name] = res.emitted.get(name, 0) + 1
        res.n_steps += 1
        if A < 2 or e_star <= 0.0:
            break

    res.products.append((A, Z))         # the surviving residue
    res.e_star_final = max(0.0, float(e_star))
    return res


def excitation_energy(cluster_total_mev: float, A: int, Z: int,
                      reference: str = "model") -> float:
    """E* = (energy as formed) - (ground-state energy) [MeV], floored at 0.

    ``reference``:
      * ``"model"`` (default) — the ground state of the *same* energy model,
        from ``qmd_energy.ground_state_zeta``.  This is the physically correct
        choice: E* is excitation above the minimum of the Hamiltonian being used.
      * ``"bwm"`` — the mass formula's ``-B(A, Z)``.  Mixes two energy scales:
        our model is shallower than BWM by +2.73 MeV/nucleon at A=16 down to
        +0.64 at A=80 (mean +1.41), and referencing E* to BWM charges that whole
        gap to the excitation, so every fragment comes out spuriously hot.

    Clusters below their ground state are treated as cold rather than as having
    negative excitation — FRIGA does the same, noting that a semi-classical
    model over-binds because "the ground state of the quantum hamiltonian is
    higher than the ground state of the classical hamiltonian".
    """
    if reference == "bwm":
        return max(0.0, cluster_total_mev + binding(A, Z))
    if reference != "model":
        raise ValueError(f"unknown reference {reference!r}; expected 'model' or 'bwm'")
    return max(0.0, cluster_total_mev - A * ground_state_zeta(A, Z))
