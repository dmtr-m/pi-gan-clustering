"""SACA baseline: simulated-annealing cluster recognition, seeded by MST.

MST links nucleons that are *close*; it has no notion of whether the group it
formed is bound.  SACA (Simulated Annealing Clusterization Algorithm) takes the
MST fragments as a starting configuration and rearranges nucleons between them
to minimize the total binding energy of the whole event, so a spatially close
but unbound group can come apart and a genuinely bound one can absorb the free
nucleons around it.

Energy
------
For a fragment ``f`` of ``N_f`` nucleons the paper's per-nucleon energy is

    zeta_f = (1/N_f) * SUM_a [ sqrt((p_a - P_f/N_f)^2 + m_a^2) - m_a
                               + (1/2) SUM_{b != a} V_ab ]

i.e. the kinetic energy in the fragment's own frame plus the pairwise potential,
and the configuration energy is ``E = SUM_f N_f * zeta_f``.  Written per
fragment that is just

    E_f = SUM_a T_a(rest frame)  +  SUM_{a<b} V_ab

which is what ``_SacaEvent.fragment_energy`` computes.  ``V`` comes from
``qmd_energy.cluster_energy``: Skyrme in the interaction density plus Coulomb,
evaluated in the **fragment** rest frame, with Yukawa and
the FRIGA asymmetry term off by default and no Pauli or momentum-dependent term.
That is deliberately **not** the potential the RL reward uses — ``physics.py``'s
pairwise Skyrme+Yukawa+Coulomb+Pauli does not saturate, and SACA minimizes its
energy directly, so it would exploit that.  See ``qmd_energy``'s docstring and
``tests/test_qmd_energy.py``.

A singleton contributes exactly zero: with ``N_f = 1`` the momentum relative to
the fragment is zero and there are no pairs.  Free nucleons are therefore
energetically neutral, and a fragment is worth forming only if its potential
outweighs the internal kinetic energy it locks in.

Note this functional never reads ``fParticles.fE``: it reconstructs the energy
from ``p`` and the nucleon mass.  That is deliberate — the stored ``fE`` is
off-shell by 18-21 MeV (it carries the source's binding), which would leak
straight into a binding-energy criterion.

Pipeline (SACA_PIPELINE.md)
---------------------------
1. MST pre-fragments + free nucleons.
2. Classify: bound if ``zeta < L_be``, with ``L_be = e_cut`` (-4 MeV/nucleon)
   for ``N_f >= 3`` and ``e_cut_light`` (0) below that — the original SACA
   criterion.  Everything else is *unstable*.
3. Pass 1 — anneal each unstable fragment on its own, allowing nucleons to be
   released or moved between its pieces.  Pieces that end up bound are kept;
   the rest are dissolved into free nucleons.
4. Pass 2 — anneal the whole event with established clusters frozen against
   loss: free nucleons may be absorbed and clusters may merge, but no cluster
   gives a nucleon back.
5. Output: bound clusters + free nucleons.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from clustering.baselines.qmd_energy import cluster_energy
from clustering.baselines.qmd_full import SPIN_FACTOR, YUKAWA, full_cluster_energy
from clustering.physics import bethe_weizsacker, pairwise_potential_matrix
from clustering.split_prediction.mst import D_CUT, P_CUT, P_FRAME, mst_clusters

MASS_PROTON = 0.938272   # GeV/c^2
MASS_NEUTRON = 0.939565  # GeV/c^2
GEV_TO_MEV = 1000.0


@dataclass
class SacaParams:
    """Annealing controls.  ``t_*`` are in MeV, the unit of the energy above."""
    t_max: float = 20.0
    t_min: float = 0.5
    alpha: float = 0.9          # geometric cooling, T <- alpha * T
    trials_per_nucleon: int = 4  # trials at each temperature = this * N
    # How T falls from t_max to t_min.  "geometric" is T <- alpha T (the paper's
    # schedule, and the default).  "linear" is T <- T - (t_max - t_min)/n_steps, so
    # it takes exactly n_steps temperature steps and ``alpha`` is unused.  Linear
    # spends most of its steps at high T on a log scale and few near t_min, the
    # opposite of geometric, which spends equally many per decade.
    cooling: str = "geometric"  # "geometric" | "linear"
    n_steps: int = 90
    # Puri & Aichelin's admissibility criterion (ALPHAXIV_FORMULA_CONVERSATION.md):
    # zeta < L_be with L_be = -4 MeV/nucleon for N_f >= 3 and L_be = 0 below that.
    # The size split is not cosmetic — a real deuteron is bound at -1.1
    # MeV/nucleon, so a uniform -4 cut forbids deuterons by construction, and
    # they are the generator's most abundant species.  SACA_PIPELINE.md's CCL
    # variant uses -4 for every size; set e_cut_light=-4.0 to reproduce it.
    e_cut: float = -4.0         # MeV/nucleon, fragments with N_f >= 3
    e_cut_light: float = 0.0    # MeV/nucleon, fragments with N_f < 3
    p_release: float = 0.3      # probability a pass-1 move is "release to free"
    two_pass: bool = True
    # FRIGA's asymmetry term, off by default because SACA proper does not have
    # it ("in contradistinction to SACA in which only the first term is used").
    # SACA's energy is isospin-blind apart from Coulomb, which penalizes protons
    # and so drives fragments neutron-rich with nothing pulling back; this is the
    # term FRIGA adds and calls "a key ingredient" for isotopic widths.
    asymmetry: bool = False
    e_0_asy: float = 23.3       # MeV, FRIGA's coefficient
    gamma_asy: float = 1.0      # FRIGA's default; it scans 0.5 / 1 / 1.5
    # SACA 2.1 (Vermani et al.): the constant admissibility cut is replaced by
    # the fragment's *own* BWM binding energy per nucleon, so the threshold
    # depends on (A, Z) instead of being flat.  "constant" keeps SACA 1.1.
    #
    # The paper never writes the 2.1 inequality — it says only "instead of a
    # constant -4 MeV/nucleon binding energy", and its Eq. (7) is a total energy
    # while the criterion is per nucleon.  The reading used here, flagged in
    # papers/notes/saca_realistic_binding.md as an inference rather than a
    # quotation, is  zeta < -bwm_scale * B_BWM(A, Z) / A.  The N_f <= 2 special
    # case is also unstated; e_cut_light still applies, as in 1.1.
    # Which quantity the admissibility test compares against, and to what.
    #   "constant"   zeta_QMD < e_cut                    (SACA 1.1)
    #   "bwm"        zeta_QMD < -bwm_scale B(A,Z)/A      (SACA 2.1)
    #   "objective"  zeta of the *annealing objective* < e_cut, i.e. the same
    #                B-corrected quantity the search minimizes rather than the
    #                bare QMD energy.  Its scale is different — with
    #                "qmd_minus_b" it runs ~ -14 MeV/nucleon, not ~ -6 — so
    #                e_cut has to be rescaled with it.
    e_cut_model: str = "constant"   # "constant" | "bwm" | "objective"
    bwm_scale: float = 1.0          # 1.0 = the full BWM binding, as published
    # What the annealing minimizes.  B is the BWM binding energy, positive for a
    # bound nuclide; the BWM *energy* is -B.  The two signs are different
    # objectives, not a convention detail:
    #
    #   "qmd"          sum_f E_QMD(f).  SACA proper: maximize binding.  Binding
    #                  is extensive, which is why it collapses onto one residue.
    #   "qmd_plus_b"   sum_f [E_QMD(f) + w B_f] = QMD - E_BWM: the excitation
    #                  above the liquid-drop ground state.  Prefers *cold*
    #                  fragments.  Beware: at fixed A, B is smallest furthest
    #                  from the valley, so this actively rewards exotic
    #                  nuclides — measured, it makes more dineutrons than
    #                  deuterons.
    #   "qmd_minus_b"  sum_f [E_QMD(f) - w B_f]: credits binding twice, once
    #                  microscopically and once from the mass formula.  At fixed
    #                  A this is maximized *on* the valley, so it pulls the
    #                  opposite way to qmd_plus_b on composition.
    #
    # `bwm_weight` (w) mixes with plain SACA: w = 0 is "qmd" for either model.
    #
    # Three more, which swap or blend the *microscopic* term (baselines_comparison.py):
    #
    #   "physics_v"          sum_f V_f: the pairwise potential of physics.py (Skyrme +
    #                        Yukawa + Coulomb + Pauli, no fragment-frame kinetic
    #                        term) — the RL reward's "QMD" energy — in place of
    #                        qmd_energy.cluster_energy.
    #   "physics_v_minus_b"  sum_f [V_f - w B_f]: the same, with BWM binding credited
    #                        (the qmd_minus_b reward on the annealer).
    #   "mix"                sum_f [alpha E_f - (1 - alpha) B_f] with E_f the
    #                        cluster_energy of the default model; alpha = 1 is plain
    #                        SACA, alpha = 0 is the mass formula alone.  Per
    #                        nucleon, zeta = alpha zeta_QMD - (1 - alpha) B/A.
    #
    # ``mix_per_nucleon`` chooses whether that objective is *extensive* (False, the
    # sum above — SACA's own N_f zeta_f) or *intensive* (True: the sum over
    # fragments of zeta_f, i.e. each fragment divided by its A).  The intensive one
    # is the "affinity" scale; weizsacker_qmd_energy's notes say it over-splits.
    # ``e_cut_model="objective"`` tests zeta of whichever objective is in force.
    #
    #   "zeta_correct"       sum_f [zeta_f + lambda * bwd_f], the paper's ζ with the
    #                        full QMD potential (qmd_full.full_cluster_energy:
    #                        rest-frame kinetic + two- and three-body Skyrme +
    #                        Yukawa + Coulomb + Pauli) plus lambda times the
    #                        Weizsäcker energy bwd = -B_BWM, negative when bound.
    #                        lambda is ``bwm_weight``.  Same sign as qmd_minus_b, on
    #                        a different QMD energy.  ``spin_factor`` scales Pauli.
    energy_model: str = "qmd"
    bwm_weight: float = 1.0
    mix_alpha: float = 1.0
    mix_per_nucleon: bool = False
    spin_factor: float = SPIN_FACTOR
    yukawa: str = YUKAWA         # zeta_correct only: "folded" | "point"


class _SacaEvent:
    """Per-event cache of the nucleon array, in numpy.

    There is no precomputed pair-potential matrix any more, and there cannot be:
    the Skyrme term is per nucleon in the interaction density, so a fragment's
    energy is not a sum over its pairs and a membership change cannot be scored
    by a submatrix sum.  Each evaluation recomputes the fragment — same O(k^2)
    order as the submatrix sum it replaces, plus the boost to the fragment rest
    frame, which is O(k) and is not optional: scoring spectator fragments in the
    lab frame biases them by ~+8.6 MeV/nucleon and makes bound ones look free.
    """

    def __init__(self, event: torch.Tensor, type_index: int = 7,
                 asymmetry: bool = False, e_0_asy: float = 23.3,
                 gamma_asy: float = 1.0, energy_model: str = "qmd",
                 bwm_weight: float = 1.0, mix_alpha: float = 1.0,
                 mix_per_nucleon: bool = False,
                 spin_factor: float = SPIN_FACTOR,
                 yukawa: str = YUKAWA) -> None:
        self.spin_factor = spin_factor
        self.yukawa = yukawa
        self.x = event.cpu().numpy().astype(np.float64)
        self._t = event.detach().cpu().float()   # physics.py wants float32 tensors
        self.mix_alpha = mix_alpha
        self.mix_per_nucleon = mix_per_nucleon
        self._v_matrix: Optional[np.ndarray] = None
        self.n = event.shape[0]
        self.asymmetry = asymmetry
        self.e_0_asy = e_0_asy
        self.gamma_asy = gamma_asy
        self.energy_model = energy_model
        self.bwm_weight = bwm_weight
        self._cache: Dict[Tuple[int, ...], float] = {}

    def fragment_energy(self, idx: np.ndarray) -> float:
        """Total energy of the fragment [MeV]; negative means bound."""
        k = len(idx)
        if k < 1:
            return 0.0
        if k == 1:
            return 0.0    # a free nucleon: no partner, and no internal motion
        key = tuple(sorted(int(i) for i in idx))
        hit = self._cache.get(key)
        if hit is None:
            if self.energy_model == "zeta_correct":
                hit = full_cluster_energy(self.x[list(key)],
                                          spin_factor=self.spin_factor,
                                          yukawa=self.yukawa).total
            else:
                hit = cluster_energy(self.x[list(key)], asymmetry=self.asymmetry,
                                     e_0_asy=self.e_0_asy,
                                     gamma_asy=self.gamma_asy).total
            self._cache[key] = hit
        return hit

    def fragment_terms(self, idx: np.ndarray) -> Tuple[float, float]:
        """(T, V) of the fragment [MeV]: rest-frame kinetic and total potential.

        The split of ``fragment_energy`` for whichever QMD energy the model uses:
        ``zeta_correct`` -> qmd_full (Skyrme 2+3 body, Yukawa, Coulomb, Pauli);
        the older models -> qmd_energy.cluster_energy (Skyrme, Coulomb, plus
        Yukawa / asymmetry when on).  ``physics_v*`` have no such split and
        return NaN.  A single nucleon is (0, 0).
        """
        if len(idx) < 2:
            return 0.0, 0.0
        if self.energy_model in ("physics_v", "physics_v_minus_b"):
            return float("nan"), float("nan")
        xs = self.x[sorted(int(i) for i in idx)]
        if self.energy_model == "zeta_correct":
            t = full_cluster_energy(xs, spin_factor=self.spin_factor, yukawa=self.yukawa)
            return t.kinetic, t.potential
        t = cluster_energy(xs, asymmetry=self.asymmetry, e_0_asy=self.e_0_asy,
                           gamma_asy=self.gamma_asy)
        return t.kinetic, t.total - t.kinetic

    def zeta(self, idx: np.ndarray) -> float:
        """Binding energy per nucleon [MeV]; the quantity ``e_cut`` tests.

        Always the plain QMD energy — the admissibility test is independent of
        whatever the annealing happens to be minimizing.
        """
        return self.fragment_energy(idx) / max(1, len(idx))

    def physics_v(self, idx: np.ndarray) -> float:
        """physics.py's pairwise QMD potential of the fragment [MeV]; 0 below two.

        ``pairwise_potential_matrix`` is evaluated once for the whole event: each
        entry is boosted to *its own pair's* rest frame, so it does not depend on
        which fragment the two nucleons end up in and a fragment's V is a
        submatrix sum.  (Unlike cluster_energy's Skyrme term, which is per
        nucleon in the interaction density.)  Equal to ``total_potential_energy``
        of the fragment — asserted in tests/test_saca_energy_models.py.
        """
        if len(idx) < 2:
            return 0.0
        if self._v_matrix is None:
            t = self._t.unsqueeze(0)
            mask = torch.ones(1, self.n, dtype=torch.bool)
            self._v_matrix = pairwise_potential_matrix(t, mask)[0].double().numpy()
        ii = np.asarray(idx, dtype=int)
        return float(self._v_matrix[np.ix_(ii, ii)].sum() / 2.0)

    def zeta_objective(self, idx: np.ndarray) -> float:
        """Objective energy per nucleon — what ``e_cut_model="objective"`` tests."""
        e = self.objective_energy(idx)
        return e if self._intensive() else e / max(1, len(idx))

    def _intensive(self) -> bool:
        return self.energy_model == "mix" and self.mix_per_nucleon

    def objective_energy(self, idx: np.ndarray) -> float:
        """The quantity the annealing minimizes, per ``energy_model``."""
        if self.energy_model in ("physics_v", "physics_v_minus_b", "mix"):
            return self._objective_new(idx)
        e = self.fragment_energy(idx)
        if self.energy_model == "qmd":
            return e
        if self.energy_model not in ("qmd_plus_b", "qmd_minus_b", "zeta_correct"):
            raise ValueError(f"unknown energy_model {self.energy_model!r}")
        A = len(idx)
        if A < 2:
            return e
        Z = int((self.x[idx, 7] == 1).sum())
        if self.energy_model == "zeta_correct":
            bwd = -bwm_binding(A, Z)          # Weizsäcker energy: negative when bound
            return e + self.bwm_weight * bwd
        b = self.bwm_weight * bwm_binding(A, Z)
        return e + b if self.energy_model == "qmd_plus_b" else e - b

    def _objective_new(self, idx: np.ndarray) -> float:
        A = len(idx)
        if A < 2:
            return 0.0
        Z = int((self.x[idx, 7] == 1).sum())
        if self.energy_model == "physics_v":
            return self.physics_v(idx)
        if self.energy_model == "physics_v_minus_b":
            return self.physics_v(idx) - self.bwm_weight * bwm_binding(A, Z)
        a = self.mix_alpha
        e = a * self.fragment_energy(idx) - (1.0 - a) * bwm_binding(A, Z)
        return e / A if self.mix_per_nucleon else e


_BWM_PER_A: Dict[Tuple[int, int], float] = {}


def bwm_zeta(A: int, Z: int) -> float:
    """-B_BWM(A, Z) / A [MeV/nucleon]: SACA 2.1's admissibility threshold.

    Negative for a bound nuclide, so it drops straight into the same ``zeta <
    cut`` comparison the constant -4 MeV/nucleon occupies in SACA 1.1.
    """
    key = (A, Z)
    hit = _BWM_PER_A.get(key)
    if hit is None:
        b = float(bethe_weizsacker(torch.tensor([float(A)]),
                                   torch.tensor([float(Z)])).item())
        hit = -b / A
        _BWM_PER_A[key] = hit
    return hit


def bwm_binding(A: int, Z: int) -> float:
    """BWM binding energy B(A, Z) [MeV], positive for a bound nuclide."""
    return -bwm_zeta(A, Z) * A


def _is_bound(ev: "_SacaEvent", idx: np.ndarray, params: SacaParams) -> bool:
    """The SACA admissibility test, with its size-dependent threshold."""
    if len(idx) < 2:
        return False
    if len(idx) < 3:
        # The three energy models added for baselines_comparison.py have no
        # meaningful cluster_energy zeta of their own, so the N_f = 2 test is
        # applied to their own objective; the older models keep the QMD zeta.
        if params.energy_model in ("physics_v", "physics_v_minus_b", "mix",
                                   "zeta_correct"):
            return ev.zeta_objective(idx) < params.e_cut_light
        return ev.zeta(idx) < params.e_cut_light
    if params.e_cut_model == "objective":
        return ev.zeta_objective(idx) < params.e_cut
    if params.e_cut_model == "bwm":
        A = len(idx)
        Z = int((ev.x[idx, 7] == 1).sum())
        return ev.zeta(idx) < params.bwm_scale * bwm_zeta(A, Z)
    if params.e_cut_model != "constant":
        raise ValueError(f"unknown e_cut_model {params.e_cut_model!r}")
    return ev.zeta(idx) < params.e_cut


def _clusters_from_labels(labels: np.ndarray) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for i, c in enumerate(labels):
        out.setdefault(int(c), []).append(i)
    return out


def _anneal(
    ev: _SacaEvent,
    clusters: Dict[int, List[int]],
    params: SacaParams,
    rng: np.random.Generator,
    *,
    allow_loss: bool,
    frozen: Optional[set] = None,
) -> Tuple[Dict[int, List[int]], int, int]:
    """Metropolis annealing over cluster membership.

    ``allow_loss=True`` is pass 1: a nucleon may be released to a singleton or
    moved to another cluster.  ``allow_loss=False`` is pass 2: only nucleons
    that are currently free may move, and whole clusters may merge, so an
    established fragment never loses mass.

    The *best* configuration visited is returned, not the last one.  Metropolis
    accepts uphill moves by design, so the final state of a cooling run is not
    generally its minimum — without this the annealer could and did end above
    the MST configuration it started from.

    Returns (best_clusters, n_proposed, n_accepted).
    """
    frozen = frozen or set()
    energy = {c: ev.objective_energy(np.asarray(m)) for c, m in clusters.items()}
    total = sum(energy.values())
    best_total = total
    best = {c: list(m) for c, m in clusters.items() if m}
    next_id = max(clusters) + 1 if clusters else 0
    n_prop = n_acc = 0

    trials = max(20, params.trials_per_nucleon * ev.n)
    T = params.t_max
    while T > params.t_min:
        for _ in range(trials):
            ids = [c for c in clusters if clusters[c]]
            multi = [c for c in ids if len(clusters[c]) > 1]
            n_prop += 1

            if allow_loss:
                # Release needs one multi-nucleon cluster; transfer needs a
                # second cluster to move into.  A single intact MST fragment is
                # the normal pass-1 starting point, so release must work there.
                if not multi:
                    break
                src = int(rng.choice(multi))
                a = int(rng.choice(clusters[src]))
                others = [c for c in ids if c != src]
                to_new = (not others) or (rng.random() < params.p_release)
                dst = -1 if to_new else int(rng.choice(others))
                new_src = [i for i in clusters[src] if i != a]
                new_dst = ([] if to_new else list(clusters[dst])) + [a]
            else:
                if len(ids) < 2:
                    break
                # Pass 2: hand a free nucleon to a cluster, or merge two whole
                # clusters.  Neither takes mass out of an established cluster.
                singles = [c for c in ids if len(clusters[c]) == 1 and c not in frozen]
                to_new = False
                if singles and rng.random() < 0.5:
                    src = int(rng.choice(singles))
                else:
                    src = int(rng.choice(ids))
                others = [c for c in ids if c != src]
                if not others:
                    break
                dst = int(rng.choice(others))
                new_src, new_dst = [], clusters[src] + clusters[dst]

            e_new_src = ev.objective_energy(np.asarray(new_src)) if new_src else 0.0
            e_new_dst = ev.objective_energy(np.asarray(new_dst))
            d = e_new_src + e_new_dst - energy[src] - (0.0 if to_new else energy[dst])

            if d < 0 or rng.random() < np.exp(-min(d / T, 700.0)):
                n_acc += 1
                if to_new:
                    dst = next_id
                    next_id += 1
                clusters[src] = new_src
                clusters[dst] = new_dst
                energy[src] = e_new_src
                energy[dst] = e_new_dst
                if not clusters[src]:
                    del clusters[src], energy[src]
                total += d
                if total < best_total - 1e-9:
                    best_total = total
                    best = {c: list(m) for c, m in clusters.items() if m}
        if params.cooling == "geometric":
            T *= params.alpha
        elif params.cooling == "linear":
            T -= (params.t_max - params.t_min) / params.n_steps
        else:
            raise ValueError(f"unknown cooling {params.cooling!r}")
    return best, n_prop, n_acc


@dataclass
class SacaResult:
    labels: np.ndarray          # cluster id per nucleon
    e_initial: float            # E_system of the MST configuration [MeV]
    e_final: float              # E_system after annealing [MeV]
    n_proposed: int = 0
    n_accepted: int = 0
    n_unstable_mst: int = 0     # MST fragments that failed the e_cut test
    # Final-state sums over the fragments of N >= 2 (free nucleons contribute 0), MeV.
    # For zeta_correct, e_final == t_sum + v_sum + bwm_weight * bwd_sum.
    t_sum: float = float("nan")     # rest-frame kinetic energy, summed
    v_sum: float = float("nan")     # QMD potential, summed
    bwd_sum: float = float("nan")   # Weizsaecker energy -B_BWM, summed (unweighted)


def saca_clusters(
    event: torch.Tensor,
    params: SacaParams = SacaParams(),
    *,
    d_cut: float = D_CUT,
    p_cut: float = P_CUT,
    metric: str = "coord",
    p_frame: str = P_FRAME,
    type_index: int = 7,
    rng: Optional[np.random.Generator] = None,
) -> SacaResult:
    """Run the full MST -> classify -> anneal -> recombine pipeline on one event."""
    rng = rng or np.random.default_rng()
    ev = _SacaEvent(event, type_index, asymmetry=params.asymmetry,
                    e_0_asy=params.e_0_asy, gamma_asy=params.gamma_asy,
                    energy_model=params.energy_model,
                    bwm_weight=params.bwm_weight, mix_alpha=params.mix_alpha,
                    mix_per_nucleon=params.mix_per_nucleon,
                    spin_factor=params.spin_factor, yukawa=params.yukawa)

    x = event.unsqueeze(0)
    mask = torch.ones(1, event.shape[0], dtype=torch.bool, device=event.device)
    mst = mst_clusters(x, mask, d_cut, p_cut, metric, p_frame)[0].cpu().numpy()
    clusters = _clusters_from_labels(mst)
    e_initial = sum(ev.objective_energy(np.asarray(m)) for m in clusters.values())

    # Step 2 — stable / unstable by binding energy per nucleon.
    stable, unstable = {}, {}
    for c, members in clusters.items():
        (stable if _is_bound(ev, np.asarray(members), params) else unstable)[c] = members
    n_unstable = sum(1 for m in unstable.values() if len(m) > 1)

    n_prop = n_acc = 0
    free: List[int] = []
    pass1: Dict[int, List[int]] = {}
    next_id = (max(clusters) + 1) if clusters else 0
    for members in unstable.values():
        if len(members) < 2:
            free += members
            continue
        # Anneal this fragment alone: every nucleon starts in one cluster.
        sub = {0: list(members)}
        sub, pr, ac = _anneal(ev, sub, params, rng, allow_loss=True)
        n_prop += pr; n_acc += ac
        for m in sub.values():
            if _is_bound(ev, np.asarray(m), params):
                pass1[next_id] = m
                next_id += 1
            else:
                free += m          # not bound enough -> dissolved

    # Step 4 — recombination over stable + pass-1 clusters + free nucleons.
    final: Dict[int, List[int]] = {}
    for members in list(stable.values()) + list(pass1.values()):
        final[next_id] = list(members); next_id += 1
    frozen = set(final)
    for i in free:
        final[next_id] = [i]; next_id += 1
    if params.two_pass and len(final) > 1:
        final, pr, ac = _anneal(ev, final, params, rng, allow_loss=False, frozen=frozen)
        n_prop += pr; n_acc += ac

    labels = np.full(ev.n, -1, dtype=np.int64)
    for new_c, (_, members) in enumerate(sorted(final.items())):
        for i in members:
            labels[i] = new_c
    e_final = sum(ev.objective_energy(np.asarray(m)) for m in final.values())
    t_sum = v_sum = bwd_sum = 0.0
    for m in final.values():
        if len(m) < 2:
            continue
        idx = np.asarray(m)
        t, v = ev.fragment_terms(idx)
        t_sum += t
        v_sum += v
        bwd_sum += -bwm_binding(len(idx), int((ev.x[idx, 7] == 1).sum()))
    return SacaResult(labels=labels, e_initial=e_initial, e_final=e_final,
                      n_proposed=n_prop, n_accepted=n_acc, n_unstable_mst=n_unstable,
                      t_sum=t_sum, v_sum=v_sum, bwd_sum=bwd_sum)


class SACABaseline:
    """SACA fragments in the shape Stage 4's report consumes.

    Same output contract as ``MSTDecayBaseline`` — a ``BaselineResult`` whose
    ``fragments`` are ``(n_i, 8)`` tensors — so the two baselines produce
    directly comparable plots, yields and traces.  ``decay`` runs the same
    stability-table evaporation afterwards, which is not part of SACA itself but
    keeps the comparison against the MST baseline like-for-like.
    """

    def __init__(
        self,
        lut,
        *,
        params: SacaParams = SacaParams(),
        d_cut: float = D_CUT,
        p_cut: float = P_CUT,
        metric: str = "coord",
        p_frame: str = P_FRAME,
        decay: bool = True,
        emit_rule: str = "hottest",
        type_index: int = 7,
        trace: bool = False,
        seed: int = 0,
    ) -> None:
        from clustering.baselines.mst_decay import decay_to_table  # circular-safe
        self._decay_to_table = decay_to_table
        self.lut = lut
        self.params = params
        self.d_cut = d_cut
        self.p_cut = p_cut
        self.metric = metric
        self.p_frame = p_frame
        self.decay = decay
        self.emit_rule = emit_rule
        self.type_index = type_index
        self.trace = trace
        self.rng = np.random.default_rng(seed)
        # Energy bookkeeping over the whole pass, for the Stage 4 summary.
        self.e_initial = 0.0
        self.e_final = 0.0
        self.n_proposed = 0
        self.n_accepted = 0

    @torch.no_grad()
    def __call__(self, event: torch.Tensor):
        from clustering.baselines.mst_decay import BaselineResult, DecayStep, _AZ
        res = saca_clusters(
            event, self.params, d_cut=self.d_cut, p_cut=self.p_cut,
            metric=self.metric, p_frame=self.p_frame, type_index=self.type_index,
            rng=self.rng,
        )
        self.e_initial += res.e_initial
        self.e_final += res.e_final
        self.n_proposed += res.n_proposed
        self.n_accepted += res.n_accepted

        labels = torch.from_numpy(res.labels).to(event.device)
        primaries = [event[labels == c] for c in labels.unique() if c >= 0]
        n_in_table = sum(1 for f in primaries
                         if self.lut.is_stable(*_AZ(f, self.type_index)))

        fragments, steps, n_evap = [], [], 0
        for j, f in enumerate(primaries):
            if self.decay:
                frag_trace = [] if self.trace else None
                pieces, n = self._decay_to_table(
                    f, self.lut, emit_rule=self.emit_rule,
                    type_index=self.type_index, trace=frag_trace,
                )
                fragments += pieces
                n_evap += n
                if frag_trace:
                    for st in frag_trace:
                        st.fragment = j
                    steps += frag_trace
            else:
                fragments.append(f)

        return BaselineResult(
            fragments=fragments, steps=steps, n_primary=len(primaries),
            n_primary_in_table=n_in_table, n_evaporated=n_evap,
        )
