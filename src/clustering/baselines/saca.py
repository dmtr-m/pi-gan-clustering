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

which is what ``_SacaEvent.fragment_energy`` computes.  ``V`` is this repo's QMD
potential (Skyrme + Yukawa + Coulomb + Pauli, each in the pair rest frame) — the
same potential the RL reward uses, so the two are on one scale.

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
                 gamma_asy: float = 1.0) -> None:
        self.x = event.cpu().numpy().astype(np.float64)
        self.n = event.shape[0]
        self.asymmetry = asymmetry
        self.e_0_asy = e_0_asy
        self.gamma_asy = gamma_asy
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
            hit = cluster_energy(self.x[list(key)], asymmetry=self.asymmetry,
                                 e_0_asy=self.e_0_asy,
                                 gamma_asy=self.gamma_asy).total
            self._cache[key] = hit
        return hit

    def zeta(self, idx: np.ndarray) -> float:
        """Binding energy per nucleon [MeV]; the quantity ``e_cut`` tests."""
        return self.fragment_energy(idx) / max(1, len(idx))


def _is_bound(ev: "_SacaEvent", idx: np.ndarray, params: SacaParams) -> bool:
    """The SACA admissibility test, with its size-dependent threshold."""
    if len(idx) < 2:
        return False
    cut = params.e_cut if len(idx) >= 3 else params.e_cut_light
    return ev.zeta(idx) < cut


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
    energy = {c: ev.fragment_energy(np.asarray(m)) for c, m in clusters.items()}
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

            e_new_src = ev.fragment_energy(np.asarray(new_src)) if new_src else 0.0
            e_new_dst = ev.fragment_energy(np.asarray(new_dst))
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
        T *= params.alpha
    return best, n_prop, n_acc


@dataclass
class SacaResult:
    labels: np.ndarray          # cluster id per nucleon
    e_initial: float            # E_system of the MST configuration [MeV]
    e_final: float              # E_system after annealing [MeV]
    n_proposed: int = 0
    n_accepted: int = 0
    n_unstable_mst: int = 0     # MST fragments that failed the e_cut test


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
                    e_0_asy=params.e_0_asy, gamma_asy=params.gamma_asy)

    x = event.unsqueeze(0)
    mask = torch.ones(1, event.shape[0], dtype=torch.bool, device=event.device)
    mst = mst_clusters(x, mask, d_cut, p_cut, metric, p_frame)[0].cpu().numpy()
    clusters = _clusters_from_labels(mst)
    e_initial = sum(ev.fragment_energy(np.asarray(m)) for m in clusters.values())

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
    e_final = sum(ev.fragment_energy(np.asarray(m)) for m in final.values())
    return SacaResult(labels=labels, e_initial=e_initial, e_final=e_final,
                      n_proposed=n_prop, n_accepted=n_acc, n_unstable_mst=n_unstable)


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
