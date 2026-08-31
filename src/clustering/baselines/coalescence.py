"""Box-like coalescence, per Kireyeu arXiv:2512.02084 section 5.

This is the third baseline family. Unlike MST and SACA it forms *only light
nuclei* — d, t, 3He, 4He — and everything it does not bind stays a free nucleon.
That is exactly the end of the spectrum the other two miss: the generator makes
10.9 fragments per collision at A = 2-4 against MST's 2.2.

The criterion is a **box, not a phase-space volume**: two independent hard cuts,
evaluated pairwise in the pair centre-of-mass frame,

    dr^2 <= R^2   AND   dp^2 <= P^2

each with its own per-species value. A further particle joins an existing
candidate by satisfying the same two cuts against it, with

    p_pair = p_1 + p_2,   r_pair = (r_1 + r_2) / 2

— momentum sum, position arithmetic mean, no mass weighting.

**The species chain is mandatory and ordered heaviest first**: 4He -> t -> 3He
-> d, with the constituents of every accepted cluster removed from the pool so
they cannot be reused. Deuterons therefore form last, out of the leftovers,
which is the paper's own explanation for why the deuteron yield is insensitive
to the cuts. (The paper's chain also carries 4LH and 3LH ahead of these; there
are no hyperons in this dataset.)

Two deviations from the paper, both forced and both documented:

* **No freeze-out-time synchronisation.** The paper propagates the
  earlier-freezing particle along a straight line to the later freeze-out time
  before measuring dr. This dataset is a single time snapshot with no
  per-particle freeze-out time, so dt = 0 and the step is a no-op.
* **No spin-isospin formation probability.** The paper applies one but does not
  print the values ("taken from the UrQMD model source code"). Rather than
  invent them, ``selection="none"`` accepts every candidate — which
  over-produces, as the paper says coalescence does. ``selection="ebind"`` is
  the paper's own alternative: its **"mixed" procedure** replaces the
  spin-isospin factor with a negative-binding-energy selection, which is fully
  specified and which we can evaluate exactly with the verified QMD energy.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from clustering.baselines.qmd_energy import cluster_energy
from clustering.physics import _pairwise_lorentz_boost

# (A, Z) of each species, in the order the chain must be applied.
SPECIES_CHAIN: Sequence[Tuple[str, int, int]] = (
    ("He4", 4, 2),
    ("t", 3, 1),
    ("He3", 3, 2),
    ("d", 2, 1),
)

# Table 3 of the paper: (dr [fm], dp [GeV/c]) per species.
CUT_SETS: Dict[str, Dict[str, Tuple[float, float]]] = {
    # M1 - imported from Reichert, Omana Kuttan et al., arXiv:2504.17389
    "M1": {"d": (4.0, 0.33), "t": (3.5, 0.45), "He3": (3.5, 0.45), "He4": (3.5, 0.55)},
    # M2 - "simulates the classical MST-based clusterization parameters"
    "M2": {"d": (2.5, 0.285), "t": (2.5, 0.285), "He3": (2.5, 0.285), "He4": (2.5, 0.285)},
    "M3": {"d": (4.0, 0.3), "t": (4.0, 0.3), "He3": (4.0, 0.3), "He4": (4.0, 0.3)},
    "M4": {"d": (4.0, 0.33), "t": (2.5, 0.285), "He3": (2.5, 0.285), "He4": (3.5, 0.3)},
}


@dataclass
class CoalescenceResult:
    labels: np.ndarray                     # cluster id per nucleon, -1 = free
    counts: Dict[str, int] = field(default_factory=dict)
    n_rejected: int = 0                    # candidates dropped by the E_bind cut


class _Pool:
    """The particle pool, shrinking as the chain consumes nucleons.

    The pairwise separations in each pair's own COM frame are computed once, in
    one vectorized boost, and reused for every species: the chain re-scans the
    survivors up to a few dozen times per event, and re-boosting every pair each
    time is what would make this unusably slow.
    """

    def __init__(self, x: np.ndarray, type_index: int = 7) -> None:
        self.x = x
        self.p = x[:, 0:3].astype(np.float64)
        self.E = x[:, 3].astype(np.float64)
        self.r = x[:, 4:7].astype(np.float64)
        self.is_p = x[:, type_index] > 0
        self.alive = np.ones(len(x), dtype=bool)

        t = torch.tensor(x, dtype=torch.float64).unsqueeze(0)
        p_b, r_b = _pairwise_lorentz_boost(t[..., 0:3], t[..., 3], t[..., 4:7])
        # p_b[0, i, j] is p_i boosted into the COM frame of the pair (i, j).
        self.DP = (p_b - p_b.transpose(1, 2)).norm(dim=-1)[0].numpy()
        self.DR = (r_b - r_b.transpose(1, 2)).norm(dim=-1)[0].numpy()

    def aggregate(self, members: List[int]) -> Tuple[np.ndarray, float, np.ndarray]:
        """A candidate cluster as a pseudo-particle: p summed, r averaged."""
        return (self.p[members].sum(0), float(self.E[members].sum()),
                self.r[members].mean(0))

    def sep_to(self, idx: np.ndarray, p_c: np.ndarray, E_c: float, r_c: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray]:
        """(dr, dp) of every particle in ``idx`` against one pseudo-particle.

        Vectorized over ``idx``; each pair gets its own COM boost, as the paper
        specifies, with the cluster treated as a single object.
        """
        p_i, E_i, r_i = self.p[idx], self.E[idx], self.r[idx]
        beta = (p_i + p_c) / np.maximum(E_i + E_c, 1e-12)[:, None]
        b2 = np.clip((beta ** 2).sum(1), 0.0, 1.0 - 1e-12)
        gamma = 1.0 / np.sqrt(1.0 - b2)
        bmag = np.sqrt(np.maximum(b2, 1e-300))
        bhat = beta / bmag[:, None]

        def boost(p, E, r):
            p_par = (p * bhat).sum(1)
            r_par = (r * bhat).sum(1)
            p_new = (p - p_par[:, None] * bhat) + (gamma * (p_par - bmag * E))[:, None] * bhat
            r_new = (r - r_par[:, None] * bhat) + (gamma * r_par)[:, None] * bhat
            return p_new, r_new

        p_a, r_a = boost(p_i, E_i, r_i)
        p_b, r_b = boost(np.broadcast_to(p_c, p_i.shape).copy(),
                         np.full(len(idx), E_c),
                         np.broadcast_to(r_c, r_i.shape).copy())
        still = b2 < 1e-16
        dr = np.linalg.norm(r_a - r_b, axis=1)
        dp = np.linalg.norm(p_a - p_b, axis=1)
        if still.any():   # degenerate zero-velocity pairs: no boost needed
            dr[still] = np.linalg.norm(r_i[still] - r_c, axis=1)
            dp[still] = np.linalg.norm(p_i[still] - p_c, axis=1)
        return dr, dp


def _find_one(pool: _Pool, A: int, Z: int, dr_max: float, dp_max: float,
              accept: Optional[Callable[[List[int]], bool]] = None
              ) -> Optional[List[int]]:
    """Greedily grow one cluster of the requested composition, or return None.

    Seeds are tried tightest-first in relative momentum: deterministic, and it
    prefers the most strongly correlated pairs — the ones the spin-isospin
    factor would most likely keep.  ``accept`` rejects a completed candidate
    (the paper's E_bind selection), and the search then continues with the next
    seed rather than abandoning the species.
    """
    idx = np.flatnonzero(pool.alive)
    if len(idx) < A:
        return None
    n_p_needed, n_n_needed = Z, A - Z

    sub_dp = pool.DP[np.ix_(idx, idx)]
    sub_dr = pool.DR[np.ix_(idx, idx)]
    ok = (sub_dr <= dr_max) & (sub_dp <= dp_max)
    ok[np.tril_indices(len(idx))] = False
    ai, bi = np.nonzero(ok)
    if len(ai) == 0:
        return None
    order = np.argsort(sub_dp[ai, bi], kind="stable")

    for s in order:
        i, j = int(idx[ai[s]]), int(idx[bi[s]])
        npr = int(pool.is_p[i]) + int(pool.is_p[j])
        if npr > n_p_needed or (2 - npr) > n_n_needed:
            continue
        members = [i, j]
        while len(members) < A:
            p_c, E_c, r_c = pool.aggregate(members)
            need_p = n_p_needed - sum(int(pool.is_p[m]) for m in members)
            need_n = n_n_needed - sum(1 - int(pool.is_p[m]) for m in members)
            cand = np.array([k for k in idx if k not in members
                             and ((pool.is_p[k] and need_p > 0)
                                  or (not pool.is_p[k] and need_n > 0))], dtype=int)
            if len(cand) == 0:
                break
            dr, dp = pool.sep_to(cand, p_c, E_c, r_c)
            passing = np.flatnonzero((dr <= dr_max) & (dp <= dp_max))
            if len(passing) == 0:
                break
            members.append(int(cand[passing[np.argmin(dp[passing])]]))
        if len(members) == A and (accept is None or accept(members)):
            return members
    return None


def coalescence_clusters(event, *, cut_set: str = "M1", selection: str = "none",
                         type_index: int = 7) -> CoalescenceResult:
    """Run the coalescence chain on one event.  ``event`` is (N, 8)."""
    if cut_set not in CUT_SETS:
        raise ValueError(f"unknown cut set {cut_set!r}; expected one of {list(CUT_SETS)}")
    if selection not in ("none", "ebind"):
        raise ValueError(f"unknown selection {selection!r}; expected 'none' or 'ebind'")

    x = event.cpu().numpy().astype(np.float64) if hasattr(event, "cpu") else np.asarray(event)
    labels = np.full(len(x), -1, dtype=np.int64)
    counts: Dict[str, int] = {}
    if len(x) < 2:
        return CoalescenceResult(labels=labels, counts=counts)

    pool = _Pool(x, type_index)
    cuts = CUT_SETS[cut_set]
    next_id, n_rejected = 0, 0
    rejected: List[int] = [0]

    def accept(members: List[int]) -> bool:
        if selection == "none":
            return True
        # The paper's "mixed" procedure: keep only candidates that are bound.
        if cluster_energy(x[members]).total < 0.0:
            return True
        rejected[0] += 1
        return False

    for name, A, Z in SPECIES_CHAIN:
        dr_max, dp_max = cuts[name]
        while True:
            members = _find_one(pool, A, Z, dr_max, dp_max, accept)
            if members is None:
                break
            for m in members:
                pool.alive[m] = False
                labels[m] = next_id
            counts[name] = counts.get(name, 0) + 1
            next_id += 1

    return CoalescenceResult(labels=labels, counts=counts, n_rejected=rejected[0])


class CoalescenceBaseline:
    """Coalescence fragments in the shape Stage 4's report consumes.

    Same output contract as ``MSTDecayBaseline`` and ``SACABaseline``, so all
    three produce directly comparable plots, yields and traces.
    """

    def __init__(self, lut, *, cut_set: str = "M1", selection: str = "none",
                 decay: bool = False, emit_rule: str = "hottest",
                 type_index: int = 7, trace: bool = False, **_ignored) -> None:
        from clustering.baselines.mst_decay import decay_to_table
        self._decay_to_table = decay_to_table
        self.lut = lut
        self.cut_set = cut_set
        self.selection = selection
        self.decay = decay
        self.emit_rule = emit_rule
        self.type_index = type_index
        self.trace = trace
        self.counts: Dict[str, int] = {}
        self.n_rejected = 0

    @torch.no_grad()
    def __call__(self, event: torch.Tensor):
        from clustering.baselines.mst_decay import BaselineResult, _AZ
        res = coalescence_clusters(event, cut_set=self.cut_set,
                                   selection=self.selection,
                                   type_index=self.type_index)
        for k, v in res.counts.items():
            self.counts[k] = self.counts.get(k, 0) + v
        self.n_rejected += res.n_rejected

        labels = torch.from_numpy(res.labels).to(event.device)
        primaries = [event[labels == c] for c in labels.unique() if c >= 0]
        # Everything the chain did not bind stays a free nucleon.
        primaries += [event[i].unsqueeze(0) for i in torch.nonzero(labels < 0).flatten()]
        n_in_table = sum(1 for f in primaries
                         if self.lut.is_stable(*_AZ(f, self.type_index)))

        fragments, steps, n_evap = [], [], 0
        for j, f in enumerate(primaries):
            if self.decay:
                frag_trace = [] if self.trace else None
                pieces, n = self._decay_to_table(
                    f, self.lut, emit_rule=self.emit_rule,
                    type_index=self.type_index, trace=frag_trace)
                fragments += pieces
                n_evap += n
                if frag_trace:
                    for st in frag_trace:
                        st.fragment = j
                    steps += frag_trace
            else:
                fragments.append(f)

        return BaselineResult(fragments=fragments, steps=steps,
                              n_primary=len(primaries),
                              n_primary_in_table=n_in_table, n_evaporated=n_evap)
