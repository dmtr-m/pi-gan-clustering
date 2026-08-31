"""Baseline: classical MST cluster recognition + secondary decay to the nuclei table.

This is the reference procedure the learned splitter has to beat.  It has no
parameters to train — everything is a cut or a table lookup — so it fixes the
scale on every Stage-3 observable (fragments/event, the A-vs-Z map, max charge)
before any model is involved.

Two stages, mirroring how transport codes are actually post-processed:

1. **Primary fragments** — connected components of the MST graph
   (``clustering.split_prediction.mst.mst_clusters``), under one of three link
   criteria: ``"coord"`` (|Δr| < ``d_cut``), ``"mstp"`` (that *and* |Δp| <
   ``p_cut`` in the pair rest frame), or ``"momentum"`` (|Δp| alone, coordinates
   ignored).

2. **Secondary decay** — a primary fragment is emitted only if its ``(A, Z)``
   appears in the known-nuclei table; otherwise it evaporates one nucleon and the
   remainder is re-tested.  MST cuts on phase-space proximity alone and has no
   notion of which nuclei exist, so it routinely produces species like ²n or a
   Z/A far off the valley; this is the step that removes them.

Why the pipeline needs this at all: `STABILITY_VALLEY_BRAINSTORM.md` records that
the model emits **primary fragments only** while experimental yields are
post-de-excitation, and calls that "the single largest known structural
difference from the baseline".  The decay here is the crude version of that —
driven by the table rather than by an excitation energy, since no E* term exists
in this codebase yet.

Conservation is exact by construction: decay only *moves* nucleons from a
fragment into free-nucleon fragments, so ΣA, ΣZ and Σp over the output equal the
input event.  Stage 3's conservation plots should read identically zero.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from clustering.physics import weizsacker_formula
from clustering.split_prediction.mst import D_CUT, P_CUT, P_FRAME, mst_clusters
from clustering.stability.lookup import StabilityLookup

NUCLEON_MASS = 0.938272  # GeV/c² — the parquet stores E and p in GeV


@dataclass
class DecayStep:
    """One evaporation step, recorded verbatim so the rule can be audited.

    Every field is what the code actually looked at when it made the decision,
    in the order it looked at it: the parent (A, Z) and its table verdict, both
    candidate daughters with their own verdicts and Weizsacker binding, which
    channel won, and which physical nucleon left.
    """
    event: int = -1          # index within the visualization pass
    fragment: int = -1       # index of the primary fragment inside that event
    step: int = 0            # 0-based, counting evaporations from this primary
    A: int = 0
    Z: int = 0
    in_table: bool = False   # always False for a step that happened
    n_daughter_in_table: bool = False   # (A-1, Z)
    n_daughter_binding: float = 0.0
    p_daughter_in_table: bool = False   # (A-1, Z-1)
    p_daughter_binding: float = 0.0
    emitted: str = ""        # "n" or "p"
    emitted_reason: str = "" # which of the two tie-breaks decided it
    emitted_index: int = -1  # row of the emitted nucleon within the fragment
    emitted_ke_mev: float = 0.0   # its kinetic energy in the fragment rest frame
    A_after: int = 0
    Z_after: int = 0
    after_in_table: bool = False


@dataclass
class BaselineResult:
    """Duck-type compatible with ``identifier.FragmentResult`` so Stage 3 can
    consume either without branching."""
    fragments: List[torch.Tensor]
    split_depths: List[int] = field(default_factory=list)
    degenerate_leaves: int = 0
    n_primary: int = 0            # fragments before decay
    n_primary_in_table: int = 0   # of those, already a known nucleus
    n_evaporated: int = 0         # nucleons emitted by the decay stage
    steps: List[DecayStep] = field(default_factory=list)  # empty unless tracing


def _AZ(frag: torch.Tensor, type_index: int = 7) -> Tuple[int, int]:
    return int(frag.shape[0]), int((frag[:, type_index] == 1).sum().item())


def _binding(A: int, Z: int) -> float:
    """Weizsäcker B(A, Z) [MeV].  0 for A < 2, where the liquid drop is meaningless."""
    if A < 2:
        return 0.0
    return float(weizsacker_formula(torch.tensor([float(A)]), torch.tensor([float(Z)])).item())


def _kinetic_in_rest_frame(frag: torch.Tensor) -> torch.Tensor:
    """Per-nucleon kinetic energy [GeV] in the fragment's own rest frame.

    Single boost with β = ΣP/ΣE (the fragment's velocity), then T = E' − m.
    Ranking by this is what makes the emitted nucleon the *hottest* one rather
    than an arbitrary member of the right isospin.
    """
    p = frag[:, 0:3]
    E = frag[:, 3]
    beta = p.sum(0) / E.sum().clamp(min=1e-8)                    # (3,)
    beta_sq = float((beta @ beta).clamp(max=1 - 1e-8))
    gamma = 1.0 / (1.0 - beta_sq) ** 0.5
    # E' = γ(E − β·p)
    E_rest = gamma * (E - p @ beta)
    return E_rest - NUCLEON_MASS


def _emission_index(frag: torch.Tensor, want_proton: bool, rule: str,
                    type_index: int = 7) -> int:
    """Which physical nucleon of the requested species leaves the fragment."""
    is_proton = frag[:, type_index] == 1
    candidates = torch.nonzero(is_proton if want_proton else ~is_proton).flatten()
    if rule == "hottest":
        score = _kinetic_in_rest_frame(frag)[candidates]
    elif rule == "outermost":
        r = frag[:, 4:7]
        score = (r[candidates] - r.mean(0)).norm(dim=-1)
    elif rule == "first":
        return int(candidates[0].item())
    else:
        raise ValueError(f"unknown emit rule {rule!r}; expected 'hottest', 'outermost' or 'first'")
    return int(candidates[int(score.argmax().item())].item())


def decay_to_table(
    frag: torch.Tensor,
    lut: StabilityLookup,
    *,
    emit_rule: str = "hottest",
    type_index: int = 7,
    trace: Optional[List[DecayStep]] = None,
) -> Tuple[List[torch.Tensor], int]:
    """Evaporate nucleons until every piece is a known nucleus (or a free nucleon).

    One step, applied to a fragment whose ``(A, Z)`` is not in the table:

    1. The two open channels are neutron emission → ``(A−1, Z)`` and proton
       emission → ``(A−1, Z−1)``.  Pick the daughter that **is in the table**;
       if both are (or neither is), pick the one with the larger Weizsäcker
       binding — equivalently the smaller separation energy, since ``B(A, Z)`` is
       common to both channels.
    2. The nucleon that actually leaves is the most energetic one of that species
       in the fragment rest frame (``emit_rule``), and is emitted as a free
       nucleon.

    Repeat on the remainder.  ``A`` strictly decreases, so this always terminates
    — at worst at ``A = 1``, which is emitted as a free nucleon.

    Returns:
        (fragments, n_evaporated).
    """
    out: List[torch.Tensor] = []
    n_evaporated = 0
    step = 0
    stack = [frag]
    while stack:
        f = stack.pop()
        A, Z = _AZ(f, type_index)
        # Step 1 — open the table.  In it (or a free nucleon): emit, done.
        if A <= 1 or lut.is_stable(A, Z):
            out.append(f)
            continue

        # Step 2 — not in the table, so it decays.  Two channels are open.
        N = A - Z
        n_ok = n_b = p_ok = p_b = None
        channels = []
        if N > 0:
            n_ok, n_b = lut.is_stable(A - 1, Z), _binding(A - 1, Z)
            channels.append((False, (n_ok, n_b)))
        if Z > 0:
            p_ok, p_b = lut.is_stable(A - 1, Z - 1), _binding(A - 1, Z - 1)
            channels.append((True, (p_ok, p_b)))
        # (in_table, binding) — lexicographic, higher is better.
        want_proton, _ = max(channels, key=lambda c: c[1])

        # Step 3 — pick the physical nucleon of that species and emit it.
        i = _emission_index(f, want_proton, emit_rule, type_index)
        keep = torch.ones(A, dtype=torch.bool, device=f.device)
        keep[i] = False
        out.append(f[~keep])          # the free nucleon
        n_evaporated += 1
        stack.append(f[keep])         # the remainder, re-tested next iteration

        if trace is not None:
            a2, z2 = A - 1, (Z - 1 if want_proton else Z)
            if n_ok is None or p_ok is None:
                reason = "only channel open"
            elif bool(n_ok) != bool(p_ok):
                reason = "table membership"
            else:
                reason = "binding (table tied)"
            trace.append(DecayStep(
                step=step, A=A, Z=Z, in_table=False,
                n_daughter_in_table=bool(n_ok) if n_ok is not None else False,
                n_daughter_binding=float(n_b) if n_b is not None else float("nan"),
                p_daughter_in_table=bool(p_ok) if p_ok is not None else False,
                p_daughter_binding=float(p_b) if p_b is not None else float("nan"),
                emitted="p" if want_proton else "n",
                emitted_reason=reason,
                emitted_index=i,
                emitted_ke_mev=float(_kinetic_in_rest_frame(f)[i].item()) * 1000.0,
                A_after=a2, Z_after=z2, after_in_table=lut.is_stable(a2, z2),
            ))
        step += 1
    return out, n_evaporated


class MSTDecayBaseline:
    """MST (or MSTp) primary fragments, optionally decayed onto the nuclei table.

    Call it on one event ``(N, 8)`` and it returns a :class:`BaselineResult`
    whose ``fragments`` field has the same shape and meaning as
    ``FragmentsIdentifier``'s, so both feed the same Stage-3 report.
    """

    def __init__(
        self,
        lut: StabilityLookup,
        *,
        d_cut: float = D_CUT,
        p_cut: float = P_CUT,
        metric: str = "mstp",
        p_frame: str = P_FRAME,
        decay: bool = True,
        emit_rule: str = "hottest",
        type_index: int = 7,
        trace: bool = False,
    ) -> None:
        self.lut = lut
        self.d_cut = d_cut
        self.p_cut = p_cut
        self.metric = metric
        self.p_frame = p_frame
        self.decay = decay
        self.emit_rule = emit_rule
        self.type_index = type_index
        self.trace = trace

    @torch.no_grad()
    def __call__(self, event: torch.Tensor) -> BaselineResult:
        # mst_clusters is batched; one event is B = 1 with an all-true mask.
        x = event.unsqueeze(0)
        mask = torch.ones(1, event.shape[0], dtype=torch.bool, device=event.device)
        labels = mst_clusters(x, mask, self.d_cut, self.p_cut,
                              self.metric, self.p_frame)[0]

        primaries = [event[labels == c] for c in labels.unique() if c >= 0]
        n_in_table = sum(
            1 for f in primaries if self.lut.is_stable(*_AZ(f, self.type_index))
        )

        fragments: List[torch.Tensor] = []
        steps: List[DecayStep] = []
        n_evaporated = 0
        for j, f in enumerate(primaries):
            if self.decay:
                frag_trace: Optional[List[DecayStep]] = [] if self.trace else None
                pieces, n_evap = decay_to_table(
                    f, self.lut, emit_rule=self.emit_rule,
                    type_index=self.type_index, trace=frag_trace,
                )
                fragments += pieces
                n_evaporated += n_evap
                if frag_trace:
                    for st in frag_trace:
                        st.fragment = j
                    steps += frag_trace
            else:
                fragments.append(f)

        return BaselineResult(
            fragments=fragments,
            steps=steps,
            n_primary=len(primaries),
            n_primary_in_table=n_in_table,
            n_evaporated=n_evaporated,
        )
