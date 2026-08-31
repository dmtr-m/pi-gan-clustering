"""Classical MST cluster recognition + supervised pretraining of the splitter.

Asking REINFORCE to discover clustering structure from scratch is a bad deal
here: only ~6% of the per-node reward variance is attributable to the *action*
(the rest is state), so the policy-gradient signal is heavily diluted.  Nuclear
physics already has cheap, established cluster-recognition algorithms, so we use
one to generate pseudo-labels and warm-start the splitter supervised, then
fine-tune with REINFORCE on the true physics reward.

MST (Minimum Spanning Tree / proximity clustering): two nucleons are bound into
the same pre-fragment if they lie within ``d_cut`` in coordinate space.  The
momentum-extended variant (MSTp) additionally requires their relative momentum
to be below ``p_cut``.  Fragments are the connected components of that graph.

Relative momentum is evaluated in the rest frame of the *pair* by default
(``p_frame="pair_cm"``): the source here is a spectator remnant carrying ~1.1
GeV/c per nucleon in the lab, and a boost stretches longitudinal momentum
differences by gamma ~ 1.5, so a lab-frame |dp| is not the quantity the p_cut
literature value refers to.  Measured on HSE SpectatorsLeft pairs within 2 fm:
median |dp| is 331 MeV/c in the lab against 280 MeV/c in the pair CM.

Refs: arXiv:1009.5452, arXiv:1706.01300.
"""
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch.utils.data import DataLoader

from clustering.physics import MOMENTUM_TO_MEV, _pairwise_lorentz_boost
from clustering.split_prediction.model import SplitPredictionModel

# Cluster-recognition cuts.
#
# The literature default is d_cut ≈ 3–4 fm, but on this UrQMD spectator source
# that *percolates*: at 3.0 fm the largest MST fragment holds ~82% of the
# nucleons (95% at 4.0 fm), so ~82% of pairs are labelled "same cluster" and a
# pretrained splitter collapses to "never split".  Measured percolation scan
# (frags/event, largest fragment, % pairs same):
#     1.5 fm -> 13.0,  28%, 14%
#     2.0 fm ->  7.8,  49%, 39%   <- near the transition, balanced target
#     2.5 fm ->  4.4,  71%, 69%
#     3.0 fm ->  2.9,  82%, 82%
#     4.0 fm ->  1.5,  95%, 94%
# 2.0 fm is the default: it breaks the giant component and gives a near-balanced
# target.  Measured downstream (pretrain -> RL fine-tune, 3 seeds, final eval
# reward vs RL-from-scratch): d=3.0 is catastrophic (-4.1 / -7.0, the collapse
# above); d=2.0 is break-even; d=1.5 is best (+0.77 at k=1/K=5).  Pretraining's
# main benefit is reliability — it cuts seed-to-seed std ~8x (±0.23 vs ±1.82).
D_CUT = 2.0    # fm
P_CUT = 250.0  # MeV/c
P_FRAME = "pair_cm"   # frame the relative momentum is measured in: "pair_cm" | "lab"

# `metric` defaults to **"coord"**, i.e. coordinate-only MST.
#
# The momentum cut used to be on by default with this note attached: "on spectator matter the
# momentum cut is inactive — nucleons within d_cut are co-moving, so every such
# pair already satisfies |Δp| < 250 MeV/c and MSTp gives labels identical to
# coordinate-only MST".  That was a description of a unit bug, not of the
# physics: the parquet stores momenta in **GeV/c** and this cut is in **MeV/c**,
# so `cdist(p) < 250` was comparing ~0.3 against 250 and was true for every pair
# on the grid.  Same class of bug as the Pauli potential (commit aa657d6).
#
# With the conversion applied, the cut is very much active — measured on HSE
# SpectatorsLeft, over pairs already within d_cut = 2 fm:
#     p_cut [MeV/c]   100    150    200    250    300    400
#     frac kept       0.045  0.121  0.241  0.399  0.567  0.846
# so MSTp at the literature 250 MeV/c would drop ~60% of the spatial bonds.
#
# Every MST warm-start run to date therefore used coordinate-only labels, and
# the d_cut percolation scan documented above was measured that way.  Keeping
# metric="coord" preserves that behaviour exactly; "mstp" and "momentum" are
# opt-in, with a cut that actually fires.


@torch.no_grad()
def relative_momenta(
    nucleons: torch.Tensor,   # (N, 8) one fragment / event, no padding
    frame: str = P_FRAME,
) -> torch.Tensor:
    """Pairwise relative momentum |p_i - p_j| in **MeV/c**.  Returns (N, N).

    ``frame``:
      - ``"pair_cm"``: both momenta are Lorentz-boosted into the rest frame of
        the pair first (the same boost the QMD Pauli term uses).  In that frame
        p_i' = -p_j', so this returns 2|p*| — the same quantity the lab-frame
        expression means to measure, without the beam boost folded in.
      - ``"lab"``: the raw difference as stored.

    The dataset stores momenta in GeV/c; the conversion to MeV/c happens here, so
    every caller compares against ``p_cut`` in the unit the literature quotes.
    """
    p = nucleons[..., 0:3]
    if frame == "lab":
        return torch.cdist(p, p) * MOMENTUM_TO_MEV
    if frame != "pair_cm":
        raise ValueError(f"unknown frame {frame!r}; expected 'pair_cm' or 'lab'")
    p_boosted, _ = _pairwise_lorentz_boost(
        p.unsqueeze(0), nucleons[..., 3].unsqueeze(0), nucleons[..., 4:7].unsqueeze(0)
    )  # (1, N, N, 3)
    return (p_boosted - p_boosted.transpose(1, 2)).norm(dim=-1)[0] * MOMENTUM_TO_MEV


METRICS = ("coord", "mstp", "momentum")


@torch.no_grad()
def mst_clusters(
    x: torch.Tensor,        # (B, N, 8)
    mask: torch.Tensor,     # (B, N)  True = real nucleon
    d_cut: float = D_CUT,
    p_cut: float = P_CUT,
    metric: str = "coord",
    p_frame: str = P_FRAME,
) -> torch.Tensor:
    """MST cluster labels under one of three link criteria.

    ``metric``:
      - ``"coord"`` (default): |Δr| < ``d_cut`` [fm].  Plain MST — what every run
        in this repo has actually used; see the note above ``D_CUT``.
      - ``"mstp"``: |Δr| < ``d_cut`` **and** |Δp| < ``p_cut`` [MeV/c].
      - ``"momentum"``: |Δp| < ``p_cut`` alone, coordinates ignored.  The
        phase-space analogue of coordinate MST for a source whose spatial
        structure you do not trust — here the freeze-out coordinates are the
        generator's, not an observable.  Note the cut has a completely different
        meaning without the spatial condition backing it: it links *every*
        co-moving pair in the event rather than only spatially adjacent ones, so
        it percolates at a far smaller ``p_cut`` (measured: a giant component
        already at 100 MeV/c).

    Returns:
        labels (B, N) int64 — connected-component id per nucleon, ``-1`` for
        padding.  Ids are only meaningful within an event (arbitrary ordering),
        which is why the pretraining objective below is pairwise.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    B, N, _ = x.shape
    labels = torch.full((B, N), -1, dtype=torch.long, device=x.device)
    r = x[..., 4:7]

    for b in range(B):
        idx = torch.nonzero(mask[b]).flatten()
        if idx.numel() == 0:
            continue
        if metric == "momentum":
            adj = relative_momenta(x[b, idx], p_frame) < p_cut
        else:
            rr = r[b, idx]
            adj = torch.cdist(rr, rr) < d_cut
            if metric == "mstp":
                adj = adj & (relative_momenta(x[b, idx], p_frame) < p_cut)
        adj = adj.clone()
        adj.fill_diagonal_(False)
        _, comp = connected_components(
            csr_matrix(adj.cpu().numpy()), directed=False
        )
        labels[b, idx] = torch.from_numpy(comp).long().to(x.device)
    return labels


def pair_targets(
    labels: torch.Tensor,  # (B, N)
    mask: torch.Tensor,    # (B, N)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pairwise co-membership target.

    Returns:
        same  (B, N, N) float — 1 if i and j share an MST fragment.
        valid (B, N, N) bool  — real, off-diagonal pairs only.
    """
    N = labels.shape[1]
    same = (labels.unsqueeze(2) == labels.unsqueeze(1)).float()
    valid = mask.unsqueeze(2) & mask.unsqueeze(1)
    eye = torch.eye(N, dtype=torch.bool, device=labels.device).unsqueeze(0)
    return same, valid & ~eye


def pair_probabilities(attn: torch.Tensor) -> torch.Tensor:
    """P(i and j land in the same slot) from soft assignments.

    Args:
        attn: (B, K, N) softmax over slots.
    Returns:
        (B, N, N) — Σ_k attn[k,i]·attn[k,j].

    This is permutation-invariant in the slot index, which is exactly what we
    need: slot identity is arbitrary, so a direct cross-entropy against MST
    labels would require Hungarian matching.  The pairwise form sidesteps that
    entirely, and degrades gracefully when MST finds more or fewer fragments
    than the model has slots.
    """
    return torch.einsum('bki,bkj->bij', attn, attn)


class MSTPretrainer:
    """Supervised warm-start: match the splitter's soft assignment to MST fragments.

    Loss is a binary cross-entropy on pairwise co-membership over real,
    off-diagonal pairs.  After this, fine-tune the same model instance with
    ``KSplitTrainer`` on the QMD reward.
    """

    def __init__(
        self,
        model: SplitPredictionModel,
        dataloader: DataLoader,
        *,
        optimizer: optim.Optimizer,
        device: str = "cpu",
        d_cut: float = D_CUT,
        p_cut: float = P_CUT,
        metric: str = "coord",
        p_frame: str = P_FRAME,
        balance_classes: bool = True,
        grad_clip: float = 1.0,
    ) -> None:
        self.model = model.to(device)
        self.dataloader = dataloader
        self.optim = optimizer
        self.device = device
        self.d_cut = d_cut
        self.p_cut = p_cut
        self.metric = metric
        self.p_frame = p_frame
        self.balance_classes = balance_classes
        self.grad_clip = grad_clip

        # Per-optimizer-step log, mirroring KSplitTrainer.  Epoch resolution is
        # useless at scale: 2 epochs over 58k events is 912 optimizer steps but
        # only 2 points.  Everything here is already computed per batch.
        self.global_step = 0
        self.epoch_end_steps: List[int] = []
        self.step_history: Dict[str, List[float]] = {
            "step": [], "loss": [], "pair_acc": [], "grad_norm": [], "entropy_frac": [],
        }

    def _step(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[float, float, float, float]:  # loss, pair_acc, grad_norm, entropy_frac
        self.model.train()
        self.optim.zero_grad()

        labels = mst_clusters(x, mask, self.d_cut, self.p_cut,
                              self.metric, self.p_frame)
        same, valid = pair_targets(labels, mask)
        if not valid.any():
            # NaN, not 0: a batch with no real pair should drop out of the plots
            # and the epoch mean rather than read as a genuine zero loss.
            return float("nan"), float("nan"), float("nan"), float("nan")

        attn = self.model.soft_assign(x, mask)               # (B, K, N)
        p_same = pair_probabilities(attn).clamp(1e-6, 1 - 1e-6)  # (B, N, N)

        # Score every pair and weight the invalid ones to zero, instead of
        # gathering with p_same[valid].  The gather form raised an intermittent
        # (~1 run in 3) "target size != input size" ValueError here that we could
        # not reproduce on demand; keeping every tensor at (B, N, N) removes that
        # failure mode by construction, and drops two ~1M-element gathers per
        # batch.  p_same is clamped above, so the log on weight-0 entries is
        # finite and 0 * finite contributes exactly nothing to loss or grad.
        #
        # The assert is what keeps a genuine upstream shape bug loud: without it
        # the weighted form would silently broadcast rather than raise.  It is
        # metadata-only, so it costs nothing per step.
        assert p_same.shape == same.shape == valid.shape, (
            f"pair shape divergence: p_same={tuple(p_same.shape)} "
            f"same={tuple(same.shape)} valid={tuple(valid.shape)}"
        )

        vf = valid.float()
        n_valid = vf.sum().clamp(min=1.0)
        if self.balance_classes:
            # The pair target is intrinsically imbalanced (the "same" class grows
            # with d_cut).  Unweighted, the majority class alone can drive the
            # model to a constant prediction — at d_cut=3.0 that means "never
            # split", which destroys the policy before RL ever starts.
            pos = ((same * vf).sum() / n_valid).clamp(1e-6, 1 - 1e-6)
            w = vf * torch.where(same > 0.5, 0.5 / pos, 0.5 / (1 - pos))
        else:
            w = vf
        # `sum` / n_valid reproduces BCE's weighted `mean` over the valid pairs.
        loss = F.binary_cross_entropy(p_same, same, weight=w, reduction="sum") / n_valid
        loss.backward()
        # clip_grad_norm_ returns the total norm *before* clipping — free diagnostic.
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optim.step()

        acc = (((p_same > 0.5).float() == same).float() * vf).sum() / n_valid
        # Slot-assignment entropy over real nucleons, normalized by log K so it is
        # comparable across n_clusters.  1.0 = uniform (the model is not committing
        # to any assignment); a collapse toward one slot drives it down.
        with torch.no_grad():
            per_nucleon = -(attn * (attn + 1e-9).log()).sum(dim=1)   # (B, N) over slots
            ent = float((per_nucleon * mask.float()).sum() / mask.sum().clamp(min=1)
                        / float(np.log(attn.shape[1])))
        return float(loss.item()), float(acc.item()), float(grad_norm), ent

    def train(self, n_epochs: int, verbose: bool = True, log_every: int = 1) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {"loss": [], "pair_acc": []}
        for ep in range(1, n_epochs + 1):
            losses, accs = [], []
            for batch in self.dataloader:
                l, a, g, h = self._step(
                    batch["x"].to(self.device), batch["mask"].to(self.device)
                )
                losses.append(l)
                accs.append(a)
                self.global_step += 1
                sh = self.step_history
                sh["step"].append(float(self.global_step))
                sh["loss"].append(l)
                sh["pair_acc"].append(a)
                sh["grad_norm"].append(g)
                sh["entropy_frac"].append(h)
            self.epoch_end_steps.append(self.global_step)
            # nanmean: skip batches that had no valid pair at all.
            history["loss"].append(float(np.nanmean(losses)))
            history["pair_acc"].append(float(np.nanmean(accs)))
            if verbose and ep % log_every == 0:
                print(
                    f"[MST] {ep:4d}/{n_epochs}  "
                    f"loss={history['loss'][-1]:.4f}  "
                    f"pair_acc={history['pair_acc'][-1]:.3f}"
                )
        return history


@torch.no_grad()
def mst_stats(dataloader: DataLoader, d_cut: float = D_CUT, p_cut: float = P_CUT,
              metric: str = "coord", p_frame: str = P_FRAME) -> Dict[str, float]:
    """Fragment-multiplicity summary of the MST labelling (sanity check the cuts)."""
    n_frags, sizes = [], []
    for batch in dataloader:
        labels = mst_clusters(batch["x"], batch["mask"], d_cut, p_cut,
                              metric, p_frame)
        for b in range(labels.shape[0]):
            lab = labels[b][batch["mask"][b]]
            if lab.numel() == 0:
                continue
            uniq, counts = torch.unique(lab, return_counts=True)
            n_frags.append(len(uniq))
            sizes.extend(counts.tolist())
    return dict(
        mean_fragments=float(np.mean(n_frags)),
        max_fragments=float(np.max(n_frags)),
        mean_size=float(np.mean(sizes)),
        max_size=float(np.max(sizes)),
        frac_singletons=float(np.mean([s == 1 for s in sizes])),
    )
