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
# NB: on spectator matter the momentum cut is inactive — nucleons within d_cut
# are co-moving, so every such pair already satisfies |Δp| < 250 MeV/c and MSTp
# gives labels identical to coordinate-only MST.


@torch.no_grad()
def mst_clusters(
    x: torch.Tensor,        # (B, N, 8)
    mask: torch.Tensor,     # (B, N)  True = real nucleon
    d_cut: float = D_CUT,
    p_cut: float = P_CUT,
    use_momentum: bool = True,
) -> torch.Tensor:
    """MST / MSTp cluster labels.

    Returns:
        labels (B, N) int64 — connected-component id per nucleon, ``-1`` for
        padding.  Ids are only meaningful within an event (arbitrary ordering),
        which is why the pretraining objective below is pairwise.
    """
    B, N, _ = x.shape
    labels = torch.full((B, N), -1, dtype=torch.long, device=x.device)
    r = x[..., 4:7]
    p = x[..., 0:3]

    for b in range(B):
        idx = torch.nonzero(mask[b]).flatten()
        if idx.numel() == 0:
            continue
        rr = r[b, idx]
        adj = torch.cdist(rr, rr) < d_cut
        if use_momentum:
            adj = adj & (torch.cdist(p[b, idx], p[b, idx]) < p_cut)
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
        use_momentum: bool = True,
        balance_classes: bool = True,
        grad_clip: float = 1.0,
    ) -> None:
        self.model = model.to(device)
        self.dataloader = dataloader
        self.optim = optimizer
        self.device = device
        self.d_cut = d_cut
        self.p_cut = p_cut
        self.use_momentum = use_momentum
        self.balance_classes = balance_classes
        self.grad_clip = grad_clip

    def _step(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float]:
        self.model.train()
        self.optim.zero_grad()

        labels = mst_clusters(x, mask, self.d_cut, self.p_cut, self.use_momentum)
        same, valid = pair_targets(labels, mask)
        if not valid.any():
            return 0.0, 0.0

        attn = self.model.soft_assign(x, mask)               # (B, K, N)
        p_same = pair_probabilities(attn).clamp(1e-6, 1 - 1e-6)  # (B, N, N)

        p, t = p_same[valid], same[valid]
        if self.balance_classes:
            # The pair target is intrinsically imbalanced (the "same" class grows
            # with d_cut).  Unweighted, the majority class alone can drive the
            # model to a constant prediction — at d_cut=3.0 that means "never
            # split", which destroys the policy before RL ever starts.
            pos = t.mean().clamp(1e-6, 1 - 1e-6)
            w = torch.where(t > 0.5, 0.5 / pos, 0.5 / (1 - pos))
            loss = F.binary_cross_entropy(p, t, weight=w)
        else:
            loss = F.binary_cross_entropy(p, t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optim.step()

        acc = ((p_same[valid] > 0.5).float() == same[valid]).float().mean()
        return float(loss.item()), float(acc.item())

    def train(self, n_epochs: int, verbose: bool = True, log_every: int = 1) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {"loss": [], "pair_acc": []}
        for ep in range(1, n_epochs + 1):
            losses, accs = [], []
            for batch in self.dataloader:
                l, a = self._step(
                    batch["x"].to(self.device), batch["mask"].to(self.device)
                )
                losses.append(l)
                accs.append(a)
            history["loss"].append(float(np.mean(losses)))
            history["pair_acc"].append(float(np.mean(accs)))
            if verbose and ep % log_every == 0:
                print(
                    f"[MST] {ep:4d}/{n_epochs}  "
                    f"loss={history['loss'][-1]:.4f}  "
                    f"pair_acc={history['pair_acc'][-1]:.3f}"
                )
        return history


@torch.no_grad()
def mst_stats(dataloader: DataLoader, d_cut: float = D_CUT, p_cut: float = P_CUT,
              use_momentum: bool = True) -> Dict[str, float]:
    """Fragment-multiplicity summary of the MST labelling (sanity check the cuts)."""
    n_frags, sizes = [], []
    for batch in dataloader:
        labels = mst_clusters(batch["x"], batch["mask"], d_cut, p_cut, use_momentum)
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
