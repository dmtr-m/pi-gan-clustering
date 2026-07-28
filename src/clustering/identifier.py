import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import List, Tuple

from clustering.stability.lookup import StabilityLookup
from clustering.split_prediction.model import SplitPredictionModel


@dataclass
class FragmentResult:
    """Result of hierarchical fragment identification."""
    fragments: List[torch.Tensor]  # each (n_i, input_dim) -- nucleons of a stable fragment
    log_prob: torch.Tensor         # scalar log prob of all decisions, for REINFORCE
    split_depths: List[int] = field(default_factory=list)  # depth of each split performed
    degenerate_leaves: int = 0     # leaves emitted because no split was possible


class FragmentsIdentifier(nn.Module):
    """
    Fragments identification module receives set of nucleon and
    hierarchicaly clusters it.

    During inference model receives set of nucleons, at each step
    model:
        1. Looks up whether the fragment's (A, Z) is a known stable nucleus.
        2. if it is not in the table, model splits nucleons into up to
           n_clusters groups and calls itself on every non-empty child;
           if it is, model stops and emits this fragment.

    Depth of the tree is limited by hyperparameter max_depth.

    Returns FragmentResult containing the list of stable fragment tensors and
    the total log probability of the split-assignment decisions (for REINFORCE
    training).  The stop/split decision is a deterministic table lookup and
    contributes no log-probability.

    Args:
        csv_path: Path to the known-nuclei CSV backing the stability lookup.
            Ignored (and may be None) when ``force_split`` is True.
        force_split: If True, bypass the stability lookup and always split
            until max_depth.
    """
    def __init__(
        self,
        max_depth: int = 8,
        input_dim: int = 8,
        type_index: int = 7,       # column index of particle type (1=proton, -1=neutron)
        hidden_dim: int = 64,
        n_iters: int = 3,
        n_clusters: int = 2,       # branching factor of a single split
        csv_path: "str | None" = None,
        force_split: bool = False,
    ) -> None:
        super().__init__()
        self.max_depth = max_depth
        self.type_index = type_index
        self.force_split = force_split
        self.stability = None if force_split else StabilityLookup(csv_path)
        self.split_prediction_module = SplitPredictionModel(input_dim, hidden_dim, n_iters, n_clusters)

    def _force_split(self, x: torch.Tensor) -> "List[torch.Tensor] | None":
        """Minimal 2-group split when slot attention collapsed everything into one.

        Peels off the single nucleon with the strongest pull toward any *other*
        slot, giving the least-disruptive non-degenerate split.  Recursion then
        continues on both parts, so e.g. an unbound dineutron becomes two free
        neutrons instead of being emitted as a bogus ²n fragment.

        Returns None if the fragment is too small to split (N < 2).
        """
        N = x.shape[0]
        if N < 2:
            return None
        with torch.no_grad():
            mask = torch.ones(1, N, dtype=torch.bool, device=x.device)
            attn = self.split_prediction_module.soft_assign(x.unsqueeze(0), mask)  # (1,K,N)
        attn = attn.squeeze(0)                      # (K, N)
        winner = int(attn.argmax(dim=0)[0].item())  # the slot that took everything
        rival = attn.clone()
        rival[winner] = -1.0                        # look only at other slots
        # Nucleon with the highest affinity to any non-winning slot.
        n_star = int(rival.max(dim=0).values.argmax().item())
        peeled = torch.zeros(N, dtype=torch.bool, device=x.device)
        peeled[n_star] = True
        return [~peeled, peeled]

    def forward(self, x: torch.Tensor, depth: int = 0) -> FragmentResult:
        """
        Args:
            x:     (N, input_dim) nucleon features for the current fragment
            depth: current recursion depth (managed internally)
        Returns:
            FragmentResult with stable leaf fragments and accumulated log prob
        """
        N = x.shape[0]
        device = x.device
        zero = torch.tensor(0.0, device=device)

        # Base cases: single nucleon or depth limit
        if N <= 1 or depth >= self.max_depth:
            return FragmentResult(fragments=[x], log_prob=zero)

        # Stop / split decision: a deterministic lookup of the fragment's
        # (A, Z) in the known-nuclei table.  Stable -> stop; otherwise split.
        # The lookup carries no gradient, so it adds nothing to the log prob.
        split_log_prob = zero
        if self.force_split:
            should_split = True
        else:
            A = int(N)
            Z = int((x[:, self.type_index] == 1).sum().item())
            should_split = not self.stability.is_stable(A, Z)

        if not should_split:
            return FragmentResult(fragments=[x], log_prob=split_log_prob)

        # SplitPredictionModel expects (B, N, D) with a boolean mask.
        x_bat = x.unsqueeze(0)                                         # (1, N, D)
        node_mask = torch.ones(1, N, dtype=torch.bool, device=device)  # (1, N)
        assignments_bat, log_probs_bat = self.split_prediction_module(x_bat, mask=node_mask)
        assignments = assignments_bat.squeeze(0)     # (N,)
        assign_log_probs = log_probs_bat.squeeze(0)  # (N,)

        # Collect the non-empty child groups of the up-to-K-way split.
        child_masks = [
            assignments == c for c in range(self.split_prediction_module.n_clusters)
        ]
        child_masks = [m for m in child_masks if m.any()]

        # Guard: degenerate split puts all nucleons in a single group.
        # Previously this silently returned the fragment as a leaf — which meant
        # *every* multi-nucleon leaf was really "wherever the splitter gave up",
        # emitted regardless of whether it was physical.  The classifier already
        # said this fragment should split, so honour that: force a minimal split
        # by peeling off the nucleon most inclined toward another slot.
        if len(child_masks) < 2:
            child_masks = self._force_split(x)
            if child_masks is None:
                return FragmentResult(
                    fragments=[x], log_prob=split_log_prob, degenerate_leaves=1
                )

        cluster_log_prob = split_log_prob + assign_log_probs.sum()

        fragments: List[torch.Tensor] = []
        total_log_prob = cluster_log_prob
        split_depths: List[int] = [depth]  # this node performed a (non-degenerate) split
        n_degenerate = 0
        for child_mask in child_masks:
            child_result = self.forward(x[child_mask], depth + 1)
            fragments += child_result.fragments
            total_log_prob = total_log_prob + child_result.log_prob
            split_depths += child_result.split_depths
            n_degenerate += child_result.degenerate_leaves

        return FragmentResult(
            fragments=fragments,
            log_prob=total_log_prob,
            split_depths=split_depths,
            degenerate_leaves=n_degenerate,
        )
