import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from typing import Dict, List, Tuple
from torch.utils.data import DataLoader

from functools import partial

from clustering.physics import (
    weizsacker_per_nucleon_formula,
    binding_energy,
    total_potential_energy,
    fragment_energy,
    weizsacker_qmd_energy,
    qmd_minus_b_energy,
    saca_qmd_minus_b_energy,
    zeta_correct_energy,
)
from clustering.split_prediction.model import SplitPredictionModel

# ─── K-level forward pass ─────────────────────────────────────────────────────

def k_level_forward(
    model: SplitPredictionModel,
    x: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    min_fragment_size: int = 2,
) -> Dict:
    """Run k levels of splitting on a padded nucleon batch.

    Each node is split into ``model.n_clusters`` children (the branching
    factor), so a depth-k tree produces up to ``n_clusters**k`` leaf nodes.
    With ``n_clusters == 2`` this is the original binary tree; with ``k == 1``
    it is a single up-to-K-way split.

    Args:
        model: SplitPredictionModel instance (used at every level, weights shared).
        x:     (B, N, D) nucleon features, padded.
        mask:  (B, N)    True for real nucleons.
        k:     number of split levels; produces up to n_clusters**k leaf nodes.

    Empty paths are pruned, and a node is only split while some batch item has
    ≥ ``2 * min_fragment_size`` nucleons there (below that it cannot yield two
    valid fragments) — such nodes become leaves.  Without pruning the BFS
    enumerates ``n_clusters ** k`` paths, which is intractable at depth 8.

    Returns dict with:
        leaf_masks     — {path_tuple: (B,N) bool} for each non-empty leaf.
                         path_tuple is a sequence of cluster indices, e.g. (0,3,1).
        path_log_probs — (B,N) accumulated log P(actual leaf | nucleon path).
        valid          — (B,) bool; True if at least 2 leaves are non-empty,
                         i.e. the root split was non-degenerate for that event.
    """
    B, N, _ = x.shape
    n_clusters = model.n_clusters
    min_splittable = 2 * min_fragment_size

    # Additive accumulation (not in-place indexed assignment) is required to
    # keep path_log_probs in the autograd graph so gradients flow back to the
    # model parameters.
    path_log_probs = torch.zeros(B, N, device=x.device, dtype=x.dtype)

    # BFS: map path-tuple -> (B,N) mask of nucleons at that node
    leaf_masks: Dict[Tuple, torch.Tensor] = {}
    current_level: Dict[Tuple, torch.Tensor] = {(): mask}

    for _ in range(k):
        next_level: Dict[Tuple, torch.Tensor] = {}
        for path, node_mask in current_level.items():
            if not node_mask.any():
                continue  # prune: empty for the whole batch
            if not (node_mask.sum(dim=1) >= min_splittable).any():
                leaf_masks[path] = node_mask  # too small to split anywhere
                continue

            # Zero out nucleons outside this node so CM centering and slot
            # attention operate only on the current sub-group.
            x_node = x * node_mask.unsqueeze(-1)

            assignments, log_probs = model(x_node, node_mask)  # (B,N), (B,N)

            # Additive update: for positions outside node_mask the float mask
            # is 0, so they receive +0 and the gradient stays clean.
            path_log_probs = path_log_probs + log_probs * node_mask.float()

            for c in range(n_clusters):
                child = (assignments == c) & node_mask
                if child.any():
                    next_level[path + (c,)] = child

        current_level = next_level
        if not current_level:
            break

    leaf_masks.update(current_level)  # anything still live at depth k is a leaf

    # Require at least 2 leaves of size >= min_fragment_size per batch item.
    # Requiring ALL leaves to qualify is too strict for wide/deep trees.
    if not leaf_masks:
        return dict(leaf_masks={}, path_log_probs=path_log_probs,
                    valid=torch.zeros(B, dtype=torch.bool, device=x.device))
    leaf_large_enough = torch.stack(
        [m.sum(dim=1) >= min_fragment_size for m in leaf_masks.values()]
    )  # (n_leaves, B)
    valid = leaf_large_enough.sum(dim=0) >= 2

    return dict(leaf_masks=leaf_masks, path_log_probs=path_log_probs, valid=valid)


def iter_split_nodes(
    model: SplitPredictionModel,
    x: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    min_fragment_size: int = 2,
    type_index: int = 7,
    energy_fn=fragment_energy,
):
    """Walk k levels of splitting, **yielding one record per split node**.

    Each node is scored with its own immediate QMD reward:

    ```
    q(node) = ( U(node) − Σ_child U(child) ) / N_node      [MeV/nucleon]
    ```

    This is a generator so the caller can backward and free each node's graph
    *before* the next node is built.  Gradients never chain across levels
    anyway — child masks come from ``multinomial`` samples and are integer/bool,
    so they carry no gradient — but materializing every node's graph at once
    made memory grow with tree size.  Yielding per node keeps the live graph
    bounded to a single split regardless of depth.

    **Pruning** (what makes deep trees tractable): a path is dropped once it is
    empty for the whole batch, and a node is not split unless some batch item
    has ≥ ``2 * min_fragment_size`` nucleons there — below that it cannot
    produce two valid fragments, so splitting it is wasted compute.  Without
    this the BFS enumerates ``n_clusters ** k`` paths (390k at K=5, k=8).

    Yields dicts with:
        q         (B,) local reward, detached.
        log_prob  (B,) Σ_n log p(assignment_n) at this node — in the autograd graph.
        valid     (B,) bool: split into ≥2 fragments of size ≥ min_fragment_size.
        mask      (B,N) node membership.
        depth     int, and critic scalars ``n``/``zfrac``/``u_per_n`` (B,).
    """
    B, N, _ = x.shape
    n_clusters = model.n_clusters
    min_splittable = 2 * min_fragment_size

    current_level: Dict[Tuple, torch.Tensor] = {(): mask}

    for depth in range(k):
        next_level: Dict[Tuple, torch.Tensor] = {}
        for path, node_mask in current_level.items():
            n_node_raw = node_mask.sum(dim=1)
            # Prune: nothing here, or too small anywhere to yield 2 fragments.
            if not (n_node_raw >= min_splittable).any():
                continue

            x_node = x * node_mask.unsqueeze(-1)
            assignments, log_probs = model(x_node, node_mask)  # (B,N), (B,N)
            child_masks = [(assignments == c) & node_mask for c in range(n_clusters)]

            U_node = energy_fn(x, node_mask, type_index)  # (B,)
            U_child = x.new_zeros(B)
            n_large_children = x.new_zeros(B)
            for cm in child_masks:
                big = cm.sum(dim=1) >= min_fragment_size  # (B,)
                n_large_children = n_large_children + big.float()
                if big.any():
                    U_child[big] = U_child[big] + energy_fn(x[big], cm[big], type_index)
            n_node = n_node_raw.float().clamp(min=1)
            q = (U_node - U_child) / n_node  # (B,)
            z = ((x[..., type_index] == 1) & node_mask).sum(dim=1).float()  # (B,)

            yield dict(
                q=q.detach(),
                log_prob=(log_probs * node_mask.float()).sum(dim=1),  # (B,), in graph
                valid=n_large_children >= 2,
                mask=node_mask,
                depth=depth,
                n=n_node,
                zfrac=z / n_node,
                u_per_n=U_node / n_node,
            )

            for c in range(n_clusters):
                if child_masks[c].any():
                    next_level[path + (c,)] = child_masks[c]

        current_level = next_level
        if not current_level:
            break  # fully fragmented; nothing left to split


def node_scalars(rec: Dict, k: int) -> torch.Tensor:
    """Physics summary features for the critic: (B, 4).

    ``[N_node, Z/A, U(node)/N, depth]``, roughly normalized to O(1).  ``U/N``
    carries most of the signal (~94% of per-node reward variance on real events).
    """
    return torch.stack(
        [
            rec["n"] / 50.0,
            rec["zfrac"],
            rec["u_per_n"] / 50.0,
            torch.full_like(rec["n"], rec["depth"] / max(1, k)),
        ],
        dim=-1,
    )


def compute_k_level_reward(
    x: torch.Tensor,
    mask: torch.Tensor,
    leaf_masks: Dict[Tuple, torch.Tensor],
    type_index: int = 7,
    min_fragment_size: int = 2,
    energy_fn=fragment_energy,
) -> torch.Tensor:
    """QMD-driven split reward.  Returns (B,) tensor.

    ```
    Q = U(parent) − Σ_i U(leaf_i)
    ```

    where ``U`` is the *extensive* total pairwise QMD potential energy
    (``total_potential_energy``).  ``Q`` equals the inter-fragment interaction
    energy broken by the split, so maximizing it keeps strongly-attractive bonds
    *inside* fragments and cuts weak or repulsive ones.

    This is position/momentum-aware (unlike the liquid-drop Weizsäcker term,
    which is blind to configuration and always prefers merging into one big
    nucleus).  It also penalizes proton clustering automatically: same-isospin
    Coulomb + Pauli repulsion makes a proton-rich fragment's internal energy
    less negative, lowering ``Σ U(leaf)`` and thus the reward.

    The result is normalized per parent nucleon (``/ N_parent``) so events of
    different size are on a comparable MeV/nucleon scale — without this offset
    removal the raw energy (tens–hundreds of MeV, size-dependent) swamps the
    split-quality signal.  ``N_parent`` is independent of the split, so this adds
    no leaf-count bias.

    ``min_fragment_size`` is retained for API compatibility; leaves below it
    (i.e. singletons) have ``U = 0`` and so do not affect ``Σ U(leaf)`` anyway.
    """
    U_parent = energy_fn(x, mask, type_index)  # (B,)
    B = x.shape[0]
    U_leaves = x.new_zeros(B)
    for lm in leaf_masks.values():
        non_empty = lm.sum(dim=1) >= min_fragment_size  # (B,)
        if non_empty.any():
            U_leaves[non_empty] = (
                U_leaves[non_empty] + energy_fn(x[non_empty], lm[non_empty], type_index)
            )
    n_parent = mask.sum(dim=1).float().clamp(min=1)  # (B,)
    return (U_parent - U_leaves) / n_parent


# ─── Reward ──────────────────────────────────────────────────────────────────

def fragments_affinity(x: torch.Tensor, mask: torch.Tensor, type_index: int = 7) -> torch.Tensor:
    """
    Args:
        x:    (B, N, d)
        mask: (B, N)

    a(fragment) = W(A, Z) − V(fragment)

    W: Weizsäcker semi-empirical binding energy  [MeV]
    V: QMD pairwise potential energy             [MeV]

    Higher affinity -> fragment is more tightly bound / coherent.
    """
    # TODO: Consider cases with A < 2.
    A = mask.sum(dim=-1).float()                                # (B,)
    Z = ((x[..., type_index] == 1) & mask).sum(dim=-1).float()  # (B,)
    w = weizsacker_per_nucleon_formula(A, Z)
    v = binding_energy(x, mask)
    return w - v


def compute_reward(
    x: torch.Tensor,
    mask: torch.Tensor,
    cluster_masks: List[torch.Tensor],
    type_index: int = 7,
    min_fragment_size: int = 2,
) -> torch.Tensor:
    """QMD split reward for a single up-to-K-way partition.

    Thin wrapper over ``compute_k_level_reward``: given the per-cluster masks,
    returns ``Q = U(parent) − Σ U(cluster_i)`` (see ``compute_k_level_reward``).
    """
    leaf_masks = {(c,): cm for c, cm in enumerate(cluster_masks)}
    return compute_k_level_reward(x, mask, leaf_masks, type_index, min_fragment_size)


def _advantage(rewards: torch.Tensor, baseline: float) -> torch.Tensor:
    """Variance-reduced, detached advantage for REINFORCE.

    Centre rewards on the running baseline, then whiten by their batch std so
    the update magnitude is invariant to the reward scale/offset (the QMD reward
    spans tens–hundreds of MeV).  With a single valid event (std undefined) fall
    back to the centred reward.
    """
    adv = rewards - baseline
    if adv.numel() > 1:
        adv = adv / (adv.std() + 1e-6)
    return adv.detach()


# ─── Stability Classifier Module ────────────────────────────────────────────────────────────

class SplitPredictionModuleTrainer:
    """
    REINFORCE training for a single up-to-K-way split.

    For each nucleon cluster the module proposes a partition into up to
    ``model.n_clusters`` fragments.  The QMD reward Q = U(parent) − Σ U(fragment)
    (see ``compute_k_level_reward``) is used as the RL signal.

    Variance reduction / stability: advantages are centred on an EMA baseline and
    whitened by their batch std, the policy loss is averaged per nucleon (not
    summed), and gradients are clipped.  Without this the raw ``.sum()`` loss
    produced ‖grad‖ ~ 1e4 and the reward failed to improve.
    """

    def __init__(
        self,
        model: SplitPredictionModel,
        dataloader: DataLoader,
        *,

        optimizer: optim.Optimizer,
        scheduler,
        device: str = "cpu",

        baseline_momentum: float = 0.99,
        grad_clip: float = 1.0,
        type_index: int = 7,
        min_fragment_size: int = 2,
    ) -> None:
        self.model = model.to(device)
        self.dataloader = dataloader
        self.optim = optimizer
        self.scheduler = scheduler
        self.device = device

        self.baseline = 0.0
        self.baseline_momentum = baseline_momentum
        self.grad_clip = grad_clip

        self.type_index = type_index
        self.min_fragment_size = min_fragment_size

    def _step(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float]:
        """Single REINFORCE update on a batch of nucleon clusters."""
        self.model.train()
        self.optim.zero_grad()

        assignments, log_probs = self.model(x, mask=mask)  # (B, N), (B, N)

        # Per-cluster masks for the up-to-K-way partition.
        cluster_masks = [
            (assignments == c) & mask for c in range(self.model.n_clusters)
        ]  # each (B, N)

        # Keep events where at least 2 fragments meet the minimum size
        # requirement (a non-degenerate split).
        large_enough = torch.stack(
            [cm.sum(dim=-1) >= self.min_fragment_size for cm in cluster_masks]
        )  # (K, B)
        valid = large_enough.sum(dim=0) >= 2  # (B,)
        if not valid.any():
            return 0.0, 0.0

        cluster_masks_v = [cm[valid] for cm in cluster_masks]
        reward_v = compute_reward(
            x[valid], mask[valid], cluster_masks_v, self.type_index, self.min_fragment_size
        )  # (V,)

        self.baseline = (
            self.baseline_momentum * self.baseline
            + (1.0 - self.baseline_momentum) * reward_v.mean().item()
        )
        advantage = _advantage(reward_v, self.baseline)  # (V,)

        mask_v = mask[valid].float()
        log_probs_v = log_probs[valid]
        loss = -(advantage.unsqueeze(-1) * log_probs_v * mask_v).sum() / mask_v.sum().clamp(min=1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optim.step()

        return float(loss.item()), reward_v.mean().item()

    def train_epoch(self) -> Tuple[float, float]:
        total_loss = 0.0
        total_reward = 0.0
        n = 0
        for batch in self.dataloader:
            loss, reward = self._step(
                batch["x"].to(self.device),
                batch["mask"].to(self.device)
            )
            total_loss += loss
            total_reward += reward
            n += 1
        return total_loss / n, total_reward / n

    def train(self, n_epochs: int, verbose: bool = True, log_every: int = 1) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {"loss": [], "reward": [], "baseline": []}
        for ep in range(1, n_epochs + 1):
            avg_loss, avg_reward = self.train_epoch()
            history["loss"].append(avg_loss)
            history["reward"].append(avg_reward)
            history["baseline"].append(self.baseline)
            if verbose and ep % log_every == 0:
                print(
                    f"[DM] {ep:4d}/{n_epochs}  "
                    f"loss={avg_loss:.4f}  "
                    f"reward={avg_reward:.4f}  "
                    f"baseline={self.baseline:.4f}"
                )
        return history


# ─── K-level REINFORCE trainer ────────────────────────────────────────────────

class KSplitTrainer:
    """Fine-tunes a SplitPredictionModel using a k-level REINFORCE objective.

    The same model is applied at every level of a depth-k split tree whose
    branching factor is ``model.n_clusters``.  The reward is computed **per split
    node** (``iter_split_nodes``): every split decision is credited with its own
    local QMD energy gain ``q(node) = (U(node) − Σ_child U(child)) / N_node``,
    rather than a single tree-wide terminal reward.

    Each valid split node is one sample, scored by its **immediate** local reward
    ``q(s)`` — a contextual bandit per split.  Nodes are streamed by
    ``iter_split_nodes`` and backwarded one at a time, so no autograd graph is
    retained across levels and memory is flat in ``k``; gradients accumulate and
    one ``optim.step()`` runs per batch.  This is what makes deep trees (k≈8)
    affordable, together with the empty/too-small node pruning in the walker.

    **Actor-Critic** (``critic`` supplied): advantage is ``q(s) − V(s)``, where
    the critic is a DeepSets value function over the node's nucleon set.  It
    replaces the scalar EMA baseline, which subtracts the same number from a
    40-nucleon root and a 4-nucleon deep fragment alike — on real events ~94% of
    the per-node reward variance is state-predictable, so a state-dependent
    baseline removes variance the EMA structurally cannot.  Per-node loss is
    ``actor_loss + value_coef · MSE(V(s), q(s))`` (both in normalized reward
    space; raw q spans tens of MeV/nucleon and would swamp the policy term).

    **Plain REINFORCE** (``critic=None``): falls back to the EMA baseline.

    ``eval_reward`` reports the global root→leaves reward
    (``compute_k_level_reward``) as an overall metric.

    Intended usage: first train a single up-to-K-way split with
    SplitPredictionModuleTrainer, then optionally fine-tune with
    KSplitTrainer(k>1) for deeper hierarchies using the same model instance.
    With k=1 this trains exactly one up-to-K-way split.

    Note: when a critic is supplied, ``optimizer`` must also cover its
    parameters, e.g.
    ``AdamW(list(model.parameters()) + list(critic.parameters()), lr=...)``.
    """

    def __init__(
        self,
        model: SplitPredictionModel,
        dataloader: DataLoader,
        *,
        optimizer: optim.Optimizer,
        scheduler,
        k: int = 2,
        device: str = "cpu",
        critic: "nn.Module | None" = None,
        value_coef: float = 0.5,
        baseline_momentum: float = 0.99,
        grad_clip: float = 1.0,
        type_index: int = 7,
        min_fragment_size: int = 2,
        reward_type: str = "qmd_asym",
        qmd_weight: float = 1.0,
        energy_scale: str = "extensive",
        bwm_weight: float = 1.0,
        bwm_form: str = "bwm",
        zeta_spin_factor: float = 0.5,
        zeta_yukawa: str = "folded",
    ) -> None:
        self.model = model.to(device)
        self.critic = critic.to(device) if critic is not None else None
        self.value_coef = value_coef
        self.dataloader = dataloader
        self.optim = optimizer
        self.scheduler = scheduler
        self.k = k
        self.device = device
        self.baseline = 0.0
        self.return_std = 1.0  # running std of returns; normalizes the critic target
        self._norm_init = False  # warm-start baseline/return_std from the first batch
        self.baseline_momentum = baseline_momentum
        self.grad_clip = grad_clip
        self.type_index = type_index
        self.min_fragment_size = min_fragment_size

        # Per-optimizer-step log.  Epoch-level series are far too coarse once the
        # dataset is large: 3 epochs over 58k events is 1,368 optimizer steps but
        # only 3 plotted points for 3.5 h of compute.  Everything here is already
        # computed per batch, so recording it costs nothing.  eval_reward stays
        # epoch-level — it is a full-dataset pass and cannot run per step.
        self.global_step = 0
        self.epoch_end_steps: List[int] = []   # x-positions for the epoch-level series
        self.step_history: Dict[str, List[float]] = {
            "step": [], "loss": [], "reward": [], "q_weighted": [], "grad_norm": [],
            "value_loss": [], "lr": [], "n_nodes": [], "node_depth": [], "valid_frac": [],
            "entropy": [], "entropy_frac": [],
        }
        # log K — the entropy of a uniform policy, and the ceiling for `entropy`.
        # `entropy_frac` normalizes by it so runs with different K compare directly.
        self._log_k = float(np.log(model.n_clusters))

        # Per-node energy U(fragment); the split reward is q = (U_parent − ΣU_child)/N.
        #   "qmd_asym"       U = QMD potential + asymmetry penalty      (the main-branch reward)
        #   "weizsacker_qmd" U = qmd_weight·V − W  (maximize Weizsäcker W, minimize QMD V),
        #                    with energy_scale = "extensive" | "per_nucleon" (see weizsacker_qmd_energy)
        #   "qmd_minus_b"    U = V − bwm_weight·B  — the SACA annealer's QMD − B
        #                    objective, with bwm_form = "bwm" | "bw" selecting the
        #                    mass formula (see qmd_minus_b_energy).  The weight sits
        #                    on B here, not on V, to match the baseline's λ scan.
        #   "saca_qmd_minus_b"  the same objective on the *baselines'* QMD energy
        #                    (qmd_energy.cluster_energy: saturating Skyrme, plus the
        #                    rest-frame kinetic term) rather than on physics.py's
        #                    non-saturating potential.  See saca_qmd_minus_b_energy.
        #   "zeta_correct"   the SACA paper's full zeta (rest-frame kinetic + Skyrme 2/3-body
        #                    + Yukawa + Coulomb + Pauli) + bwm_weight·(−B_BWM); the same
        #                    energy as SacaParams(energy_model="zeta_correct").  See
        #                    zeta_correct_energy.
        if reward_type == "qmd_asym":
            self.energy_fn = fragment_energy
        elif reward_type == "zeta_correct":
            if zeta_yukawa not in ("folded", "point", "off"):
                raise ValueError(
                    f"unknown zeta_yukawa {zeta_yukawa!r}; expected 'folded', 'point' or 'off'"
                )
            self.energy_fn = partial(
                zeta_correct_energy, bwm_weight=bwm_weight,
                spin_factor=zeta_spin_factor, yukawa=zeta_yukawa,
            )
        elif reward_type == "weizsacker_qmd":
            if energy_scale not in ("extensive", "per_nucleon"):
                raise ValueError(
                    f"unknown energy_scale {energy_scale!r}; expected 'extensive' or 'per_nucleon'"
                )
            self.energy_fn = partial(
                weizsacker_qmd_energy, qmd_weight=qmd_weight, scale=energy_scale
            )
        elif reward_type in ("qmd_minus_b", "saca_qmd_minus_b"):
            if bwm_form not in ("bwm", "bw"):
                raise ValueError(
                    f"unknown bwm_form {bwm_form!r}; expected 'bwm' or 'bw'"
                )
            fn = (qmd_minus_b_energy if reward_type == "qmd_minus_b"
                  else saca_qmd_minus_b_energy)
            self.energy_fn = partial(fn, bwm_weight=bwm_weight, binding=bwm_form)
        else:
            raise ValueError(
                f"unknown reward_type {reward_type!r}; expected 'qmd_asym', "
                f"'weizsacker_qmd', 'qmd_minus_b', 'saca_qmd_minus_b' or 'zeta_correct'"
            )
        self.reward_type = reward_type
        self.energy_scale = energy_scale
        self.bwm_weight = bwm_weight
        self.bwm_form = bwm_form

    def _step(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[float, "torch.Tensor | None", float, float]:
        """One Actor-Critic (or REINFORCE) update. Returns (loss, rewards | None, value_loss).

        Each split node is backwarded **as it is produced**, so its graph is freed
        before the next node is built and memory stays flat in tree depth.
        Gradients are accumulated across nodes and applied with a single
        ``optim.step()`` per batch.
        """
        self.model.train()
        if self.critic is not None:
            self.critic.train()
        self.optim.zero_grad()

        rewards_seen: List[torch.Tensor] = []
        total_loss = 0.0
        total_value_loss = 0.0
        n_nodes = 0
        # Diagnostics for the falling-`reward` question.  `reward` is an
        # *unweighted* mean over split nodes, while eval_reward telescopes the
        # whole tree weighted by node size.  Since U is extensive pairwise, the
        # energy a split breaks scales ~N^2 and q divides by N, so q grows
        # ~linearly with node size: a deepening tree adds many small low-q nodes
        # and drags the unweighted mean down even when every split improved.
        # q_weighted (Sigma q*N / Sigma N) is the size-weighted counterpart — if it
        # holds up while `reward` falls, the drop is that averaging artifact and
        # not a regression.  node_depth/n_nodes show the population shifting;
        # valid_frac shows how often a visited node produces a scorable split
        # (invalid ones are masked out entirely, so they never reach the loss).
        wq_sum = 0.0     # Sigma q * N over valid (node, item) pairs
        wn_sum = 0.0     # Sigma N
        depth_sum = 0.0  # Sigma depth, over valid pairs
        n_valid_items = 0
        n_seen_items = 0
        # Policy entropy, free from the sampled log-probs.  H = E_{a~p}[-log p(a)],
        # and log_prob is already Sigma_n log p(a_n) over the node, so the
        # size-weighted mean per-nucleon entropy is just Sigma(-log_prob)/Sigma N —
        # no second forward pass.  Valid only under sampling (train mode); the
        # argmax path would give -log p(argmax), which is not an entropy.
        # Worth watching: H near log(K) means the policy is barely committing to
        # any assignment, and a sharp drop is how slot collapse announces itself.
        ent_sum = 0.0

        for rec in iter_split_nodes(
            self.model, x, mask, self.k, self.min_fragment_size, self.type_index,
            energy_fn=self.energy_fn,
        ):
            v = rec["valid"]
            n_seen_items += int(v.numel())
            if not v.any():
                continue
            n_valid_items += int(v.sum().item())
            _n = rec["n"][v]
            wq_sum += float((rec["q"][v] * _n).sum().item())
            wn_sum += float(_n.sum().item())
            depth_sum += float(rec["depth"]) * int(v.sum().item())
            ent_sum += float(-rec["log_prob"][v].detach().sum().item())

            q = rec["q"][v]                  # (m,) detached, immediate local reward
            log_probs = rec["log_prob"][v]   # (m,) in graph

            # Warm-start the reward normalizer from the first node seen.  Left at
            # the 0.0/1.0 defaults, the critic spends its early updates chasing a
            # raw target through a per-epoch-lagging normalizer and never catches up.
            if not self._norm_init and q.numel() > 1:
                self.baseline = float(q.mean().item())
                self.return_std = max(float(q.std().item()), 1e-6)
                self._norm_init = True

            # Normalized reward: raw q spans tens of MeV/nucleon, so an
            # unnormalized MSE would dwarf the policy loss in the shared backward.
            q_norm = (q - self.baseline) / (self.return_std + 1e-6)

            if self.critic is not None:
                V = self.critic(x[v], rec["mask"][v], node_scalars(rec, self.k)[v])  # (m,)
                advantage = (q_norm - V).detach()   # state-dependent baseline
                value_loss = ((V - q_norm) ** 2).mean()
                node_loss = -(advantage * log_probs).mean() + self.value_coef * value_loss
                total_value_loss += float(value_loss.item())
            else:
                # No critic: centre on the running EMA baseline (per-epoch) and
                # whiten within this node's batch slice.
                advantage = q_norm - q_norm.mean() if q_norm.numel() > 1 else q_norm
                advantage = advantage.detach()
                node_loss = -(advantage * log_probs).mean()

            # Backward per node: frees this node's graph immediately.  Gradients
            # accumulate into .grad across nodes; one step() per batch below.
            node_loss.backward()

            total_loss += float(node_loss.item())
            rewards_seen.append(q.detach())
            n_nodes += 1

        stats = {
            "n_nodes": float(n_nodes),
            "wq_sum": wq_sum,
            "wn_sum": wn_sum,
            "depth_sum": depth_sum,
            "ent_sum": ent_sum,
            "n_valid_items": float(n_valid_items),
            "n_seen_items": float(n_seen_items),
        }

        if n_nodes == 0:
            return 0.0, None, 0.0, 0.0, stats

        # Clip actor and critic separately: a shared clip lets the critic's much
        # larger gradient throttle the policy gradient down to nothing.
        # clip_grad_norm_ returns the total norm *before* clipping — capture the
        # actor's as the gradient-norm diagnostic (no extra backward needed).
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        if self.critic is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.optim.step()

        return (
            total_loss / n_nodes,
            torch.cat(rewards_seen),
            total_value_loss / n_nodes,
            float(grad_norm),
            stats,
        )

    def train_epoch(self) -> Tuple[float, float, float, float, Dict[str, float]]:
        total_loss = 0.0
        total_value_loss = 0.0
        total_grad_norm = 0.0
        all_rewards: List[float] = []
        agg = {"n_nodes": 0.0, "wq_sum": 0.0, "wn_sum": 0.0, "depth_sum": 0.0,
               "ent_sum": 0.0, "n_valid_items": 0.0, "n_seen_items": 0.0}
        n = 0
        for batch in self.dataloader:
            loss, rewards, value_loss, grad_norm, stats = self._step(
                batch["x"].to(self.device),
                batch["mask"].to(self.device),
            )
            total_loss += loss
            total_value_loss += value_loss
            total_grad_norm += grad_norm
            for key in agg:
                agg[key] += stats[key]
            if rewards is not None:
                all_rewards.extend(rewards.cpu().tolist())
            n += 1

            # One record per optim.step().  `rewards` is None when the batch
            # produced no scorable node at all — log NaN so the point is dropped
            # from the plot rather than reading as a genuine zero.
            self.global_step += 1
            sh = self.step_history
            sh["step"].append(float(self.global_step))
            sh["loss"].append(loss)
            sh["reward"].append(
                float(rewards.mean().item()) if rewards is not None else float("nan"))
            sh["q_weighted"].append(
                stats["wq_sum"] / stats["wn_sum"] if stats["wn_sum"] else float("nan"))
            sh["grad_norm"].append(grad_norm)
            sh["value_loss"].append(value_loss)
            sh["lr"].append(self.optim.param_groups[0]["lr"])
            sh["n_nodes"].append(stats["n_nodes"])
            sh["node_depth"].append(
                stats["depth_sum"] / stats["n_valid_items"]
                if stats["n_valid_items"] else float("nan"))
            sh["valid_frac"].append(
                stats["n_valid_items"] / stats["n_seen_items"]
                if stats["n_seen_items"] else float("nan"))
            _h = stats["ent_sum"] / stats["wn_sum"] if stats["wn_sum"] else float("nan")
            sh["entropy"].append(_h)
            sh["entropy_frac"].append(_h / self._log_k)

        # One EMA update per epoch: baseline tracks epoch-mean reward,
        # so the plotted curve is as smooth as the epoch-average reward.
        # return_std tracks the spread, used to z-score the critic's target.
        if all_rewards:
            epoch_mean = float(np.mean(all_rewards))
            epoch_std = float(np.std(all_rewards))
            self.baseline = (
                self.baseline_momentum * self.baseline
                + (1.0 - self.baseline_momentum) * epoch_mean
            )
            self.return_std = (
                self.baseline_momentum * self.return_std
                + (1.0 - self.baseline_momentum) * max(epoch_std, 1e-6)
            )

        avg_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
        diag = {
            # Size-weighted twin of avg_reward.  Same q values, weighted by node
            # size instead of counted equally — the direct test of whether the
            # fall in avg_reward is an averaging artifact of a deepening tree.
            "q_weighted": agg["wq_sum"] / agg["wn_sum"] if agg["wn_sum"] else 0.0,
            "n_nodes": agg["n_nodes"] / n if n else 0.0,          # per batch
            "node_depth": (agg["depth_sum"] / agg["n_valid_items"]
                           if agg["n_valid_items"] else 0.0),
            "valid_frac": (agg["n_valid_items"] / agg["n_seen_items"]
                           if agg["n_seen_items"] else 0.0),
            "entropy": agg["ent_sum"] / agg["wn_sum"] if agg["wn_sum"] else 0.0,
            "entropy_frac": ((agg["ent_sum"] / agg["wn_sum"]) / self._log_k
                             if agg["wn_sum"] else 0.0),
        }
        return total_loss / n, avg_reward, total_value_loss / n, total_grad_norm / n, diag

    @torch.no_grad()
    def eval_reward(self) -> float:
        """Mean deterministic (argmax) split reward over the dataset.

        The training curve tracks *sampled* reward, which stays roughly flat as
        the policy sharpens (exploration noise cancels the learned gain).  This
        eval-mode pass measures the reward of the greedy policy the model
        actually deploys, so genuine improvement is visible.
        """
        self.model.eval()
        rewards: List[float] = []
        for batch in self.dataloader:
            x = batch["x"].to(self.device)
            mask = batch["mask"].to(self.device)
            result = k_level_forward(self.model, x, mask, self.k, self.min_fragment_size)
            valid = result["valid"]
            if not valid.any():
                continue
            r = compute_k_level_reward(
                x[valid], mask[valid],
                {p: m[valid] for p, m in result["leaf_masks"].items()},
                self.type_index, self.min_fragment_size,
                energy_fn=self.energy_fn,
            )
            rewards.extend(r.cpu().tolist())
        return float(np.mean(rewards)) if rewards else 0.0

    def train(self, n_epochs: int, verbose: bool = True, log_every: int = 1) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {
            "loss": [], "reward": [], "eval_reward": [], "baseline": [], "value_loss": [],
            "grad_norm": [], "lr": [],
            # Diagnostics for the falling-`reward` question — see _step.
            "q_weighted": [], "n_nodes": [], "node_depth": [], "valid_frac": [],
            "entropy": [], "entropy_frac": [],
        }
        for ep in range(1, n_epochs + 1):
            lr = self.optim.param_groups[0]["lr"]  # LR used for this epoch
            avg_loss, avg_reward, avg_value_loss, avg_grad_norm, diag = self.train_epoch()
            self.epoch_end_steps.append(self.global_step)  # anchor epoch series on the step axis
            eval_reward = self.eval_reward()
            if self.scheduler is not None:
                self.scheduler.step()  # advance the LR schedule once per epoch
            history["loss"].append(avg_loss)
            history["reward"].append(avg_reward)
            history["eval_reward"].append(eval_reward)
            history["baseline"].append(self.baseline)
            history["value_loss"].append(avg_value_loss)
            history["grad_norm"].append(avg_grad_norm)
            history["lr"].append(lr)
            for key in ("q_weighted", "n_nodes", "node_depth", "valid_frac",
                        "entropy", "entropy_frac"):
                history[key].append(diag[key])
            if verbose and ep % log_every == 0:
                critic_msg = f"  v_loss={avg_value_loss:.3f}" if self.critic is not None else ""
                print(
                    f"[KS] {ep:4d}/{n_epochs}  "
                    f"loss={avg_loss:.4f}  "
                    f"reward={avg_reward:.4f}  "
                    f"qw={diag['q_weighted']:.4f}  "
                    f"eval={eval_reward:.4f}  "
                    f"baseline={self.baseline:.4f}"
                    f"  grad={avg_grad_norm:.3f}"
                    f"  nodes={diag['n_nodes']:.1f}"
                    f"  depth={diag['node_depth']:.2f}"
                    f"  valid={diag['valid_frac']:.2f}"
                    f"  H={diag['entropy']:.3f}/{self._log_k:.3f}"
                    f"  lr={lr:.2e}"
                    f"{critic_msg}"
                )
        return history
