import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class SlotAttention(nn.Module):
    """
    Iterative slot attention over a batch of sets with ``n_clusters`` learnable
    slots.  Slots compete for inputs; the final attention distribution gives
    soft assignments used both for hard clustering (argmax) and REINFORCE log
    probs.

    With ``n_clusters == 2`` this reduces to the original binary split.  For
    ``n_clusters == K`` each nucleon is assigned to one of K slots, so a single
    forward pass partitions the cluster into up to K fragments.  A slot may win
    zero nucleons, hence "up to" K rather than exactly K.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_iters: int = 3, n_clusters: int = 2) -> None:
        super().__init__()
        assert n_clusters >= 2, "n_clusters must be at least 2 to define a split"
        self.n_iters = n_iters
        self.hidden_dim = hidden_dim
        self.n_clusters = n_clusters
        self.scale = hidden_dim ** -0.5

        self.norm_inputs = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(hidden_dim)
        self.norm_mlp = nn.LayerNorm(hidden_dim)

        self.project_k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.project_v = nn.Linear(input_dim, hidden_dim, bias=False)
        self.project_q = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.slots_mu = nn.Parameter(torch.randn(n_clusters, hidden_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(n_clusters, hidden_dim))

        # GRUCell treats B*n_clusters as the batch dimension
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        """Run the slot refinement loop and return the final soft attention.

        Args:
            x: (B, N, input_dim)
        Returns:
            attn: (B, n_clusters, N) — softmax over the slot dim, so column n is
            the categorical distribution over slots for nucleon n.

        Exposed separately from ``forward`` because supervised pretraining needs
        the *soft* assignment (see the pairwise MST objective), not a sample.
        """
        B, N, _ = x.shape
        D = self.hidden_dim
        K = self.n_clusters

        normed = self.norm_inputs(x)
        k = self.project_k(normed)  # (B, N, D)
        v = self.project_v(normed)  # (B, N, D)

        if self.training:
            noise = torch.randn(B, K, D, device=x.device, dtype=x.dtype)
            slots = self.slots_mu + torch.exp(self.slots_log_sigma) * noise  # (B, K, D)
        else:
            slots = self.slots_mu.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, K, D)

        for _ in range(self.n_iters):
            slots_prev = slots
            q = self.project_q(self.norm_slots(slots))  # (B, K, D)

            # Softmax over slot dim so slots compete for each input
            attn = F.softmax(
                torch.einsum('bsd,bnd->bsn', q, k) * self.scale, dim=1
            )  # (B, K, N)
            attn_norm = attn / (attn.sum(dim=2, keepdim=True) + 1e-8)  # (B, K, N)

            updates = torch.einsum('bsn,bnd->bsd', attn_norm, v)  # (B, K, D)
            slots = self.gru(
                updates.reshape(B * K, D),
                slots_prev.reshape(B * K, D),
            ).reshape(B, K, D)
            slots = slots + self.mlp(self.norm_mlp(slots))

        # Final attention with refined slots
        q = self.project_q(self.norm_slots(slots))
        return F.softmax(
            torch.einsum('bsd,bnd->bsn', q, k) * self.scale, dim=1
        )  # (B, K, N)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, input_dim)
        Returns:
            assignments: (B, N) cluster assignments in {0, ..., n_clusters-1}
            log_probs:   (B, N) log prob of each assignment under the attention policy
        """
        B, N, _ = x.shape
        attn = self.attention(x)  # (B, K, N)
        K = self.n_clusters

        if self.training:
            # Sample from the per-nucleon slot distribution for REINFORCE.
            # Sampling prevents slot collapse: argmax lets one slot dominate
            # until the degenerate-split guard silences all gradients.
            assignments = torch.multinomial(
                attn.permute(0, 2, 1).reshape(B * N, K), num_samples=1
            ).reshape(B, N)  # (B, N)
        else:
            assignments = attn.argmax(dim=1)  # (B, N)

        log_probs = torch.log(
            attn.gather(dim=1, index=assignments.unsqueeze(1)).squeeze(1) + 1e-8
        )  # (B, N)

        return assignments, log_probs


class SplitPredictionModel(nn.Module):
    """
    Receives a batch of nucleon sets and predicts per-nucleon cluster
    assignments in {0, ..., n_clusters-1} to achieve more stable fragments than
    the initial fragment.  With ``n_clusters == 2`` this is a binary split;
    larger values split the cluster into up to ``n_clusters`` fragments at once.

    Based on SlotAttention; trained with REINFORCE.
    """
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        n_iters: int = 3,
        n_clusters: int = 2,
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.slot_attention = SlotAttention(input_dim, hidden_dim, n_iters, n_clusters)

    @torch.no_grad()
    def _center_cluster(
        self,
        x: torch.Tensor,                   # (B, N, 8)
        mask: torch.Tensor | None = None,  # (B, N)  True = real nucleon
    ) -> torch.Tensor:
        """Thin wrapper over the module-level :func:`center_cluster`."""
        return center_cluster(x, mask)

    def forward(
        self,
        x: torch.Tensor,    # (B, N, input_dim)
        mask: torch.Tensor, # (B, N)  True = real nucleon
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    (B, N, input_dim) batch of nucleon features
            mask: (B, N) True = real nucleon
        Returns:
            assignments: (B, N) assignments in {0, ..., n_clusters-1}
            log_probs:   (B, N) per-nucleon log prob for REINFORCE
        """
        return self.slot_attention(self._center_cluster(x, mask))

    def soft_assign(
        self,
        x: torch.Tensor,     # (B, N, input_dim)
        mask: torch.Tensor,  # (B, N)  True = real nucleon
    ) -> torch.Tensor:
        """Soft cluster assignment (B, n_clusters, N) — used by MST pretraining."""
        return self.slot_attention.attention(self._center_cluster(x, mask))


@torch.no_grad()
def center_cluster(
    x: torch.Tensor,                   # (B, N, 8)
    mask: torch.Tensor | None = None,  # (B, N)  True = real nucleon
) -> torch.Tensor:
    """
    Lorentz-boost all nucleons to the cluster CM frame, then subtract the
    spatial centroid from positions.  The ``type`` column (index 7) is left
    unchanged.  Padding tokens (mask=False) are excluded from both the CM
    and centroid computations.  γ = 1/√(1−β²) is computed in float64 on
    CPU to avoid cancellation when β ≈ 1; the transformation itself runs
    in float32 (MPS-safe).

    Shared by the actor (SplitPredictionModel) and the critic
    (SplitValueCritic) so both see the same frame-invariant representation.

    The boost is computed as:
      β⃗ = Σp / ΣE  (summed over real nucleons only)
      for each nucleon:
        p∥ = p · β̂,  p⊥ = p − p∥ β̂
        p' = p⊥ + γ(p∥ − β E) β̂
        E' = γ(E − β p∥)
        r∥ = r · β̂,  r⊥ = r − r∥ β̂
        r' = r⊥ + γ r∥ β̂        (evaluated at t = 0)
    When β ≈ 0 the formula reduces to the identity via β̂ → 0⃗, γ → 1.
    """
    p = x[..., 0:3]   # (B, N, 3)
    E = x[..., 3]     # (B, N)
    r = x[..., 4:7]   # (B, N, 3)

    # ── CM frame velocity ─────────────────────────────────────────────────
    if mask is not None:
        mf    = mask.float()                                         # (B, N)
        E_tot = (E * mf).sum(dim=-1).clamp(min=1e-8)                 # (B,)
        p_tot = (p * mf.unsqueeze(-1)).sum(dim=-2)                   # (B, 3)
    else:
        E_tot = E.sum(dim=-1).clamp(min=1e-8)                        # (B,)
        p_tot = p.sum(dim=-2)                                        # (B, 3)

    beta_vec = p_tot / E_tot.unsqueeze(-1)                           # (B, 3) f32

    # TODO: Consider rewriting math with logarithms to avoid copying data redundantly

    # γ = 1/√(1−β²) suffers catastrophic cancellation when β≈1 in f32.
    # Move the scalar computation to CPU in f64, then cast back to f32.
    beta_sq_f64 = (beta_vec.cpu().double() ** 2).sum(dim=-1).clamp(max=1.0 - 1e-8)
    gamma = (1.0 - beta_sq_f64).rsqrt().float().to(x.device)         # (B,)
    beta_sq = beta_sq_f64.float().to(x.device)                       # (B,)

    beta    = beta_sq.sqrt()                                         # (B,)
    # When beta≈0, beta_hat→0⃗ so p_par=r_par=0 and the boost is identity
    beta_hat = beta_vec / beta.unsqueeze(-1).clamp(min=1e-10)        # (B, 3)

    # ── Lorentz boost ─────────────────────────────────────────────────────
    g  = gamma.unsqueeze(-1)        # (B, 1)   for broadcasting over N
    b  = beta.unsqueeze(-1)         # (B, 1)
    bh = beta_hat.unsqueeze(-2)     # (B, 1, 3)

    p_par = (p * bh).sum(dim=-1)    # (B, N)  parallel component of p
    r_par = (r * bh).sum(dim=-1)    # (B, N)  parallel component of r

    p_new = (p - p_par.unsqueeze(-1) * bh) + (g * (p_par - b * E)).unsqueeze(-1) * bh  # (B, N, 3)
    E_new = g * (E - b * p_par)                                                        # (B, N)
    r_new = (r - r_par.unsqueeze(-1) * bh) + (g * r_par).unsqueeze(-1) * bh            # (B, N, 3)

    # ── Spatial centroid ──────────────────────────────────────────────────
    if mask is not None:
        mf3    = mask.float().unsqueeze(-1)                             # (B, N, 1)
        n_real = mf3.sum(dim=-2).clamp(min=1.0)                         # (B, 1)
        r_mean = (r_new * mf3).sum(dim=-2) / n_real                     # (B, 3)
    else:
        r_mean = r_new.mean(dim=-2)                                     # (B, 3)
    r_new = r_new - r_mean.unsqueeze(-2)

    # ── Reassemble ────────────────────────────────────────────────────────
    out = x.clone()
    out[..., 0:3] = p_new
    out[..., 3]   = E_new
    out[..., 4:7] = r_new
    # out[..., 7] (type) left unchanged

    return out


class SplitValueCritic(nn.Module):
    """DeepSets value function V(s) over a split node's nucleon set.

    Predicts the expected **subtree return** R(s) = q(s) + Σ_children R(c) of
    splitting node ``s`` — i.e. the total QMD reward collected in s's subtree.
    Used as a state-dependent baseline: ``advantage = R(s) − V(s)``.

    Architecture: a permutation-invariant DeepSets encoder over the CM-centered
    nucleons (per-nucleon MLP → masked mean-pool), concatenated with cheap
    physics scalars, then an MLP value head.  The scalars matter: measured on
    real events, ``U(node)/N`` alone linearly explains ~94% of the per-node
    reward variance, so handing it to the critic directly is nearly free signal.

    Args:
        input_dim:  nucleon feature width (8).
        hidden_dim: width of the encoder / head.
        n_scalars:  number of physics scalars concatenated to the pooled vector.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, n_scalars: int = 4) -> None:
        super().__init__()
        self.n_scalars = n_scalars
        self.norm_inputs = nn.LayerNorm(input_dim)
        self.phi = nn.Sequential(          # per-nucleon encoder
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.rho = nn.Sequential(          # pooled set + scalars -> value
            nn.Linear(hidden_dim + n_scalars, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,        # (B, N, input_dim) raw (uncentered) nucleons
        mask: torch.Tensor,     # (B, N) True = nucleon belongs to this node
        scalars: torch.Tensor,  # (B, n_scalars) physics summary features
    ) -> torch.Tensor:
        """Returns V(s) of shape (B,)."""
        centered = center_cluster(x, mask)
        mf = mask.unsqueeze(-1).float()                       # (B, N, 1)
        h = self.phi(self.norm_inputs(centered)) * mf         # (B, N, H); padding zeroed
        pooled = h.sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)  # (B, H) masked mean
        return self.rho(torch.cat([pooled, scalars], dim=-1)).squeeze(-1)  # (B,)
