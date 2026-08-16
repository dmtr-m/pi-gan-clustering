import math

import torch
import torch.nn as nn

import numpy as np


def weizsacker_formula(A: torch.Tensor, Z: torch.Tensor):
    """Weizsäcker semi-empirical binding energy B(A, Z) [MeV] (extensive).

    volume − surface − Coulomb − asymmetry + pairing.

    The pairing term δ (the parity correction) is now implemented, vectorized in
    torch — previously a disabled Python-scalar ``if`` TODO:
        δ = +a_p·A^(−3/4)  even-even,   −a_p·A^(−3/4)  odd-odd,   0  when A is odd.
    """
    a_1 = 15.75  # MeV  volume
    a_2 = 17.80  # MeV  surface
    a_3 = 0.711  # MeV  Coulomb
    a_4 = 23.70  # MeV  asymmetry  (pairs with the standard (A − 2Z)² form below)
    a_p = 34.0   # MeV  pairing

    assert (A > 0).all()

    A = A.float()
    Z = Z.float()
    N = A - Z
    even_even = (torch.remainder(Z, 2) == 0) & (torch.remainder(N, 2) == 0)
    odd_odd = (torch.remainder(Z, 2) == 1) & (torch.remainder(N, 2) == 1)
    pairing = a_p * torch.pow(A, -3.0 / 4.0)
    delta = torch.where(even_even, pairing, torch.zeros_like(A))
    delta = torch.where(odd_odd, -pairing, delta)

    weizsacker_energy = (
        a_1 * A
        - a_2 * torch.pow(A, 2 / 3)
        - a_3 * Z**2 / torch.pow(A, 1 / 3)
        # (A − 2Z)², not (A/2 − Z)².  They differ by a factor of 4, and 23.70 MeV is
        # the literature coefficient for *this* form.  Paired with the halved form
        # the penalty came out 4x too weak, Coulomb won, and the predicted valley
        # slid neutron-rich — Z*(128) = 36 against 53 in the HSE nuclei table.
        - a_4 * (A - 2 * Z) ** 2 / A
        + delta
    )

    return weizsacker_energy


def weizsacker_per_nucleon_formula(A: torch.Tensor, Z: torch.Tensor):
    return weizsacker_formula(A, Z) / A


A_SYM = 23.70  # MeV — Weizsäcker asymmetry coefficient (a_4)


def asymmetry_energy(A: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """Isospin-asymmetry penalty  ``+a₄·(A/2 − Z)²/A``  [MeV].  Always ≥ 0.

    This is the one Weizsäcker term the QMD potential cannot express.  The QMD
    pairwise potential penalizes *proton* clustering through Coulomb + Pauli
    repulsion, but neutrons carry no charge — a pure-neutron blob sees only the
    attractive Skyrme/Yukawa terms, so an energy-only reward happily clumps
    neutrons into species that do not exist (²n, ³n, …).  This term supplies the
    missing repulsion symmetrically in isospin: it vanishes for N = Z and grows
    quadratically with |A/2 − Z|.

    Only the asymmetry term is used — *not* the full liquid-drop formula.  The
    volume/surface terms are what made an extensive Weizsäcker reward prefer
    merging everything into one big low-surface nucleus (Σ Aᵢ^(2/3) > (ΣA)^(2/3)),
    which fights the spatial fragmentation we want.  The asymmetry term has no
    such bias: for isospin-balanced splits it is 0 on both sides of the split,
    so it only ever penalizes isospin *segregation*.

    Returns 0 for A < 2: the liquid-drop picture is meaningless for a single
    nucleon, and free nucleons are physical.
    """
    A = A.float()
    Z = Z.float()
    penalty = A_SYM * (A / 2 - Z) ** 2 / A.clamp(min=1.0)
    return torch.where(A >= 2, penalty, torch.zeros_like(penalty))


alpha = 0.1152  # fm^-2
t_1 = - 84.5  # MeV fm^3   (two-body Skyrme)
t_gamma = 188.2  # MeV fm^6  (three-body Skyrme)
gamma = 1.46
V_0_Yuk = -85.1  # MeV fm
gamma_Y = 1.0  # fm

V_0_Pauli = 98.95  # MeV
q_0 = 2.16  # fm
p_0 = 120  # MeV / c

planck_constant = 197.3269804  # MeV * fm  (ℏc)
e_sq = 1.4399644  # MeV * fm  (e² / 4πε₀ in natural units)


def binding_energy(
    nucleons: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    N = mask.shape[1]
    off_diag = ~torch.eye(N, dtype=torch.bool, device=mask.device).unsqueeze(0)  # (1, N, N)
    mask_ij = mask.unsqueeze(2) & mask.unsqueeze(1) & off_diag  # (B, N, N)

    momenta = nucleons[..., 0:3]
    energies = nucleons[..., 3]
    coords = nucleons[..., 4:7]
    types = nucleons[..., 7]

    momenta_pairwise_boosted, coords_pairwise_boosted = _pairwise_lorentz_boost(momenta, energies, coords)

    # Same-isospin mask (both proton or both neutron) — for Pauli
    same_type_matrix = (types.unsqueeze(2) == types.unsqueeze(1)).float()  # (B, N, N)
    # Proton-proton mask — for Coulomb (Z_i * Z_j, neutrons have Z=0)
    proton_proton_mask = (
        (types == 1).unsqueeze(2)
        & (types == 1).unsqueeze(1)
    ).float()  # (B, N, N)

    # Pairwise potentials (B, N, N) — eqs. 3.9, 3.13, 3.14, 3.15
    skyrme = _calculate_skyrme_potential(
        coords_pairwise_boosted,
    )
    yukawa = _calculate_yukawa_potential(
        coords_pairwise_boosted,
    )
    coulomb = _caclulate_coulomb_potential(
        coords_pairwise_boosted,
        proton_proton_mask,
    )
    pauli = _caclulate_pauli_potential(
        momenta_pairwise_boosted,
        coords_pairwise_boosted,
        same_type_matrix,
    )

    # Element-wise sum of pairwise potentials, weighted by embedding similarity
    pairwise_potential = skyrme + yukawa + coulomb + pauli  # (B, N, N)
    pair_contributions = pairwise_potential * mask_ij.float()

    n_per_event = mask.sum(dim=1).clamp(min=1)  # (B,)
    n_pairs_per_event = (n_per_event * (n_per_event - 1)).clamp(min=1)  # (B,), i!=j pairs

    loss_pair = pair_contributions.sum(dim=(1, 2)) / n_pairs_per_event  # (B,)
    return loss_pair


def total_potential_energy(
    nucleons: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Extensive total pairwise QMD potential energy [MeV] of the masked set.

    Sum over unordered real pairs (i < j) of the four pairwise potentials
    (Skyrme + Yukawa + Coulomb + Pauli).  Unlike ``binding_energy`` — which
    returns the per-ordered-pair *mean* (an intensive quantity) — this is
    *extensive*: the energies of disjoint fragments add up, so a fragmentation
    can be scored as a proper energy balance against the parent cloud.

    ``binding_energy`` computes ``Σ_{i≠j} v / (N(N-1))`` (ordered pairs), so the
    total over unordered pairs is ``binding_energy · N(N-1) / 2``.  Sets with
    fewer than two real nucleons have no pairs and return 0.
    """
    N = mask.sum(dim=1).float()                       # (B,)
    n_ordered_pairs = N * (N - 1)                      # 0 when N < 2
    return binding_energy(nucleons, mask) * n_ordered_pairs / 2.0


def fragment_energy(
    nucleons: torch.Tensor,
    mask: torch.Tensor,
    type_index: int = 7,
) -> torch.Tensor:
    """Extensive energy of a fragment [MeV]: QMD potential + asymmetry penalty.

    ``E(S) = U(S) + a₄·(A/2 − Z)²/A``

    ``U`` is configuration-aware but isospin-blind for neutrons; the asymmetry
    term adds the missing isospin cost.  Both are extensive, so fragment
    energies add and a split can be scored as an energy balance:
    ``Q = (E(parent) − Σ E(leaf)) / N_parent``.  Lower ``E`` = more bound, so
    forming a neutron-rich fragment raises ``Σ E(leaf)`` and lowers ``Q``.
    """
    A = mask.sum(dim=1).float()                                       # (B,)
    Z = ((nucleons[..., type_index] == 1) & mask).sum(dim=1).float()  # (B,)
    return total_potential_energy(nucleons, mask) + asymmetry_energy(A, Z)


def weizsacker_qmd_energy(
    nucleons: torch.Tensor,
    mask: torch.Tensor,
    type_index: int = 7,
    qmd_weight: float = 1.0,
    scale: str = "extensive",
) -> torch.Tensor:
    """Per-node energy for the Weizsäcker reward:  U = qmd_weight·V − W.

    The split reward ``q = (U_parent − Σ U_child) / N`` maximizes W and minimizes
    V (weighted by ``qmd_weight``) — "maximize Weizsäcker binding, minimize QMD".

    ``scale`` selects how W and V are measured:

    - ``"extensive"`` (default): ``W`` = ``weizsacker_formula`` (total binding
      B(A,Z) [MeV]), ``V`` = ``total_potential_energy`` (Σ over pairs [MeV]).
      Both are **extensive** (additive over disjoint fragments), so a split's
      reward is a genuine energy balance: ``V_parent − ΣV_child`` is exactly the
      inter-fragment interaction the split breaks, and W's surface term makes
      merging into fewer, larger nuclei favorable — both resist over-splitting.
    - ``"per_nucleon"``: ``W`` = ``weizsacker_per_nucleon_formula`` (B/A),
      ``V`` = ``binding_energy`` (mean pairwise) — intensive / "affinity" scale.
      Kept for comparison; it washes out the extensive energy signal and tends to
      over-split into free nucleons (see the analysis in the branch history).

    Note on sign: on the spectator data ``V`` is Pauli-repulsion-dominated and
    **positive**, so "minimize QMD" means reducing repulsion between over-close
    nucleons (which favors splitting) — the opposite pressure to W's merge bias;
    ``qmd_weight`` tunes the balance.

    W is undefined for A < 2 (``weizsacker_formula`` asserts A > 0 and the
    liquid-drop picture is meaningless for a free nucleon), so W = 0 there; the
    QMD energy already returns 0 for sets with no pairs.
    """
    A = mask.sum(dim=1).float()                                       # (B,)
    Z = ((nucleons[..., type_index] == 1) & mask).sum(dim=1).float()  # (B,)
    W = torch.zeros_like(A)
    big = A >= 2
    if scale == "extensive":
        if big.any():
            W[big] = weizsacker_formula(A[big], Z[big])
        V = total_potential_energy(nucleons, mask)
    elif scale == "per_nucleon":
        if big.any():
            W[big] = weizsacker_per_nucleon_formula(A[big], Z[big])
        V = binding_energy(nucleons, mask)
    else:
        raise ValueError(f"unknown scale {scale!r}; expected 'extensive' or 'per_nucleon'")
    return qmd_weight * V - W


def _calculate_skyrme_potential(
    coords_pairwise_boosted: torch.Tensor,  # (B, N, N, 3)
) -> torch.Tensor:
    """
    Two-body Skyrme pairwise contribution (eq. 3.9).
    """
    diff = coords_pairwise_boosted - coords_pairwise_boosted.transpose(1, 2)  # (B, N, N, 3)
    dist_sq = (diff ** 2).sum(dim=-1)  # (B, N, N)
    skyrme_potential = (
        t_1
        * (alpha / math.pi) ** 1.5
        * torch.exp(- alpha * dist_sq)
    )
    return skyrme_potential

def _caclulate_pauli_potential(
    momenta_pairwise_boosted: torch.Tensor,    # (B, N, N, 3)
    coords_pairwise_boosted: torch.Tensor,     # (B, N, N, 3)
    particle_type_equality_matrix: torch.Tensor,  # (B, N, N)
) -> torch.Tensor:
    """Pauli potential (eq. 3.15), evaluated in the CM frame of each pair."""
    diff_r = coords_pairwise_boosted - coords_pairwise_boosted.transpose(1, 2)   # (B, N, N, 3)
    diff_p = momenta_pairwise_boosted - momenta_pairwise_boosted.transpose(1, 2)  # (B, N, N, 3)

    dist_r_sq = (diff_r ** 2).sum(dim=-1)  # (B, N, N)
    dist_p_sq = (diff_p ** 2).sum(dim=-1)  # (B, N, N)

    pauli_potential = (
        V_0_Pauli
        * (planck_constant / (q_0 * p_0)) ** 3
        * torch.exp(
            - dist_r_sq / (2 * q_0 ** 2)
            - dist_p_sq / (2 * p_0 ** 2)
        )
        * particle_type_equality_matrix
    )
    return pauli_potential

def _caclulate_coulomb_potential(
    coords_pairwise_boosted: torch.Tensor,  # (B, N, N, 3)
    proton_proton_mask: torch.Tensor,       # (B, N, N)
) -> torch.Tensor:
    """Coulomb potential (eq. 3.14): V_Coul^ij = Z_i*Z_j*e² / |r_i - r_j|."""
    diff = coords_pairwise_boosted - \
        coords_pairwise_boosted.transpose(1, 2)  # (B, N, N, 3)
    dist = torch.norm(diff, p=2, dim=-1).clamp(min=1e-8)  # (B, N, N)
    return proton_proton_mask * e_sq / dist

def _calculate_yukawa_potential(
    coords_pairwise_boosted: torch.Tensor,  # (B, N, N, 3)
) -> torch.Tensor:
    """Yukawa potential (eq. 3.13): V_Yuk^ij = V_0^Yuk * exp(-|r_ij|/γ_Y) / |r_ij|."""
    diff = coords_pairwise_boosted - \
        coords_pairwise_boosted.transpose(1, 2)  # (B, N, N, 3)
    dist = torch.norm(diff, p=2, dim=-1).clamp(min=1e-8)  # (B, N, N)
    return V_0_Yuk * torch.exp(-dist / gamma_Y) / dist

def _pairwise_lorentz_boost(
    momenta: torch.Tensor,
    energies: torch.Tensor,
    coords: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Lorentz boost each nucleon's momentum and coordinate to the center-of-mass
    frame of its pair.

    Returns:
        momenta_prime  (B, N, N, 3): momenta_prime[b,i,j,:] = LorentzBoost(p_i, beta_ij)
        coords_prime   (B, N, N, 3): coords_prime[b,i,j,:]  = LorentzBoost(x_i, beta_ij)
    """
    p_i = momenta.unsqueeze(2)   # (B, N, 1, 3)
    p_j = momenta.unsqueeze(1)   # (B, 1, N, 3)
    E_i = energies.unsqueeze(2)  # (B, N, 1)
    E_j = energies.unsqueeze(1)  # (B, 1, N)

    # Pairwise CM frame velocity
    E_sum = (E_i + E_j).clamp(min=1e-8)
    beta_vec = (p_i + p_j) / E_sum.unsqueeze(-1)  # (B, N, N, 3)

    beta_sq = (beta_vec ** 2).sum(dim=-1, keepdim=True).clamp(max=1 - 1e-8)  # (B, N, N, 1)
    beta = torch.sqrt(beta_sq)  # (B, N, N, 1)
    beta_hat = beta_vec / beta.clamp(min=1e-10)  # (B, N, N, 3)
    gamma = 1.0 / torch.sqrt(1.0 - beta_sq)  # (B, N, N, 1)

    # --- Momentum boost ---
    p_parallel = (p_i * beta_hat).sum(dim=-1, keepdim=True)  # (B, N, N, 1)
    p_perp = p_i - p_parallel * beta_hat                     # (B, N, N, 3)
    momenta_prime = p_perp + gamma * \
        (p_parallel - beta * E_i.unsqueeze(-1)) * beta_hat

    # --- Coordinate boost ---
    x_i = coords.unsqueeze(2)                                # (B, N, 1, 3)
    x_parallel = (x_i * beta_hat).sum(dim=-1, keepdim=True)  # (B, N, N, 1)
    x_perp = x_i - x_parallel * beta_hat                     # (B, N, N, 3)
    coords_prime = x_perp + gamma * x_parallel * beta_hat

    return momenta_prime, coords_prime
