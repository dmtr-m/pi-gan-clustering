import math

import torch
import torch.nn as nn

import numpy as np

# Acyclic: qmd_energy imports numpy only, never this module.  It is the
# energy the SACA baselines minimize, and saca_qmd_minus_b_energy below
# exposes it as a reward so the annealer and the policy can share one.
from clustering.baselines.qmd_energy import cluster_energy
from clustering.baselines.qmd_full import full_cluster_energy


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


# ─── Bethe-Weizsäcker mass formulas as used by SACA 2.1 ───────────────────────
#
# Vermani, Dhawan, Goyal, Puri & Aichelin, arXiv:0912.5130, Eqs. (3)-(10).
# These are the paper's own coefficients and forms, kept separate from
# ``weizsacker_formula`` above (which the RL reward uses and which has different
# coefficients, a Z^2 Coulomb term and an A^(-3/4) pairing term).
#
# Two differences from the plain BW formula, and only two — the coefficients are
# identical between them:
#   * asymmetry is damped at low A by  1 / (1 + exp(-A/17))
#   * pairing is damped at low A by    (1 - exp(-A/30))
#
# Verified against the paper's own worked value: Fe-56 -> 489.4 MeV total,
# 8.74 MeV/nucleon.  See tests/test_binding_formulas.py.
BWM_A_V = 15.777   # MeV   volume
BWM_A_S = 18.34    # MeV   surface
BWM_A_C = 0.71     # MeV   Coulomb, paired with Z(Z-1) — NOT Z^2
BWM_A_SYM = 23.21  # MeV   asymmetry
BWM_A_P = 12.0     # MeV   pairing, paired with A^(-1/2) — NOT A^(-3/4)


def bethe_weizsacker(A: torch.Tensor, Z: torch.Tensor,
                     modified: bool = True) -> torch.Tensor:
    """Binding energy [MeV], positive for a bound nucleus.

    ``modified=True`` is the BWM form of Samanta & Adhikari that SACA 2.1 uses
    as its admissibility criterion; ``False`` is the plain BW form.  Note the
    threshold in SACA is *per nucleon*, so divide by A before comparing with
    ``zeta``.
    """
    A = A.float()
    Z = Z.float()
    N = A - Z

    asym_denom = A * (1.0 + torch.exp(-A / 17.0)) if modified else A
    pairing = BWM_A_P * torch.pow(A, -0.5)
    if modified:
        pairing = pairing * (1.0 - torch.exp(-A / 30.0))

    even_even = (torch.remainder(Z, 2) == 0) & (torch.remainder(N, 2) == 0)
    odd_odd = (torch.remainder(Z, 2) == 1) & (torch.remainder(N, 2) == 1)
    delta = torch.where(even_even, pairing, torch.zeros_like(A))
    delta = torch.where(odd_odd, -pairing, delta)
    # delta is 0 for odd A, which both parity masks already exclude.

    return (
        BWM_A_V * A
        - BWM_A_S * torch.pow(A, 2.0 / 3.0)
        - BWM_A_C * Z * (Z - 1.0) / torch.pow(A, 1.0 / 3.0)
        - BWM_A_SYM * (A - 2.0 * Z) ** 2 / asym_denom
        + delta
    )


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

# The parquet files store momenta and energies in GeV/c and GeV (fE spans
# 0.94–3.83), while every constant here is MeV-based and coordinates are in fm.
# Only the Pauli term reads momenta — Skyrme and Yukawa use coordinates alone and
# Coulomb is e_sq [MeV*fm] / r [fm] — so this conversion is applied there and
# nowhere else.  The Lorentz boosts are unaffected either way: they use β = p/E,
# which is dimensionless as long as p and E share a unit.
MOMENTUM_TO_MEV = 1000.0  # GeV/c (as stored) -> MeV/c (as p_0 expects)


def pairwise_potential_matrix(
    nucleons: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Pairwise QMD potential ``V_ij`` [MeV] for every real pair.  Returns (B, N, N).

    Skyrme + Yukawa + Coulomb + Pauli, each evaluated in the rest frame of the
    pair, with padding and the diagonal zeroed.  ``binding_energy`` and
    ``total_potential_energy`` are sums over this matrix; SACA needs the matrix
    itself, because its annealing moves change one nucleon's cluster membership
    at a time and the energy change is then a row sum rather than a full
    recomputation.
    """
    N = mask.shape[1]
    off_diag = ~torch.eye(N, dtype=torch.bool, device=mask.device).unsqueeze(0)
    mask_ij = mask.unsqueeze(2) & mask.unsqueeze(1) & off_diag  # (B, N, N)

    momenta = nucleons[..., 0:3]
    energies = nucleons[..., 3]
    coords = nucleons[..., 4:7]
    types = nucleons[..., 7]

    momenta_pairwise_boosted, coords_pairwise_boosted = _pairwise_lorentz_boost(
        momenta, energies, coords
    )
    same_type_matrix = (types.unsqueeze(2) == types.unsqueeze(1)).float()
    proton_proton_mask = (
        (types == 1).unsqueeze(2) & (types == 1).unsqueeze(1)
    ).float()

    pairwise = (
        _calculate_skyrme_potential(coords_pairwise_boosted)
        + _calculate_yukawa_potential(coords_pairwise_boosted)
        + _caclulate_coulomb_potential(coords_pairwise_boosted, proton_proton_mask)
        + _caclulate_pauli_potential(
            momenta_pairwise_boosted, coords_pairwise_boosted, same_type_matrix
        )
    )
    return pairwise * mask_ij.float()


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
      inter-fragment interaction the split breaks.  The two terms pull in
      *opposite* directions — W's surface term favors merging into fewer, larger
      nuclei, while V rewards cutting the weak or mildly repulsive long-range
      bonds between clusters.  W wins on cold matter (see below), so the pair
      still resists over-splitting overall, but not for the reason the earlier
      version of this note gave.
    - ``"per_nucleon"``: ``W`` = ``weizsacker_per_nucleon_formula`` (B/A),
      ``V`` = ``binding_energy`` (mean pairwise) — intensive / "affinity" scale.
      Kept for comparison; it washes out the extensive energy signal and tends to
      over-split into free nucleons (see the analysis in the branch history).

    Both W and V are in **MeV** — every QMD constant carries its units
    (``t_1`` MeV·fm³, ``V_0_Yuk`` MeV·fm, ``e_sq`` MeV·fm, ``V_0_Pauli`` MeV) and
    the Weizsäcker coefficients are MeV against dimensionless A and Z.  So
    ``qmd_weight`` is a genuine physics weight, not a unit conversion.

    **What each term does** (measured on 512 HSE SpectatorsLeft events, MST at
    d_cut=2.0 against a random 6-way assignment).  Writing the reward as

        q·N = V_cut + (ΣW_leaf − W_parent),   V_cut ≡ V_parent − ΣV_leaf

    makes the two pressures explicit — ``V_cut`` is the summed pair potential of
    the bonds the split breaks, and ΔW the binding gained or lost by reshaping
    the mass into new fragments:

                        V(parent)   V_cut      ΔW      q·N
        MST d_cut=2.0    -1831.8    +68.9   -119.7    -50.7
        random 6-way     -1831.8   -1522.8  -213.9  -1736.7

    ``V(parent)`` is **negative**: the spectator source is bound.  ``V_cut`` is
    *positive* for MST because MST cuts at 2–3 fm, where the pair potential is
    mildly repulsive (+0.96 MeV at 2 fm), so severing those bonds releases
    energy.  A random assignment instead tears through the attractive short-range
    core and pays -1522.8 for it, which is why the reward separates physical
    clustering from noise by more than an order of magnitude.

    What resists fragmentation is therefore **W's surface term**, not the QMD
    potential: more fragments means more total surface (Σ Aᵢ^⅔ > (ΣA)^⅔), so
    ΔW = -119.7 outweighs V_cut = +68.9 and q·N stays negative.  That is correct
    for *cold* matter — and it is why the global optimum of this reward is not to
    split at all (q = 0 identically when the single leaf is the parent).  Real
    multifragmentation is driven by excitation energy paying that surface cost,
    and no E* term exists here yet.  See ``STABILITY_VALLEY_BRAINSTORM.md``.

    HISTORY: this note used to read "V is Pauli-repulsion-dominated and positive,
    so minimize-QMD favors splitting".  That was an accurate description of a
    unit bug — momenta are stored in GeV/c while ``p_0`` is MeV/c, so the Pauli
    term never switched off and the pair potential was positive at short range,
    inverting the objective.  Fixed in the commit that rewrote this docstring;
    every reward recorded before it is on the inverted scale.

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


def qmd_minus_b_energy(
    nucleons: torch.Tensor,
    mask: torch.Tensor,
    type_index: int = 7,
    bwm_weight: float = 1.0,
    binding: str = "bwm",
) -> torch.Tensor:
    """Per-fragment energy of the SACA "QMD - B" objective:  ``E = V - w*B``.

    This is the baseline objective from ``experiments/qmd_minus_b.py``
    (``energy_model="qmd_minus_b"`` in ``SacaParams``) carried over to the RL
    reward, so the divisive slot-attention policy and the annealer minimize the
    same quantity.  ``V`` is the extensive QMD pairwise potential and ``B`` the
    mass-formula binding energy, positive for a bound nuclide, so subtracting it
    credits binding twice: once microscopically, once phenomenologically.  The
    split reward ``q = (E_parent - sum E_child) / N`` therefore rewards cutting
    weak or repulsive bonds *and* leaving behind fragments that sit on the
    stability valley.

    Two things differ from ``weizsacker_qmd_energy``, which has the same shape
    (``qmd_weight*V - W``) and is *not* changed by this function:

    * **which mass formula.**  ``binding="bwm"`` is ``bethe_weizsacker(...,
      modified=True)`` — the Samanta-Adhikari form SACA 2.1 uses and the one the
      QMD - B baseline ran on.  ``binding="bw"`` is ``weizsacker_formula``, what
      the ``weizsacker_qmd`` reward uses.  They are interchangeable above A ~ 9
      and very different below it; see the table below.
    * **where the weight sits.**  The baseline scans lambda on *B*
      (``bwm_weight``), not on V, so lambda = 0 degenerates to the plain QMD
      potential rather than to minus the mass formula.  ``binding="bw"`` with
      ``bwm_weight=1`` reproduces ``weizsacker_qmd_energy(scale="extensive",
      qmd_weight=1)`` exactly; ``tests/test_qmd_minus_b.py`` asserts it.

    **The two formulas disagree where this model lives.**  B [MeV], against the
    experimental value where the nuclide exists:

        nuclide     A   Z       BW      BWM      exp
        d           2   1   -17.54    +1.89     2.22
        t           3   1    +1.83    +4.97     8.48
        He-4        4   2   +28.38   +16.75    28.30
        H-4         4   1   -18.02    +3.18    (does not exist)
        3n          3   0   -60.88   -28.70    (does not exist)
        Li-7        7   3   +38.38   +39.11    39.24
        Fe-56      56  26  +490.72  +489.40   492.25

    So the swap is **not** a uniform improvement, and should not be sold as one:

    * BW declares the deuteron unbound by 17.5 MeV.  Under ``weizsacker_qmd``
      the reward is therefore actively hostile to deuterons, which is worth
      knowing given that A = 2-4 comes out at ~0.55x in every clusterizer here.
      BWM puts it at +1.89 against 2.22 measured.
    * BWM underbinds He-4 by 11.6 MeV, where BW happens to land within 0.1 MeV
      of experiment (its A^(-3/4) pairing term is large, and even-even alpha is
      the case that flatters).
    * BWM calls H-4 bound (+3.18) — a nuclide that does not exist — and halves
      BW's penalty on neutron blobs.  BW's harshness there is a feature, not an
      accident, for a source that arrives at Z/A = 0.41.

    **B is extensive, so lambda is a collapse knob.**  Measured on the annealer
    (``figures/az_qmd_minus_b.png``, 1000 collisions): as lambda goes
    0.25 -> 0.5 -> 1 -> 1.5 the fragment multiplicity falls 3.73 -> 3.44 ->
    2.69 -> 2.20 per collision, bound mass climbs 127 -> 141 against a target of
    102.8, and the 7-bin mass RMS degrades 0.632 -> 0.973.  What it buys is
    composition: the A-Z band comes out narrow and on the valley, where the
    opposite sign (``qmd_plus_b``) scatters off it.  Expect the same trade here
    — the reward's global optimum is already "do not split" (q = 0 for the
    single-leaf partition), and -w*B deepens that minimum.

    B = 0 for A < 2: the liquid-drop picture is meaningless for a free nucleon,
    and ``V`` already returns 0 for a set with no pairs.
    """
    if binding not in ("bwm", "bw"):
        raise ValueError(f"unknown binding {binding!r}; expected 'bwm' or 'bw'")
    A = mask.sum(dim=1).float()                                       # (B,)
    Z = ((nucleons[..., type_index] == 1) & mask).sum(dim=1).float()  # (B,)
    B = torch.zeros_like(A)
    big = A >= 2
    if big.any():
        B[big] = (bethe_weizsacker(A[big], Z[big], modified=True) if binding == "bwm"
                  else weizsacker_formula(A[big], Z[big]))
    return total_potential_energy(nucleons, mask) - bwm_weight * B


def saca_qmd_minus_b_energy(
    nucleons: torch.Tensor,
    mask: torch.Tensor,
    type_index: int = 7,
    bwm_weight: float = 1.0,
    binding: str = "bwm",
    eos: str = "soft",
    yukawa: bool = False,
    asymmetry: bool = False,
) -> torch.Tensor:
    """QMD - B on the **baselines'** QMD energy:  ``E = zeta_QMD*A - w*B``.

    ``qmd_minus_b_energy`` above ports the baseline's *B* term onto the reward's
    existing potential.  This one ports the whole objective: the energy is
    ``clustering.baselines.qmd_energy.cluster_energy``, which is what
    ``experiments/qmd_minus_b.py`` and the SACA annealer actually minimize.
    The two differ in ways that are not cosmetic — measured on coordinate-MST
    fragments of real events:

        fragment          V (physics.py)   E (qmd_energy)
        A=126 Z=51            -33.4 MeV/A       -7.75 MeV/A
        A=105 Z=43            -36.8             -6.25
        A=4   Z=0 (4n)        -19.6             +10.34
        A=2   Z=2 (2p)         -3.65            +7.80

    ``physics.py``'s potential is Skyrme(two-body) + Yukawa + Coulomb + Pauli
    with **no density-dependent Skyrme term**, so nothing opposes compression:
    it over-binds the residue by roughly a factor of four against the empirical
    -8 MeV/nucleon, and it binds four free neutrons at -19.6 MeV/nucleon.
    ``tests/test_nuclear_matter.py`` fails on it for exactly this reason and
    always has — the density-dependent term was added to ``qmd_energy.py`` for
    the baselines and never to the reward.  ``cluster_energy`` also carries the
    internal kinetic energy in the fragment rest frame, which the reward's
    potential-only V omits entirely.

    Cost is not the reason to prefer one: measured at A = 130, ``cluster_energy``
    takes 318 us per fragment against 315 us per event for the batched torch
    potential.  It is numpy and **not differentiable**, which is fine — the split
    reward is detached before it reaches the loss (REINFORCE differentiates only
    log pi), so no gradient ever flowed through the energy anyway.

    Sign convention matches the rest of the reward: lower is more bound, and the
    split reward is ``q = (E_parent - sum E_child) / N_parent``.  A set with no
    real nucleons scores 0; a single nucleon scores 0 too (its rest-frame
    kinetic energy vanishes identically and it has no interaction partner).

    ``eos``/``yukawa``/``asymmetry`` are passed through to ``cluster_energy``;
    its defaults are the self-consistent BQMD set (see its docstring on why
    Yukawa is off).
    """
    if binding not in ("bwm", "bw"):
        raise ValueError(f"unknown binding {binding!r}; expected 'bwm' or 'bw'")
    x = nucleons.detach().cpu().double().numpy()
    m = mask.detach().cpu().numpy()
    out = np.zeros(x.shape[0], dtype=np.float64)
    for b in range(x.shape[0]):
        sub = x[b][m[b]]
        if sub.shape[0] == 0:
            continue
        e = cluster_energy(sub, eos=eos, yukawa=yukawa, asymmetry=asymmetry).total
        A = sub.shape[0]
        if A >= 2 and bwm_weight != 0.0:
            Z = int((sub[:, type_index] == 1).sum())
            A_t, Z_t = torch.tensor([float(A)]), torch.tensor([float(Z)])
            B = float(bethe_weizsacker(A_t, Z_t, modified=True) if binding == "bwm"
                      else weizsacker_formula(A_t, Z_t))
            e = e - bwm_weight * B
        out[b] = e
    return torch.as_tensor(out, dtype=nucleons.dtype, device=nucleons.device)


def zeta_correct_energy(
    nucleons: torch.Tensor,
    mask: torch.Tensor,
    type_index: int = 7,
    bwm_weight: float = 1.0,
    spin_factor: float = 0.5,
    yukawa: str = "folded",
) -> torch.Tensor:
    """The SACA paper's zeta with the full QMD, plus a Weizsacker term, as a reward.

    ``E(f) = T + V_Sk2 + V_Sk3 + V_Yuk + V_Coul + V_Pau + lambda * bwd(A, Z)``,
    with ``bwd = -B_BWM`` (negative when bound), everything in the fragment rest
    frame.  This is exactly ``SacaParams(energy_model="zeta_correct")`` — the
    energy is ``baselines.qmd_full.full_cluster_energy`` and B is
    ``bethe_weizsacker(modified=True)`` — so the annealer and the policy
    minimize the same quantity (FORMULAS.md 5, 5a).  The split reward built on
    it is, as for every reward here, ``q = (E_parent - sum E_child) / N_parent``.

    ``spin_factor`` and ``yukawa`` are the two modelling readings documented in
    ``qmd_full`` (no spin in the dataset; Yukawa point vs Gaussian-folded).  The
    defaults are the ones behind the ``az_zeta_correct_yukawa_*`` figures.

    Not differentiable (numpy), which is fine: the reward is detached before the
    loss.  A set with no nucleons scores 0, a single nucleon scores 0, and B is 0
    below A = 2, as in ``saca_qmd_minus_b_energy``.
    """
    x = nucleons.detach().cpu().double().numpy()
    m = mask.detach().cpu().numpy()
    out = np.zeros(x.shape[0], dtype=np.float64)
    for b in range(x.shape[0]):
        sub = x[b][m[b]]
        A = sub.shape[0]
        if A < 2:
            continue
        e = full_cluster_energy(sub, spin_factor=spin_factor, yukawa=yukawa).total
        if bwm_weight != 0.0:
            Z = int((sub[:, type_index] == 1).sum())
            B = float(bethe_weizsacker(torch.tensor([float(A)]),
                                       torch.tensor([float(Z)]), modified=True))
            e = e - bwm_weight * B
        out[b] = e
    return torch.as_tensor(out, dtype=nucleons.dtype, device=nucleons.device)


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
    """Pauli potential (eq. 3.15), evaluated in the CM frame of each pair.

    ``p_0`` is in MeV/c and the prefactor ``(ℏc / (q_0·p_0))³`` is dimensionless
    only in that unit, so the momentum difference must be converted from the
    GeV/c the dataset stores.  Without the conversion Δp² is ~10⁶ times too small
    against ``2·p_0²``, the momentum Gaussian never leaves 1, and the Pauli
    repulsion stays switched on for *every* same-isospin pair.  It then swamps
    the Yukawa attraction at all separations (+28 MeV at 1 fm), which inverts the
    sign of the pair potential at short range — and with it the split reward,
    which is driven by the potential of the bonds a split breaks.  Measured
    consequence: a random partition scored 7x better than MST cluster
    recognition, and an untrained network beat every trained one.
    """
    diff_r = coords_pairwise_boosted - coords_pairwise_boosted.transpose(1, 2)   # (B, N, N, 3)
    diff_p = (
        momenta_pairwise_boosted - momenta_pairwise_boosted.transpose(1, 2)
    ) * MOMENTUM_TO_MEV  # (B, N, N, 3)

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
