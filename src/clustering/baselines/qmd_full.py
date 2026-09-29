"""The SACA fragment energy ζ with the full QMD potential.

Written to the four formulas in the SACA paper (Puri & Aichelin) and the
"with Pauli-potential" column of its parameter table, and nothing else:

    ζ_f = Σ_α [ sqrt((p_α − P_f/N_f)² + m_α²) − m_α  +  ½ Σ_{β≠α} V_αβ ]  <  L_be N_f

    V_Yuk  = V₀^Yuk exp(−r/γ_Y) / r
    V_Coul = Z_i Z_j e² / r
    V_Pau  = V₀^Pau (ħ/(q₀p₀))³ exp(−r²/2q₀² − Δp²/2p₀²) δ_ττ' δ_σσ'
    V_Sk   = (1/2!) t₁ Σ' δ(x_j−x_k) + (1/3!) t₂ Σ' δ(x_j−x_k) δ(x_j−x_l)

Every term is evaluated in the **fragment rest frame**, the frame the paper
computes E_bind in.  The two existing energies in this repo each miss part of
this: ``physics.py`` has no density-dependent Skyrme term (so nothing saturates,
see tests/test_nuclear_matter.py) and no kinetic term, and
``qmd_energy.cluster_energy`` has no Pauli term, no Yukawa, and a Skyrme range
four times narrower in α than the table's (its exp(−r²/L) with L = 2.16 is
α = 0.46 fm⁻², the table has α = 0.1152).

What "the Skyrme term" is here
------------------------------
The δ-functions folded with Gaussian wave packets of width α give

    V_Sk² = t₁ Σ_{i<j} (α/π)^{3/2} exp(−α r_ij²)                        two-body
    V_Sk³ = t_γ/(γ+1) Σ_i ρ_i^γ,   ρ_i = (α/π)^{3/2} Σ_{j≠i} exp(−α r_ij²)

The 3-body term of the image (t₂ with three δ's) is the γ = 2 case of the
density-dependent term; the table's (t_γ, γ) = (188.2, 1.46) is the generalised
form, so that is what is used.  The (α/π)^{3/2} normalises the Gaussian to unit
integral, so ρ_i is a density in fm⁻³.  *This is my reading of the generalised
Skyrme term from the standard QMD literature, not something printed in the
images.*

Known limitation — this parameter set does not saturate
--------------------------------------------------------
With the table's parameters the Yukawa term is about -89 MeV/nucleon at ρ₀ and
linear in ρ (∫V_Yuk = 4π V₀ γ_Y² = -1069 MeV fm³, 12× the two-body Skyrme's
t₁ = -84.5), the Pauli energy is roughly flat in ρ, and the density-dependent
Skyrme term grows only as ρ^1.46 with a small coefficient.  Nothing opposes
compression: ζ/A falls monotonically with density (tests/test_qmd_full.py prints
the scan), the same failure as ``physics.py`` (tests/test_nuclear_matter.py).

Measured, with the default folded Yukawa:

* Random-Fermi-sphere nuclei at ρ₀ are *unbound*: ζ/A = +13.1 (O-16), +11.3
  (Ca-40), +7.7 (Sn-119).  On real MST fragments (30 events) the median ζ/A is
  +15 (A 3-9), +16 (A 10-39), +9 (A ≥ 40).  The folded Yukawa reaches about 2 fm,
  comparable to a small nucleus's radius, so a droplet loses most of the bulk
  Yukawa (Ca-40: -28 MeV/A against -89 in bulk).
* Random sampling is not a ground state of this Hamiltonian, so those numbers do
  not say the parameters cannot bind.  Minimising the classical energy over
  positions and momenta (Adam, 4 seeds, local minima) gives, for O-16 / Ca-40:
  point Yukawa -36 / -186 (runaway collapse), folded -1.9 / -27 at
  spin_factor 1 and -8.4 / -38 at 0.5, against -8.0 / -8.6 empirical.  Folding
  removes the runaway; it does not give saturation, since ζ/A still falls by
  25-30 MeV from O-16 to Ca-40.
* With the point Yukawa the earlier "reasonable" cold nuclei (Ca-40 -13.0) were
  partly the r → 0 divergence compensating for surface loss.

The Yukawa term is Gaussian-folded
----------------------------------
The image writes V_Yuk for point particles, V₀ e^{−r/γ_Y}/r.  That diverges as
r → 0 and made ζ of small MST fragments wildly scattered (A = 3-9: 10th percentile
−58 MeV/A), because two nucleons that happen to sit 0.3 fm apart contribute
−210 MeV.  The nucleons are Gaussian wave packets, so the pair interaction is the
Yukawa convolved with the *same* relative Gaussian the Skyrme term uses,
exp(−α r²), i.e. σ² = 1/(2α) per component (μ = 1/γ_Y):

    V(r) = V₀/(2r) e^{−r²/2σ²} [ erfcx(A) − erfcx(B) ],
           A = (μσ² − r)/(√2 σ),   B = (μσ² + r)/(√2 σ)

(Aichelin's closed form, rewritten with erfcx = e^{x²} erfc so it does not
overflow.)  Checked against 2-D quadrature of the convolution to 0.2 % at
r = 0.3-8 fm, and ∫V d³r = 4π V₀ γ_Y² is preserved exactly, so the bulk Yukawa
energy is *unchanged* (measured −87.7 point vs −89.0 folded MeV/A at ρ₀) — this
fixes the short-range divergence and nothing else, and it makes finite fragments
much less bound (see below).  ``yukawa="point"`` gives the image's literal
formula.  V(0) = −4.8 MeV where the point form gives −210 at 0.3 fm.

Two things the images do not settle
-----------------------------------
* **Spin.**  V_Pau carries δ_σσ' and the dataset has no spin.  ``spin_factor``
  multiplies the Pauli term.  0.5 is the average over randomly assigned spins;
  1.0 treats every same-isospin pair as same-spin (what ``physics.py`` does) and
  is the default *only because it is what physics.py does*.  With the point
  Yukawa it also gave the closest cold nuclei; with the folded Yukawa that
  argument no longer holds (the classical O-16 minimum is -8.4 at 0.5 and -1.9
  at 1.0, but Ca-40 is -38 vs -27), so this default is unsettled.
* **Off-shell energies.**  The boost to the fragment frame uses β = ΣP/ΣE.
  The stored ``fE`` is off-shell by 18-21 MeV, so E is rebuilt from p and the
  nucleon mass — the same on-shell energy the kinetic term already uses.

Sign convention: negative means bound.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import erfcx

from clustering.baselines.qmd_energy import M_N, M_P, boost_to_rest_frame

# Parameters: SACA paper, "with Pauli-potential" column.  physics.py holds the
# same values; they are repeated here so this module has no torch dependency.
ALPHA = 0.1152        # fm^-2
T_1 = -84.5           # MeV fm^3     two-body Skyrme
T_GAMMA = 188.2       # MeV fm^(3γ)  density-dependent Skyrme
GAMMA = 1.46
V_0_YUK = -85.1       # MeV fm
GAMMA_Y = 1.0         # fm
V_0_PAULI = 98.95     # MeV
Q_0 = 2.16            # fm
P_0 = 120.0           # MeV/c
HBARC = 197.3269804   # MeV fm
E_SQ = 1.4399644      # MeV fm
GEV_TO_MEV = 1000.0

SPIN_FACTOR = 1.0     # see module docstring
YUKAWA = "folded"     # "folded" | "point"; see yukawa_pair


@dataclass
class FullTerms:
    """Breakdown in MeV, so a wrong term can be found without guessing."""
    kinetic: float = 0.0
    skyrme2: float = 0.0
    skyrme3: float = 0.0
    yukawa: float = 0.0
    coulomb: float = 0.0
    pauli: float = 0.0

    @property
    def potential(self) -> float:
        return self.skyrme2 + self.skyrme3 + self.yukawa + self.coulomb + self.pauli

    @property
    def total(self) -> float:
        return self.kinetic + self.potential


def yukawa_pair(d: np.ndarray, mode: str = YUKAWA) -> np.ndarray:
    """Pair Yukawa potential [MeV] at separations ``d`` [fm]; 0 where d is inf.

    ``"point"`` is V₀ e^{−d/γ_Y}/d.  ``"folded"`` is the same convolved with the
    wave-packet Gaussian (see the module docstring).  ``"off"`` is 0.
    """
    d = np.asarray(d, dtype=np.float64)
    out = np.zeros_like(d)
    ok = np.isfinite(d)
    r = np.maximum(d[ok], 1e-6)
    if mode == "off":
        return out
    if mode == "point":
        out[ok] = V_0_YUK * np.exp(-r / GAMMA_Y) / r
        return out
    if mode != "folded":
        raise ValueError(f"unknown yukawa mode {mode!r}; expected 'folded', 'point' or 'off'")
    mu = 1.0 / GAMMA_Y
    s2 = 1.0 / (2.0 * ALPHA)
    s = np.sqrt(s2)
    a = (mu * s2 - r) / (np.sqrt(2.0) * s)
    b = (mu * s2 + r) / (np.sqrt(2.0) * s)
    # erfcx(a) overflows for a < ~-26.  By then the second term is e^{-r²/2σ²}-
    # suppressed to nothing and erfc(a) -> 2, leaving V₀ e^{μ²σ²/2 - μr}/r.
    far = a < -20.0
    with np.errstate(over="ignore", invalid="ignore"):
        near = V_0_YUK / (2.0 * r) * np.exp(-r ** 2 / (2.0 * s2)) * (erfcx(a) - erfcx(b))
    tail = V_0_YUK / r * np.exp(mu ** 2 * s2 / 2.0 - mu * r)
    out[ok] = np.where(far, tail, near)
    return out


def full_cluster_energy(nucleons: np.ndarray, *, spin_factor: float = SPIN_FACTOR,
                        yukawa: str = YUKAWA, boost: bool = True) -> FullTerms:
    """ζ_f × N_f [MeV] for one fragment.  ``nucleons`` is (n, 8): p(3) E x(3) type.

    Momenta in GeV/c, coordinates in fm, type = +1 proton / −1 neutron.
    """
    n = nucleons.shape[0]
    terms = FullTerms()
    if n == 0:
        return terms

    p = nucleons[:, 0:3].astype(np.float64)
    r = nucleons[:, 4:7].astype(np.float64)
    is_p = nucleons[:, 7] > 0
    m = np.where(is_p, M_P, M_N)
    E = np.sqrt((p ** 2).sum(1) + m ** 2)          # on shell, not the stored fE

    if boost:
        p, r = boost_to_rest_frame(p, E, r)

    p_rel = p - p.sum(0) / n
    terms.kinetic = float((np.sqrt((p_rel ** 2).sum(1) + m ** 2) - m).sum()) * GEV_TO_MEV
    if n == 1:
        return terms   # a free nucleon has no partner

    d2 = ((r[:, None, :] - r[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)                    # kills the self term everywhere
    d = np.sqrt(d2)

    gauss = (ALPHA / np.pi) ** 1.5 * np.exp(-ALPHA * d2)     # (n, n), 0 on the diagonal
    terms.skyrme2 = 0.5 * T_1 * float(gauss.sum())           # Σ_{i<j}
    rho = gauss.sum(1)
    terms.skyrme3 = T_GAMMA / (GAMMA + 1.0) * float((rho ** GAMMA).sum())

    terms.yukawa = 0.5 * float(yukawa_pair(d, yukawa).sum())

    pp = np.outer(is_p, is_p).astype(np.float64)
    np.fill_diagonal(pp, 0.0)
    terms.coulomb = 0.5 * float((E_SQ * pp / d).sum())

    # Pauli: same isospin only.  Δp in MeV/c, since p_0 is.
    dp2 = (((p[:, None, :] - p[None, :, :]) * GEV_TO_MEV) ** 2).sum(-1)
    same = (is_p[:, None] == is_p[None, :]).astype(np.float64)
    np.fill_diagonal(same, 0.0)
    pauli = (V_0_PAULI * (HBARC / (Q_0 * P_0)) ** 3
             * np.exp(-d2 / (2 * Q_0 ** 2) - dp2 / (2 * P_0 ** 2)) * same)
    terms.pauli = spin_factor * 0.5 * float(pauli.sum())
    return terms
