"""Check the SACA 2.1 mass formula against the paper's own numbers.

Run:  .venv/bin/python tests/test_binding_formulas.py

Reference: Vermani et al., arXiv:0912.5130, Eqs. (3)-(10).  The paper works out
Fe-56 explicitly, which pins every coefficient, the Z(Z-1) Coulomb form, the
A^(-1/2) pairing power and both damping factors at once — get any of them wrong
and the total moves by MeV.
"""
import torch

from clustering.physics import bethe_weizsacker

T = lambda v: torch.tensor([float(v)])

# (name, A, Z, experimental B/A [MeV/nucleon], AME2020)
NUCLEI = [
    ("He-4",   4,  2, 7.074),
    ("C-12",  12,  6, 7.680),
    ("O-16",  16,  8, 7.976),
    ("Ca-40", 40, 20, 8.551),
    ("Fe-56", 56, 26, 8.790),
    ("Sn-120", 120, 50, 8.505),
    ("Xe-132", 132, 54, 8.428),
]


def main() -> bool:
    ok = True

    # The paper's own worked value.
    fe = float(bethe_weizsacker(T(56), T(26), modified=True).item())
    hit = abs(fe - 489.4) < 0.1
    ok &= hit
    print(f"Fe-56 BWM total : {fe:.2f} MeV  (paper: 489.4)  "
          f"-> {'PASS' if hit else 'FAIL'}")
    print(f"Fe-56 BWM per A : {fe / 56:.3f} MeV/nucleon  (paper: 8.74)\n")

    print(f"{'nucleus':<9}{'BWM B/A':>10}{'BW B/A':>9}{'experiment':>12}{'BWM err':>10}")
    worst = 0.0
    for name, A, Z, exp_ba in NUCLEI:
        bwm = float(bethe_weizsacker(T(A), T(Z), modified=True).item()) / A
        bw = float(bethe_weizsacker(T(A), T(Z), modified=False).item()) / A
        err = bwm - exp_ba
        worst = max(worst, abs(err))
        print(f"{name:<9}{bwm:10.3f}{bw:9.3f}{exp_ba:12.3f}{err:+10.3f}")

    # The liquid-drop model is not expected to describe the lightest nuclei —
    # He-4 is a closed-shell four-nucleon system and the formula misses it by
    # ~2.9 MeV/nucleon no matter how correctly it is transcribed.  The paper's
    # own damping factors (1+e^{-A/17}, 1-e^{-A/30}) exist to soften exactly
    # this region, not to fix it.  So the tolerance applies from A >= 12, where
    # the formula is meant to hold; Fe-56 above is the exact check on the
    # transcription itself.
    heavy_worst = max(abs(float(bethe_weizsacker(T(A), T(Z), True).item()) / A - e)
                      for _, A, Z, e in NUCLEI if A >= 12)
    mass_ok = heavy_worst < 0.6
    ok &= mass_ok
    print(f"\n  worst |BWM - experiment|, A >= 12 = {heavy_worst:.3f} MeV/nucleon  "
          f"-> {'PASS' if mass_ok else 'FAIL'}")
    print(f"  (He-4 is off by {worst:.2f}; expected, the LDM does not cover A < 12)")
    return bool(ok)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
