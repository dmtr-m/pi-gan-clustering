"""Candidate sharpenings of the Weizsäcker landscape (total B, the reward's default scale).

D(A,Z) = B_max(A) - B_smooth(A,Z) >= 0 is the deficit from the valley, computed from
the smooth part of B (no pairing, so the sharpened shapes are not jagged).
Constants are *uncalibrated guesses* chosen so dZ=1 at A=100 costs ~5% of B_max.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clustering.physics import weizsacker_formula  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--per-nucleon", action="store_true", help="normalise everything by A (B/A)")
PN = ap.parse_args().per_nucleon

A_MAX, Z_MAX = 200, 200
A = torch.arange(1, A_MAX + 1)
Z = torch.arange(0, Z_MAX + 1)
AA, ZZ = torch.meshgrid(A, Z, indexing="xy")
Af, Zf = AA.float(), ZZ.float()
valid = (ZZ <= AA).numpy()

S = Af.numpy() if PN else 1.0                               # per-nucleon divisor
B = weizsacker_formula(AA, ZZ).numpy() / S                  # true B, with pairing
# smooth B: same formula without pairing (a_p -> 0 equivalent: use even/odd average = drop delta)
Bs = (15.75 * Af - 17.80 * Af ** (2 / 3) - 0.711 * Zf ** 2 / Af ** (1 / 3)
      - 23.70 * (Af - 2 * Zf) ** 2 / Af).numpy() / S
Bs_v = np.where(valid, Bs, -np.inf)
zstar = Bs_v.argmax(axis=0)                                  # (A,)
Bmax = Bs_v.max(axis=0)[None, :]                             # (1, A)
D = np.where(valid, Bmax - Bs, np.nan)

if PN:   # same 5%-at-dZ=1-at-A=100 target, rescaled to MeV/nucleon
    KAPPA, EPS, BETA, TAU = 3.8, 0.05, 0.43, 0.25
    UNIT, SPAN, YL, YZ = "B/A", 12, (-8, 0.3), (-0.8, 0.05)
else:
    KAPPA, EPS, BETA, TAU = 38.0, 0.5, 40.0, 25.0
    UNIT, SPAN, YL, YZ = "B", 1500, (-700, 20), (-120, 5)
dz = np.abs(ZZ.numpy() - zstar[None, :])

variants = {
    f"original  {UNIT}": np.where(valid, B, np.nan),
    "cone  $B_{max}-\\kappa\\sqrt{D+\\epsilon^2}$": Bmax - KAPPA * np.sqrt(D + EPS ** 2),
    f"additive  {UNIT}$-\\beta|Z-Z^*|$": np.where(valid, B - BETA * dz, np.nan),
    "exp  $B_{max}\\,e^{-D/\\tau}$": Bmax * np.exp(-D / TAU),
}
variants = {k: np.where(valid, v, np.nan) for k, v in variants.items()}

fig, axs = plt.subplots(2, 3, figsize=(16, 9.5))
vmax = max(np.nanmax(v) for v in variants.values())
for ax, (name, v) in zip(axs.flat[:4], variants.items()):
    im = ax.imshow(v, origin="lower", aspect="auto", cmap="viridis",
                   vmin=vmax - SPAN, vmax=vmax, extent=[0.5, A_MAX + .5, -.5, Z_MAX + .5])
    ax.plot(A, zstar, "r--", lw=.8)
    ax.set_title(name); ax.set_xlabel("A"); ax.set_ylabel("Z")
fig.colorbar(im, ax=axs[:, :], location="top", shrink=.5, pad=.04, label="reward-scale [MeV], common colour scale")

# slice at fixed A: loss relative to the peak, vs dZ
a = 100
ax = axs.flat[4]
zz = np.arange(0, 2 * 44 + 1)
for name, v in variants.items():
    col = v[:, a - 1]
    ax.plot(zz - zstar[a - 1], (col - np.nanmax(col))[zz], marker=".", label=name)
ax.set_xlim(-15, 15); ax.set_ylim(*YL)
ax.set_title(f"slice at A={a}: reward − peak vs $Z-Z^*$"); ax.set_xlabel("Z − Z*"); ax.set_ylabel("MeV / nucleon" if PN else "MeV")
ax.legend(fontsize=7); ax.grid(alpha=.3)

# zoom, near the peak
ax = axs.flat[5]
for name, v in variants.items():
    col = v[:, a - 1]
    ax.plot(zz - zstar[a - 1], (col - np.nanmax(col))[zz], marker=".", label=name)
ax.set_xlim(-4, 4); ax.set_ylim(*YZ)
ax.set_title("same, zoom on the top"); ax.set_xlabel("Z − Z*"); ax.grid(alpha=.3)

out = ROOT / "figures" / ("weizsacker_sharpened_per_nucleon.png" if PN else "weizsacker_sharpened.png")
fig.savefig(out, dpi=130, bbox_inches="tight"); print(out)
