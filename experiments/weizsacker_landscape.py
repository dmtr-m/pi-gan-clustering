"""Landscape of the Weizsäcker binding energy over (A, Z).

Uses `clustering.physics.weizsacker_formula` — the formula the RL reward uses
(not the SACA BWM variant).  Z > A is unphysical and left blank.

    python experiments/weizsacker_landscape.py            # B/A  -> weizsacker_landscape.png
    python experiments/weizsacker_landscape.py --total    # B    -> weizsacker_landscape_total.png
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
ap.add_argument("--total", action="store_true", help="plot B (extensive) instead of B/A")
total = ap.parse_args().total

A_MAX, Z_MAX = 200, 200
A = torch.arange(1, A_MAX + 1)
Z = torch.arange(0, Z_MAX + 1)
AA, ZZ = torch.meshgrid(A, Z, indexing="xy")          # shape (Z, A)
B = weizsacker_formula(AA, ZZ)
if not total:
    B = B / AA.float()
B = B.numpy()
B = np.where(ZZ.numpy() <= AA.numpy(), B, np.nan)      # mask Z > A

# valley of stability: Z maximising B at each A (same Z* for B and B/A)
zstar = np.nanargmax(np.where(np.isnan(B), -np.inf, B), axis=0)

fig, ax = plt.subplots(figsize=(8, 6.5))
# data-driven range; clipped below so the deep negative corners far from the
# valley don't wash out the structure near it
vmax = np.nanmax(B)
span = 1500 if total else 12
vmin = max(np.nanmin(B), vmax - span)
im = ax.imshow(B, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
               extent=[0.5, A_MAX + 0.5, -0.5, Z_MAX + 0.5])
ax.plot(A, zstar, "r--", lw=1.2, label="valley $Z^*(A)$")
ax.plot([1, A_MAX], [1, A_MAX], "k:", lw=1, label="$Z=A$")
ax.set_xlabel("A"); ax.set_ylabel("Z")
name = "$B(A,Z)$" if total else "$B(A,Z)/A$"
ax.set_title(f"Weizsäcker binding energy {'' if total else 'per nucleon '}{name} [MeV]")
ax.legend(loc="upper left")
fig.colorbar(im, ax=ax, extend="min",
             label=f"{'B' if total else 'B/A'} [MeV]  (clipped below at vmax-{span})")
out = ROOT / "figures" / ("weizsacker_landscape_total.png" if total else "weizsacker_landscape.png")
fig.tight_layout(); fig.savefig(out, dpi=150)
print(out, "max = %.3f at A=%d" % (vmax, A[np.nanargmax(np.nanmax(B, 0))]))
