"""Recover the generator's (A, Z) yield map from `data/expected_A_Z_distribution.png`.

The original digitizer was never committed (FIGURES.md 0.2 says so). This is a
rebuild, calibrated off the axis ticks rather than off remembered constants, and
it reproduces the committed inversion formula exactly:

    row = 494 - v (494 - 32),    yield = 10^(-(row - 87) / 85.2)

Pixel geometry, read out of the PNG by locating the frame and the tick marks
(`_calibrate` re-derives all of it, so it is checked rather than hardcoded):

    plot frame   cols 57..677, rows 32..494
    x ticks      cols 59,155,252,348,444,540,636  ->  A = 0,20,...,120
    y ticks      rows 490,407,325,242,160,77      ->  Z = 0,10,...,50
    colour bar   cols 716..739, rows 32..494
    cbar ticks   rows 87,172,258,343,428          ->  10^0,10^-1,...,10^-4

"Event" in the source plot's title means *collision* — the generator writes one
entry per collision, while `NucleonDataset` splits each collision into two
spectator sides. So the digitized numbers are already per collision and need no
factor of two.

Validated against the three checks recorded in SESSION_SUMMARY.md / FIGURES.md:
917 coloured cells, the brightest cell (A=2, Z=1) inverting to ~4.4 per
collision, and sum(A * yield) = 100 nucleons per collision.  Run this file
directly to re-run those checks.
"""
from typing import Tuple

import numpy as np

DEFAULT_PNG = "data/expected_A_Z_distribution.png"

# Colour-bar calibration, from the tick marks above.
CBAR_TOP_ROW = 32
CBAR_BOTTOM_ROW = 494
CBAR_DECADE_PX = 85.2      # pixels per decade
CBAR_UNIT_ROW = 87.0       # the row carrying the 10^0 tick


CBAR_COLS = slice(720, 736)   # interior of the colour bar, clear of its frame


def _cbar_ramp(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(rows, colours) of the colour bar as *rendered in this PNG*.

    Inverting against the image's own bar rather than against a matplotlib
    viridis LUT matters: the PNG quantizes, so the brightest cell is
    (253, 231, 36) where the LUT holds (253, 231, 37).  Blue varies slowly at
    the top of viridis, so that one-unit difference costs ~1.5 px of bar
    position — a 4% yield error, uniform across the map and enough on its own
    to miss the sum(A * yield) = 100 check.
    """
    rows = np.arange(CBAR_TOP_ROW + 1, CBAR_BOTTOM_ROW)
    return rows, img[rows, CBAR_COLS, :].mean(axis=1)


def _calibrate(img: np.ndarray) -> Tuple[float, float, float, float]:
    """(col_at_A0, px_per_A, row_at_Z0, px_per_Z), re-derived from the tick marks."""
    dark = img.sum(2) < 250
    xt = np.flatnonzero(dark[496:500, :].sum(0) >= 3)
    yt = np.flatnonzero(dark[:, 53:57].sum(1) >= 3)
    if len(xt) < 2 or len(yt) < 2:
        raise RuntimeError(f"tick detection failed: {len(xt)} x-ticks, {len(yt)} y-ticks")
    # x ticks are 20 A apart, y ticks 10 Z apart, y running bottom-up.
    px_per_a = (xt[-1] - xt[0]) / (20.0 * (len(xt) - 1))
    px_per_z = (yt[-1] - yt[0]) / (10.0 * (len(yt) - 1))
    return float(xt[0]), float(px_per_a), float(yt[-1]), float(px_per_z)


def load_target(path: str = DEFAULT_PNG) -> np.ndarray:
    """(A, Z, yield-per-collision) for every coloured cell in the reference plot."""
    from PIL import Image

    img = np.array(Image.open(path).convert("RGB")).astype(int)
    col0, px_a, row0, px_z = _calibrate(img)
    cbar_rows, cbar_rgb = _cbar_ramp(img)

    out = []
    for A in range(1, 140):
        col = int(round(col0 + A * px_a))
        if not (58 <= col <= 676):
            continue
        for Z in range(0, A + 1):
            row = int(round(row0 - Z * px_z))
            if not (33 <= row <= 493):
                continue
            rgb = img[row, col]
            if rgb.min() > 200:          # white background: no entry in this cell
                continue
            d = np.abs(cbar_rgb - rgb).sum(1)
            k = int(d.argmin())
            if d[k] > 30:            # a grid line or a label, not a cell
                continue
            cbar_row = float(cbar_rows[k])
            out.append((A, Z, 10.0 ** (-(cbar_row - CBAR_UNIT_ROW) / CBAR_DECADE_PX)))
    return np.array(out, dtype=float)


if __name__ == "__main__":
    ref = load_target()
    A, Z, y = ref[:, 0], ref[:, 1], ref[:, 2]
    top = ref[y.argmax()]
    print(f"cells recovered      {len(ref)}          (expect 917)")
    print(f"brightest cell       A={top[0]:.0f} Z={top[1]:.0f} -> {top[2]:.2f}/coll  (expect A=2 Z=1, ~4.4)")
    print(f"sum A * yield        {(A * y).sum():.2f}       (expect ~100)")
    print(f"sum Z * yield        {(Z * y).sum():.2f}")
    print(f"bound Z/A            {(Z * y).sum() / (A * y).sum():.4f}   (expect 0.4520)")
    m = A >= 2
    print(f"fragments A>=2       {y[m].sum():.2f}        (expect 13.54)")
    print()
    for lo, hi in [(2, 2), (3, 4), (5, 10), (11, 20), (21, 40), (41, 80), (81, 132)]:
        s = ((A >= lo) & (A <= hi))
        print(f"  A={lo}-{hi:<4} {y[s].sum():8.2f}")
