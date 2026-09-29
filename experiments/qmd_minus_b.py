"""SACA on QMD - B (the other sign), and with the admissibility cut on zeta_B.

    .venv/bin/python experiments/qmd_minus_b.py --n-events 1000

Two changes from `qmd_minus_bwm.py`, which ran the objective

    sum_f [E_QMD(f) + w B_f]     ("qmd_plus_b", = QMD - E_BWM, the excitation)

and found that it fixes SACA's multiplicity collapse but produces more
dineutrons than deuterons: at fixed A the BWM binding B is *smallest* furthest
from the valley, so minimizing E_QMD + B rewards exotic nuclides.

  row 2  **the opposite sign**, sum_f [E_QMD(f) - w B_f] ("qmd_minus_b").  This
         credits binding twice — once microscopically, once from the mass
         formula — and at fixed A it is largest *on* the valley, so it should
         pull the opposite way on composition.  The risk is the mirror image:
         double-counted binding is still extensive, so it may restore SACA's
         collapse onto one residue.  lambda scan, QMD-zeta cut at -4.

  row 3  **the admissibility test moved onto the same B-corrected quantity**
         the annealing minimizes, instead of the bare QMD zeta
         (`e_cut_model="objective"`).  Its scale is different and the cut has to
         move with it: measured over coordinate-MST fragments, zeta_QMD has
         median -4.97 while zeta of QMD - B has median -10.98 (10-90%: -15.40 to
         +4.17), so the scan runs -6 to -15 rather than around -4.

Reported alongside the usual four numbers: the isotopic band (mean and sd of
Z - Z*(A)) and the most abundant species.  The 7-bin RMS is a *mass* metric and
cannot see charge at all — it scored the dineutron-dominated run as the best in
the previous figure — so a composition diagnostic is printed with it.
"""
import argparse
import os
import time
from collections import Counter
from multiprocessing import get_context
from typing import Counter as CounterT, Dict, List, Tuple

import numpy as np
import torch

from clustering.baselines.saca import SacaParams, saca_clusters
from clustering.physics import weizsacker_formula
from clustering.split_prediction.dataset import NucleonDataset

from baseline_grid import BINS, load_events, print_table, render_grid, stats
from target_reference import load_target

LAMBDAS = [0.25, 0.5, 1.0, 1.5]
ZETA_B_CUTS = [-6.0, -9.0, -12.0, -15.0]

_ZSTAR: Dict[int, float] = {}


def zstar(A: int) -> float:
    """Z of the Weizsacker valley at mass A."""
    if A not in _ZSTAR:
        Z = np.arange(1, max(A, 2))
        B = weizsacker_formula(torch.full((len(Z),), float(A)),
                               torch.tensor(Z, dtype=torch.float))
        _ZSTAR[A] = float(Z[int(B.argmax())])
    return _ZSTAR[A]


def band(counter, lo: int = 5, hi: int = 60) -> Tuple[float, float]:
    """(mean, sd) of Z - Z*(A), yield-weighted, over a mass window."""
    dev = [(Z - zstar(A), v) for (A, Z), v in counter.items() if lo <= A <= hi]
    # Guard on the number of distinct species, not the weight sum: the target's
    # weights are yields per collision (order 1), not raw counts.
    if len(dev) < 5:
        return float("nan"), float("nan")
    d = np.array([x for x, _ in dev], float)
    w = np.array([v for _, v in dev], float)
    m = float(np.average(d, weights=w))
    return m, float(np.sqrt(np.average((d - m) ** 2, weights=w)))


def panel_specs() -> List[Tuple[Tuple[int, int], str, dict]]:
    out: List[Tuple[Tuple[int, int], str, dict]] = [
        ((0, 1), "control: SACA (QMD, cut $-$4)", {"e_cut": -4.0}),
        ((0, 2), "control: QMD $+$ B, BWM cut $\\times$0.75",
         {"energy_model": "qmd_plus_b", "bwm_weight": 1.0,
          "e_cut_model": "bwm", "bwm_scale": 0.75}),
    ]
    for j, lam in enumerate(LAMBDAS):
        out.append(((1, j), f"QMD $-$ {lam:g}$\\cdot$B,  cut $-$4",
                    {"e_cut": -4.0, "energy_model": "qmd_minus_b",
                     "bwm_weight": lam}))
    for j, c in enumerate(ZETA_B_CUTS):
        out.append(((2, j), f"QMD $-$ B,  $\\zeta_B$ cut {c:g}",
                    {"energy_model": "qmd_minus_b", "bwm_weight": 1.0,
                     "e_cut_model": "objective", "e_cut": c}))
    return out


def run_saca(events, seed: int, **kw) -> CounterT[Tuple[int, int]]:
    params = SacaParams(e_cut_light=0.0, **kw)
    rng = np.random.default_rng(seed)
    out: CounterT[Tuple[int, int]] = Counter()
    for e in events:
        if len(e) < 2:
            continue
        lab = saca_clusters(e, params, d_cut=2.0, metric="coord", rng=rng).labels
        xn = e.numpy()
        for c in np.unique(lab[lab >= 0]):
            idx = np.flatnonzero(lab == c)
            out[(len(idx), int((xn[idx, 7] == 1).sum()))] += 1
    return out


def _worker(job):
    chunk, lo, hi, path, seed = job
    np.random.seed(seed * 100003 + chunk)
    events = load_events(path, lo, hi)
    out = {key: run_saca(events, seed + chunk, **kw)
           for key, _title, kw in panel_specs()}
    return chunk, len(events), out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-events", type=int, default=1000)
    ap.add_argument("--data", default="data/xecs_hse.parquet")
    ap.add_argument("--out", default="figures/az_qmd_minus_b.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    ref = load_target()
    g = np.array([ref[(ref[:, 0] >= lo) & (ref[:, 0] <= hi), 2].sum() for lo, hi in BINS])

    specs = panel_specs()
    n_per_side = min(args.n_events,
                     len(NucleonDataset(args.data, particle_type="SpectatorsLeft")))
    edges = np.linspace(0, n_per_side, args.jobs + 1).round().astype(int)
    jobs = [(c, int(edges[c]), int(edges[c + 1]), args.data, args.seed)
            for c in range(args.jobs) if edges[c + 1] > edges[c]]
    print(f"{n_per_side} events per side over {len(jobs)} worker(s)\n")

    merged: Dict[Tuple[int, int], CounterT[Tuple[int, int]]] = {
        key: Counter() for key, _, _ in specs}
    n_events_seen = 0
    t0 = time.time()
    if len(jobs) == 1:
        results = [_worker(jobs[0])]
    else:
        with get_context("spawn").Pool(len(jobs)) as pool:
            results = []
            for chunk, n_ev, out in pool.imap_unordered(_worker, jobs):
                results.append((chunk, n_ev, out))
                print(f"  chunk {chunk:2d} done ({n_ev} events)  "
                      f"{len(results)}/{len(jobs)}  {time.time() - t0:7.1f} s")
    for _chunk, n_ev, out in results:
        n_events_seen += n_ev
        for key, c in out.items():
            merged[key] += c

    n_coll = n_events_seen / 2.0
    print(f"\n{n_events_seen} events = {n_coll:.0f} collisions in "
          f"{(time.time() - t0) / 60:.1f} min")
    panels = [(key[0], key[1], title, merged[key]) for key, title, _ in specs]

    render_grid(panels, ref, n_coll, g, args.out,
                f"SACA on QMD $-$ B — {n_coll:.0f} collisions", n_rows=3, n_cols=4)
    print_table(panels, ref, n_coll, g)

    # --- composition, which the RMS cannot see -------------------------------
    tm, tsd = band({(int(A), int(Z)): y for A, Z, y in ref})
    print(f"\n{'panel':<38}{'band mean':>10}{'sd':>7}   top species (A,Z)")
    print(f"{'Target':<38}{tm:10.2f}{tsd:7.2f}")
    for _i, _j, title, c in panels:
        m, sd = band(c)
        top = sorted(((v, A, Z) for (A, Z), v in c.items() if A >= 2), reverse=True)[:4]
        clean = title.replace("$", "").replace("\\", "").replace("{", "").replace("}", "")
        print(f"{clean:<38}{m:10.2f}{sd:7.2f}   "
              + ", ".join(f"({A},{Z}):{v}" for v, A, Z in top))


if __name__ == "__main__":
    main()
