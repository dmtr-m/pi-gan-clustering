"""SACA annealing on QMD - BWM: minimize excitation instead of maximizing binding.

    .venv/bin/python experiments/qmd_minus_bwm.py --n-events 1000

SACA proper minimizes the total microscopic energy, so the search maximizes
binding — and binding is extensive, which is why it collapses onto one heavy
residue (NOTES.md: 4.05 fragments per collision against the generator's 13.5).

This changes the objective to

    sum_f [ E_QMD(f) - E_BWM(A_f, Z_f) ]  =  sum_f [ E_QMD(f) + B(A_f, Z_f) ]

BWM's *energy* is -B, B being the positive binding, so subtracting it adds B.
The quantity is the fragment's excitation above the liquid-drop ground state,
and minimizing it prefers the partition whose fragments are collectively as
*cold* as possible rather than as deeply bound as possible.  The bias toward one
big fragment goes away by construction: a bigger fragment carries more BWM
binding to credit against its own energy.

``bwm_weight`` (lambda) mixes the two objectives; lambda = 0 is plain SACA, so
row 2 starts from the published algorithm and walks away from it.

The admissibility test is left on the QMD zeta throughout.  Excitation is >= 0
almost everywhere, so a "zeta < -4" cut applied to it would dissolve everything.

  row 1  controls: SACA as published, and SACA with the BWM admissibility cut
  row 2  QMD - BWM objective, lambda scan, constant -4 MeV/nucleon cut
  row 3  QMD - BWM objective at lambda = 1, scanned over the BWM cut scale

Row 3 is the pairing that should matter most: the strict BWM cut is what marks
the heavy residue unstable in the first place, and SACA only ever anneals
fragments it has marked unstable — with the published -4 cut the residue is
declared bound and the objective never gets to touch it.
"""
import argparse
import os
import time
from collections import Counter
from multiprocessing import get_context
from typing import Counter as CounterT, Dict, List, Tuple

import numpy as np

from clustering.baselines.saca import SacaParams, saca_clusters
from clustering.split_prediction.dataset import NucleonDataset

from baseline_grid import BINS, load_events, print_table, render_grid
from target_reference import load_target

LAMBDAS = [0.25, 0.5, 1.0, 1.5]
CUT_SCALES = [0.25, 0.5, 0.75, 1.0]


def panel_specs() -> List[Tuple[Tuple[int, int], str, dict]]:
    out: List[Tuple[Tuple[int, int], str, dict]] = [
        ((0, 1), "control: SACA (QMD, cut $-$4)", {"e_cut": -4.0}),
        ((0, 2), "control: SACA (QMD, BWM cut $\\times$1)",
         {"e_cut_model": "bwm", "bwm_scale": 1.0}),
    ]
    for j, lam in enumerate(LAMBDAS):
        out.append(((1, j), f"QMD $-$ {lam:g}$\\cdot$BWM,  cut $-$4",
                    {"e_cut": -4.0, "energy_model": "qmd_plus_b",
                     "bwm_weight": lam}))
    for j, sc in enumerate(CUT_SCALES):
        out.append(((2, j), f"QMD $-$ BWM,  BWM cut $\\times${sc:g}",
                    {"energy_model": "qmd_plus_b", "bwm_weight": 1.0,
                     "e_cut_model": "bwm", "bwm_scale": sc}))
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
    ap.add_argument("--out", default="figures/az_qmd_minus_bwm.png")
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
                f"SACA annealing on QMD $-$ BWM — {n_coll:.0f} collisions",
                n_rows=3, n_cols=4)
    print_table(panels, ref, n_coll, g)


if __name__ == "__main__":
    main()
