"""SACA with Weizsacker energetics in place of FRIGA's B_asy.

    .venv/bin/python experiments/weizsacker_saca.py --n-events 1000

"Weizsacker instead of B_asy" has two distinct readings, and they change
different parts of the algorithm, so both are run here rather than guessed
between.  Row 1 carries the two controls the rows are measured against.

  row 2  **BWM admissibility cut** — SACA 2.1 (Vermani, Dhawan, Goyal, Puri &
         Aichelin).  The constant -4 MeV/nucleon threshold is replaced by the
         fragment's own BWM binding energy per nucleon, so the cut depends on
         (A, Z).  Scanned over a scale factor on that threshold.

         The scale is not decoration.  The published cut is the *full* BWM
         binding, ~-8.5 MeV/nucleon across the mid-mass range, while our QMD
         energy under-binds finite nuclei by ~2.5 MeV/nucleon relative to BWM
         (NOTES.md 4.1: Ca-40 at -6.00 against BWM's -8.48).  At scale 1.0 the
         threshold is therefore below anything this energy model can reach and
         every fragment should dissolve.  Scale 0.5 puts the mid-mass cut near
         the -4.2 MeV/nucleon that SACA 1.1 uses.

  row 3  **Weizsacker asymmetry term in the energy** — the apples-to-apples
         swap for B_asy, since FRIGA's

             B_asy = E_0 ((rho_n - rho_p)/rho_B)^2 (<rho_B>/rho_0)^gamma

         reduces to the Weizsacker symmetry energy E_0 (N - Z)^2 / A exactly
         when gamma = 0.  So this row needs no new energy code at all: it is
         `asymmetry=True, gamma_asy=0`, scanned over the coefficient, with
         a_sym = 23.70 MeV the literature value from `physics.weizsacker_formula`.
         Row 4 of `baseline_grid.py` is the same term with FRIGA's density
         weighting switched back on.

Both rows sit on the same coordinate MST seed (d_cut = 2.0 fm) as the published
SACA panel, so the only difference from the control is the ingredient named.
"""
import argparse
import os
import time
from collections import Counter
from multiprocessing import get_context
from typing import Counter as CounterT, Dict, List, Tuple

import numpy as np

from clustering.baselines.saca import SacaParams, bwm_zeta, saca_clusters
from clustering.split_prediction.dataset import NucleonDataset

from baseline_grid import (BINS, load_events, print_table, render_grid, stats)
from target_reference import load_target

# Scale on the BWM binding-energy threshold.  1.0 is as published.
BWM_SCALES = [0.25, 0.5, 0.75, 1.0]

# Weizsacker asymmetry coefficient a_4 [MeV].  23.70 is the literature value;
# the others bracket it by factors of two, as BAND_WIDTH.md 2 did for E_0.
A_SYMS = [11.85, 23.70, 47.40, 94.80]


def panel_specs() -> List[Tuple[Tuple[int, int], str, dict]]:
    """((row, col), title, kwargs) for every panel below Target."""
    out: List[Tuple[Tuple[int, int], str, dict]] = [
        ((0, 1), "control: SACA  $e_{cut}$ = $-$4",
         {"e_cut": -4.0}),
        ((0, 2), "control: SACA + $B_{asy}$  ($\\gamma$ = 0.5)",
         {"e_cut": -4.0, "asymmetry": True, "e_0_asy": 23.3, "gamma_asy": 0.5}),
    ]
    for j, sc in enumerate(BWM_SCALES):
        out.append(((1, j), f"BWM cut  $\\times${sc:g}",
                    {"e_cut_model": "bwm", "bwm_scale": sc}))
    for j, a in enumerate(A_SYMS):
        out.append(((2, j), f"Weizsacker asym  $a_4$ = {a:g} MeV",
                    {"e_cut": -4.0, "asymmetry": True,
                     "e_0_asy": a, "gamma_asy": 0.0}))
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
    ap.add_argument("--n-events", type=int, default=1000,
                    help="events per spectator side; one collision = two events")
    ap.add_argument("--data", default="data/xecs_hse.parquet")
    ap.add_argument("--out", default="figures/az_weizsacker.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    print("BWM thresholds -B/A [MeV/nucleon] at scale 1.0:")
    print("   " + "  ".join(f"A={A}:{bwm_zeta(A, Z):.2f}"
                            for A, Z in [(4, 2), (12, 6), (40, 18), (100, 44)]))

    ref = load_target()
    g = np.array([ref[(ref[:, 0] >= lo) & (ref[:, 0] <= hi), 2].sum() for lo, hi in BINS])

    specs = panel_specs()
    n_per_side = min(args.n_events,
                     len(NucleonDataset(args.data, particle_type="SpectatorsLeft")))
    edges = np.linspace(0, n_per_side, args.jobs + 1).round().astype(int)
    jobs = [(c, int(edges[c]), int(edges[c + 1]), args.data, args.seed)
            for c in range(args.jobs) if edges[c + 1] > edges[c]]
    print(f"\n{n_per_side} events per side over {len(jobs)} worker(s)\n")

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
                f"SACA with Weizsacker energetics — {n_coll:.0f} collisions",
                n_rows=3, n_cols=4)
    print_table(panels, ref, n_coll, g)


if __name__ == "__main__":
    main()
