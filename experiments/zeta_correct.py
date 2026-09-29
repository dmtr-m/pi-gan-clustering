"""SACA on the paper's full-QMD zeta, Yukawa dropped: zeta_correct = zeta + lambda * bwd.

    .venv/bin/python experiments/zeta_correct.py --n-events 1000

The energy is `qmd_full.full_cluster_energy` (rest-frame kinetic + two- and
three-body Skyrme + Coulomb + Pauli) with `yukawa="off"`, plus lambda times the
BWM Weizsacker energy bwd = -B.  The Yukawa is off because with the table's V_0
it either diverges (point form) or leaves every fragment unbound (folded), and the
SACA papers use it only as a surface correction anyway.

Without it nothing attractive is left: measured on coordinate-MST fragments the
zeta/A medians are +22 (A 3-9), +31 (10-39), +47 (>= 40) MeV/A at spin_factor 1,
and +18, +27, +36 at 0.5, against bwd/A of -0.8, -6.7, -8.5.  So lambda has to be
large before anything binds, and the light fragments (bwd/A ~ -1) cannot bind at
any lambda in this range.  The lambda values below are first guesses read off
those numbers, not tuned.  The admissibility cut is on the annealing objective
(`e_cut_model="objective"`), as in qmd_minus_b.

Usage: scan the admissibility cut at one lambda, or lambda at one cut.  ``--yukawa``
selects off (default, historical), folded, or point.

    .venv/bin/python experiments/zeta_correct.py --spin 0.5 --lams 4 --cuts=-8,-6,-2,0,2,4,8,12
    .venv/bin/python experiments/zeta_correct.py --spin 0.5 --yukawa folded --lams 2,4,8,16 --cuts=-4

Panels: row 0 is Target and two controls (plain SACA; the first lambda at cut -4), then four cuts per row.
"""
import argparse
import csv
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

DEFAULT_CUTS = [-8.0, -6.0, -2.0, 0.0, 2.0, 4.0, 8.0, 12.0]

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


def panel_specs(spin: float, lams: List[float], cuts: List[float], yukawa: str
                ) -> List[Tuple[Tuple[int, int], str, dict]]:
    """Two controls, then one panel per (lambda, cut), four to a row from row 1.

    Vary one of the two: pass several lambdas and one cut, or one lambda and
    several cuts.  The second control is the first lambda at cut -4.
    """
    zc = {"energy_model": "zeta_correct", "yukawa": yukawa, "spin_factor": spin,
          "e_cut_model": "objective"}
    out: List[Tuple[Tuple[int, int], str, dict]] = [
        ((0, 1), "control: SACA (QMD, cut $-$4)", {"e_cut": -4.0}),
        ((0, 2), f"control: $\\zeta$ + {lams[0]:g}$\\cdot$bwd, cut $-$4",
         dict(zc, bwm_weight=lams[0], e_cut=-4.0)),
    ]
    combos = [(lam, c) for lam in lams for c in cuts]
    for k, (lam, c) in enumerate(combos):
        out.append(((1 + k // 4, k % 4), f"$\\zeta$ + {lam:g}$\\cdot$bwd, cut {c:+g}",
                    dict(zc, bwm_weight=lam, e_cut=c)))
    return out


def run_saca(events, seed: int, **kw):
    """(species Counter, sums) with sums = [T, V, bwd, E_objective] over all events.

    The four sums are the *final-state* values, summed over the fragments of
    N >= 2 in every event (free nucleons contribute 0): rest-frame kinetic
    energy, QMD potential, Weizsaecker energy -B_BWM (unweighted), and the
    annealing objective, so that E = T + V + lambda * bwd for zeta_correct.
    """
    params = SacaParams(e_cut_light=0.0, **kw)
    rng = np.random.default_rng(seed)
    out: CounterT[Tuple[int, int]] = Counter()
    sums = np.zeros(4)
    for e in events:
        if len(e) < 2:
            continue
        res = saca_clusters(e, params, d_cut=2.0, metric="coord", rng=rng)
        lab = res.labels
        sums += (res.t_sum, res.v_sum, res.bwd_sum, res.e_final)
        xn = e.numpy()
        for c in np.unique(lab[lab >= 0]):
            idx = np.flatnonzero(lab == c)
            out[(len(idx), int((xn[idx, 7] == 1).sum()))] += 1
    return out, sums


def _worker(job):
    chunk, lo, hi, path, seed, specs = job
    np.random.seed(seed * 100003 + chunk)
    events = load_events(path, lo, hi)
    out = {key: run_saca(events, seed + chunk, **kw)
           for key, _title, kw in specs}
    return chunk, len(events), out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-events", type=int, default=1000)
    ap.add_argument("--data", default="data/xecs_hse.parquet")
    ap.add_argument("--out", default="figures/az_zeta_correct_cuts.png")
    ap.add_argument("--spin", type=float, default=0.5, help="Pauli spin_factor")
    ap.add_argument("--lams", default="4", help="comma-separated lambdas on bwd")
    ap.add_argument("--cuts", default=",".join(f"{c:g}" for c in DEFAULT_CUTS),
                    help="comma-separated admissibility cuts on the objective, MeV/nucleon")
    ap.add_argument("--yukawa", default="off", choices=["off", "folded", "point"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    ref = load_target()
    g = np.array([ref[(ref[:, 0] >= lo) & (ref[:, 0] <= hi), 2].sum() for lo, hi in BINS])

    cuts = [float(c) for c in args.cuts.split(",")]
    lams = [float(x) for x in args.lams.split(",")]
    specs = panel_specs(args.spin, lams, cuts, args.yukawa)
    n_rows = 1 + (len(lams) * len(cuts) + 3) // 4
    n_per_side = min(args.n_events,
                     len(NucleonDataset(args.data, particle_type="SpectatorsLeft")))
    edges = np.linspace(0, n_per_side, args.jobs + 1).round().astype(int)
    jobs = [(c, int(edges[c]), int(edges[c + 1]), args.data, args.seed, specs)
            for c in range(args.jobs) if edges[c + 1] > edges[c]]
    print(f"{n_per_side} events per side over {len(jobs)} worker(s)\n")

    merged: Dict[Tuple[int, int], CounterT[Tuple[int, int]]] = {
        key: Counter() for key, _, _ in specs}
    sums_merged = {key: np.zeros(4) for key, _, _ in specs}
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
        for key, (c, sm) in out.items():
            merged[key] += c
            sums_merged[key] += sm

    n_coll = n_events_seen / 2.0
    print(f"\n{n_events_seen} events = {n_coll:.0f} collisions in "
          f"{(time.time() - t0) / 60:.1f} min")
    panels = [(key[0], key[1], title, merged[key]) for key, title, _ in specs]

    render_grid(panels, ref, n_coll, g, args.out,
                f"SACA on $\\zeta$ + $\\lambda\\cdot$bwd, spin {args.spin:g}, Yukawa {args.yukawa} — {n_coll:.0f} collisions", n_rows=n_rows, n_cols=4)
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

    # --- final-state energy terms, per collision ------------------------------
    # T, V and bwd are summed over the final fragments (N >= 2); E is the annealing
    # objective, so E = T + V + lambda*bwd for the zeta_correct panels.  The SACA
    # control has no bwd term in its objective (E = T + V), but bwd is still shown.
    rows = []
    print(f"\nFinal-state sums per collision [MeV]   ({n_coll:.0f} collisions)")
    print(f"{'panel':<38}{'lambda':>7}{'cut':>6}{'frags':>7}{'sum T':>10}{'sum V':>11}"
          f"{'sum bwd':>10}{'lam*bwd':>10}{'E':>10}")
    for (_i, _j, title, c), (key, _t, kw) in zip(panels, specs):
        tsum, vsum, bsum, esum = sums_merged[key] / n_coll
        lam = kw.get("bwm_weight", 0.0) if kw.get("energy_model") == "zeta_correct" else 0.0
        frags = sum(v for (A, _Z), v in c.items() if A >= 2) / n_coll
        clean = title.replace("$", "").replace("\\", "").replace("{", "").replace("}", "")
        print(f"{clean:<38}{lam:7g}{kw.get('e_cut', float('nan')):6g}{frags:7.2f}"
              f"{tsum:10.1f}{vsum:11.1f}{bsum:10.1f}{lam * bsum + 0.0:10.1f}{esum:10.1f}")
        rows.append(dict(panel=clean, spin=args.spin, yukawa=args.yukawa, lam=lam,
                         cut=kw.get("e_cut"), frags_per_coll=frags, T=tsum, V=vsum,
                         bwd=bsum, lam_bwd=lam * bsum, E=esum, n_coll=n_coll))
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
