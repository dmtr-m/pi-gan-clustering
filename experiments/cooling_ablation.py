"""Does a colder annealing schedule change the "/A" objective's histograms?

    .venv/bin/python experiments/cooling_ablation.py --n-events 1500

The per-nucleon ("/A") objective sums each fragment's zeta over fragments, so its
energy differences are ~0.1-0.5 MeV per move — but the default schedule
(T 20 -> 0.5 MeV) never cools below 0.5, and stays hot for the whole run.  On 120
events a colder schedule found a lower final energy (-10.02 vs -9.60, paired
-0.42 +- 0.11), at ~40x the proposals.  This asks whether that shows up in the
(A, Z) distribution.

Four columns per (alpha, cut) row, all on the same events:

  extensive          sum_f [alpha E_f - (1 - alpha) B_f], default schedule
  /A, default        the per-nucleon objective, T 20 -> 0.5, factor 0.90
  /A, geometric      the same objective,        T  1 -> 0.01, factor 0.95
  /A, linear         the same objective,        T  1 -> 0.01, 90 equal steps

The extensive column is the reference the /A columns are read against.  Costs
~10 ms/event (alpha = 0.2) to ~330 ms/event (alpha = 0.8, cut -4) for the colder
schedule, against 1.5-4.5 ms for the default, so this runs at reduced statistics.
"""
import argparse

import numpy as np

from baseline_grid import BINS, print_bands, print_table, render_grid, _install_term_handler
from baselines_comparison import _objective, compute
from target_reference import load_target

ROWS = [(0.2, -4.0), (0.2, -2.0), (0.5, -4.0), (0.5, -2.0),
        (0.8, -4.0), (0.8, -2.0), (0.6, 6.0)]
COLD = dict(t_max=1.0, t_min=0.01, alpha=0.95)     # `alpha` here is the cooling factor
# The same T range in 90 equal steps: ln(100)/ln(1/0.95) = 90, so both schedules
# take the same number of temperature steps and cost about the same.
LINEAR = dict(t_max=1.0, t_min=0.01, cooling="linear", n_steps=90)


def main() -> None:
    _install_term_handler()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-events", type=int, default=1500)
    ap.add_argument("--data", default="data/xecs_hse.parquet")
    ap.add_argument("--out", default="figures/az_alpha_cooling.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=10, help="chunks (part of the cache key)")
    ap.add_argument("--workers", type=int, default=4, help="chunks run at once")
    args = ap.parse_args()

    ref = load_target()
    g = np.array([ref[(ref[:, 0] >= lo) & (ref[:, 0] <= hi), 2].sum() for lo, hi in BINS])

    specs, row_labels = {}, {}
    for i, (a, cut) in enumerate(ROWS, start=1):
        row_labels[i] = f"$\\alpha$ = {a:g}, cut {cut:g}"
        base = dict(energy_model="mix", mix_alpha=a)
        specs[(i, 0)] = ("extensive", _objective(cut, **base))
        specs[(i, 1)] = ("/A, default T 20$\\to$0.5",
                         _objective(cut, mix_per_nucleon=True, **base))
        specs[(i, 2)] = ("/A, geometric T 1$\\to$0.01",
                         _objective(cut, mix_per_nucleon=True, **COLD, **base))
        specs[(i, 3)] = ("/A, linear T 1$\\to$0.01",
                         _objective(cut, mix_per_nucleon=True, **LINEAR, **base))

    merged, n_coll = compute(specs, args.data, args.n_events, args.seed, args.jobs, args.workers)
    panels = [(k[0], k[1], f"{row_labels[k[0]]}: {t}", merged[k])
              for k, (t, _kw) in specs.items()]
    render_grid(panels, ref, n_coll, g, args.out,
                f"Annealing schedules on the /A objective — {n_coll:.0f} collisions",
                n_rows=1 + len(ROWS), n_cols=4, row_labels=row_labels, panel_in=4.5)
    print_table(panels, ref, n_coll, g)
    print_bands(panels, ref)


if __name__ == "__main__":
    main()
