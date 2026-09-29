"""Five baselines side by side, then the alpha grid.

    .venv/bin/python experiments/baselines_comparison.py --figure compare
    .venv/bin/python experiments/baselines_comparison.py --figure alpha
    .venv/bin/python experiments/baselines_comparison.py --figure ablation

Every panel's counts go through `baseline_grid`'s per-(configuration, chunk)
cache, and rendering is a separate, cheap step — so reordering rows or columns is
`--order ...` / `--alphas ...` and costs no compute.

"momenta" and "QMD" are the two energies this project has, and the names are the
user's:

  momenta  qmd_energy.cluster_energy: internal kinetic energy in the fragment
           rest frame (from the momenta) plus the saturating Skyrme + Coulomb
           potential.  This is the SACA of the papers, and what every earlier
           SACA figure ran on.
  QMD      physics.py's pairwise potential V (Skyrme + Yukawa + Coulomb + Pauli),
           the energy the RL reward uses.  It has no internal-kinetic term.

Figure 1 — five rows, four cuts each:

  1  MST, momentum only            p_cut                           (cached, full)
  2  SACA, QMD                     sum_f V_f                        new
  3  SACA, momenta                 sum_f E_f                        (cached, full)
  4  SACA, momenta + Weizsacker    sum_f [E_f - B_f]                (cached, full)
  5  SACA, QMD + Weizsacker        sum_f [V_f - B_f]                new

B is the BWM binding energy (SACA 2.1's mass formula) in 4 and 5, the same as
the cached QMD - B run, so 4 versus 5 changes only which microscopic energy sits
beside it.  Rows 2 and 5 are annealed on their own objective and the
admissibility test is that same per-nucleon objective (`e_cut_model="objective"`),
as in row 4; the cut scales differ because the objectives do, and were read off
the zeta distribution of coordinate-MST fragments (median V/A = -23.8,
(V-B)/A = -30.6, against E/A = -4.1).

Figure 2 — the alpha grid.  SACA on

    zeta = alpha * zeta_momenta + (1 - alpha) * zeta_Weizsacker
         = alpha * E/A  -  (1 - alpha) * B/A

for alpha = 0.1 ... 0.9 and ten cuts on that zeta (-12 ... +6).  alpha = 1 would be row 3 and
alpha = 0 the mass formula alone.  The annealer minimizes the *extensive*
objective sum_f N_f zeta_f, SACA's own.

Figure 3 — the ablation of "divide by A".  For three alphas the objective is
instead the sum over fragments of zeta_f itself (each fragment divided by its own
A, so a 100-nucleon residue weighs the same as a deuteron).  The cut is on the
same zeta in both, so the two objectives differ only in how the annealer trades
one big fragment against several small ones.
"""
import argparse
import os
import time
from collections import Counter
from multiprocessing import get_context
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from baseline_grid import (BINS, _cache_load, _cache_path, _cache_store,
                           _install_term_handler, load_events, print_bands,
                           print_table, render_grid, run_momentum, run_saca)
from clustering.split_prediction.dataset import NucleonDataset
from target_reference import load_target

Spec = Tuple[str, dict]      # (panel title, kwargs incl. "kind")

# --- figure 1 ---------------------------------------------------------------
P_CUTS = [60.0, 90.0, 120.0, 150.0]
MOMENTA_CUTS = [-6.0, -4.0, -2.0, 0.0]            # as cached
MOMENTA_W_CUTS = [-6.0, -9.0, -12.0, -15.0]       # as cached
QMD_CUTS = [-30.0, -20.0, -10.0, -5.0]            # V/A
QMD_W_CUTS = [-40.0, -30.0, -20.0, -10.0]         # (V - B)/A


def _saca(**kw) -> dict:
    return {"kind": "saca", **kw}


def _objective(cut: float, **kw) -> dict:
    return _saca(e_cut_model="objective", e_cut=cut, **kw)


ROWS: Dict[str, Tuple[str, List[Spec]]] = {
    "mst_momentum": ("MST momentum", [
        (f"MST momentum  $p$ = {p:.0f} MeV/c", {"kind": "momentum", "p_cut": p})
        for p in P_CUTS]),
    "saca_qmd": ("SACA QMD", [
        (f"SACA QMD  $\\zeta_V$ cut {c:g}",
         _objective(c, energy_model="physics_v")) for c in QMD_CUTS]),
    "saca_momenta": ("SACA momenta", [
        (f"SACA momenta  $e_{{cut}}$ = {c:g}", _saca(e_cut=c))
        for c in MOMENTA_CUTS]),
    "saca_momenta_w": ("SACA momenta + Weizsacker", [
        (f"SACA momenta + W  $\\zeta$ cut {c:g}",
         _objective(c, energy_model="qmd_minus_b", bwm_weight=1.0))
        for c in MOMENTA_W_CUTS]),
    "saca_qmd_w": ("SACA QMD + Weizsacker", [
        (f"SACA QMD + W  $\\zeta$ cut {c:g}",
         _objective(c, energy_model="physics_v_minus_b", bwm_weight=1.0))
        for c in QMD_W_CUTS]),
}
DEFAULT_ORDER = list(ROWS)

# --- figures 2 and 3 --------------------------------------------------------
ALPHAS = [round(0.1 * i, 1) for i in range(1, 10)]
# On zeta_mix [MeV/nucleon].  The original scan was -8..-2 and the RMS was still
# improving at the loose end, so the range was extended on both sides.  Over MST
# fragments zeta_mix lies roughly between -8.6 (B/A of a heavy nucleus) and +3
# (q90), so below ~-9 no fragment can pass and above ~+3 every one does — the
# outer rows are the limits, not physical choices.
ALPHA_CUTS = [-12.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0]
ABLATION_CUTS = [-8.0, -6.0, -4.0, -2.0]
ABLATION_ALPHAS = [0.2, 0.5, 0.8]


def mix_spec(alpha: float, cut: float, per_nucleon: bool = False) -> Spec:
    kw = _objective(cut, energy_model="mix", mix_alpha=alpha)
    if per_nucleon:
        kw["mix_per_nucleon"] = True
    tag = "  /A" if per_nucleon else ""
    return (f"$\\alpha$ = {alpha:g}{tag}  $\\zeta$ cut {cut:g}", kw)


# ---------------------------------------------------------------------------
def _lower_priority() -> None:
    os.nice(10)


def _worker(job):
    chunk, lo, hi, path, seed, specs, n_side = job
    torch.set_num_threads(1)     # 10 workers; torch's own pool would oversubscribe
    np.random.seed(seed * 100003 + chunk)
    files = {key: _cache_path(dict(kw), path, chunk, lo, hi, seed)
             for key, kw in specs}
    out, n_hit = {}, 0
    for key, _kw in specs:
        hit = _cache_load(files[key]) if files[key].exists() else None
        if hit is not None:
            out[key] = hit
            n_hit += 1
    if len(out) < len(specs):
        events = load_events(path, lo, hi)
        n_ev = len(events)
        for key, kw in specs:
            if key in out:
                continue
            kw = dict(kw)
            t0 = time.time()
            kind = kw.pop("kind")
            out[key] = (run_momentum(events, kw["p_cut"]) if kind == "momentum"
                        else run_saca(events, seed + chunk, **kw))
            _cache_store(files[key], out[key])
            print(f"    chunk {chunk:2d} {key}  {time.time() - t0:6.0f} s", flush=True)
    else:
        n_ev = 2 * (min(hi, n_side) - lo)
    return chunk, n_ev, out, n_hit


def compute(specs: Dict[Tuple[int, int], Spec], data: str, n_events: int, seed: int,
            jobs_n: int, workers: int = 4) -> Tuple[Dict[Tuple[int, int], Counter], float]:
    """Counts per panel key, merged over chunks, and the number of collisions.

    ``jobs_n`` is the number of chunks and is part of every cache key, so it must
    stay at 10 to reuse the cached panels.  ``workers`` is how many of them run
    at once — it changes nothing in the results, only how much of the machine is
    used, and the workers run at lowered priority.
    """
    n_side = min(n_events, len(NucleonDataset(data, particle_type="SpectatorsLeft")))
    edges = np.linspace(0, n_side, jobs_n + 1).round().astype(int)
    payload = [(key, kw) for key, (_t, kw) in specs.items()]
    jobs = [(c, int(edges[c]), int(edges[c + 1]), data, seed, payload, n_side)
            for c in range(jobs_n) if edges[c + 1] > edges[c]]
    print(f"{len(payload)} panels, {n_side} events per side over {len(jobs)} chunk(s)\n",
          flush=True)
    merged = {key: Counter() for key, _ in payload}
    n_events_seen, t0 = 0, time.time()
    if len(jobs) == 1:
        results = [_worker(jobs[0])]
    else:
        with get_context("spawn").Pool(min(workers, len(jobs)),
                                       initializer=_lower_priority) as pool:
            results = []
            for chunk, n_ev, out, n_hit in pool.imap_unordered(_worker, jobs):
                results.append((chunk, n_ev, out, n_hit))
                print(f"  chunk {chunk:2d} done ({n_ev} events, {n_hit}/{len(payload)} "
                      f"cached)  {len(results)}/{len(jobs)}  {time.time() - t0:7.1f} s",
                      flush=True)
    for _c, n_ev, out, _h in results:
        n_events_seen += n_ev
        for key, c in out.items():
            merged[key] += c
    n_coll = n_events_seen / 2.0
    print(f"\n{n_events_seen} events = {n_coll:.0f} collisions in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return merged, n_coll


def main() -> None:
    _install_term_handler()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", choices=("compare", "alpha", "ablation"),
                    required=True)
    ap.add_argument("--n-events", type=int, default=58374,
                    help="events per spectator side; the default is the whole "
                         "dataset, which is what the cached panels used")
    ap.add_argument("--data", default="data/xecs_hse.parquet")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=10,
                    help="chunks == worker processes.  The cached panels were "
                         "split 10 ways; changing this recomputes them")
    ap.add_argument("--workers", type=int, default=4,
                    help="chunks run at once; only affects CPU use, not results")
    ap.add_argument("--order", default=",".join(DEFAULT_ORDER),
                    help="figure 1 rows, top to bottom, from: " + ", ".join(ROWS))
    ap.add_argument("--alphas", default=",".join(f"{a:g}" for a in ALPHAS),
                    help="figures 2 and 3, left to right")
    ap.add_argument("--cuts", default=None,
                    help="figures 2 and 3, top to bottom (default: the full scan)")
    args = ap.parse_args()

    ref = load_target()
    g = np.array([ref[(ref[:, 0] >= lo) & (ref[:, 0] <= hi), 2].sum() for lo, hi in BINS])

    specs: Dict[Tuple[int, int], Spec] = {}
    row_labels: Dict[int, str] = {}
    if args.figure == "compare":
        order = [r.strip() for r in args.order.split(",") if r.strip()]
        for i, name in enumerate(order, start=1):
            label, row = ROWS[name]
            row_labels[i] = label
            for j, spec in enumerate(row):
                specs[(i, j)] = spec
        n_rows, n_cols, panel_in = 1 + len(order), 4, 5.0
        out = args.out or "figures/az_baseline_comparison.png"
        title = "Five baselines"
    else:
        alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
        if args.figure == "alpha":
            cols = [(a, False) for a in alphas]
            title = "SACA on $\\zeta = \\alpha\\,\\zeta_{momenta} + (1-\\alpha)\\,\\zeta_{Weizsacker}$"
        else:
            cols = [(a, pn) for a in ABLATION_ALPHAS for pn in (False, True)]
            title = ("SACA on $\\alpha\\,\\zeta_{momenta} + (1-\\alpha)\\,\\zeta_{Weizsacker}$: "
                     "extensive objective vs each fragment divided by $A$")
        cuts = ([float(c) for c in args.cuts.split(",") if c.strip()] if args.cuts
                else ALPHA_CUTS if args.figure == "alpha" else ABLATION_CUTS)
        for j, (a, pn) in enumerate(cols):
            for i, cut in enumerate(cuts, start=1):
                specs[(i, j)] = mix_spec(a, cut, pn)
        for i, cut in enumerate(cuts, start=1):
            row_labels[i] = f"$\\zeta$ cut {cut:g}"
        n_rows, n_cols, panel_in = 1 + len(cuts), max(len(cols), 1), 4.0
        out = args.out or f"figures/az_{'alpha_grid' if args.figure == 'alpha' else 'alpha_ablation'}.png"

    merged, n_coll = compute(specs, args.data, args.n_events, args.seed, args.jobs, args.workers)
    panels = [(k[0], k[1], t, merged[k]) for k, (t, _kw) in specs.items()]
    render_grid(panels, ref, n_coll, g, out, f"{title} — {n_coll:.0f} collisions",
                n_rows=n_rows, n_cols=n_cols, row_labels=row_labels, panel_in=panel_in)
    print_table(panels, ref, n_coll, g)
    print_bands(panels, ref)


if __name__ == "__main__":
    main()
