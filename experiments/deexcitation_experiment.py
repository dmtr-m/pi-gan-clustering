"""Experiment: what does secondary de-excitation do to the baselines?

    .venv/bin/python experiments/deexcitation_experiment.py --n-events 500

Standalone on purpose — it does not touch the Stage 4 pipeline, so the baseline
numbers in NOTES.md stay reproducible while this is being tuned.

The chain under test:

    clusterizer  ->  primary (A, Z) + E*  ->  Weisskopf evaporation  ->  cold (A, Z)

E* is the fragment's energy as formed (``qmd_energy.cluster_energy``, verified in
tests/test_qmd_energy.py) minus its ground-state energy from the BWM mass formula
(verified in tests/test_binding_formulas.py). Everything the evaporation itself
does is checked in tests/test_deexcitation.py.

Three predictions are tested at once, one per known discrepancy:

  1. the A = 3-4 yield should *rise* (evaporation makes light fragments),
  2. the heavy residue should *fall* (mass moves down),
  3. the isotopic band should *narrow* (neutron-rich fragments shed neutrons).
"""
import argparse
from collections import Counter

import numpy as np
import torch

from clustering.baselines.deexcitation import evaporate, excitation_energy
from clustering.baselines.qmd_energy import cluster_energy
from clustering.physics import weizsacker_formula
from clustering.split_prediction.dataset import NucleonDataset
from clustering.split_prediction.mst import mst_clusters

BINS = [(2, 2), (3, 4), (5, 10), (11, 20), (21, 40), (41, 80), (81, 132)]
_ZSTAR = {}


def zstar(A):
    """Z of the valley of stability at mass A."""
    if A not in _ZSTAR:
        Z = np.arange(1, max(A, 2))
        B = weizsacker_formula(torch.full((len(Z),), float(A)),
                               torch.tensor(Z, dtype=torch.float))
        _ZSTAR[A] = float(Z[int(B.argmax())])
    return _ZSTAR[A]


def generator_reference(path):
    """(A, Z, yield-per-collision) digitized from the reference plot, if present."""
    try:
        ref = np.load(path)
    except (FileNotFoundError, OSError):
        return None
    row = 494 - ref[:, 2] * (494 - 32)
    return np.column_stack([ref[:, 0], ref[:, 1], 10 ** (-(row - 87.0) / 85.2)])


def binned(counter, n_collisions):
    """Yield per collision in each mass bin."""
    return np.array([sum(v for (A, _), v in counter.items() if lo <= A <= hi)
                     for lo, hi in BINS]) / n_collisions


def band(counter, lo=20, hi=60):
    """(mean, sd) of Z - Z*(A), yield-weighted, over a mass window."""
    dev, w = [], []
    for (A, Z), n in counter.items():
        if lo <= A <= hi:
            dev.append(Z - zstar(A))
            w.append(n)
    if sum(w) < 5:
        return float("nan"), float("nan")
    dev, w = np.array(dev, float), np.array(w, float)
    m = np.average(dev, weights=w)
    return m, float(np.sqrt(np.average((dev - m) ** 2, weights=w)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-events", type=int, default=500, help="events per spectator side")
    ap.add_argument("--metric", default="mstp", choices=("coord", "mstp", "momentum"))
    ap.add_argument("--d-cut", type=float, default=3.0)
    ap.add_argument("--p-cut", type=float, default=150.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/xecs_hse.parquet")
    ap.add_argument("--ref", default=None, help="digitized generator .npy (optional)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    events = []
    for side in ("SpectatorsLeft", "SpectatorsRight"):
        ds = NucleonDataset(args.data, particle_type=side)
        events += [ds[i] for i in range(min(args.n_events, len(ds)))]
    n_coll = len(events) / 2.0
    print(f"{len(events)} events = {n_coll:.0f} collisions; "
          f"clusterizer = {args.metric} d={args.d_cut} p={args.p_cut}\n")

    primary, cold = Counter(), Counter()
    e_star_per_a, n_emitted = [], Counter()
    sum_a_in = sum_a_out = sum_z_in = sum_z_out = 0

    for e in events:
        if len(e) < 2:
            continue
        x = e.unsqueeze(0)
        mask = torch.ones(1, len(e), dtype=torch.bool)
        lab = mst_clusters(x, mask, args.d_cut, args.p_cut, args.metric)[0].numpy()
        xn = e.numpy().astype(np.float64)

        for c in np.unique(lab[lab >= 0]):
            idx = np.flatnonzero(lab == c)
            A = len(idx)
            Z = int((xn[idx, 7] == 1).sum())
            primary[(A, Z)] += 1
            sum_a_in += A
            sum_z_in += Z
            if A < 2:
                cold[(A, Z)] += 1
                sum_a_out += A
                sum_z_out += Z
                continue
            e_star = excitation_energy(cluster_energy(xn[idx]).total, A, Z)
            e_star_per_a.append(e_star / A)
            res = evaporate(A, Z, e_star, rng)
            for name, k in res.emitted.items():
                n_emitted[name] += k
            for (a_p, z_p) in res.products:
                cold[(a_p, z_p)] += 1
                sum_a_out += a_p
                sum_z_out += z_p

    # --- conservation, first: nothing below means anything if this fails -----
    print(f"conservation: A {sum_a_in} -> {sum_a_out}, Z {sum_z_in} -> {sum_z_out}"
          f"  {'OK' if (sum_a_in == sum_a_out and sum_z_in == sum_z_out) else 'BROKEN'}")
    e_star_per_a = np.array(e_star_per_a)
    print(f"excitation: <E*/A> = {e_star_per_a.mean():.2f} MeV over "
          f"{len(e_star_per_a)} fragments; {100 * (e_star_per_a == 0).mean():.0f}% "
          f"came out cold")
    print(f"emitted: {dict(sorted(n_emitted.items()))}\n")

    # --- mass spectrum -------------------------------------------------------
    y_pri, y_cold = binned(primary, n_coll), binned(cold, n_coll)
    ref = generator_reference(args.ref) if args.ref else None
    head = f"{'A bin':<10}{'primary':>10}{'after decay':>13}{'change':>9}"
    if ref is not None:
        g = np.array([ref[(ref[:, 0] >= lo) & (ref[:, 0] <= hi), 2].sum() for lo, hi in BINS])
        head += f"{'generator':>11}{'ratio pri':>11}{'ratio cold':>12}"
    print(head)
    for i, (lo, hi) in enumerate(BINS):
        name = f"A={lo}" if lo == hi else f"A={lo}-{hi}"
        line = f"{name:<10}{y_pri[i]:10.2f}{y_cold[i]:13.2f}{y_cold[i] / max(y_pri[i], 1e-9):8.2f}x"
        if ref is not None:
            line += f"{g[i]:11.2f}{y_pri[i] / g[i]:10.2f}x{y_cold[i] / g[i]:11.2f}x"
        print(line)
    line = f"{'total':<10}{y_pri.sum():10.2f}{y_cold.sum():13.2f}{y_cold.sum() / y_pri.sum():8.2f}x"
    if ref is not None:
        line += f"{g.sum():11.2f}{y_pri.sum() / g.sum():10.2f}x{y_cold.sum() / g.sum():11.2f}x"
    print(line)

    if ref is not None:
        def rms(y):
            r = np.where(y > 0, y / g, 1e-3)
            return float(np.sqrt((np.log10(r) ** 2).mean()))
        print(f"\nRMS(log10 ratio):  primary {rms(y_pri):.3f}  ->  "
              f"after decay {rms(y_cold):.3f}")

    # --- isotopic band -------------------------------------------------------
    print(f"\n{'band A=20-60':<16}{'mean':>8}{'sd':>7}")
    for label, c in (("primary", primary), ("after decay", cold)):
        m, sd = band(c)
        print(f"{label:<16}{m:8.2f}{sd:7.2f}")
    if ref is not None:
        dev = ref[(ref[:, 0] >= 20) & (ref[:, 0] <= 60)]
        d = dev[:, 1] - np.array([zstar(int(a)) for a in dev[:, 0]])
        m = np.average(d, weights=dev[:, 2])
        sd = float(np.sqrt(np.average((d - m) ** 2, weights=dev[:, 2])))
        print(f"{'generator':<16}{m:8.2f}{sd:7.2f}")


if __name__ == "__main__":
    main()
