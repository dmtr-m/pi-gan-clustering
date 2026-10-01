"""Spectator nucleons vs fragments per collision, from the raw (A, Z) counters.

    PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python experiments/frag_ratio.py --sweep kK

Same events, models and fragment counting as az_sweep.py (both spectator sides, --n-events per side,
one collision = two events).  Unlike the az_sweep table, A = 1 leaves are counted too, so the total
nucleon count per collision is exact.  Columns: nucleons/collision, leaves (all), fragments (A>=2),
free nucleons (A=1), bound fraction, mean A of a fragment, mean A of a leaf.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from az_policy import add_fragments, labels_to_leaves  # noqa: E402
from az_sweep import SWEEPS  # noqa: E402

from clustering.split_prediction.dataset import NucleonDataset, collate_fn  # noqa: E402
from clustering.split_prediction.model import SplitPredictionModel  # noqa: E402
from clustering.split_prediction.mst import mst_clusters  # noqa: E402
from clustering.split_prediction.trainer import k_level_forward  # noqa: E402


KS = (2, 3, 5)


def label(sweep: str, key) -> str:
    if sweep == "kK":
        return f"k={key[0]} K={KS[key[1]]}"
    if sweep == "pretrain_len":
        return f"pretrain {key[0]} ep"
    return f"lam={key[0]:g} seed {key[1]}"


SIZE_BINS = [(2, 2), (3, 3), (4, 4), (5, 9), (10, 19), (20, 49), (50, 200)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", choices=list(SWEEPS))
    ap.add_argument("--run-dir", help="a single run instead of a sweep")
    ap.add_argument("--n-events", type=int, default=5000, help="events per spectator side")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--data", default="data/xecs_hse.parquet")
    args = ap.parse_args()
    assert args.sweep or args.run_dir, "give --sweep or --run-dir"
    runs = {("run", 0): args.run_dir} if args.run_dir else SWEEPS[args.sweep][0]

    models = {}
    for key, d in runs.items():
        cfg = yaml.safe_load(open(f"{d}/resolved_config.yaml"))["config"]
        m = SplitPredictionModel(input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"],
                                 n_iters=cfg["n_iters"], n_clusters=cfg["n_clusters"])
        m.load_state_dict(torch.load(f"{d}/dm_model.pt", map_location="cpu"))
        m.eval()
        models[Path(d).name if args.run_dir else label(args.sweep, key)] = (m, cfg["split_k"])
    counters = {name: Counter() for name in models}
    counters["MST d=2.0"], counters["MSTp d=3.0,p=150"] = Counter(), Counter()
    total_nuc, n_events = 0, 0
    with torch.no_grad():
        for side in ("SpectatorsLeft", "SpectatorsRight"):
            loader = DataLoader(NucleonDataset(args.data, particle_type=side, n_events=args.n_events),
                                batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
            for batch in loader:
                x, mask = batch["x"], batch["mask"]
                n_events += x.shape[0]
                total_nuc += int(mask.sum())
                for name, (m, k) in models.items():
                    add_fragments(counters[name], x, k_level_forward(m, x, mask, k, 2)["leaf_masks"])
                add_fragments(counters["MST d=2.0"], x, labels_to_leaves(
                    mst_clusters(x, mask, d_cut=2.0, metric="coord"), mask))
                add_fragments(counters["MSTp d=3.0,p=150"], x, labels_to_leaves(
                    mst_clusters(x, mask, d_cut=3.0, p_cut=150.0, metric="mstp"), mask))
    n_coll = n_events / 2.0
    print(f"{n_events} events = {n_coll:.0f} collisions; {total_nuc / n_coll:.2f} spectator "
          f"nucleons per collision (from the mask)\n")
    print(f"Per collision (mean over {n_coll:.0f}); fragment = group of >=2 nucleons, free = group of 1.\n")
    print(f"{'run':<22}{'nucleons':>9}{'fragments':>10}{'free':>8}{'in frags':>10}{'% in frags':>11}")
    for name, c in counters.items():
        nuc = sum(a * v for (a, _), v in c.items()) / n_coll
        fr = sum(v for (a, _), v in c.items() if a >= 2) / n_coll
        bound = sum(a * v for (a, _), v in c.items() if a >= 2) / n_coll
        print(f"{name:<22}{nuc:9.1f}{fr:10.2f}{nuc - bound:8.1f}{bound:10.1f}{100 * bound / nuc:10.1f}%")
    print("\nFragment-size distribution: fragments per collision in each size bin\n")
    hdr = "".join(f"{(str(lo) if lo == hi else f'{lo}-{hi}'):>8}" for lo, hi in SIZE_BINS)
    print(f"{'run':<22}{hdr}{'median':>8}{'max':>6}")
    for name, c in counters.items():
        row = "".join(f"{sum(v for (a, _), v in c.items() if lo <= a <= hi) / n_coll:8.2f}"
                      for lo, hi in SIZE_BINS)
        sizes = sorted((a, v) for (a, _), v in c.items() if a >= 2)
        tot, acc, med = sum(v for _, v in sizes), 0, 0
        for a, v in sizes:
            acc += v
            if acc >= tot / 2:
                med = a
                break
        print(f"{name:<22}{row}{med:8d}{max((a for a, _ in sizes), default=0):6d}")


if __name__ == "__main__":
    main()
