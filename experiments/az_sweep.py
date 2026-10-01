"""(A, Z) yield maps for every model in the depth sweep, on one figure.

    PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python experiments/az_sweep.py --n-events 5000

Rows are depths k = 1, 2, 4, 8 (--sweep depth) or bwm_weight = 0, 0.25, 0.5, 1 (--sweep lambda) and columns are seeds 0, 1, 2 (the greedy/argmax policy).  The
top row is Target, MST d=2.0 and MSTp d=3.0,p=150.  Every model scores the same batches, so the
panels are paired.  Same renderer, target and stat box as baseline_grid.py / az_policy.py; both
spectator sides are used so yields are per collision, and the policy (trained on
SpectatorsLeft only) is out of distribution on the right side.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from az_policy import add_fragments, labels_to_leaves  # noqa: E402
from baseline_grid import BINS, print_bands, print_table, render_grid  # noqa: E402
from target_reference import load_target  # noqa: E402

from clustering.split_prediction.dataset import NucleonDataset, collate_fn  # noqa: E402
from clustering.split_prediction.model import SplitPredictionModel  # noqa: E402
from clustering.split_prediction.mst import mst_clusters  # noqa: E402
from clustering.split_prediction.progress import StepProgress  # noqa: E402
from clustering.split_prediction.trainer import k_level_forward  # noqa: E402

RUNS_DEPTH = {(1, 0): "outputs/sweep_depth/k1_s0", (1, 1): "outputs/sweep_depth/k1_s1",
        (1, 2): "outputs/sweep_depth/k1_s2", (2, 0): "outputs/sweep_depth/k2_s0",
        (2, 1): "outputs/sweep_depth/k2_s1", (2, 2): "outputs/sweep_depth/k2_s2",
        (4, 0): "outputs/sweep_depth/k4_s0", (4, 1): "outputs/sweep_depth/k4_s1",
        (4, 2): "outputs/sweep_depth/k4_s2", (8, 0): "outputs/2026-09-29/17-11-42",
        (8, 1): "outputs/2026-09-29/17-40-17", (8, 2): "outputs/sweep_depth/k8_s2"}
# lambda sweep (k=4): rows are bwm_weight; lambda=0.5 is the k=4 row of the depth sweep.
RUNS_LAMBDA = {(0.0, s): f"outputs/sweep_lambda/l0_s{s}" for s in range(3)}
RUNS_LAMBDA.update({(0.25, s): f"outputs/sweep_lambda/l0.25_s{s}" for s in range(3)})
RUNS_LAMBDA.update({(0.5, s): f"outputs/sweep_depth/k4_s{s}" for s in range(3)})
RUNS_LAMBDA.update({(1.0, s): f"outputs/sweep_lambda/l1.0_s{s}" for s in range(3)})
# warm-start-only control vs REINFORCE (k=4): rows 0/1/2 are named in ROW_NAMES.
RUNS_CONTROL = {(0, s): f"outputs/control_warmstart/s{s}" for s in range(3)}
RUNS_CONTROL.update({(1, s): f"outputs/sweep_depth/k4_s{s}" for s in range(3)})
RUNS_CONTROL.update({(2, s): f"outputs/sweep_lambda/l0_s{s}" for s in range(3)})
# no-warm-start sweep (pretrain_epochs=0, k=4): rows are bwm_weight.
RUNS_NOPRETRAIN = {(l, s): f"outputs/sweep_nopretrain/l{l}_s{s}"
                   for l in (0, 0.25, 0.5, 1.0) for s in range(3)}
# depth x branching sweep (no warm start, lambda=0.5, seed 0): rows are k, columns are K.
KS = (2, 3, 5)
RUNS_KK = {(k, j): f"outputs/sweep_kK/k{k}_K{K}" for k in (2, 4) for j, K in enumerate(KS)}
# MST warm-start length sweep (k=4, K=2, lambda=0.5, seed 0): one seed, flat grid.
RUNS_PRELEN = {(p, 0): f"outputs/sweep_pretrain_len/p{p}" for p in (1, 3, 8, 25)}
# wide lambda sweep: one seed, flat grid (3 panels per row)
RUNS_WIDE = {(l, 0): (("outputs/sweep_lambda/l0_s0" if l == 0 else "outputs/sweep_lambda/l1.0_s0")
                      if l < 2 else f"outputs/sweep_lambda_wide/l{l}_s0") for l in range(11)}
FLAT = {"lambda_wide", "pretrain_len"}
ROW_NAMES = {"control": {0: "warm-start only", 1: "REINFORCE λ=0.5", 2: "REINFORCE λ=0"}}
SWEEPS = {"depth": (RUNS_DEPTH, "k", "Depth sweep", "figures/az_sweep_depth.png"),
          "lambda": (RUNS_LAMBDA, "λ", "Weizsäcker-weight sweep (k=4)", "figures/az_sweep_lambda.png"),
          "lambda_wide": (RUNS_WIDE, "λ", "Wide Weizsäcker-weight sweep (k=4, seed 0)",
                          "figures/az_sweep_lambda_wide.png"),
          "nopretrain": (RUNS_NOPRETRAIN, "λ", "No warm start, REINFORCE only (k=4)",
                         "figures/az_sweep_nopretrain.png"),
          "kK": (RUNS_KK, "k", "Depth k x branching K, no warm start (λ=0.5, seed 0)",
                 "figures/az_sweep_kK.png"),
          "pretrain_len": (RUNS_PRELEN, "pre-train epochs",
                           "MST warm-start length (k=4, K=2, λ=0.5, seed 0)",
                           "figures/az_sweep_pretrain_len.png"),
          "control": (RUNS_CONTROL, "", "Warm-start-only control vs REINFORCE (k=4)",
                      "figures/az_control_warmstart.png")}
A_MAX, Z_MAX = 132, 60


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-events", type=int, default=5000, help="events per spectator side")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--data", default="data/xecs_hse.parquet")
    ap.add_argument("--sweep", choices=list(SWEEPS), default="depth")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    RUNS, row_name, title, default_out = SWEEPS[args.sweep]
    out_path = args.out or default_out

    models = {}
    for key, d in RUNS.items():
        cfg = yaml.safe_load(open(f"{d}/resolved_config.yaml"))["config"]
        m = SplitPredictionModel(input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"],
                                 n_iters=cfg["n_iters"], n_clusters=cfg["n_clusters"])
        m.load_state_dict(torch.load(f"{d}/dm_model.pt", map_location="cpu"))
        m.eval()
        models[key] = (m, cfg["split_k"])
        if args.sweep == "depth":
            assert cfg["split_k"] == key[0], f"{d}: split_k {cfg['split_k']} != {key[0]}"
        elif args.sweep == "kK":
            assert (cfg["split_k"], cfg["n_clusters"]) == (key[0], KS[key[1]]), f"{d}: k/K mismatch"
        elif args.sweep in ("lambda", "lambda_wide", "nopretrain"):
            assert abs(cfg["bwm_weight"] - key[0]) < 1e-9, f"{d}: bwm_weight != {key[0]}"

    loaders = [DataLoader(NucleonDataset(args.data, particle_type=side, n_events=args.n_events),
                          batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
               for side in ("SpectatorsLeft", "SpectatorsRight")]
    total = sum(len(l) for l in loaders)
    print(f"{len(models)} models, {args.n_events} events per side x 2 sides in {total} batches "
          f"of {args.batch}; every model scores the same batches", flush=True)

    counters = {key: Counter() for key in models}
    mst2, mstp = Counter(), Counter()
    n_events = 0
    prog = StepProgress(total, "A-Z sweep", every=10)
    with torch.no_grad():
        for loader in loaders:
            for batch in loader:
                x, mask = batch["x"], batch["mask"]
                n_events += x.shape[0]
                for key, (m, k) in models.items():
                    add_fragments(counters[key], x, k_level_forward(m, x, mask, k, 2)["leaf_masks"])
                add_fragments(mst2, x, labels_to_leaves(
                    mst_clusters(x, mask, d_cut=2.0, metric="coord"), mask))
                add_fragments(mstp, x, labels_to_leaves(
                    mst_clusters(x, mask, d_cut=3.0, p_cut=150.0, metric="mstp"), mask))
                prog.update(1)
    prog.close()

    n_coll = n_events / 2.0
    amax = max(a for c in [mst2, mstp, *counters.values()] for (a, _z) in c)
    zmax = max(z for c in [mst2, mstp, *counters.values()] for (_a, z) in c)
    print(f"\n{n_events} events = {n_coll:.0f} collisions; largest fragment A={amax}, Z={zmax} "
          f"(plot limits A<={A_MAX}, Z<={Z_MAX}{'  -- CROPPED!' if amax > A_MAX or zmax > Z_MAX else ''})")

    ref = load_target()
    g = np.array([ref[(ref[:, 0] >= lo) & (ref[:, 0] <= hi), 2].sum() for lo, hi in BINS])
    depths = sorted({k for k, _ in models})   # the row values (depths, lambdas or row ids)
    names = ROW_NAMES.get(args.sweep)
    name_of = (lambda k: names[k]) if names else (lambda k: f"{row_name}={k:g}")
    panels = [(0, 1, "MST d=2.0", mst2), (0, 2, "MSTp d=3.0, p=150", mstp)]
    if args.sweep in FLAT:      # one seed per row value: fill a 3-column grid left to right
        for i, k in enumerate(depths):
            panels.append((1 + i // 3, i % 3, name_of(k), counters[(k, 0)]))
        n_rows, row_labels = 1 + (len(depths) + 2) // 3, None
    else:
        for i, k in enumerate(depths):
            for s in range(3):
                col = f"K={KS[s]}" if args.sweep == "kK" else f"seed {s}"
                panels.append((i + 1, s, f"{name_of(k)}, {col}", counters[(k, s)]))
        n_rows, row_labels = 1 + len(depths), {i + 1: name_of(k) for i, k in enumerate(depths)}
    render_grid(panels, ref, n_coll, g, out_path,
                f"{title} — fragment (A, Z) yield per collision, {n_coll:.0f} collisions",
                n_rows=n_rows, n_cols=3, a_max=A_MAX, z_max=Z_MAX,
                row_labels=row_labels, panel_in=4.2)
    print_table(panels, ref, n_coll, g)
    print_bands(panels, ref)


if __name__ == "__main__":
    main()
