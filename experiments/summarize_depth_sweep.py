"""Aggregate the depth sweep from the per-event CSVs eval_final_sum.py wrote.

    .venv/bin/python experiments/summarize_depth_sweep.py

One row per run (policy R, its paired difference from the no-split reference, the share
of events left unsplit, fragments per event), then mean +- sd over the three seeds at each
depth.  R = -sum E(leaf)/N, higher is better; "vs no-split" > 0 means the policy beat
leaving every event whole.  The MST d=2.0 column is the best reference partition.
"""
import numpy as np
import pandas as pd

RUNS = {(1, 0): "outputs/sweep_depth/k1_s0", (1, 1): "outputs/sweep_depth/k1_s1",
        (1, 2): "outputs/sweep_depth/k1_s2", (2, 0): "outputs/sweep_depth/k2_s0",
        (2, 1): "outputs/sweep_depth/k2_s1", (2, 2): "outputs/sweep_depth/k2_s2",
        (4, 0): "outputs/sweep_depth/k4_s0", (4, 1): "outputs/sweep_depth/k4_s1",
        (4, 2): "outputs/sweep_depth/k4_s2", (8, 0): "outputs/2026-09-29/17-11-42",
        (8, 1): "outputs/2026-09-29/17-40-17", (8, 2): "outputs/sweep_depth/k8_s2"}


def sem(a):
    return float(np.std(a, ddof=1) / np.sqrt(len(a)))


rows = []
for (k, s), d in RUNS.items():
    df = pd.read_csv(f"{d}/eval_final_sum_5000.csv")
    p, ps = df["R[policy]"], df["R[policy sampled]"]
    ns, mst = df["R[all together]"], df["R[MST d=2.0]"]
    rows.append(dict(k=k, seed=s, n=len(df), R=p.mean(), sem=sem(p),
                     vs_nosplit=(p - ns).mean(), vs_nosplit_sem=sem(p - ns),
                     vs_mst2=(p - mst).mean(), R_sampled=ps.mean(),
                     unsplit=100 * df["unsplit"].mean(), frags=df["frags[policy]"].mean(),
                     largest=df["largest[policy]"].mean(), nosplit=ns.mean(), mst2=mst.mean()))
r = pd.DataFrame(rows)
print("Per run (5000 events; R = -sum E/N, higher better):")
print(f"{'k':>2} {'seed':>4} {'R':>6} {'±sem':>5} {'vs no-split':>13} {'vs MST2.0':>10} "
      f"{'R sampled':>10} {'unsplit%':>9} {'frags':>6} {'largest':>8}")
for x in rows:
    print(f"{x['k']:>2} {x['seed']:>4} {x['R']:6.2f} {x['sem']:5.2f} "
          f"{x['vs_nosplit']:+8.3f}±{x['vs_nosplit_sem']:.3f} {x['vs_mst2']:+10.2f} "
          f"{x['R_sampled']:10.2f} {x['unsplit']:9.1f} {x['frags']:6.2f} {x['largest']:8.1f}")
print("\nMean ± sd over the 3 seeds at each depth:")
print(f"{'k':>2} {'R':>14} {'vs no-split':>16} {'vs MST2.0':>14} {'unsplit%':>13} {'frags':>13}")
for k, g in r.groupby("k"):
    f = lambda c: f"{g[c].mean():.2f} ± {g[c].std(ddof=1):.2f}"
    print(f"{k:>2} {f('R'):>14} {g['vs_nosplit'].mean():+.3f} ± {g['vs_nosplit'].std(ddof=1):.3f}"
          f" {f('vs_mst2'):>14} {f('unsplit'):>13} {f('frags'):>13}")
print(f"\nreference (mean over runs): no-split R = {r['nosplit'].mean():.2f}, "
      f"MST d=2.0 R = {r['mst2'].mean():.2f}")
