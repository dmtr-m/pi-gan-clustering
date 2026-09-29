"""Aggregate the Weizsacker-weight sweep from the per-event CSVs eval_final_sum.py wrote.

    .venv/bin/python experiments/summarize_lambda_sweep.py

k=4 throughout.  lambda=0.5 is the k=4 runs of the depth sweep.  R = -sum E(leaf)/N with the run's
own lambda, so R is NOT comparable across lambdas; the meaningful numbers are within a lambda:
the paired gain over no-split, the gap to the best reference partition, the unsplit share.
"""
import numpy as np
import pandas as pd

RUNS = {(0.0, s): f"outputs/sweep_lambda/l0_s{s}" for s in range(3)}
RUNS.update({(0.25, s): f"outputs/sweep_lambda/l0.25_s{s}" for s in range(3)})
RUNS.update({(0.5, s): f"outputs/sweep_depth/k4_s{s}" for s in range(3)})
RUNS.update({(1.0, s): f"outputs/sweep_lambda/l1.0_s{s}" for s in range(3)})
REFS = ["all together", "MST d=2.0", "MST d=1.5", "MSTp d=3.0/p=150", "singletons"]


def sem(a):
    return float(np.std(a, ddof=1) / np.sqrt(len(a)))


rows = []
for (lam, s), d in RUNS.items():
    df = pd.read_csv(f"{d}/eval_final_sum_5000.csv")
    p = df["R[policy]"]
    ref_means = {r: df[f"R[{r}]"].mean() for r in REFS}
    best = max(ref_means, key=ref_means.get)
    rows.append(dict(lam=lam, seed=s, R=p.mean(), vs_ns=(p - df["R[all together]"]).mean(),
                     vs_ns_sem=sem(p - df["R[all together]"]), best=best,
                     vs_best=(p - df[f"R[{best}]"]).mean(), sampled=df["R[policy sampled]"].mean(),
                     unsplit=100 * df["unsplit"].mean(), frags=df["frags[policy]"].mean(),
                     largest=df["largest[policy]"].mean(), **{f"ref_{r}": v for r, v in ref_means.items()}))
r = pd.DataFrame(rows)
print("Per run (5000 events, k=4):")
print(f"{'lam':>5} {'seed':>4} {'R':>6} {'vs no-split':>14} {'best ref':>14} {'vs best':>8} "
      f"{'R sampled':>9} {'unsplit%':>9} {'frags':>6} {'largest':>8}")
for x in rows:
    print(f"{x['lam']:>5g} {x['seed']:>4} {x['R']:6.2f} {x['vs_ns']:+8.3f}±{x['vs_ns_sem']:.3f} "
          f"{x['best']:>14} {x['vs_best']:+8.2f} {x['sampled']:9.2f} {x['unsplit']:9.1f} "
          f"{x['frags']:6.2f} {x['largest']:8.1f}")
print("\nMean ± sd over the 3 seeds at each lambda:")
print(f"{'lam':>5} {'vs no-split':>16} {'vs best ref':>14} {'unsplit%':>14} {'frags':>13} {'largest':>13}")
for lam, g in r.groupby("lam"):
    f = lambda c, n=2: f"{g[c].mean():.{n}f} ± {g[c].std(ddof=1):.{n}f}"
    print(f"{lam:>5g} {f('vs_ns', 3):>16} {f('vs_best'):>14} {f('unsplit'):>14} {f('frags'):>13} "
          f"{f('largest', 1):>13}")
print("\nReference R at each lambda (mean over runs) -- the room available to the policy:")
print(f"{'lam':>5} " + " ".join(f"{x:>17}" for x in REFS))
for lam, g in r.groupby("lam"):
    print(f"{lam:>5g} " + " ".join(f"{g['ref_' + x].mean():17.2f}" for x in REFS))
