"""Warm-start-only control vs the trained (REINFORCE) models, at each lambda.

    .venv/bin/python experiments/summarize_control.py

Control: 25 supervised MST epochs, 0 REINFORCE epochs, k=4, seeds 0-2, each scored at lambda
0, 0.25, 0.5, 1 (a warm-start model does not depend on lambda).  Trained: the k=4 runs of the
depth / lambda sweeps at the same lambda.  Same 5000 events, same reward, per-event paired.
R is not comparable across lambda, so read within a row.
"""
import numpy as np
import pandas as pd

LAMS = [0.0, 0.25, 0.5, 1.0]
trained_dir = {0.0: "outputs/sweep_lambda/l0_s{s}", 0.25: "outputs/sweep_lambda/l0.25_s{s}",
               0.5: "outputs/sweep_depth/k4_s{s}", 1.0: "outputs/sweep_lambda/l1.0_s{s}"}
REFS = ["all together", "MST d=2.0", "MST d=1.5", "MSTp d=3.0/p=150"]


def row(df):
    p = df["R[policy]"]
    return dict(R=p.mean(), vs_ns=(p - df["R[all together]"]).mean(), sampled=df["R[policy sampled]"].mean(),
                unsplit=100 * df["unsplit"].mean(), frags=df["frags[policy]"].mean(),
                largest=df["largest[policy]"].mean(), **{r: df[f"R[{r}]"].mean() for r in REFS})


print("Per model (5000 events): R, paired gain over no-split, unsplit %, frags/event")
print(f"{'lam':>5} {'model':>10} {'seed':>4} {'R':>6} {'vs no-split':>12} {'R sampled':>9} "
      f"{'unsplit%':>9} {'frags':>6} {'largest':>8}")
agg = {}
for lam in LAMS:
    for kind in ("control", "trained"):
        rs = []
        for s in range(3):
            path = (f"outputs/control_warmstart/s{s}/eval_final_sum_5000_lam{lam:g}.csv"
                    if kind == "control" else trained_dir[lam].format(s=s) + "/eval_final_sum_5000.csv")
            r = row(pd.read_csv(path))
            rs.append(r)
            print(f"{lam:>5g} {kind:>10} {s:>4} {r['R']:6.2f} {r['vs_ns']:+12.3f} {r['sampled']:9.2f} "
                  f"{r['unsplit']:9.1f} {r['frags']:6.2f} {r['largest']:8.1f}")
        agg[(lam, kind)] = pd.DataFrame(rs)

print("\nMean ± sd over 3 seeds:")
print(f"{'lam':>5} {'model':>10} {'vs no-split':>15} {'unsplit%':>14} {'frags':>13} {'largest':>13}")
for lam in LAMS:
    for kind in ("control", "trained"):
        g = agg[(lam, kind)]
        f = lambda c, n=2: f"{g[c].mean():.{n}f} ± {g[c].std(ddof=1):.{n}f}"
        print(f"{lam:>5g} {kind:>10} {f('vs_ns', 3):>15} {f('unsplit'):>14} {f('frags'):>13} {f('largest', 1):>13}")

print("\nReference R (control-run events; the room available):")
print(f"{'lam':>5} " + " ".join(f"{r:>17}" for r in REFS))
for lam in LAMS:
    g = agg[(lam, "control")]
    print(f"{lam:>5g} " + " ".join(f"{g[r].mean():17.2f}" for r in REFS))
