"""Wide Weizsacker-weight sweep (lambda = 0..10, seed 0, k=4) from the 5000-event evaluations.

    .venv/bin/python experiments/summarize_lambda_wide.py

R = -sum E(leaf)/N with each run's own lambda, so R is NOT comparable across lambdas; read the
paired gain over no-split, the gap to the best reference partition, and the unsplit share.
"""
import numpy as np
import pandas as pd

PATH = {0: "outputs/sweep_lambda/l0_s0", 1: "outputs/sweep_lambda/l1.0_s0"}
PATH.update({l: f"outputs/sweep_lambda_wide/l{l}_s0" for l in range(2, 11)})
REFS = ["all together", "MST d=2.0", "MST d=1.5", "MSTp d=3.0/p=150"]


def sem(a):
    return float(np.std(a, ddof=1) / np.sqrt(len(a)))


print(f"{'lam':>3} {'R':>7} {'vs no-split':>13} {'R sampled':>9} {'unsplit%':>9} {'frags':>6} "
      f"{'largest':>8}   " + " ".join(f"{r:>15}" for r in REFS) + "   best ref / policy gap")
for lam, d in PATH.items():
    df = pd.read_csv(f"{d}/eval_final_sum_5000.csv")
    p = df["R[policy]"]
    ref = {r: df[f"R[{r}]"].mean() for r in REFS}
    best = max(ref, key=ref.get)
    print(f"{lam:>3} {p.mean():7.2f} {(p - df['R[all together]']).mean():+8.3f}±{sem(p - df['R[all together]']):.3f}"
          f" {df['R[policy sampled]'].mean():9.2f} {100 * df['unsplit'].mean():9.1f}"
          f" {df['frags[policy]'].mean():6.2f} {df['frags[policy]'].shape[0] and df['largest[policy]'].mean():8.1f}   "
          + " ".join(f"{ref[r]:15.2f}" for r in REFS) + f"   {best} / {p.mean() - ref[best]:+.2f}")
