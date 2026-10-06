"""Summarise the logs/zn night queue: unseen-event eval + fragment table per run.
    .venv/bin/python experiments/zn_summary.py > logs/zn/summary.txt
"""
import re, glob, os, statistics as st
def ev(n):
    t = open(f"logs/zn/eval_{n}.log").read().split("=== UNSEEN")[-1]
    g = lambda lab: float(re.search(rf"^{re.escape(lab)}\s+(-?[\d.]+)", t, re.M).group(1))
    un = float(re.search(r"unsplit: ([\d.]+)%", t).group(1))
    h = re.search(r"^policy\s+(-?[\d.]+)\s+([\d.]+)", t, re.M)
    return dict(R=float(h.group(1)), sem=float(h.group(2)), nosplit=g("all together"), mst=g("MST d=2.0"), unsplit=un)
def fr(n):
    for l in open(f"logs/zn/frag_{n}.log"):
        p = l.split()
        if p and p[0] == n and len(p) >= 6 and p[-1].endswith("%"):
            return dict(frags=float(p[2]), free=float(p[3]), pct=p[-1])
    return {}
def fin(n):  # final-epoch training eval from the run log
    ls = [l for l in open(f"logs/zn/{n}.log") if l.startswith("[KS]")]
    m = re.search(r"H=([\d.]+)", ls[-1]); return float(m.group(1)) if m else float("nan")
def main():
    names = sorted(os.path.basename(f)[:-5] for f in glob.glob("logs/zn/P*.done"))
    print(f"{'run':30s} {'R':>7s} {'gain':>7s} {'unsplit%':>8s} {'frags':>6s} {'free':>6s} {'%inFrag':>7s} {'H':>6s}")
    for n in names:
        e, f = ev(n), fr(n)
        print(f"{n:30s} {e['R']:7.2f} {e['R']-e['nosplit']:+7.2f} {e['unsplit']:8.1f} {f.get('frags',float('nan')):6.2f} {f.get('free',float('nan')):6.1f} {f.get('pct','?'):>7s} {fin(n):6.3f}")
    print("\nPart 1 mean ± sd over seeds (gain = policy - no-split, unseen):")
    for p in (0, 1, 5, 10):
        g = [ev(f"P1_pre{p}_s{s}") for s in (0, 1, 2)]
        gains = [x['R']-x['nosplit'] for x in g]; un = [x['unsplit'] for x in g]
        print(f"pre{p:<3d} R {st.mean(x['R'] for x in g):.3f}  gain {st.mean(gains):+.3f} ± {st.stdev(gains):.3f}  unsplit {st.mean(un):.1f}%  (no-split {g[0]['nosplit']:.3f}, MST {g[0]['mst']:.3f})")


if __name__ == "__main__":
    main()
