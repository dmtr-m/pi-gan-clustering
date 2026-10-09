"""Impact-parameter distributions and how event content depends on B.

    PYTHONPATH=src .venv/bin/python experiments/b_distributions.py \\
        --parquet-dir data/xecs_hse_nested --out figures/b_dist

Reads the nested Parquet from clustering.data (one row per event) and writes three
sheets: <out>_overview.png, <out>_counts.png, <out>_spectators.png.  Every axis range
is taken from the data.  Nucleons = PDG 2112/2212; spectator = fStatus == 0, side by
the sign of fPz; participant = fStatus > 0.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
BLUE, ORANGE, GREEN = "#2a6fb0", "#d9822b", "#3a8f5c"  # fixed order: L / R / participants
PDG, ST = "fParticles.fPdg", "fParticles.fStatus"
PX, PY, PZ = "fParticles.fPx", "fParticles.fPy", "fParticles.fPz"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "figure.dpi": 130,
})


def aggregates(parquet_dir: Path) -> pl.DataFrame:
    """One row per event: B and per-event content counts, from the list columns."""
    # the masks combine several list columns elementwise, so explode once and group back
    base = pl.scan_parquet(parquet_dir / "part-*.parquet").select(
        "event_id", PDG, ST, PX, PY, PZ).explode(PDG, ST, PX, PY, PZ)
    is_nuc = pl.col(PDG).is_in([2112, 2212])
    spec = is_nuc & (pl.col(ST) == 0)
    sl, sr, part = spec & (pl.col(PZ) < 0), spec & (pl.col(PZ) > 0), is_nuc & (pl.col(ST) > 0)
    pt = (pl.col(PX) ** 2 + pl.col(PY) ** 2).sqrt()
    g = base.group_by("event_id", maintain_order=True).agg(
        n_nuc=is_nuc.sum(), n_other=(~is_nuc).sum(),
        n_sl=sl.sum(), n_sr=sr.sum(), n_part=part.sum(),
        zl=(sl & (pl.col(PDG) == 2212)).sum(),
        ptl=pt.filter(sl).mean(), pzl=pl.col(PZ).filter(sl).mean(),
        coll=pl.col(ST).filter(part).mean(),
    )
    b = pl.scan_parquet(parquet_dir / "part-*.parquet").select("event_id", "fB")
    d = b.join(g, on="event_id").collect()
    # counts come out unsigned; cast so differences cannot wrap around
    return d.with_columns(pl.col("n_nuc", "n_other", "n_sl", "n_sr", "n_part", "zl").cast(pl.Int64))


def mean_line(ax, x, y, bins=24, color=INK):
    """Binned mean of y vs x as a thin line (data-driven bins; empty bins skipped)."""
    ok = np.isfinite(y)
    e = np.linspace(x[ok].min(), x[ok].max(), bins + 1)
    k = np.digitize(x[ok], e[1:-1])
    xs = [(e[i] + e[i + 1]) / 2 for i in range(bins) if (k == i).sum() > 5]
    ys = [y[ok][k == i].mean() for i in range(bins) if (k == i).sum() > 5]
    ax.plot(xs, ys, color=color, lw=1.6)


def density(ax, x, y, xlabel, ylabel, title, bins=70, cmap="Blues"):
    y = np.asarray(y, float)
    ok = np.isfinite(y)  # ratios / means are undefined for events with no such particles
    ybins = bins
    if np.all(y[ok] == np.round(y[ok])):  # integer counts: edges on half-integers, no aliasing stripes
        lo, hi = y[ok].min(), y[ok].max()
        step = max(1, int(np.ceil((hi - lo + 1) / bins)))
        ybins = np.arange(lo - 0.5, hi + 0.5 + step, step)
    h = ax.hist2d(x[ok], y[ok], bins=[bins, ybins], cmin=1, cmap=cmap, norm="log")
    mean_line(ax, x, y)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    ax.grid(False)
    return h[3]


def sheet_overview(d, path):
    b = d["fB"].to_numpy()
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    a = ax[0, 0]
    a.hist(b, bins=56, color=BLUE, lw=0)
    a.set(xlabel="B (fm)", ylabel="events"); a.set_title(f"B distribution ({len(b):,} events)", loc="left")
    a = ax[0, 1]
    a.hist(b ** 2, bins=56, color=BLUE, lw=0)
    a.set(xlabel="B² (fm²)", ylabel="events")
    a.set_title("B²: flat if B was sampled with dσ ∝ b db", loc="left")
    a = ax[1, 0]
    s = np.sort(b)
    a.plot(s, np.arange(1, len(s) + 1) / len(s), color=BLUE, lw=1.8)
    for q in (0.05, 0.1, 0.25, 0.5, 0.75):
        a.axhline(q, color=GRID, lw=0.8, zorder=0); a.text(s[0], q, f" {q:.0%} below B={np.quantile(b, q):.1f}", va="bottom", fontsize=7.5, color=MUTED)
    a.set(xlabel="B (fm)", ylabel="fraction of events with fB ≤ B"); a.set_title("Cumulative: where central events run out", loc="left")
    a = ax[1, 1]
    for n, c, lab in (("n_sl", BLUE, "left spectators"), ("n_sr", ORANGE, "right spectators"), ("n_part", GREEN, "participants")):
        a.hist(d[n].to_numpy(), bins=60, histtype="step", color=c, lw=1.6, label=lab)
    a.set(xlabel="nucleons per event", ylabel="events"); a.legend(frameon=False)
    a.set_title("Nucleon counts per event, B pooled", loc="left")
    fig.savefig(path); plt.close(fig)


def sheet_counts(d, path):
    b = d["fB"].to_numpy()
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.4), constrained_layout=True)
    specs = [("n_sl", "left spectators"), ("n_sr", "right spectators"), ("n_part", "participants"),
             ("n_nuc", "all nucleons"), ("n_other", "non-nucleons (π, K, …)")]
    last = None
    for a, (k, lab) in zip(ax.flat, specs):
        last = density(a, b, d[k].to_numpy(), "B (fm)", lab, f"{lab} vs B")
    a = ax.flat[5]
    tot = d["n_sl"].to_numpy() + d["n_sr"].to_numpy() + d["n_part"].to_numpy()
    a.hist(d["n_nuc"].to_numpy() - tot, bins=40, color=BLUE, lw=0)
    a.set(xlabel="n_nuc − (L + R + participants)", ylabel="events"); a.set_title("Nucleon bookkeeping check (0 = closed)", loc="left")
    fig.colorbar(last, ax=ax, shrink=0.6, label="events per cell (log)")
    fig.savefig(path); plt.close(fig)


def sheet_spectators(d, path):
    b = d["fB"].to_numpy()
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.4), constrained_layout=True)
    a = ax[0, 0]
    sc = a.scatter(d["n_sl"].to_numpy(), d["n_sr"].to_numpy(), c=b, s=3, cmap="viridis", alpha=0.35, lw=0, rasterized=True)
    a.set(xlabel="left spectators", ylabel="right spectators"); a.set_title("Left vs right spectators, coloured by B", loc="left")
    a.grid(False); fig.colorbar(sc, ax=a, label="B (fm)")
    density(ax[0, 1], b, (d["n_sl"] - d["n_sr"]).to_numpy(), "B (fm)", "left − right spectators", "Left/right asymmetry vs B")
    density(ax[0, 2], b, (d["zl"] / d["n_sl"]).to_numpy(), "B (fm)", "proton fraction (left spectators)", "Spectator Z/A vs B")
    density(ax[1, 0], b, d["ptl"].to_numpy(), "B (fm)", "mean pT of left spectators (GeV/c)", "Spectator transverse kick vs B")
    density(ax[1, 1], b, d["pzl"].to_numpy(), "B (fm)", "mean pz of left spectators (GeV/c)", "Spectator longitudinal momentum vs B")
    density(ax[1, 2], b, d["coll"].to_numpy(), "B (fm)", "mean fStatus of participants", "Collisions per participant vs B")
    fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", default="data/xecs_hse_nested")
    ap.add_argument("--out", default="figures/b_dist")
    a = ap.parse_args()
    d = aggregates(Path(a.parquet_dir))
    print(d.describe())
    for name, fn in (("overview", sheet_overview), ("counts", sheet_counts), ("spectators", sheet_spectators)):
        fn(d, f"{a.out}_{name}.png"); print("wrote", f"{a.out}_{name}.png")


if __name__ == "__main__":
    main()
