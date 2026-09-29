"""The (A, Z) baseline grid: Target, momentum-only MST, SACA, and one variant.

    .venv/bin/python experiments/baseline_grid.py --n-events 58374 --row4 both

Four rows, four columns, one common logarithmic colour scale in yield per
collision:

  row 1  Target            the digitized generator distribution (single panel)
  row 2  Momentum MST      |dp| < p_cut alone, four cut values
  row 3  SACA              four admissibility cuts, QMD energy
  row 4  --row4            SACA + B_asy, or one of the BWM-corrected objectives

`--row4` picks the bottom row: "basy" is FRIGA's asymmetry term at the same four
cuts (a controlled A/B against row 3), "qmd_minus_b" and "qmd_plus_b" are the
two signs of the BWM-corrected annealing objective, each scanned over its own
admissibility cut, and "both" computes the two in one pass and writes a figure
for each — rows 2 and 3 are shared and are only evaluated once.

Unlike the composite figures in `FIGURES.md`, this one is a committed script —
the previous grid was assembled in a scratchpad that was not kept, which is why
none of its panels could be regenerated.

Panel captions use the four numbers defined in FIGURES.md 0.3, over A >= 2:
fragments per collision, bound A per collision, bound Z/A, and RMS(log10 ratio)
across the seven mass bins with the 1e-3 floor for an empty bin.
"""
import argparse
import hashlib
import inspect
import json
import os
import pathlib
import signal
import time
from collections import Counter
from multiprocessing import get_context
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from clustering.baselines.saca import SacaParams, saca_clusters
from clustering.physics import weizsacker_formula
from clustering.split_prediction.dataset import NucleonDataset
from clustering.split_prediction.mst import mst_clusters

from target_reference import load_target

BINS = [(2, 2), (3, 4), (5, 10), (11, 20), (21, 40), (41, 80), (81, 132)]

# Row 2: momentum-only MST.  p=90 is the tuned value from NOTES.md; the others
# bracket it, since the momentum cut is where nearly all the discrimination is.
P_CUTS = [60.0, 90.0, 120.0, 150.0]

# Rows 3 and 4: SACA's admissibility cut.  A fragment counts as bound when
# zeta < e_cut MeV/nucleon.  -4 is Puri & Aichelin's published value; the scan
# probes SACA's known failure, that it stops fragmenting and keeps one heavy
# over-bound residue.  e_cut_light stays at 0 throughout so deuterons remain
# allowed (a uniform -4 forbids them by construction).
E_CUTS = [-6.0, -4.0, -2.0, 0.0]

# FRIGA's asymmetry coefficient, and the gamma its own scan (0.5 / 1 / 1.5)
# found best here: BAND_WIDTH.md 1 measures sd 2.57 -> 1.94 at gamma = 0.5.
E_0_ASY = 23.3
GAMMA_ASY = 0.5

# Row 4 alternatives to B_asy.  Both anneal on a BWM-corrected objective; each
# is scanned over its own natural admissibility cut, whose scale differs because
# the objectives differ (zeta_QMD has median -4.97 over coordinate-MST
# fragments, zeta of QMD - B has median -10.98).
ZETA_B_CUTS = [-6.0, -9.0, -12.0, -15.0]     # for QMD - B
BWM_CUT_SCALES = [0.25, 0.5, 0.75, 1.0]      # for QMD + B

ROW4_CHOICES = ("basy", "qmd_minus_b", "qmd_plus_b", "both")

CACHE_DIR = pathlib.Path(os.environ.get("GRID_CACHE", ".cache/grid"))
_FINGERPRINT: Optional[str] = None


def algorithm_fingerprint() -> str:
    """Hash of everything whose change would alter a panel's counts.

    The cache is keyed on this, so editing the clusterizer, the energy model or
    the dataset invalidates it automatically.  Editing the plotting or the stats
    does not — those are downstream of the counts.  Without this a cache is a
    trap: the whole point of the runs is that the physics keeps changing, and a
    silently stale hit is exactly the class of bug this project keeps finding.
    """
    global _FINGERPRINT
    if _FINGERPRINT is None:
        from clustering import physics
        from clustering.baselines import qmd_energy, saca
        from clustering.split_prediction import dataset, mst
        h = hashlib.sha1()
        for mod in (saca, qmd_energy, physics, mst, dataset):
            h.update(pathlib.Path(inspect.getfile(mod)).read_bytes())
        for fn in (run_momentum, run_saca, load_events):
            h.update(inspect.getsource(fn).encode())
        _FINGERPRINT = h.hexdigest()[:16]
    return _FINGERPRINT


# The fingerprint the full-statistics momentum-MST, SACA and QMD - B panels were
# cached under (58374 events per side, 10 chunks, seed 0).  Those panels use only
# the original code paths, and adding the physics_v / physics_v_minus_b / mix
# energy models to saca.py changes no output of them — checked byte for byte on
# four configs before and after the edit.  Re-hashing saca.py would nonetheless
# orphan ~150 cached chunks (hours of compute), so their key is pinned here.
# Anything that touches a new code path is keyed on the live hash instead, and a
# genuine change to the *legacy* paths must bump this by hand.
LEGACY_FINGERPRINT = "c1cad292d64e84ba"
_LEGACY_MODELS = (None, "qmd", "qmd_plus_b", "qmd_minus_b")
_LEGACY_KEYS = {"kind", "p_cut", "e_cut", "asymmetry", "energy_model",
                "bwm_weight", "e_cut_model", "bwm_scale"}


def _is_legacy(kw: dict) -> bool:
    return (kw.get("energy_model") in _LEGACY_MODELS
            and set(kw) <= _LEGACY_KEYS)


def _cache_path(kw: dict, path: str, chunk: int, lo: int, hi: int, seed: int
                ) -> pathlib.Path:
    fp = LEGACY_FINGERPRINT if _is_legacy(kw) else algorithm_fingerprint()
    key = json.dumps([fp, sorted(kw.items()), path,
                      chunk, lo, hi, seed], sort_keys=True, default=str)
    return CACHE_DIR / f"{hashlib.sha1(key.encode()).hexdigest()}.json"


def _cache_load(f: pathlib.Path) -> Optional[Counter]:
    try:
        raw = json.loads(f.read_text())
    except (OSError, ValueError):
        return None            # missing or half-written: just recompute
    return Counter({(int(k.split(",")[0]), int(k.split(",")[1])): v
                    for k, v in raw.items()})


def _cache_store(f: pathlib.Path, c: Counter) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps({f"{A},{Z}": v for (A, Z), v in c.items()}))
    tmp.replace(f)             # atomic, so a killed run leaves no partial file


def panel_specs(row4: str = "basy") -> List[Tuple[Tuple[int, int], str, dict]]:
    """Every panel below row 1, as ((row, col), title, kwargs) — the unit of work.

    ``row4`` selects what occupies the bottom row.  "both" computes the two
    BWM-corrected objectives at once, on rows 3 and 4, so a single pass can
    render one figure per choice without recomputing rows 1 and 2.
    """
    out = []
    for j, p in enumerate(P_CUTS):
        out.append(((1, j), f"Momentum  $p$ = {p:.0f} MeV/c",
                    {"kind": "momentum", "p_cut": p}))
    for j, ec in enumerate(E_CUTS):
        out.append(((2, j), f"SACA  $e_{{cut}}$ = {ec:.0f}",
                    {"kind": "saca", "e_cut": ec}))

    def basy(row):
        return [((row, j), f"SACA + $B_{{asy}}$  $e_{{cut}}$ = {ec:.0f}",
                 {"kind": "saca", "e_cut": ec, "asymmetry": True})
                for j, ec in enumerate(E_CUTS)]

    def qmd_minus_b(row):
        return [((row, j), f"QMD $-$ B  $\\zeta_B$ cut {c:g}",
                 {"kind": "saca", "energy_model": "qmd_minus_b", "bwm_weight": 1.0,
                  "e_cut_model": "objective", "e_cut": c})
                for j, c in enumerate(ZETA_B_CUTS)]

    def qmd_plus_b(row):
        return [((row, j), f"QMD $+$ B  BWM cut $\\times${sc:g}",
                 {"kind": "saca", "energy_model": "qmd_plus_b", "bwm_weight": 1.0,
                  "e_cut_model": "bwm", "bwm_scale": sc})
                for j, sc in enumerate(BWM_CUT_SCALES)]

    if row4 == "basy":
        out += basy(3)
    elif row4 == "qmd_minus_b":
        out += qmd_minus_b(3)
    elif row4 == "qmd_plus_b":
        out += qmd_plus_b(3)
    elif row4 == "both":
        out += qmd_minus_b(3) + qmd_plus_b(4)
    else:
        raise ValueError(f"unknown row4 {row4!r}; expected one of {ROW4_CHOICES}")
    return out


def load_events(path: str, lo: int, hi: int) -> List[torch.Tensor]:
    """Events with dataset index in [lo, hi), taken from both spectator sides."""
    events = []
    for side in ("SpectatorsLeft", "SpectatorsRight"):
        ds = NucleonDataset(path, particle_type=side)
        events += [ds[i] for i in range(lo, min(hi, len(ds)))]
    return events


def _worker(job: Tuple[int, int, int, str, int]) -> Tuple[int, int, Dict[Tuple[int, int], Counter]]:
    """Run *every* panel over one slice of events.

    The split is by event, not by configuration, on purpose: `e_cut = -6` leaves
    the most fragments unstable and so costs several times what `e_cut = 0` does,
    and splitting by configuration would leave that one panel as the critical
    path.  Every worker doing every panel over an equal slice is balanced by
    construction, and it keeps each worker's event list to 1/jobs of the total.
    """
    chunk, lo, hi, path, seed, row4, n_side = job
    # Each chunk seeds the legacy global RNG that NucleonDataset permutes with,
    # so the whole run is reproducible for a given (seed, jobs, n_events).
    np.random.seed(seed * 100003 + chunk)
    specs = [(key, dict(kw)) for key, _t, kw in panel_specs(row4)]
    files = {key: _cache_path(kw, path, chunk, lo, hi, seed) for key, kw in specs}
    out, n_hit = {}, 0
    for key, _kw in specs:
        hit = _cache_load(files[key]) if files[key].exists() else None
        if hit is not None:
            out[key] = hit
            n_hit += 1

    # Only touch the data if something actually has to be computed — loading a
    # slice costs ~13 s per worker and a full cache hit should cost nothing.
    if len(out) < len(specs):
        events = load_events(path, lo, hi)
        n_ev = len(events)
        for key, kw in specs:
            if key in out:
                continue
            kind = kw.pop("kind")
            out[key] = (run_momentum(events, kw["p_cut"]) if kind == "momentum"
                        else run_saca(events, seed + chunk, **kw))
            _cache_store(files[key], out[key])
    else:
        # Nothing was loaded, so count what the slice *would* hold.  n_side is
        # passed in from main rather than hardcoded: a literal sized for one
        # dataset silently miscounts the next one.
        n_ev = 2 * (min(hi, n_side) - lo)
    return chunk, n_ev, out, n_hit


def run_momentum(events, p_cut: float) -> Counter:
    """Momentum-only MST: link i, j if |dp| < p_cut in the pair rest frame."""
    out = Counter()
    for e in events:
        if len(e) < 2:
            continue
        x = e.unsqueeze(0)
        mask = torch.ones(1, len(e), dtype=torch.bool)
        lab = mst_clusters(x, mask, 0.0, p_cut, "momentum")[0].numpy()
        xn = e.numpy()
        for c in np.unique(lab[lab >= 0]):
            idx = np.flatnonzero(lab == c)
            out[(len(idx), int((xn[idx, 7] == 1).sum()))] += 1
    return out


def run_saca(events, seed: int = 0, *, asymmetry: bool = False, **kw) -> Counter:
    """SACA on a coordinate MST seed at d_cut = 2.0 fm, as published."""
    params = SacaParams(e_cut_light=0.0, asymmetry=asymmetry,
                        e_0_asy=E_0_ASY, gamma_asy=GAMMA_ASY, **kw)
    rng = np.random.default_rng(seed)
    out = Counter()
    for e in events:
        if len(e) < 2:
            continue
        lab = saca_clusters(e, params, d_cut=2.0, metric="coord", rng=rng).labels
        xn = e.numpy()
        for c in np.unique(lab[lab >= 0]):
            idx = np.flatnonzero(lab == c)
            out[(len(idx), int((xn[idx, 7] == 1).sum()))] += 1
    return out


def stats(counter: Counter, n_coll: float, g: np.ndarray) -> Dict[str, float]:
    """frags, bound A, bound Z/A and RMS, per FIGURES.md 0.3.  A >= 2 only."""
    n = sum(v for (A, _), v in counter.items() if A >= 2) / n_coll
    a = sum(A * v for (A, _), v in counter.items() if A >= 2) / n_coll
    z = sum(Z * v for (A, Z), v in counter.items() if A >= 2) / n_coll
    y = np.array([sum(v for (A, _), v in counter.items() if lo <= A <= hi)
                  for lo, hi in BINS]) / n_coll
    r = np.where(y > 0, y / g, 1e-3)
    return {"frags": n, "bound_a": a, "z_over_a": z / max(a, 1e-9),
            "rms": float(np.sqrt((np.log10(r) ** 2).mean()))}


def counter_to_map(counter: Counter, n_coll: float, a_max: int, z_max: int) -> np.ndarray:
    """(z_max+1, a_max+1) map of yield per collision; 0 where nothing was made."""
    m = np.zeros((z_max + 1, a_max + 1))
    for (A, Z), v in counter.items():
        if 2 <= A <= a_max and 0 <= Z <= z_max:
            m[Z, A] += v / n_coll
    return m


_ZSTAR: Dict[int, float] = {}


def zstar(A: int) -> float:
    """Z of the Weizsacker valley at mass A."""
    if A not in _ZSTAR:
        Z = np.arange(1, max(A, 2))
        B = weizsacker_formula(torch.full((len(Z),), float(A)),
                               torch.tensor(Z, dtype=torch.float))
        _ZSTAR[A] = float(Z[int(B.argmax())])
    return _ZSTAR[A]


def band(counter, lo: int = 5, hi: int = 60) -> Tuple[float, float]:
    """(mean, sd) of Z - Z*(A), yield-weighted, over a mass window.

    Reported next to the RMS because the RMS is a *mass* metric and is blind to
    charge: it ranked a dineutron-dominated configuration best in the QMD + B
    scan, where this diagnostic showed it as by far the worst composition.
    """
    dev = [(Z - zstar(A), v) for (A, Z), v in counter.items() if lo <= A <= hi]
    # Guard on the number of distinct species, not the weight sum: the target's
    # weights are yields per collision (order 1), not raw counts.
    if len(dev) < 5:
        return float("nan"), float("nan")
    d = np.array([x for x, _ in dev], float)
    w = np.array([v for _, v in dev], float)
    m = float(np.average(d, weights=w))
    return m, float(np.sqrt(np.average((d - m) ** 2, weights=w)))


def print_bands(panels, ref) -> None:
    tm, tsd = band({(int(A), int(Z)): y for A, Z, y in ref})
    print(f"\n{'panel':<34}{'band mean':>10}{'sd':>7}   top species (A,Z)")
    print(f"{'Target':<34}{tm:10.2f}{tsd:7.2f}")
    for _i, _j, title, c in panels:
        m, sd = band(c)
        top = sorted(((v, A, Z) for (A, Z), v in c.items() if A >= 2), reverse=True)[:4]
        clean = title.replace("$", "").replace("\\", "").replace("{", "").replace("}", "")
        print(f"{clean:<34}{m:10.2f}{sd:7.2f}   "
              + ", ".join(f"({A},{Z}):{v}" for v, A, Z in top))


def render_grid(panels, ref, n_coll, g, out_path, suptitle, n_rows: int = 4,
                n_cols: int = 4, a_max: int = 132, z_max: int = 60,
                row_labels: Optional[Dict[int, str]] = None,
                panel_in: float = 5.0) -> None:
    """Draw Target plus one panel per entry, on one shared log colour scale.

    ``panels`` is [(row, col, title, Counter)] with row 0 reserved for Target,
    which is drawn at (0, 0); any unused cell in row 0 is switched off.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    target_map = np.zeros((z_max + 1, a_max + 1))
    for A, Z, y in ref:
        if 2 <= A <= a_max and 0 <= Z <= z_max:
            target_map[int(Z), int(A)] += y

    maps = {(i, j): counter_to_map(c, n_coll, a_max, z_max) for i, j, _, c in panels}
    pos = np.concatenate([target_map.ravel()] + [m.ravel() for m in maps.values()])
    pos = pos[pos > 0]
    norm = LogNorm(vmin=max(pos.min(), 1e-5), vmax=pos.max())

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_in * n_cols, panel_in * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    mesh = None

    def draw(ax, m, title, s):
        nonlocal mesh
        mesh = ax.pcolormesh(np.arange(a_max + 2) - 0.5, np.arange(z_max + 2) - 0.5,
                             np.ma.masked_where(m <= 0, m), norm=norm, cmap="viridis")
        ax.set_title(title, fontsize=11)
        ax.grid(color="white", lw=0.3, alpha=0.35)
        ax.set_axisbelow(False)
        if s is not None:
            rms = "—" if s["rms"] is None else f"{s['rms']:.3f}"
            ax.text(0.03, 0.96,
                    f"{s['frags']:.2f} frags | bound A {s['bound_a']:.1f}\n"
                    f"Z/A {s['z_over_a']:.4f} | RMS {rms}",
                    transform=ax.transAxes, va="top", fontsize=9, color="0.15")

    ta = ref[:, 0] >= 2
    draw(axes[0][0], target_map, "Target",
         {"frags": ref[ta, 2].sum(),
          "bound_a": (ref[ta, 0] * ref[ta, 2]).sum(),
          "z_over_a": (ref[ta, 1] * ref[ta, 2]).sum() / (ref[ta, 0] * ref[ta, 2]).sum(),
          "rms": None})   # the target is the reference; no ratio to itself
    used = {(i, j) for i, j, _, _ in panels} | {(0, 0)}
    for i, j, title, c in panels:
        draw(axes[i][j], maps[(i, j)], title, stats(c, n_coll, g))
    for i in range(n_rows):
        for j in range(n_cols):
            if (i, j) not in used:
                axes[i][j].axis("off")

    for j in range(n_cols):
        axes[-1][j].set_xlabel("$A$")
    for i in range(n_rows):
        axes[i][0].set_ylabel("$Z$")
        if row_labels and i in row_labels:
            axes[i][0].set_ylabel(f"{row_labels[i]}\n$Z$", fontsize=12)
    axes[0][0].set_xlim(0, a_max)
    axes[0][0].set_ylim(0, z_max)
    # Reserve a *fixed* strip for the title, not the default 12% of the figure:
    # the figure is 5 in per row, so the fractional default leaves a 2-inch white
    # band on a 4-row grid and half that on a 3-row one.  Both of these must run
    # before the colorbar, which lays out from the axes' current positions.
    title_in = 0.75
    fig.subplots_adjust(top=1.0 - title_in / fig.get_figheight())
    fig.suptitle(suptitle, fontsize=15, y=1.0 - 0.25 * title_in / fig.get_figheight())
    fig.colorbar(mesh, ax=axes, label="yield per collision", fraction=0.02, pad=0.01)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\nwrote {out_path}")


def print_table(panels, ref, n_coll, g) -> None:
    ta = ref[:, 0] >= 2
    print(f"\n{'panel':<38}{'frags':>8}{'bound A':>10}{'Z/A':>9}{'RMS':>8}")
    print(f"{'Target':<38}{ref[ta, 2].sum():8.2f}{(ref[ta, 0] * ref[ta, 2]).sum():10.1f}"
          f"{(ref[ta, 1] * ref[ta, 2]).sum() / (ref[ta, 0] * ref[ta, 2]).sum():9.4f}{'—':>8}")
    for _i, _j, title, c in panels:
        s = stats(c, n_coll, g)
        clean = title.replace("$", "").replace("{", "").replace("}", "").replace("\\", "")
        print(f"{clean:<38}{s['frags']:8.2f}{s['bound_a']:10.1f}"
              f"{s['z_over_a']:9.4f}{s['rms']:8.3f}")


def _install_term_handler() -> None:
    """Turn SIGTERM into SystemExit so the Pool context manager still runs.

    Without this, a plain `kill` on the parent leaves every worker orphaned and
    still burning a core: under the "spawn" start method their command line is
    `spawn_main`, not this script, so they also survive `pkill -f baseline_grid`.
    Two killed runs left 22 orphans competing for 11 cores here, which looked
    exactly like thermal throttling.  Raising SystemExit lets `with Pool(...)`
    call terminate() on the way out.
    """
    def die(signum, _frame):
        raise SystemExit(f"terminated by signal {signum}")
    signal.signal(signal.SIGTERM, die)


def main() -> None:
    _install_term_handler()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-events", type=int, default=100,
                    help="events per spectator side; one collision = two events")
    ap.add_argument("--data", default="data/xecs_hse.parquet")
    ap.add_argument("--out", default="figures/az_grid.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="worker processes; events are split evenly between them")
    ap.add_argument("--row4", default="basy", choices=ROW4_CHOICES,
                    help="what fills the bottom row; 'both' writes one figure each")
    args = ap.parse_args()

    ref = load_target()
    g = np.array([ref[(ref[:, 0] >= lo) & (ref[:, 0] <= hi), 2].sum() for lo, hi in BINS])

    specs = panel_specs(args.row4)
    # Read the length rather than hardcoding it — a literal sized for one
    # dataset silently crops or over-runs the next one.
    n_per_side = min(args.n_events,
                     len(NucleonDataset(args.data, particle_type="SpectatorsLeft")))
    edges = np.linspace(0, n_per_side, args.jobs + 1).round().astype(int)
    jobs = [(c, int(edges[c]), int(edges[c + 1]), args.data, args.seed, args.row4,
             n_per_side) for c in range(args.jobs) if edges[c + 1] > edges[c]]
    print(f"{n_per_side} events per side over {len(jobs)} worker(s)\n")

    merged: Dict[Tuple[int, int], Counter] = {key: Counter() for key, _, _ in specs}
    n_events_seen = 0
    t0 = time.time()
    if len(jobs) == 1:
        results = [_worker(jobs[0])]
    else:
        # "spawn" is the macOS default and the safe choice here: the workers each
        # load their own slice, so there is nothing to inherit and forking a
        # process that has already imported torch is not worth the risk.
        with get_context("spawn").Pool(len(jobs)) as pool:
            results = []
            for chunk, n_ev, out, n_hit in pool.imap_unordered(_worker, jobs):
                results.append((chunk, n_ev, out, n_hit))
                print(f"  chunk {chunk:2d} done ({n_ev} events, {n_hit}/{len(specs)} "
                      f"cached)  {len(results)}/{len(jobs)}  {time.time() - t0:7.1f} s")
    for _chunk, n_ev, out, _n_hit in results:
        n_events_seen += n_ev
        for key, c in out.items():
            merged[key] += c

    n_coll = n_events_seen / 2.0
    print(f"\n{n_events_seen} events = {n_coll:.0f} collisions in "
          f"{(time.time() - t0) / 60:.1f} min\n")
    panels: List[Tuple[int, int, str, Counter]] = [
        (key[0], key[1], title, merged[key]) for key, title, _ in specs]

    suptitle = f"Fragment (A, Z) yield per collision — {n_coll:.0f} collisions"
    if args.row4 == "both":
        # One figure per bottom row, out of the same merged counts: rows 1 and 2
        # are identical between them and are only computed once.
        for tag, keep, drop in (("qmd_minus_b", 3, 4), ("qmd_plus_b", 4, 3)):
            sel = [(3 if i == keep else i, j, t, c)
                   for i, j, t, c in panels if i != drop]
            out = args.out.replace(".png", f"_{tag}.png")
            render_grid(sel, ref, n_coll, g, out, suptitle)
            print(f"\n=== bottom row: {tag} ===")
            print_table(sel, ref, n_coll, g)
            print_bands(sel, ref)
    else:
        render_grid(panels, ref, n_coll, g, args.out, suptitle)
        print_table(panels, ref, n_coll, g)
        print_bands(panels, ref)


if __name__ == "__main__":
    main()
