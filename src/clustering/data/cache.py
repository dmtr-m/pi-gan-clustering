"""Stage 1: nested Parquet -> per-selection memory-mappable cache.

``cache/<key>/`` holds ``X.npy`` (all selected particles, float32), ``offsets.npy``
(event i is ``X[offsets[i]:offsets[i+1]]``), ``events.npz`` (every numeric event-level
column + ``b_bin``) and ``meta.json`` (selection, source fingerprint, normalisation
statistics).  The key hashes the selection, the source ``schema.json`` and
``CACHE_VERSION``, so any change gives a new directory instead of reusing stale data.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

CACHE_VERSION = 1
NUCLEON_PDGS = (2112, 2212)
PDG, STATUS, PZ = "fParticles.fPdg", "fParticles.fStatus", "fParticles.fPz"
DEFAULT_FEATURES = tuple(f"fParticles.{c}" for c in ("fPx", "fPy", "fPz", "fE", "fX", "fY", "fZ", "fPdg"))
ParticleType = Literal["SpectatorsLeft", "SpectatorsRight", "Participants", "All"]


@dataclass(frozen=True)
class Selection:
    particle_type: ParticleType = "All"
    nucleons_only: bool = True
    b_range: tuple[float, float] | None = None  # keep events with lo <= fB < hi
    b_edges: tuple[float, ...] = ()  # b_bin = np.digitize(fB, b_edges); () -> one bin
    features: tuple[str, ...] = DEFAULT_FEATURES
    n_events: int | None = None  # first n events after the B cut

    def to_json(self) -> dict:
        return asdict(self)

    def key(self, source_fingerprint: str) -> str:
        blob = json.dumps([self.to_json(), source_fingerprint, CACHE_VERSION], sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]


def source_fingerprint(parquet_dir: Path) -> str:
    return hashlib.sha1((parquet_dir / "schema.json").read_bytes()).hexdigest()


def _needed_particle_cols(sel: Selection) -> list[str]:
    cols = list(sel.features)
    if sel.nucleons_only:
        cols.append(PDG)
    if sel.particle_type in ("Participants", "SpectatorsLeft", "SpectatorsRight"):
        cols.append(STATUS)
    if sel.particle_type in ("SpectatorsLeft", "SpectatorsRight"):
        cols.append(PZ)
    return list(dict.fromkeys(cols))


def particle_mask(sel: Selection) -> pl.Expr:
    m = pl.lit(True)
    if sel.nucleons_only:
        m = m & pl.col(PDG).is_in(NUCLEON_PDGS)
    if sel.particle_type == "Participants":
        m = m & (pl.col(STATUS) > 0)
    elif sel.particle_type == "SpectatorsLeft":
        m = m & (pl.col(STATUS) == 0) & (pl.col(PZ) < 0)
    elif sel.particle_type == "SpectatorsRight":
        m = m & (pl.col(STATUS) == 0) & (pl.col(PZ) > 0)
    return m


def encode_pdg(pdg: np.ndarray) -> np.ndarray:
    """neutron (2112) -> -1, proton (2212) -> +1 (same encoding as the legacy dataset)."""
    return (pdg - 2112) / 50 - 1


def build_cache(parquet_dir: str | Path, sel: Selection, cache_root: str | Path = ".cache/datasets") -> Path:
    parquet_dir, cache_root = Path(parquet_dir), Path(cache_root)
    schema = json.loads((parquet_dir / "schema.json").read_text())
    fp = source_fingerprint(parquet_dir)
    out = cache_root / sel.key(fp)
    if (out / "meta.json").exists():
        return out

    event_cols = [b["column"] for b in schema["branches"] if b["level"] == "event"]
    part_cols = _needed_particle_cols(sel)
    missing = [c for c in part_cols if c not in {b["column"] for b in schema["branches"]}]
    if missing:
        raise KeyError(f"columns not in {parquet_dir}: {missing}")

    lf = pl.scan_parquet(parquet_dir / "part-*.parquet").select(["event_id", *event_cols, *part_cols])
    if sel.b_range is not None:
        lf = lf.filter((pl.col("fB") >= sel.b_range[0]) & (pl.col("fB") < sel.b_range[1]))
    if sel.n_events is not None:
        lf = lf.head(sel.n_events)
    ev = lf.collect()
    if not ev["event_id"].is_sorted():
        raise ValueError("event_id must be increasing in file order")

    flat = ev.select("event_id", *part_cols).explode(part_cols).filter(particle_mask(sel))
    ids = ev["event_id"].to_numpy()
    flat_ids = flat["event_id"].to_numpy()
    counts = np.searchsorted(flat_ids, ids, "right") - np.searchsorted(flat_ids, ids, "left")
    keep = counts > 0
    n_empty = int((~keep).sum())
    counts = counts[keep]
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    X = np.stack([flat[c].to_numpy().astype(np.float64) for c in sel.features], axis=1)
    if PDG in sel.features:
        j = sel.features.index(PDG)
        X[:, j] = encode_pdg(X[:, j])
    X = X.astype(np.float32)

    events = {c: ev[c].to_numpy()[keep] for c in ["event_id", *event_cols]
              if ev.schema[c].is_numeric() or ev.schema[c] == pl.Boolean}
    b_bin = np.digitize(events["fB"], np.asarray(sel.b_edges, dtype=float)).astype(np.int64)
    events["b_bin"] = b_bin

    n_bins = len(sel.b_edges) + 1
    particle_bin = np.repeat(b_bin, counts)
    skip = {PDG}
    mean = np.zeros((n_bins, len(sel.features)))
    std = np.ones((n_bins, len(sel.features)))
    for k in range(n_bins):
        rows = X[particle_bin == k]
        if len(rows) == 0:
            continue
        for j, name in enumerate(sel.features):
            if name in skip:
                continue
            mean[k, j] = rows[:, j].mean()
            std[k, j] = rows[:, j].std() or 1.0

    meta = {
        "cache_version": CACHE_VERSION,
        "selection": sel.to_json(),
        "source": str(parquet_dir),
        "source_fingerprint": fp,
        "n_events": int(keep.sum()),
        "n_events_dropped_empty": n_empty,
        "n_particles": int(len(X)),
        "event_columns": list(events),
        "norm": {"features": list(sel.features), "skipped": sorted(skip & set(sel.features)),
                 "mean": mean.tolist(), "std": std.tolist()},
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    tmp = cache_root / f".tmp-{uuid.uuid4().hex}"
    tmp.mkdir()
    np.save(tmp / "X.npy", X)
    np.save(tmp / "offsets.npy", offsets)
    np.savez(tmp / "events.npz", **events)
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
    try:
        tmp.rename(out)
    except OSError:  # another process finished first
        shutil.rmtree(tmp)
    return out
