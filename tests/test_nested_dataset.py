"""Synthetic-ROOT checks for clustering.data (no real UrQMD file needed).

Run from the repo root:  PYTHONPATH=src .venv/bin/python tests/test_nested_dataset.py

Writes a small ROOT file with uproot (event scalars + jagged per-particle branches,
including empty events), then checks each stage against a direct numpy reference:
1. ROOT -> Parquet loses nothing and keeps the per-event lengths (== fNpa);
2. the cache selection (type / nucleons / B window / B bins) matches a hand-built mask;
3. the cache is reused, and a different selection gets a different key;
4. shuffling is deterministic per (seed, event, epoch) and leaves the global RNG alone;
5. normalisation is per B bin and leaves PDG at +-1;
6. splits are disjoint, cover everything, and every B bin is in every split.
"""
import json
import tempfile
from pathlib import Path

import awkward as ak
import numpy as np
import polars as pl
import torch
import uproot

from clustering.data import NucleonDataset, Selection, collate_fn, split_by_event
from clustering.data.root_to_parquet import convert

N_EV = 60
F = "fParticles."


def make_root(path, seed=0, n=N_EV):
    rng = np.random.default_rng(seed)
    npa = rng.integers(0, 12, n)
    npa[3] = 0  # an empty event
    tot = int(npa.sum())
    jag = lambda a: ak.unflatten(a, npa)
    cols = {
        "fEventNr": np.arange(n, dtype=np.int32),
        "fB": rng.uniform(0, 12, n).astype(np.float32),
        "fNpa": npa.astype(np.int32),
        F + "fPdg": jag(rng.choice([2112, 2212, 211], tot).astype(np.int32)),
        F + "fStatus": jag(rng.integers(0, 3, tot).astype(np.int32)),
        **{F + c: jag(rng.normal(size=tot)) for c in ("fPx", "fPy", "fPz", "fE", "fX", "fY", "fZ")},
    }
    with uproot.recreate(path) as f:
        f.mktree("events", {k: v.type if isinstance(v, ak.Array) else v.dtype for k, v in cols.items()})
        f["events"].extend(cols)
    return cols


def close(a, b, tol=1e-5):
    assert np.allclose(a, b, atol=tol), (a, b)


def main():
    tmp = Path(tempfile.mkdtemp())
    cols = make_root(tmp / "a.root")
    out = convert([tmp / "a.root"], tmp / "nested", chunk_events=25)
    cache = tmp / "cache"

    # 1. ROOT -> Parquet
    s = json.loads((out / "schema.json").read_text())
    assert s["n_events"] == N_EV and len(s["parts"]) == 3, s
    df = pl.read_parquet(out / "part-*.parquet")
    assert df["event_id"].to_list() == list(range(N_EV))
    close(df["fB"].to_numpy(), cols["fB"])
    for ev in (0, 3, 40):
        close(df[F + "fPx"][ev].to_list(), ak.to_list(cols[F + "fPx"][ev]))
    assert df[F + "fPdg"].list.len().to_list() == df["fNpa"].to_list()
    assert s["fNpa_mismatches"] == 0
    assert {b["level"] for b in s["branches"]} == {"event", "particle"}
    try:
        convert([tmp / "a.root"], out)
        raise AssertionError("expected FileExistsError")
    except FileExistsError:
        pass
    print("1. root->parquet ok:", s["n_events"], "events,", len(s["branches"]), "branches")

    # 2. selection vs reference
    sel = Selection(particle_type="SpectatorsLeft", b_edges=(4.0, 8.0))
    ds = NucleonDataset(out, sel, cache, shuffle_particles=False)
    assert len(ds) > 0
    for i in range(len(ds)):
        e = int(ds.event_id[i])
        pdg, st, pz = (ak.to_numpy(cols[F + k][e]) for k in ("fPdg", "fStatus", "fPz"))
        m = np.isin(pdg, (2112, 2212)) & (st == 0) & (pz < 0)
        x = ds[i]["x"].numpy()
        assert len(x) == m.sum() > 0
        close(x[:, 2], pz[m])
        assert ds.b_bin[i] == np.digitize(cols["fB"][e], (4.0, 8.0))
        assert abs(ds[i]["b"].item() - cols["fB"][e]) < 1e-5
    assert "fEventNr" in ds.event_params
    ds2 = NucleonDataset(out, Selection(b_range=(2, 6)), cache)
    assert ((ds2.fB >= 2) & (ds2.fB < 6)).all()
    ds3 = NucleonDataset(out, Selection(b_edges=(6.0,)), cache, b_bin=1)
    assert len(ds3) and (ds3.fB[ds3.idx] >= 6).all()
    print(f"2. selection ok: {len(ds)} spectator-left events; empties dropped:",
          ds.meta["n_events_dropped_empty"])

    # 3. cache reuse / key
    a = NucleonDataset(out, Selection(), cache).cache_dir
    t = (a / "meta.json").stat().st_mtime_ns
    assert NucleonDataset(out, Selection(), cache).cache_dir == a
    assert (a / "meta.json").stat().st_mtime_ns == t
    assert NucleonDataset(out, Selection(b_range=(0, 6)), cache).cache_dir != a
    print("3. cache reuse ok")

    # 4. shuffle
    d = NucleonDataset(out, Selection(), cache, seed=1)
    state = np.random.get_state()[1].copy()
    x0 = d[0]["x"]
    assert torch.equal(x0, d[0]["x"])
    d.set_epoch(1)
    assert not torch.equal(x0, d[0]["x"])
    d.set_epoch(0)
    assert torch.equal(x0, d[0]["x"])
    assert (np.random.get_state()[1] == state).all()
    ref = NucleonDataset(out, Selection(), cache, shuffle_particles=False)[0]["x"]
    assert torch.equal(x0.sort(0).values, ref.sort(0).values)
    print("4. shuffle ok")

    # 5. per-bin normalisation
    sel = Selection(b_edges=(6.0,))
    n = NucleonDataset(out, sel, cache, normalize=True, shuffle_particles=False)
    j, jp = sel.features.index(F + "fPx"), sel.features.index(F + "fPdg")
    for k in (0, 1):
        xs = torch.cat([n[i]["x"] for i in range(len(n)) if n.b_bin[i] == k])[:, j]
        assert abs(xs.mean()) < 1e-3 and abs(xs.std() - 1) < 1e-2, (k, xs.mean(), xs.std())
    pdgs = torch.cat([n[i]["x"] for i in range(len(n))])[:, jp].unique().tolist()
    assert set(pdgs) <= {-1.0, 1.0}, pdgs
    print("5. normalisation ok")

    # 6. collate + split
    ds = NucleonDataset(out, Selection(b_edges=(4.0, 8.0)), cache)
    sets = [set(sub.indices) for sub in split_by_event(ds, (0.6, 0.2, 0.2), seed=0)]
    assert sum(map(len, sets)) == len(ds) == len(set().union(*sets))
    for k in np.unique(ds.b_bin):
        assert all((ds.b_bin[list(sub)] == k).any() for sub in sets), k
    batch = collate_fn([ds[i] for i in range(5)])
    assert batch["mask"].sum(1).tolist() == [len(ds[i]["x"]) for i in range(5)]
    assert batch["x"].shape[0] == batch["b"].shape[0] == batch["b_bin"].shape[0] == 5
    print("6. collate + split ok")
    print("ALL OK")


if __name__ == "__main__":
    main()
