"""Stage 0: ROOT -> nested Parquet, nothing dropped.

Output is a directory holding ``part-00000.parquet, ...`` (one row per event) plus
``schema.json``.  Event-level branches become scalar columns; per-particle branches
become list columns whose length is the number of particles in that event.

Column names are the ROOT leaf names with the path prefix removed
(``event/fB`` -> ``fB``, ``event/fParticles/fParticles.fPdg`` -> ``fParticles.fPdg``);
the full ROOT path is kept in ``schema.json``.  Three columns are added:
``event_id`` (sequential across all input files), ``source_file`` and ``entry``.

    python -m clustering.data.root_to_parquet run1.root run2.root -o data/xexe_nested
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import polars as pl
import uproot

ADDED_COLUMNS = ("event_id", "source_file", "entry")
_ROW = "__row"


def _find_tree(f: uproot.ReadOnlyDirectory, name: str | None):
    """The TTree to read.  ROOT keeps older cycles of a tree as ``name;N``; with no
    explicit ``name`` the highest cycle of the single tree name is used.  Pass
    ``name="events;16"`` to pick a cycle yourself."""
    if name is not None:
        return f[name]
    cycles: dict[str, list[int]] = {}
    for k, cls in f.classnames().items():
        if cls == "TTree":
            base, _, cyc = k.partition(";")
            cycles.setdefault(base, []).append(int(cyc or 1))
    if len(cycles) != 1:
        raise ValueError(f"{f.file_path}: expected exactly one TTree name, found {sorted(cycles)}; pass tree=")
    (base, cycs), = cycles.items()
    if len(cycs) > 1:
        print(f"{f.file_path}: tree '{base}' has cycles {sorted(cycs)}; using {max(cycs)}")
    return f[f"{base};{max(cycs)}"]


def _leaf_keys(tree) -> tuple[list[str], list[str]]:
    """(leaf branch paths, container branch paths).  A container has sub-branches."""
    leaves, containers = [], []
    for key in tree.keys(recursive=True):
        (containers if len(tree[key].branches) else leaves).append(key)
    return leaves, containers


def _column_name(path: str) -> str:
    return path.split("/")[-1]


def _jagged_to_lists(counts: np.ndarray, cols: dict[str, np.ndarray]) -> pl.DataFrame:
    """Flat per-particle arrays + per-event counts -> one list column per array."""
    n = len(counts)
    flat = pl.DataFrame({_ROW: np.repeat(np.arange(n), counts), **{k: pl.Series(k, v) for k, v in cols.items()}})
    agg = flat.group_by(_ROW, maintain_order=True).agg(pl.exclude(_ROW))
    out = pl.DataFrame({_ROW: np.arange(n)}).join(agg, on=_ROW, how="left").sort(_ROW)
    # events with zero particles come out of the join as null
    for k in cols:
        out = out.with_columns(pl.col(k).fill_null(pl.Series([[]], dtype=out.schema[k])).alias(k)) \
            if out[k].null_count() else out
    return out.drop(_ROW)


def _chunk_to_frame(arrays: dict, paths: list[str], levels: dict[str, str]) -> pl.DataFrame:
    event_cols, part_cols, counts = {}, {}, None
    for p in paths:
        a = arrays[p]
        col = _column_name(p)
        if levels[p] == "event":
            event_cols[col] = ak.to_numpy(a)
        else:
            c = ak.to_numpy(ak.num(a, axis=1))
            if counts is None:
                counts = c
            elif not np.array_equal(counts, c):
                raise ValueError(f"{p}: per-event length differs from the other particle branches")
            part_cols[col] = ak.to_numpy(ak.flatten(a, axis=1))
    df = pl.DataFrame({k: pl.Series(k, v) for k, v in event_cols.items()})
    if part_cols:
        df = pl.concat([df, _jagged_to_lists(counts, part_cols)], how="horizontal")
    return df


def convert(root_files: list[str | Path], out_dir: str | Path, tree: str | None = None,
            chunk_events: int = 5000, overwrite: bool = False) -> Path:
    """Convert ROOT files to a nested Parquet directory.  Returns ``out_dir``."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{out_dir} is not empty (overwrite=True to replace)")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in list(out_dir.glob("part-*.parquet")) + list(out_dir.glob("schema.json")):
        old.unlink()

    root_files = [Path(p) for p in root_files]
    paths = containers = levels = None
    n_events, parts, files, npa_mismatch = 0, [], [], 0

    for rf in root_files:
        with uproot.open(rf) as f:
            t = _find_tree(f, tree)
            leaves, conts = _leaf_keys(t)
            if paths is None:
                paths, containers = leaves, conts
                names = [_column_name(p) for p in paths]
                if len(set(names)) != len(names) or set(names) & set(ADDED_COLUMNS):
                    raise ValueError(f"column-name collision after dropping ROOT paths: {names}")
            elif leaves != paths:
                raise ValueError(f"{rf}: branch list differs from {root_files[0]}")

            n_file = t.num_entries
            files.append({"name": rf.name, "size": rf.stat().st_size, "mtime": rf.stat().st_mtime,
                          "tree": t.name, "object_path": t.object_path, "num_entries": n_file})
            for start in range(0, n_file, chunk_events):
                stop = min(start + chunk_events, n_file)
                arrays = t.arrays(paths, library="ak", entry_start=start, entry_stop=stop)
                if levels is None:
                    levels = {p: "event" if arrays[p].ndim == 1 else "particle" for p in paths}
                else:
                    for p in paths:
                        if ("event" if arrays[p].ndim == 1 else "particle") != levels[p]:
                            raise ValueError(f"{p}: event/particle level changed between chunks")
                df = _chunk_to_frame(arrays, paths, levels)
                m = stop - start
                df = df.with_columns(
                    pl.Series("event_id", np.arange(n_events, n_events + m, dtype=np.int64)),
                    pl.lit(rf.name).alias("source_file"),
                    pl.Series("entry", np.arange(start, stop, dtype=np.int64)),
                ).select([*ADDED_COLUMNS, *[c for c in df.columns]])
                if "fNpa" in df.columns:
                    first_part = next(_column_name(p) for p in paths if levels[p] == "particle")
                    npa_mismatch += int((df["fNpa"] != df[first_part].list.len()).sum())
                part = out_dir / f"part-{len(parts):05d}.parquet"
                df.write_parquet(part)
                parts.append(part.name)
                n_events += m

    first = pl.read_parquet(out_dir / parts[0])
    schema = {
        "format": 1,
        "tree": files[0]["tree"],
        "n_events": n_events,
        "files": files,
        "parts": parts,
        "uproot_version": uproot.__version__,
        "containers_skipped": containers,
        "added_columns": list(ADDED_COLUMNS),
        "fNpa_mismatches": npa_mismatch,
        "branches": [
            {"root_path": p, "column": _column_name(p), "level": levels[p],
             "dtype": str(first.schema[_column_name(p)])}
            for p in paths
        ],
    }
    (out_dir / "schema.json").write_text(json.dumps(schema, indent=2))
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root_files", nargs="+")
    ap.add_argument("-o", "--out-dir", required=True)
    ap.add_argument("--tree", default=None, help="TTree name (default: the only one in the file)")
    ap.add_argument("--chunk-events", type=int, default=5000)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    out = convert(a.root_files, a.out_dir, a.tree, a.chunk_events, a.overwrite)
    s = json.loads((out / "schema.json").read_text())
    print(f"{s['n_events']} events, {len(s['parts'])} parts, {len(s['branches'])} branches -> {out}")
    if s["fNpa_mismatches"]:
        print(f"WARNING: fNpa != particle count in {s['fNpa_mismatches']} events")


if __name__ == "__main__":
    main()
