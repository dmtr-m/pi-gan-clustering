"""Stage 2: torch Dataset over a cache directory, B-stratified splits, collate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Subset

from .cache import Selection, build_cache


class NucleonDataset(Dataset):
    """One item per event: ``{"x": (n, F) float32, "b": fB, "b_bin", "event_id"}``.

    Every numeric event-level column of the ROOT file is available as
    ``ds.event_params[name][i]``.  ``b_bin`` restricts the dataset to one impact-parameter
    bin (bins come from ``Selection.b_edges``).  With ``normalize=True`` features are
    standardised with the statistics of the event's own B bin (PDG is left as +-1).
    Particle order is shuffled per item with a generator seeded by
    (seed, event_id, epoch) -- the global numpy RNG is never touched; call
    ``set_epoch`` to get a new order each epoch.
    """

    def __init__(self, parquet_dir: str | Path, selection: Selection,
                 cache_root: str | Path = ".cache/datasets", b_bin: int | None = None,
                 normalize: bool = False, shuffle_particles: bool = True, seed: int = 0):
        self.selection = selection
        self.cache_dir = build_cache(parquet_dir, selection, cache_root)
        self.meta = json.loads((self.cache_dir / "meta.json").read_text())
        self.X = np.load(self.cache_dir / "X.npy", mmap_mode="r")
        self.offsets = np.load(self.cache_dir / "offsets.npy")
        with np.load(self.cache_dir / "events.npz") as z:
            self.event_params = {k: z[k] for k in z.files}
        self.fB, self.b_bin = self.event_params["fB"], self.event_params["b_bin"]
        self.event_id = self.event_params["event_id"]
        self.idx = np.arange(len(self.fB)) if b_bin is None else np.flatnonzero(self.b_bin == b_bin)
        self.features = list(selection.features)
        self.shuffle_particles, self.seed, self.epoch = shuffle_particles, seed, 0
        self._mean = self._std = None
        if normalize:
            self._mean = np.asarray(self.meta["norm"]["mean"], dtype=np.float32)
            self._std = np.asarray(self.meta["norm"]["std"], dtype=np.float32)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        e = int(self.idx[i])
        x = np.array(self.X[self.offsets[e]:self.offsets[e + 1]])  # copy off the mmap
        if self._mean is not None:
            k = int(self.b_bin[e])
            x = (x - self._mean[k]) / self._std[k]
        if self.shuffle_particles:
            rng = np.random.default_rng((self.seed, int(self.event_id[e]), self.epoch))
            x = x[rng.permutation(len(x))]
        return {"x": torch.from_numpy(x), "b": torch.tensor(self.fB[e], dtype=torch.float32),
                "b_bin": torch.tensor(self.b_bin[e]), "event_id": int(self.event_id[e])}


def collate_fn(batch):
    """Pad variable-length events; ``mask`` is True on real particles."""
    xs = [s["x"] for s in batch]
    x = pad_sequence(xs, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([len(v) for v in xs])
    return {"x": x, "mask": torch.arange(x.shape[1])[None] < lengths[:, None],
            "b": torch.stack([s["b"] for s in batch]),
            "b_bin": torch.stack([s["b_bin"] for s in batch]),
            "event_id": [s["event_id"] for s in batch]}


def split_by_event(ds: NucleonDataset, fracs=(0.8, 0.1, 0.1), seed: int = 0) -> list[Subset]:
    """Split events (never particles) into len(fracs) Subsets, stratified by b_bin."""
    if abs(sum(fracs) - 1) > 1e-9:
        raise ValueError("fracs must sum to 1")
    rng = np.random.default_rng(seed)
    parts = [[] for _ in fracs]
    bins = ds.b_bin[ds.idx]
    cuts_f = np.cumsum(fracs)[:-1]
    for k in np.unique(bins):
        pos = rng.permutation(np.flatnonzero(bins == k))
        cuts = np.round(cuts_f * len(pos)).astype(int)
        for part, chunk in zip(parts, np.split(pos, cuts)):
            part.extend(chunk.tolist())
    return [Subset(ds, sorted(p)) for p in parts]
