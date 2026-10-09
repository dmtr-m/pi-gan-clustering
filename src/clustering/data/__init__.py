"""UrQMD event data: ROOT -> nested Parquet -> per-selection cache -> torch Dataset."""
from .cache import Selection, build_cache
from .dataset import NucleonDataset, collate_fn, split_by_event

__all__ = ["Selection", "build_cache", "NucleonDataset", "collate_fn", "split_by_event"]
