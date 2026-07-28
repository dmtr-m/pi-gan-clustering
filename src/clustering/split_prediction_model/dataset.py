import torch
from torch.utils.data import Dataset
import polars as pl
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from typing import Literal, List, Tuple

class NucleonDataset(Dataset):
    def __init__(
            self,
            file_path: str,
            particle_type: Literal["SpectatorsLeft", "SpectatorsRight", "Participants", "All"],
            n_events: int = None,  # type: ignore
    ):
        """
        Dataset for nucleons.
        Args:
            file_path: Path to parquet file.
            n_events: Optional limit on the number of events to load.
        """
        self.df = pl.read_parquet(file_path)
        self.event_ids = self.df["event_id"].unique().to_list()

        self.df = (
            self.df.filter(
                pl.col("fParticles.fPdg").is_in([2112, 2212])
            )
        )

        if n_events is not None:
            self.event_ids = self.event_ids[:n_events]
            # Filter dataframe to only include selected events
            self.df = self.df.filter(pl.col("event_id").is_in(self.event_ids))
        
        if particle_type == "Participants":
            self.df = (
                self.df
                .filter(
                    pl.col("fParticles.fStatus") > 0
                )
            )
        elif particle_type == "SpectatorsLeft":
            self.df = (
                self.df
                .filter(
                    (pl.col("fParticles.fStatus") == 0) & (pl.col("fParticles.fPz") < 0)
                )
            )
        elif particle_type == "SpectatorsRight":
            self.df = (
                self.df
                .filter(
                    (pl.col("fParticles.fStatus") == 0) & (pl.col("fParticles.fPz") > 0)
                )
            )

        # Normalize PDG code to be either -1 (neutron) or 1 (proton)
        self.df = (
            self.df
            .with_columns(
                ((pl.col("fParticles.fPdg") - 2112) / 50 - 1).alias("fParticles.fPdg")
            )
        )

    def __len__(self):
        return len(self.event_ids)

    def __getitem__(self, idx):
        event_id = self.event_ids[idx]
        event_data = self.df.filter(pl.col("event_id") == event_id)

        # Extract nucleon features: Px, Py, Pz, Energy, X, Y, Z and PDG code
        feature_cols = [
            "fParticles.fPx",
            "fParticles.fPy",
            "fParticles.fPz",
            "fParticles.fE",
            "fParticles.fX",
            "fParticles.fY",
            "fParticles.fZ",
            "fParticles.fPdg",
        ]
        available_cols = [c for c in feature_cols if c in event_data.columns]
        features = event_data.select(available_cols).to_numpy()

        indices = np.random.permutation(len(features))
        shuffled_features = features[indices]

        return torch.tensor(shuffled_features, dtype=torch.float32)


def collate_fn(batch):
    """
    Collate function to pad variable length events and create masks.
    """
    xs = [item for item in batch]

    # Pad sequences with 0.
    # xs: (batch_size, max_particles, n_features)
    x_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)

    # Create mask: True for real particles, False for padding
    # mask: (batch_size, max_particles)
    mask = torch.zeros(x_padded.shape[:2], dtype=torch.bool)
    for i, x in enumerate(xs):
        mask[i, :len(x)] = True

    return {
        "x": x_padded,
        "mask": mask
    }
