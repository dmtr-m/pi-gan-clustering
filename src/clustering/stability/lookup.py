import pandas as pd

from typing import FrozenSet, Tuple


class StabilityLookup:
    """Oracle stability test backed by the table of known nuclei.

    A fragment ``(A, Z)`` is *stable* — the identifier should stop splitting and
    emit it — iff that pair appears in ``csv_path``.

    This replaces the learned ``StabilityClassifier``.  The (A, Z) domain is
    small and the CSV enumerates it exactly, so a table is exact, needs no
    training, and cannot hallucinate stable species between the data points —
    the failure mode the MLP had, where it extrapolated a "stable" ridge along
    the valley of stability into cells that were never sampled as negatives.
    """

    def __init__(self, csv_path: str) -> None:
        df = pd.read_csv(csv_path)
        self.stable: FrozenSet[Tuple[int, int]] = frozenset(
            (int(a), int(z)) for a, z in zip(df["A"], df["Z"])
        )

    def is_stable(self, A: int, Z: int) -> bool:
        return (A, Z) in self.stable

    def __len__(self) -> int:
        return len(self.stable)
