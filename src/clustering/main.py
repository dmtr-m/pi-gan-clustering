"""Hierarchical clustering experiment: up-to-K-way nucleon fragmentation.

Partitions a nucleon cloud into nuclear fragments with a slot-attention split
model trained by REINFORCE.  Each split assigns nucleons to one of ``n_clusters``
(K) slots, so a single pass yields up to K fragments (K=2 is the binary case).

Pipeline
--------
  Stage 1  build the stability lookup table from the known-nuclei CSV (diagnostic)
  Stage 2  train the up-to-K-way SplitPredictionModel (REINFORCE) -> dm_model.pt
  Stage 3  run FragmentsIdentifier and plot fragment distributions

The stability oracle is now a table lookup (no training, no sc_model.pt); Stage 3
reads the CSV directly, so Stage 1 is purely a diagnostic and is not a
prerequisite for Stage 3.

Figures are written to ``<out_dir>/figures`` (this is a script, so nothing is
shown interactively).

Examples
--------
  python main.py                         # run all three stages
  python main.py --stages 2 3            # retrain split + viz
  python main.py --n-clusters 4 --split-epochs 120
  python main.py --stages 3 --force-split   # viz bypassing the stability lookup
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # headless: save figures instead of showing them
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from split_prediction_model.dataset import NucleonDataset, collate_fn
from split_prediction_model.module import SplitPredictionModel, SplitValueCritic
from split_prediction_model.mst import MSTPretrainer
from split_prediction_model.trainer import KSplitTrainer
from stability_classifier.module import StabilityLookup
from module import FragmentsIdentifier


# ─── Paths ────────────────────────────────────────────────────────────────────
# Anchor data to the repo layout rather than the current working directory, so
# `python main.py` works from anywhere.  Layout:
#   pi-gan-experiments/
#     data/                                     <- DATA_DIR
#     src/hierarchical_clustering_v4/main.py     <- this file
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parents[1]            # pi-gan-experiments/
DATA_DIR = PROJECT_ROOT / "data"


# ─── Configuration ────────────────────────────────────────────────────────────

def _default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class Config:
    # Paths (absolute, anchored to the repo layout — cwd-independent)
    data_path: str = str(DATA_DIR / "xexe_urqmd_5fm.parquet")
    csv_path: str = str(DATA_DIR / "existing_nuclei_amc_5fm.csv")
    out_dir: str = str(_HERE)
    device: str = field(default_factory=_default_device)

    # Data
    n_events: int = 500
    particle_type: str = "SpectatorsLeft"

    # Stage 1 — Stability lookup table (no training; diagnostic plot only)
    sc_a_max: int = 30           # upper A for the stability-map diagnostic grid

    # Stage 2 — SplitPredictionModel (up-to-K-way)
    n_clusters: int = 5          # K: max fragments produced per split
    hidden_dim: int = 32
    n_iters: int = 3
    split_epochs: int = 40
    split_batch_size: int = 64
    split_lr: float = 3e-4
    # Tree depth for training.  Every split node is rewarded and backwarded
    # independently (no autograd graph is held across levels), and empty /
    # too-small nodes are pruned, so depth is cheap: K=5,k=8 visits ~19 live
    # nodes per batch instead of the 97k an unpruned BFS would enumerate.
    split_k: int = 8
    baseline_momentum: float = 0.95

    # Actor-Critic: DeepSets V(s) replaces the scalar EMA baseline
    use_critic: bool = True
    critic_hidden_dim: int = 64
    value_coef: float = 0.5

    # MST supervised warm-start (runs before REINFORCE; 0 disables)
    pretrain_epochs: int = 25
    pretrain_lr: float = 1e-3
    mst_d_cut: float = 2.0       # fm; >2.5 percolates -> collapses to "never split"

    # Stage 3 — FragmentsIdentifier visualization
    # depth 8 is enough to peel every nucleon free if the model wants to
    # (2^8 = 256 > the largest event), so the depth limit never binds
    max_depth: int = 8
    n_vis: int = 200
    force_split: bool = False    # True bypasses the stability lookup

    type_index: int = 7
    input_dim: int = 8

    # Derived paths -----------------------------------------------------------
    @property
    def dm_model_path(self) -> Path:
        return Path(self.out_dir) / "dm_model.pt"

    @property
    def critic_model_path(self) -> Path:
        return Path(self.out_dir) / "critic_model.pt"

    @property
    def fig_dir(self) -> Path:
        d = Path(self.out_dir) / "figures"
        d.mkdir(parents=True, exist_ok=True)
        return d


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_dataset(cfg: Config) -> NucleonDataset:
    ds = NucleonDataset(cfg.data_path, particle_type=cfg.particle_type, n_events=cfg.n_events)
    sizes = [ds[i].shape[0] for i in range(len(ds))]
    print(f"Events         : {len(ds)}")
    print(f"Nucleons/event : min={min(sizes)}, mean={np.mean(sizes):.1f}, max={max(sizes)}")
    return ds


# ─── Stage 1 — Stability lookup table ──────────────────────────────────────────

def build_stability_lookup(cfg: Config) -> StabilityLookup:
    print("\n=== Stage 1 — Stability lookup table ===")
    lut = StabilityLookup(cfg.csv_path)
    print(f"Loaded {len(lut)} known nuclei from {cfg.csv_path}")
    _plot_stability_table(cfg, lut)
    return lut


def _plot_stability_table(cfg: Config, lut: StabilityLookup) -> None:
    """Map of the lookup over the (A, Z) plane: green = stable (in table)."""
    import pandas as pd

    df_stable = pd.read_csv(cfg.csv_path)
    grid = [(A, Z) for A in range(2, cfg.sc_a_max + 1) for Z in range(0, A + 1)]
    stable = np.array([lut.is_stable(A, Z) for A, Z in grid], dtype=float)
    g = np.array(grid, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(g[:, 0], g[:, 1], c=stable, cmap="RdYlGn",
                    vmin=0, vmax=1, s=30, alpha=0.8)
    ax.scatter(df_stable["A"], df_stable["Z"], s=60, edgecolors="k",
               facecolors="none", lw=1.5, label="table entries")
    fig.colorbar(sc, ax=ax, label="stable (lookup)")
    ax.set(xlabel="A", ylabel="Z", title="Stability lookup table")
    ax.legend()
    _save(cfg, fig, "stability_table.png")


# ─── Stage 2 — Up-to-K-way split model ─────────────────────────────────────────

def train_split_model(cfg: Config, dataset: NucleonDataset) -> SplitPredictionModel:
    print(f"\n=== Stage 2 — Up-to-K-way split (K={cfg.n_clusters}, k={cfg.split_k}) ===")
    model = SplitPredictionModel(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        n_iters=cfg.n_iters,
        n_clusters=cfg.n_clusters,
    )
    loader = DataLoader(dataset, batch_size=cfg.split_batch_size, shuffle=True, collate_fn=collate_fn)

    # Supervised warm-start against classical MST fragments before REINFORCE.
    # Only ~6% of the per-node reward variance is action-driven, so RL alone is a
    # weak channel to discover clustering structure from scratch; this hands the
    # model that structure up front and cuts seed-to-seed variance ~8x.
    if cfg.pretrain_epochs > 0:
        print(f"--- MST warm-start ({cfg.pretrain_epochs} epochs, d_cut={cfg.mst_d_cut} fm) ---")
        MSTPretrainer(
            model,
            loader,
            optimizer=optim.AdamW(model.parameters(), lr=cfg.pretrain_lr),
            device=cfg.device,
            d_cut=cfg.mst_d_cut,
        ).train(n_epochs=cfg.pretrain_epochs, log_every=max(1, cfg.pretrain_epochs // 5))

    # Actor-Critic: the critic's parameters must share the optimizer with the actor.
    critic = (
        SplitValueCritic(input_dim=cfg.input_dim, hidden_dim=cfg.critic_hidden_dim)
        if cfg.use_critic
        else None
    )
    params = list(model.parameters()) + (list(critic.parameters()) if critic else [])
    print(f"--- REINFORCE ({'Actor-Critic' if critic else 'EMA baseline'}) ---")

    optimizer = optim.AdamW(params, lr=cfg.split_lr)
    scheduler = optim.lr_scheduler.ConstantLR(optimizer)
    trainer = KSplitTrainer(
        model,
        loader,
        optimizer=optimizer,
        scheduler=scheduler,
        k=cfg.split_k,
        critic=critic,
        value_coef=cfg.value_coef,
        baseline_momentum=cfg.baseline_momentum,
        type_index=cfg.type_index,
        device=cfg.device,
    )
    history = trainer.train(n_epochs=cfg.split_epochs)
    torch.save(model.state_dict(), cfg.dm_model_path)
    print(f"Saved {cfg.dm_model_path}")
    if critic is not None:
        torch.save(critic.state_dict(), cfg.critic_model_path)
        print(f"Saved {cfg.critic_model_path}")

    _plot_split_history(cfg, history)
    return model


def _plot_split_history(cfg: Config, history: Dict[str, List[float]]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["eval_reward"], label="eval reward (argmax split)", color="C2")
    ax1.plot(history["reward"], label="sampled reward", color="C0", alpha=0.5)
    ax1.plot(history["baseline"], label="baseline (EMA)", linestyle="--", color="C1")
    ax1.set(xlabel="Epoch", ylabel="Q [MeV/nucleon]",
            title=f"KSplitTrainer (K={cfg.n_clusters}) — QMD reward")
    ax1.legend()
    ax2.plot(history["loss"], label="policy loss")
    if cfg.use_critic and any(history["value_loss"]):
        # On the z-scored target, value_loss ~= 1 means the critic is no better
        # than predicting the mean; -> 0 means it explains the reward.
        ax2.plot(history["value_loss"], label="critic value loss", linestyle="--")
        ax2.axhline(1.0, color="grey", lw=0.8, alpha=0.6)
        ax2.legend()
    ax2.set(xlabel="Epoch", ylabel="Loss", title=f"KSplitTrainer (K={cfg.n_clusters}) — Losses")
    _save(cfg, fig, "split_history.png")


# ─── Stage 3 — Fragment identification & visualization ─────────────────────────

def build_identifier(cfg: Config) -> FragmentsIdentifier:
    fi = FragmentsIdentifier(
        max_depth=cfg.max_depth,
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        n_iters=cfg.n_iters,
        n_clusters=cfg.n_clusters,
        csv_path=cfg.csv_path,
        force_split=cfg.force_split,
    )
    fi.split_prediction_module.load_state_dict(torch.load(cfg.dm_model_path))
    fi.to(cfg.device)
    fi.eval()
    return fi


def visualize_fragments(cfg: Config) -> None:
    print("\n=== Stage 3 — Fragment identification & visualization ===")
    fi = build_identifier(cfg)

    vis_datasets = {
        "SpectatorsLeft": NucleonDataset(cfg.data_path, particle_type="SpectatorsLeft"),
        "SpectatorsRight": NucleonDataset(cfg.data_path, particle_type="SpectatorsRight"),
    }

    nuclei_A: List[int] = []
    nuclei_Z: List[int] = []
    nuclei_eta: List[float] = []
    nucleon_eta: List[float] = []
    n_frags_per_event: List[int] = []
    split_depths: List[int] = []

    # Event-level diagnostics (one entry per event) ---------------------------
    ev_n_nucleons: List[int] = []      # nucleons per event
    ev_max_z: List[int] = []           # largest fragment charge in the event
    ev_pn_ratio: List[float] = []      # protons / neutrons
    ev_nucleon_pt: List[float] = []    # mean nucleon pt
    ev_frag_pt_pn: List[float] = []    # mean fragment pt per nucleon
    cons_dA: List[float] = []          # conservation residuals: Σ fragments − Σ nucleons
    cons_dZ: List[float] = []
    cons_dpx: List[float] = []
    cons_dpy: List[float] = []
    cons_dpz: List[float] = []
    cons_dE: List[float] = []

    for ds in vis_datasets.values():
        for idx in range(min(cfg.n_vis, len(ds))):
            event = ds[idx].to(cfg.device)  # (N, 8)
            ev_np = event.cpu().numpy()

            p = ev_np[:, :3]
            p_norm = np.linalg.norm(p, axis=1)
            ok = p_norm > 0
            nucleon_eta.extend((-np.arctanh(p[ok, 2] / p_norm[ok])).tolist())

            # Input (nucleon) event-level quantities.
            n_nuc = ev_np.shape[0]
            z_in = int((ev_np[:, cfg.type_index] == 1).sum())
            n_neutrons = n_nuc - z_in
            p_in_sum = p.sum(0)
            e_in_sum = float(ev_np[:, 3].sum())
            ev_n_nucleons.append(n_nuc)
            ev_pn_ratio.append(z_in / n_neutrons if n_neutrons > 0 else np.nan)
            ev_nucleon_pt.append(float(np.hypot(p[:, 0], p[:, 1]).mean()))

            with torch.no_grad():
                result = fi(event)

            n_frags_per_event.append(len(result.fragments))
            split_depths.extend(result.split_depths)

            # Output (fragment) totals, accumulated over this event.
            sum_a = sum_z = 0
            p_out_sum = np.zeros(3)
            e_out_sum = 0.0
            max_z = 0
            frag_pt_pn: List[float] = []
            for frag in result.fragments:
                frag_np = frag.cpu().numpy()
                A = frag_np.shape[0]
                Z = int((frag_np[:, cfg.type_index] == 1).sum())
                nuclei_A.append(A)
                nuclei_Z.append(Z)
                p_f = frag_np[:, :3].sum(0)
                pn = float(np.linalg.norm(p_f))
                if pn > 0:
                    nuclei_eta.append(float(-np.arctanh(p_f[2] / pn)))
                sum_a += A
                sum_z += Z
                p_out_sum += p_f
                e_out_sum += float(frag_np[:, 3].sum())
                max_z = max(max_z, Z)
                frag_pt_pn.append(float(np.hypot(p_f[0], p_f[1]) / A))  # per nucleon

            ev_max_z.append(max_z)
            ev_frag_pt_pn.append(float(np.mean(frag_pt_pn)) if frag_pt_pn else np.nan)
            cons_dA.append(sum_a - n_nuc)
            cons_dZ.append(sum_z - z_in)
            cons_dpx.append(float(p_out_sum[0] - p_in_sum[0]))
            cons_dpy.append(float(p_out_sum[1] - p_in_sum[1]))
            cons_dpz.append(float(p_out_sum[2] - p_in_sum[2]))
            cons_dE.append(e_out_sum - e_in_sum)

    nuclei_A_arr = np.array(nuclei_A)
    nuclei_Z_arr = np.array(nuclei_Z)
    nuclei_N_arr = nuclei_A_arr - nuclei_Z_arr
    n_frags_arr = np.array(n_frags_per_event)
    split_depths_arr = np.array(split_depths)

    print(f"Events           : {cfg.n_vis * len(vis_datasets)}")
    print(f"Fragments total  : {len(nuclei_A_arr)}  "
          f"(A=1: {(nuclei_A_arr == 1).sum()}, A>=2: {(nuclei_A_arr >= 2).sum()})")
    print(f"Fragments/event  : min={n_frags_arr.min()}, "
          f"mean={n_frags_arr.mean():.1f}, max={n_frags_arr.max()}")
    print(f"Splits total     : {len(split_depths_arr)}  "
          f"(max tree level: {split_depths_arr.max() if len(split_depths_arr) else 0})")

    _plot_n_fragments(cfg, n_frags_arr)
    _plot_splits_per_level(cfg, split_depths_arr)
    _plot_eta(cfg, np.array(nucleon_eta), np.array(nuclei_eta))
    _plot_az_nz(cfg, nuclei_A_arr, nuclei_Z_arr, nuclei_N_arr)

    # Event-level diagnostics
    _plot_conservation(cfg, np.array(cons_dA), np.array(cons_dZ),
                       np.array(cons_dpx), np.array(cons_dpy),
                       np.array(cons_dpz), np.array(cons_dE))
    _plot_mean_pt(cfg, np.array(ev_nucleon_pt), np.array(ev_frag_pt_pn))
    _plot_max_charge(cfg, np.array(ev_max_z))
    _plot_n_nucleons(cfg, np.array(ev_n_nucleons))
    _plot_pn_ratio(cfg, np.array(ev_pn_ratio))


def _plot_splits_per_level(cfg: Config, split_depths: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(split_depths):
        levels = np.arange(0, split_depths.max() + 2)
        ax.hist(split_depths, bins=levels - 0.5, color="slateblue",
                edgecolor="black", lw=0.5)
        ax.set_xticks(np.arange(0, split_depths.max() + 1))
    ax.set(xlabel="Tree level (split depth)", ylabel="Number of splits",
           title="Splits per tree level")
    _save(cfg, fig, "splits_per_level.png")


def _plot_n_fragments(cfg: Config, n_frags: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(n_frags.min(), n_frags.max() + 2) - 0.5
    ax.hist(n_frags, bins=bins, color="mediumseagreen", edgecolor="black", lw=0.5)
    ax.axvline(n_frags.mean(), color="crimson", linestyle="--", lw=1.5,
               label=f"mean = {n_frags.mean():.1f}")
    ax.set(xlabel="Fragments per event", ylabel="Events",
           title="Fragment multiplicity")
    ax.legend()
    _save(cfg, fig, "fragment_multiplicity.png")


def _plot_eta(cfg: Config, nucleon_eta: np.ndarray, nuclei_eta: np.ndarray) -> None:
    fig, (ax_nu, ax_nuc) = plt.subplots(1, 2, figsize=(14, 5))
    ax_nu.hist(nucleon_eta, bins=100, color="steelblue")
    ax_nu.set(xlabel="Pseudorapidity η", ylabel="Count", title="Nucleons (pre-split)")
    ax_nuc.hist(nuclei_eta, bins=100, color="darkorange")
    ax_nuc.set(xlabel="Pseudorapidity η", ylabel="Count", title="Identified fragments (all A)")
    _save(cfg, fig, "fragment_eta.png")


def _plot_az_nz(cfg: Config, A: np.ndarray, Z: np.ndarray, N: np.ndarray) -> None:
    fig, (ax_AZ, ax_NZ) = plt.subplots(1, 2, figsize=(12, 5))
    bins_A = np.arange(0, 22) - 0.5
    bins_Z = np.arange(0, 15) - 0.5
    bins_N = np.arange(0, 15) - 0.5
    h1 = ax_AZ.hist2d(A, Z, bins=[bins_A, bins_Z], cmap="viridis", norm=LogNorm())
    ax_AZ.set(xlabel="A", ylabel="Z", title="A vs Z")
    fig.colorbar(h1[3], ax=ax_AZ, label="Count")
    h2 = ax_NZ.hist2d(N, Z, bins=[bins_N, bins_Z], cmap="viridis", norm=LogNorm())
    ax_NZ.set(xlabel="N", ylabel="Z", title="N vs Z")
    fig.colorbar(h2[3], ax=ax_NZ, label="Count")
    _save(cfg, fig, "fragment_az_nz.png")


# ─── Event-level diagnostics ────────────────────────────────────────────────────

def _plot_conservation(cfg: Config, dA: np.ndarray, dZ: np.ndarray,
                       dpx: np.ndarray, dpy: np.ndarray, dpz: np.ndarray,
                       dE: np.ndarray) -> None:
    """Per-event residuals Σ(fragments) − Σ(nucleons).

    Clustering only partitions the nucleons, so every conserved quantity should
    stay pinned at 0 (up to float round-off).  A non-zero spread here is a bug —
    a nucleon dropped, double-counted, or a fragment's momentum recomputed rather
    than summed.
    """
    panels = [
        ("ΔA (baryon number)", dA),
        ("ΔZ (charge)", dZ),
        ("Δpₓ [GeV/c]", dpx),
        ("Δp_y [GeV/c]", dpy),
        ("Δp_z [GeV/c]", dpz),
        ("ΔE [GeV]", dE),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (title, data) in zip(axes.ravel(), panels):
        ax.hist(data, bins=41, color="teal", edgecolor="black", lw=0.3)
        ax.axvline(0, color="crimson", ls="--", lw=1)
        ax.set(title=f"{title}   max|Δ| = {np.abs(data).max():.1e}", ylabel="Events")
    fig.suptitle("Conservation laws  (Σ fragments − Σ nucleons, per event)", fontsize=14)
    _save(cfg, fig, "conservation.png")


def _plot_mean_pt(cfg: Config, nucleon_pt: np.ndarray, frag_pt_pn: np.ndarray) -> None:
    frag_pt_pn = frag_pt_pn[np.isfinite(frag_pt_pn)]
    fig, ax = plt.subplots(figsize=(8, 5))
    hi = max(nucleon_pt.max(), frag_pt_pn.max()) if len(frag_pt_pn) else nucleon_pt.max()
    bins = np.linspace(0, hi, 40)
    ax.hist(nucleon_pt, bins=bins, alpha=0.6, color="steelblue",
            label=f"nucleons  (mean {nucleon_pt.mean():.3f})")
    ax.hist(frag_pt_pn, bins=bins, alpha=0.6, color="darkorange",
            label=f"fragments / nucleon  (mean {frag_pt_pn.mean():.3f})")
    ax.set(xlabel="⟨pₜ⟩ per event [GeV/c]", ylabel="Events",
           title="Mean transverse momentum")
    ax.legend()
    _save(cfg, fig, "mean_pt.png")


def _plot_max_charge(cfg: Config, max_z: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0, max_z.max() + 2) - 0.5
    ax.hist(max_z, bins=bins, color="indianred", edgecolor="black", lw=0.5)
    ax.axvline(max_z.mean(), color="navy", ls="--", lw=1.5,
               label=f"mean = {max_z.mean():.1f}")
    ax.set(xlabel="Maximum fragment charge Z", ylabel="Events",
           title="Maximum charge per event")
    ax.legend()
    _save(cfg, fig, "max_charge.png")


def _plot_n_nucleons(cfg: Config, n_nuc: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(n_nuc.min(), n_nuc.max() + 2) - 0.5
    ax.hist(n_nuc, bins=bins, color="mediumpurple", edgecolor="black", lw=0.5)
    ax.axvline(n_nuc.mean(), color="crimson", ls="--", lw=1.5,
               label=f"mean = {n_nuc.mean():.1f}")
    ax.set(xlabel="Nucleons per event", ylabel="Events",
           title="Number of nucleons")
    ax.legend()
    _save(cfg, fig, "n_nucleons.png")


def _plot_pn_ratio(cfg: Config, pn_ratio: np.ndarray) -> None:
    pn_ratio = pn_ratio[np.isfinite(pn_ratio)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pn_ratio, bins=40, color="seagreen", edgecolor="black", lw=0.3)
    ax.axvline(pn_ratio.mean(), color="crimson", ls="--", lw=1.5,
               label=f"mean = {pn_ratio.mean():.2f}")
    ax.set(xlabel="Z / N  (protons / neutrons)", ylabel="Events",
           title="Proton-to-neutron ratio")
    ax.legend()
    _save(cfg, fig, "pn_ratio.png")


# ─── Plot helper ───────────────────────────────────────────────────────────────

def _save(cfg: Config, fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    path = cfg.fig_dir / name
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


# ─── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: Config, stages: List[int]) -> None:
    print(f"Device: {cfg.device}")
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    dataset = None
    if 1 in stages:
        build_stability_lookup(cfg)
    if 2 in stages:
        dataset = load_dataset(cfg)
        train_split_model(cfg, dataset)
    if 3 in stages:
        visualize_fragments(cfg)


def _parse_args() -> argparse.Namespace:
    cfg = Config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stages", type=int, nargs="+", choices=[1, 2, 3], default=[1, 2, 3],
                   help="Which stages to run (default: all).")
    p.add_argument("--device", default=cfg.device, help="torch device (default: auto).")
    p.add_argument("--out-dir", default=cfg.out_dir, help="Where to write models and figures/.")
    p.add_argument("--data-path", default=cfg.data_path)
    p.add_argument("--csv-path", default=cfg.csv_path)
    p.add_argument("--n-events", type=int, default=cfg.n_events)
    # Stage 2 knobs (the focus of this experiment)
    p.add_argument("--n-clusters", type=int, default=cfg.n_clusters,
                   help="K: max fragments produced per split.")
    p.add_argument("--split-k", type=int, default=cfg.split_k,
                   help="Split-tree depth; every node is rewarded independently. "
                        "k=1 is a single up-to-K split.")
    p.add_argument("--no-critic", action="store_true",
                   help="Disable the Actor-Critic value head; use the scalar EMA baseline.")
    p.add_argument("--split-epochs", type=int, default=cfg.split_epochs)
    p.add_argument("--pretrain-epochs", type=int, default=cfg.pretrain_epochs,
                   help="MST supervised warm-start epochs before REINFORCE (0 disables).")
    p.add_argument("--mst-d-cut", type=float, default=cfg.mst_d_cut,
                   help="MST coordinate cut [fm]; >2.5 percolates into one giant fragment.")
    p.add_argument("--hidden-dim", type=int, default=cfg.hidden_dim)
    p.add_argument("--max-depth", type=int, default=cfg.max_depth)
    p.add_argument("--n-vis", type=int, default=cfg.n_vis)
    p.add_argument("--force-split", action="store_true",
                   help="Stage 3: always split, bypassing the stability lookup.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = Config(
        data_path=args.data_path,
        csv_path=args.csv_path,
        out_dir=args.out_dir,
        device=args.device,
        n_events=args.n_events,
        n_clusters=args.n_clusters,
        hidden_dim=args.hidden_dim,
        split_epochs=args.split_epochs,
        pretrain_epochs=args.pretrain_epochs,
        mst_d_cut=args.mst_d_cut,
        split_k=args.split_k,
        use_critic=not args.no_critic,
        max_depth=args.max_depth,
        n_vis=args.n_vis,
        force_split=args.force_split,
    )
    run(cfg, args.stages)


if __name__ == "__main__":
    main()
