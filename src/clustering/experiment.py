"""Hierarchical clustering experiment: up-to-K-way nucleon fragmentation.

Partitions a nucleon cloud into nuclear fragments with a slot-attention split
model trained by REINFORCE.  Each split assigns nucleons to one of ``n_clusters``
(K) slots, so a single pass yields up to K fragments (K=2 is the binary case).

Pipeline
--------
  Stage 1  build the stability lookup table from the known-nuclei CSV (diagnostic)
  Stage 2  train the up-to-K-way SplitPredictionModel (REINFORCE) -> dm_model.pt
  Stage 3  run FragmentsIdentifier and plot fragment distributions

The stability oracle is a table lookup (no training, no sc_model.pt); Stage 3
reads the CSV directly, so Stage 1 is purely a diagnostic and is not a
prerequisite for Stage 3.

Harness
-------
Configuration is Hydra (`conf/config.yaml`, structured schema ``ClusteringConfig``)
and every run is tracked with Aim into a single project-root ``.aim`` store.  All
artifacts (checkpoints, figures, the resolved config) are written into the
per-run Hydra output dir — never a fixed path.  See README.md.

After `pip install -e .`, run from anywhere via the `clustering-run` console
script (equivalently `python -m clustering.experiment`).

Examples
--------
  clustering-run                              # run all three stages (Actor-Critic)
  clustering-run use_critic=false             # raw REINFORCE (EMA baseline), no code fork
  clustering-run stages=[2,3] n_clusters=4 split_epochs=120
  clustering-run +experiment=quick            # tiny smoke test
  clustering-run stages=[3] load_split_from=/path/to/prev_run   # viz an old checkpoint
  clustering-run --multirun split_lr=1e-4,3e-4,1e-3             # sweep, one Aim run each
"""
from __future__ import annotations

import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless: save figures instead of showing them
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import aim
import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from clustering.split_prediction.dataset import NucleonDataset, collate_fn
from clustering.split_prediction.model import SplitPredictionModel, SplitValueCritic
from clustering.split_prediction.mst import MSTPretrainer
from clustering.split_prediction.trainer import KSplitTrainer
from clustering.stability.lookup import StabilityLookup
from clustering.identifier import FragmentsIdentifier


# ─── Paths ────────────────────────────────────────────────────────────────────
# Anchor data and the Aim store to the repo layout rather than the current
# working directory, so the code works no matter where it is run from.  Layout:
#   pi-gan/                            <- REPO_ROOT
#     data/                            <- DATA_DIR
#     .aim/                            <- single Aim store (pinned; see README)
#     src/clustering/experiment.py     <- this file
_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[1]               # pi-gan/
DATA_DIR = REPO_ROOT / "data"
AIM_REPO = str(REPO_ROOT / ".aim")


# ─── Configuration ────────────────────────────────────────────────────────────

def _default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _seed_everything(seed: int) -> None:
    """Seed python / numpy / torch so a run is reproducible from its logged seed.

    Not full bit-determinism (no ``use_deterministic_algorithms`` — MPS/GPU kernels
    can still vary slightly), but it pins the model init, data shuffling, and
    action sampling, which is what makes seed sweeps meaningful.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds the MPS/CUDA generators in recent torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ClusteringConfig:
    """Structured (typed) Hydra schema — mirrors the old ``Config`` dataclass.

    Every field is overridable on the CLI.  Derived paths (checkpoints, figures)
    are NOT config fields: they live under the per-run Hydra output dir and are
    computed at runtime by ``Ctx`` — see ``main``.
    """

    # Which of the 3 pipeline stages to run.
    stages: List[int] = field(default_factory=lambda: [1, 2, 3])

    # RNG seed. None -> a fresh random seed is drawn each run and recorded in the
    # resolved config + Aim hparams, so any run (even a lucky one) is reproducible
    # by re-running with seed=<that value>.  Set an int to fix it directly, e.g.
    # for a seed sweep: --multirun seed=0,1,2,3,4.
    seed: Optional[int] = None

    # Paths (absolute, anchored to the repo layout — cwd-independent)
    data_path: str = str(DATA_DIR / "xexe_urqmd_5fm.parquet")
    csv_path: str = str(DATA_DIR / "existing_nuclei_amc_5fm.csv")
    device: str = "auto"          # "auto" -> mps / cuda / cpu (resolved at runtime)

    # Data
    n_events: int = 500
    particle_type: str = "SpectatorsLeft"

    # Stage 1 — Stability lookup table (no training; diagnostic plot only)
    sc_a_max: int = 30            # upper A for the stability-map diagnostic grid

    # Stage 2 — SplitPredictionModel (up-to-K-way)
    n_clusters: int = 5           # K: max fragments produced per split
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

    # Reward definition for the per-split-node energy U (reward q = (U_parent −
    # ΣU_child)/N).  "qmd_asym" = QMD potential + asymmetry penalty (the main
    # branch's reward); "weizsacker_qmd" = qmd_weight·V − W (maximize Weizsäcker
    # binding W, minimize QMD potential V).  qmd_weight is λ on the QMD term.
    # energy_scale (weizsacker_qmd only): "extensive" (total B + total pairwise
    # QMD, additive — resists over-splitting) or "per_nucleon" (B/A + mean
    # pairwise, the affinity scale that tends to over-split).
    reward_type: str = "weizsacker_qmd"
    qmd_weight: float = 1.0
    energy_scale: str = "extensive"

    # Actor-Critic: DeepSets V(s) replaces the scalar EMA baseline.
    # use_critic=false is a pure config flip to raw REINFORCE (no code fork).
    use_critic: bool = True
    critic_hidden_dim: int = 64
    value_coef: float = 0.5

    # MST supervised warm-start (runs before REINFORCE; 0 disables)
    pretrain_epochs: int = 25
    pretrain_lr: float = 1e-3
    mst_d_cut: float = 2.0        # fm; >2.5 percolates -> collapses to "never split"

    # Stage 3 — FragmentsIdentifier visualization
    # depth 8 is enough to peel every nucleon free if the model wants to
    # (2^8 = 256 > the largest event), so the depth limit never binds
    max_depth: int = 8
    n_vis: int = 200
    force_split: bool = False     # True bypasses the stability lookup

    type_index: int = 7
    input_dim: int = 8

    # Checkpoint reuse: load split-model weights from a previous run instead of
    # retraining, so Stage 3 can run standalone.  Accepts a run dir (looks for
    # dm_model.pt inside) or a direct .pt path.  None -> use this run's dm_model.pt.
    load_split_from: Optional[str] = None


# ─── Runtime context ──────────────────────────────────────────────────────────

class Ctx:
    """Ties a resolved config to its per-run output dir and Aim run.

    Everything that writes to disk or tracks a metric goes through here, so no
    function ever hard-codes an output path.  ``cfg`` stays a plain ``DictConfig``
    for hyperparameter reads (``ctx.cfg.n_clusters`` etc.).
    """

    def __init__(self, cfg: DictConfig, out_dir: Path, run: "aim.Run | None") -> None:
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.run = run

    @property
    def dm_model_path(self) -> Path:
        return self.out_dir / "dm_model.pt"

    @property
    def critic_model_path(self) -> Path:
        return self.out_dir / "critic_model.pt"

    @property
    def fig_dir(self) -> Path:
        d = self.out_dir / "figures"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def track(self, value, name: str, step: "int | None" = None, context: "dict | None" = None) -> None:
        if self.run is not None:
            self.run.track(value, name=name, step=step, context=context or {})

    def save_fig(self, fig: plt.Figure, name: str) -> None:
        fig.tight_layout()
        path = self.fig_dir / name
        fig.savefig(path, dpi=120)
        # Logging figures to Aim is nice-to-have; never let it break a run.
        if self.run is not None:
            try:
                self.run.track(aim.Image(fig), name=Path(name).stem,
                               context={"stage": "figures"})
            except Exception as exc:  # pragma: no cover - best-effort
                print(f"[aim] skipped image {name}: {exc}")
        plt.close(fig)
        print(f"Saved {path}")


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_dataset(exp: Ctx) -> NucleonDataset:
    cfg = exp.cfg
    ds = NucleonDataset(cfg.data_path, particle_type=cfg.particle_type, n_events=cfg.n_events)
    sizes = [ds[i].shape[0] for i in range(len(ds))]
    print(f"Events         : {len(ds)}")
    print(f"Nucleons/event : min={min(sizes)}, mean={np.mean(sizes):.1f}, max={max(sizes)}")
    return ds


# ─── Stage 1 — Stability lookup table ──────────────────────────────────────────

def build_stability_lookup(exp: Ctx) -> StabilityLookup:
    cfg = exp.cfg
    print("\n=== Stage 1 — Stability lookup table ===")
    lut = StabilityLookup(cfg.csv_path)
    print(f"Loaded {len(lut)} known nuclei from {cfg.csv_path}")
    _plot_stability_table(exp, lut)
    return lut


def _plot_stability_table(exp: Ctx, lut: StabilityLookup) -> None:
    """Map of the lookup over the (A, Z) plane: green = stable (in table)."""
    import pandas as pd

    cfg = exp.cfg
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
    exp.save_fig(fig, "stability_table.png")


# ─── Stage 2 — Up-to-K-way split model ─────────────────────────────────────────

def train_split_model(exp: Ctx, dataset: NucleonDataset) -> SplitPredictionModel:
    cfg = exp.cfg
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
        reward_type=cfg.reward_type,
        qmd_weight=cfg.qmd_weight,
        energy_scale=cfg.energy_scale,
    )
    history = trainer.train(n_epochs=cfg.split_epochs)
    torch.save(model.state_dict(), exp.dm_model_path)
    print(f"Saved {exp.dm_model_path}")
    if critic is not None:
        torch.save(critic.state_dict(), exp.critic_model_path)
        print(f"Saved {exp.critic_model_path}")

    _track_split_history(exp, history)
    _plot_split_history(exp, history)
    return model


def _track_split_history(exp: Ctx, history: Dict[str, List[float]]) -> None:
    """Log Stage-2 per-epoch curves to Aim (step = epoch, context stage=split)."""
    use_critic = exp.cfg.use_critic
    n = len(history["loss"])
    for i in range(n):
        epoch = i + 1
        ctx = {"stage": "split"}
        exp.track(history["reward"][i], name="reward", step=epoch, context=ctx)
        exp.track(history["eval_reward"][i], name="eval_reward", step=epoch, context=ctx)
        exp.track(history["baseline"][i], name="baseline", step=epoch, context=ctx)
        exp.track(history["loss"][i], name="loss", step=epoch, context=ctx)
        if use_critic:
            exp.track(history["value_loss"][i], name="value_loss", step=epoch, context=ctx)


def _plot_split_history(exp: Ctx, history: Dict[str, List[float]]) -> None:
    cfg = exp.cfg
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
    exp.save_fig(fig, "split_history.png")


# ─── Stage 3 — Fragment identification & visualization ─────────────────────────

def _resolve_split_ckpt(exp: Ctx) -> Path:
    """Path to the split-model weights Stage 3 should load.

    ``load_split_from`` (a previous run dir or a direct .pt) takes precedence, so
    Stage 3 can run without retraining; otherwise this run's ``dm_model.pt``.
    """
    src = exp.cfg.load_split_from
    if src:
        p = Path(src)
        ckpt = p / "dm_model.pt" if p.is_dir() else p
    else:
        ckpt = exp.dm_model_path
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Split-model checkpoint not found: {ckpt}. Run Stage 2 first, or set "
            f"load_split_from=<previous run dir> to reuse an existing dm_model.pt."
        )
    return ckpt


def build_identifier(exp: Ctx) -> FragmentsIdentifier:
    cfg = exp.cfg
    ckpt = _resolve_split_ckpt(exp)
    fi = FragmentsIdentifier(
        max_depth=cfg.max_depth,
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        n_iters=cfg.n_iters,
        n_clusters=cfg.n_clusters,
        csv_path=cfg.csv_path,
        force_split=cfg.force_split,
    )
    print(f"Loading split model from {ckpt}")
    fi.split_prediction_module.load_state_dict(torch.load(ckpt))
    fi.to(cfg.device)
    fi.eval()
    return fi


def visualize_fragments(exp: Ctx) -> None:
    cfg = exp.cfg
    print("\n=== Stage 3 — Fragment identification & visualization ===")
    fi = build_identifier(exp)

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

    n_events_seen = cfg.n_vis * len(vis_datasets)
    n_frag_total = len(nuclei_A_arr)
    n_a1 = int((nuclei_A_arr == 1).sum())
    n_a2 = int((nuclei_A_arr >= 2).sum())
    max_level = int(split_depths_arr.max()) if len(split_depths_arr) else 0

    print(f"Events           : {n_events_seen}")
    print(f"Fragments total  : {n_frag_total}  (A=1: {n_a1}, A>=2: {n_a2})")
    print(f"Fragments/event  : min={n_frags_arr.min()}, "
          f"mean={n_frags_arr.mean():.1f}, max={n_frags_arr.max()}")
    print(f"Splits total     : {len(split_depths_arr)}  (max tree level: {max_level})")

    # Stage-3 summary scalars -> Aim (single values, context stage=fragments).
    frag_ctx = {"stage": "fragments"}
    exp.track(n_events_seen, name="events", context=frag_ctx)
    exp.track(n_frag_total, name="fragments_total", context=frag_ctx)
    exp.track(n_a1, name="fragments_A1", context=frag_ctx)
    exp.track(n_a2, name="fragments_A2plus", context=frag_ctx)
    exp.track(float(n_frags_arr.mean()), name="fragments_per_event_mean", context=frag_ctx)
    exp.track(int(n_frags_arr.min()), name="fragments_per_event_min", context=frag_ctx)
    exp.track(int(n_frags_arr.max()), name="fragments_per_event_max", context=frag_ctx)
    exp.track(len(split_depths_arr), name="splits_total", context=frag_ctx)
    exp.track(max_level, name="max_tree_level", context=frag_ctx)
    exp.track(float(np.nanmean(ev_max_z)), name="max_charge_mean", context=frag_ctx)

    _plot_n_fragments(exp, n_frags_arr)
    _plot_splits_per_level(exp, split_depths_arr)
    _plot_eta(exp, np.array(nucleon_eta), np.array(nuclei_eta))
    _plot_az_nz(exp, nuclei_A_arr, nuclei_Z_arr, nuclei_N_arr)

    # Event-level diagnostics
    _plot_conservation(exp, np.array(cons_dA), np.array(cons_dZ),
                       np.array(cons_dpx), np.array(cons_dpy),
                       np.array(cons_dpz), np.array(cons_dE))
    _plot_mean_pt(exp, np.array(ev_nucleon_pt), np.array(ev_frag_pt_pn))
    _plot_max_charge(exp, np.array(ev_max_z))
    _plot_n_nucleons(exp, np.array(ev_n_nucleons))
    _plot_pn_ratio(exp, np.array(ev_pn_ratio))


def _plot_splits_per_level(exp: Ctx, split_depths: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(split_depths):
        levels = np.arange(0, split_depths.max() + 2)
        ax.hist(split_depths, bins=levels - 0.5, color="slateblue",
                edgecolor="black", lw=0.5)
        ax.set_xticks(np.arange(0, split_depths.max() + 1))
    ax.set(xlabel="Tree level (split depth)", ylabel="Number of splits",
           title="Splits per tree level")
    exp.save_fig(fig, "splits_per_level.png")


def _plot_n_fragments(exp: Ctx, n_frags: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(n_frags.min(), n_frags.max() + 2) - 0.5
    ax.hist(n_frags, bins=bins, color="mediumseagreen", edgecolor="black", lw=0.5)
    ax.axvline(n_frags.mean(), color="crimson", linestyle="--", lw=1.5,
               label=f"mean = {n_frags.mean():.1f}")
    ax.set(xlabel="Fragments per event", ylabel="Events",
           title="Fragment multiplicity")
    ax.legend()
    exp.save_fig(fig, "fragment_multiplicity.png")


def _plot_eta(exp: Ctx, nucleon_eta: np.ndarray, nuclei_eta: np.ndarray) -> None:
    fig, (ax_nu, ax_nuc) = plt.subplots(1, 2, figsize=(14, 5))
    ax_nu.hist(nucleon_eta, bins=100, color="steelblue")
    ax_nu.set(xlabel="Pseudorapidity η", ylabel="Count", title="Nucleons (pre-split)")
    ax_nuc.hist(nuclei_eta, bins=100, color="darkorange")
    ax_nuc.set(xlabel="Pseudorapidity η", ylabel="Count", title="Identified fragments (all A)")
    exp.save_fig(fig, "fragment_eta.png")


def _plot_az_nz(exp: Ctx, A: np.ndarray, Z: np.ndarray, N: np.ndarray) -> None:
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
    exp.save_fig(fig, "fragment_az_nz.png")


# ─── Event-level diagnostics ────────────────────────────────────────────────────

def _plot_conservation(exp: Ctx, dA: np.ndarray, dZ: np.ndarray,
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
    exp.save_fig(fig, "conservation.png")


def _plot_mean_pt(exp: Ctx, nucleon_pt: np.ndarray, frag_pt_pn: np.ndarray) -> None:
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
    exp.save_fig(fig, "mean_pt.png")


def _plot_max_charge(exp: Ctx, max_z: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0, max_z.max() + 2) - 0.5
    ax.hist(max_z, bins=bins, color="indianred", edgecolor="black", lw=0.5)
    ax.axvline(max_z.mean(), color="navy", ls="--", lw=1.5,
               label=f"mean = {max_z.mean():.1f}")
    ax.set(xlabel="Maximum fragment charge Z", ylabel="Events",
           title="Maximum charge per event")
    ax.legend()
    exp.save_fig(fig, "max_charge.png")


def _plot_n_nucleons(exp: Ctx, n_nuc: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(n_nuc.min(), n_nuc.max() + 2) - 0.5
    ax.hist(n_nuc, bins=bins, color="mediumpurple", edgecolor="black", lw=0.5)
    ax.axvline(n_nuc.mean(), color="crimson", ls="--", lw=1.5,
               label=f"mean = {n_nuc.mean():.1f}")
    ax.set(xlabel="Nucleons per event", ylabel="Events",
           title="Number of nucleons")
    ax.legend()
    exp.save_fig(fig, "n_nucleons.png")


def _plot_pn_ratio(exp: Ctx, pn_ratio: np.ndarray) -> None:
    pn_ratio = pn_ratio[np.isfinite(pn_ratio)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pn_ratio, bins=40, color="seagreen", edgecolor="black", lw=0.3)
    ax.axvline(pn_ratio.mean(), color="crimson", ls="--", lw=1.5,
               label=f"mean = {pn_ratio.mean():.2f}")
    ax.set(xlabel="Z / N  (protons / neutrons)", ylabel="Events",
           title="Proton-to-neutron ratio")
    ax.legend()
    exp.save_fig(fig, "pn_ratio.png")


# ─── Reproducibility ───────────────────────────────────────────────────────────

def _git_info() -> Dict[str, object]:
    """Current commit SHA + dirty-tree flag, for reproducibility.  Warns if dirty."""
    def _git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args], stderr=subprocess.DEVNULL
        ).decode().strip()

    try:
        sha = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain"))
    except Exception:
        print("[git] not a git repo or git unavailable; SHA not recorded.")
        return {"sha": None, "dirty": None}

    if dirty:
        print("[git] WARNING: working tree is dirty — this run is not reproducible "
              "from the recorded SHA alone.")
    return {"sha": sha, "dirty": dirty}


# ─── Entry point ───────────────────────────────────────────────────────────────

def run(exp: Ctx) -> None:
    cfg = exp.cfg
    stages = list(cfg.stages)
    print(f"Device: {cfg.device}")
    print(f"Output dir: {exp.out_dir}")
    if 1 in stages:
        build_stability_lookup(exp)
    if 2 in stages:
        dataset = load_dataset(exp)
        train_split_model(exp, dataset)
    if 3 in stages:
        visualize_fragments(exp)


# Register the structured schema at import time so it is available whether the
# entrypoint is reached via the `clustering-run` console script, `python -m
# clustering.experiment`, or direct execution.
ConfigStore.instance().store(name="base_config", node=ClusteringConfig)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Resolve device before anything else so it lands in both the dumped config
    # and the Aim hparams.
    if cfg.device == "auto":
        cfg.device = _default_device()

    # Draw-and-record the seed (if unset) before dumping the config, so the run is
    # reproducible from its logged seed.  Seed the RNGs before any model/data use.
    if cfg.seed is None:
        cfg.seed = random.randrange(2**31 - 1)
    _seed_everything(cfg.seed)
    print(f"Seed: {cfg.seed}")

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    git = _git_info()

    # Dump the fully-resolved config (+ git provenance) into THIS run's dir.
    resolved = OmegaConf.to_container(cfg, resolve=True)
    dump = OmegaConf.create({"config": resolved, "git": git})
    (out_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(dump))

    # One Aim run per invocation, pinned to the single project-root store so that
    # Hydra's per-run cwd/output dir can't scatter runs into separate .aim stores
    # (which would silently break cross-run comparison — the whole point).
    aim_run = aim.Run(repo=AIM_REPO, experiment="clustering")
    aim_run["hparams"] = resolved
    aim_run["git"] = git
    aim_run["output_dir"] = str(out_dir)
    if git["sha"]:
        aim_run.add_tag(f"sha:{git['sha'][:8]}")
    if git["dirty"]:
        aim_run.add_tag("dirty")
    aim_run.add_tag(f"particle:{cfg.particle_type}")
    aim_run.add_tag(f"critic:{'on' if cfg.use_critic else 'off'}")
    aim_run.add_tag(f"reward:{cfg.reward_type}")
    if cfg.reward_type == "weizsacker_qmd":
        aim_run.add_tag(f"scale:{cfg.energy_scale}")

    exp = Ctx(cfg, out_dir, aim_run)
    try:
        run(exp)
    finally:
        aim_run.close()


if __name__ == "__main__":
    main()
