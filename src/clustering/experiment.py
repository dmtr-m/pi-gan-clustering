"""Hierarchical clustering experiment: up-to-K-way nucleon fragmentation.

Partitions a nucleon cloud into nuclear fragments with a slot-attention split
model trained by REINFORCE.  Each split assigns nucleons to one of ``n_clusters``
(K) slots, so a single pass yields up to K fragments (K=2 is the binary case).

Pipeline
--------
  Stage 1  build the stability lookup table from the known-nuclei CSV (diagnostic)
  Stage 2  train the up-to-K-way SplitPredictionModel (REINFORCE) -> dm_model.pt
  Stage 3  run FragmentsIdentifier and plot fragment distributions
  Stage 4  classical MST + stability-decay baseline, same plots (no training)

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
  clustering-run stages=[4]                   # classical baseline only, no checkpoint
  clustering-run --multirun split_lr=1e-4,3e-4,1e-3             # sweep, one Aim run each
"""
from __future__ import annotations

import dataclasses
import math
import random
import time
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless: save figures instead of showing them
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.patches import Patch
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import aim
import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from clustering.baselines.mst_decay import BaselineResult, DecayStep, MSTDecayBaseline
from clustering.baselines.coalescence import CoalescenceBaseline
from clustering.baselines.saca import SACABaseline, SacaParams
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


def _build_scheduler(optimizer: optim.Optimizer, cfg: DictConfig):
    """LambdaLR: optional linear warmup, then constant or cosine decay.

    Stepped once per epoch by the trainer.  Returns a factor on ``split_lr``:
    warmup ramps 0 -> 1 over ``lr_warmup_epochs``; afterwards it is 1 (constant)
    or a cosine from 1 down to ``lr_min_factor`` over the remaining epochs.
    ``constant`` with no warmup reproduces a fixed LR (the previous behavior).
    """
    if cfg.lr_schedule not in ("constant", "cosine"):
        raise ValueError(
            f"unknown lr_schedule {cfg.lr_schedule!r}; expected 'constant' or 'cosine'"
        )
    warmup = int(cfg.lr_warmup_epochs)
    total = int(cfg.split_epochs)
    lo = float(cfg.lr_min_factor)

    def lr_lambda(epoch: int) -> float:  # epoch = scheduler.last_epoch (0-indexed)
        if warmup > 0 and epoch < warmup:
            return (epoch + 1) / warmup
        if cfg.lr_schedule == "cosine":
            progress = (epoch - warmup) / max(1, total - warmup)
            progress = min(max(progress, 0.0), 1.0)
            return lo + (1.0 - lo) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@dataclass
class ClusteringConfig:
    """Structured (typed) Hydra schema — mirrors the old ``Config`` dataclass.

    Every field is overridable on the CLI.  Derived paths (checkpoints, figures)
    are NOT config fields: they live under the per-run Hydra output dir and are
    computed at runtime by ``Ctx`` — see ``main``.
    """

    # Which pipeline stages to run (4 = classical baseline, off by default).
    stages: List[int] = field(default_factory=lambda: [1, 2, 3])

    # RNG seed. None -> a fresh random seed is drawn each run and recorded in the
    # resolved config + Aim hparams, so any run (even a lucky one) is reproducible
    # by re-running with seed=<that value>.  Set an int to fix it directly, e.g.
    # for a seed sweep: --multirun seed=0,1,2,3,4.
    seed: Optional[int] = None

    # Paths (absolute, anchored to the repo layout — cwd-independent)
    data_path: str = str(DATA_DIR / "xecs_hse.parquet")
    csv_path: str = str(DATA_DIR / "existing_nuclei_XeCs_HSE_Hybrid_mcini.csv")
    device: str = "auto"          # "auto" -> mps / cuda / cpu (resolved at runtime)

    # Data
    # HSE (Xe+Cs) is far less homogeneous than the UrQMD Xe+Xe set it replaced:
    # 56% of nucleons are spectators (vs 18%) and a SpectatorsLeft event carries
    # ~75 nucleons on average (vs ~23), spanning 2..132.  500 events no longer
    # samples that spread, hence 2000.  Cost scales ~linearly with this.
    n_events: int = 2000
    particle_type: str = "SpectatorsLeft"

    # Stage 1 — Stability lookup table (no training; diagnostic plot only)
    # None -> size the grid to the nuclei table itself, plus a margin of
    # max(10%, 5 nucleons) so the map shows some unstable space beyond the
    # heaviest known nucleus.  Any fixed value crops silently when the dataset
    # changes: 30 suited amc_5fm (A<=18) but hid most of HSE (A<=128), and 130
    # would in turn crop Au (A=197) or Pb (A=208).  Set an int to zoom in.
    sc_a_max: Optional[int] = None  # upper A for the stability-map diagnostic grid

    # Stage 2 — SplitPredictionModel (up-to-K-way)
    n_clusters: int = 5           # K: max fragments produced per split
    hidden_dim: int = 32
    n_iters: int = 3
    split_epochs: int = 40
    # 128 on HSE: one optim.step() runs per batch, so a wider batch cuts the
    # per-step Python/kernel-launch overhead that dominates this small model.
    split_batch_size: int = 128
    split_lr: float = 3e-4
    # Tree depth for training.  Every split node is rewarded and backwarded
    # independently (no autograd graph is held across levels), and empty /
    # too-small nodes are pruned, so depth is cheap: K=5,k=8 visits ~19 live
    # nodes per batch instead of the 97k an unpruned BFS would enumerate.
    split_k: int = 8
    baseline_momentum: float = 0.95
    # Max total gradient norm; actor and critic are clipped separately to this.
    # Track it against the grad_norm plot: the actor norm has been ~40x this, so
    # updates are clip-limited — raise to let more gradient through, or lower to
    # tighten. Sweep it, e.g. --multirun grad_clip=0.5,1,5,20.
    grad_clip: float = 1.0
    # Learning-rate schedule (Stage 2), stepped once per epoch. LR is the real
    # step-size lever under AdamW (grad_clip is scale-invariant), so this is the
    # main stability knob to sweep.
    #   lr_schedule: "constant" (default, LR = split_lr throughout) or "cosine"
    #                (decay from split_lr to lr_min_factor*split_lr over training).
    #   lr_warmup_epochs: linearly ramp LR 0 -> split_lr over the first N epochs.
    #   lr_min_factor: cosine floor as a fraction of split_lr.
    lr_schedule: str = "constant"
    lr_warmup_epochs: int = 0
    lr_min_factor: float = 0.0

    # Reward definition for the per-split-node energy U (reward q = (U_parent −
    # ΣU_child)/N).  "qmd_asym" = QMD potential + asymmetry penalty (the main
    # branch's reward); "weizsacker_qmd" = qmd_weight·V − W (maximize Weizsäcker
    # binding W, minimize QMD potential V).  qmd_weight is λ on the QMD term.
    # energy_scale (weizsacker_qmd only): "extensive" (total B + total pairwise
    # QMD, additive — resists over-splitting) or "per_nucleon" (B/A + mean
    # pairwise, the affinity scale that tends to over-split).
    # "qmd_minus_b" = V − bwm_weight·B, the SACA annealer's QMD − B objective
    # (experiments/qmd_minus_b.py) on the divisive policy.  bwm_form picks the
    # mass formula: "bwm" is SACA 2.1's Samanta-Adhikari form, "bw" is the plain
    # Weizsäcker the weizsacker_qmd reward uses.  They differ only below A ~ 9,
    # and that is exactly where this model's fragments are — BW calls the
    # deuteron unbound by 17.5 MeV, BWM puts it at +1.89 (2.22 measured), while
    # BWM calls the non-existent H-4 bound.  See qmd_minus_b_energy.
    # On the annealer, λ = bwm_weight is a collapse knob: 0.25 → 1.5 took
    # multiplicity 3.73 → 2.20 per collision.  Sweep it, e.g.
    # --multirun bwm_weight=0.25,0.5,1.0,1.5.
    # "saca_qmd_minus_b" = the same objective on the *baselines'* QMD energy
    # (qmd_energy.cluster_energy) instead of physics.py's potential, which has no
    # density-dependent Skyrme term and does not saturate.  Measured on MST
    # fragments the two differ by ~21 MeV/nucleon, and under physics.py's V the
    # best classical clusterizer (MSTp) scores within 10% of a random partition;
    # under the baselines' energy the gap is 53x.  See saca_qmd_minus_b_energy
    # and experiments/reward_probe.py.
    reward_type: str = "weizsacker_qmd"
    qmd_weight: float = 1.0
    energy_scale: str = "extensive"
    bwm_weight: float = 1.0
    bwm_form: str = "bwm"
    # "zeta_correct" = the SACA paper's full zeta (kinetic + Skyrme 2/3-body +
    # Yukawa + Coulomb + Pauli) + bwm_weight·(−B_BWM); bwm_form is ignored (BWM).
    # zeta_spin_factor scales the Pauli term (no spin in the data: 1.0 = all
    # same-isospin pairs same-spin, 0.5 = random-spin average); zeta_yukawa is
    # "folded" | "point" | "off".  See baselines/qmd_full.py and FORMULAS.md 5a.
    zeta_spin_factor: float = 0.5
    # How the energy becomes a reward.  "node_diff" (historical): per split node,
    # q = (U(node) − ΣU(child))/N.  "final_sum": one terminal return per event,
    # R = −Σ_leaves U(leaf)/N, shared by every split in its tree (unsplit events
    # are scored, not masked).  Ranks partitions identically within an event; what
    # changes is the credit assignment.  eval_reward is on a different scale from
    # node_diff runs — do not compare them.
    reward_mode: str = "node_diff"
    zeta_yukawa: str = "folded"

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

    # Stage 4 — classical baseline: MST cluster recognition + decay onto the
    # nuclei table.  Nothing is trained and no checkpoint is read, so
    # `stages=[4]` runs standalone and fixes the scale for every Stage-3
    # observable.  Uses n_vis events per spectator side, like Stage 3.
    #   baseline_metric: "coord" (|Δr| < d_cut), "mstp" (that and |Δp| < p_cut),
    #     or "momentum" (|Δp| alone — coordinates ignored; percolates at a much
    #     smaller p_cut than mstp, so do not carry the 250 MeV/c value over).
    #   baseline_p_frame: "pair_cm" (boost to the pair rest frame) or "lab".
    #   baseline_decay: False emits the primary fragments untouched, which is the
    #     ablation that isolates what the decay stage contributes.
    #   baseline_emit_rule: which nucleon of the chosen species evaporates —
    #     "hottest" (max kinetic energy in the fragment rest frame), "outermost",
    #     or "first".
    baseline_d_cut: float = 2.0
    baseline_metric: str = "mstp"
    baseline_p_cut: float = 250.0
    baseline_p_frame: str = "pair_cm"
    baseline_decay: bool = True
    baseline_emit_rule: str = "hottest"
    #   baseline_trace: record every evaporation step to baseline_decay_trace.csv
    #     — one row per emitted nucleon, carrying the parent (A, Z), its table
    #     verdict, both candidate daughters, and why the channel was chosen.
    baseline_trace: bool = False
    # baseline_algo: "mst" = MST/MSTp fragments straight to the decay stage;
    # "saca" = MST seeds rearranged by simulated annealing first (SACA_PIPELINE.md).
    # The SACA knobs are its annealing schedule; e_cut is the binding-energy per
    # nucleon below which a fragment counts as bound.
    baseline_algo: str = "mst"     # "mst" | "saca" | "coalescence"
    # Coalescence (Kireyeu arXiv:2512.02084 Table 3).  coal_cut_set picks the
    # per-species (dr, dp) box; coal_selection is "none" (no formation
    # probability — the paper's factors are not published) or "ebind", the
    # paper's "mixed" procedure, which keeps only bound candidates.
    coal_cut_set: str = "M1"
    coal_selection: str = "none"
    # Wall-clock budget for the Stage 3/4 event loop, in minutes (0 = no limit).
    # A SACA run is minutes-to-hours depending on n_vis, and an over-running job
    # is indistinguishable from a hung one; this stops cleanly on the event
    # boundary and reports on what it actually processed.
    report_time_budget_min: float = 40.0
    saca_t_max: float = 20.0
    saca_t_min: float = 0.5
    saca_alpha: float = 0.9
    saca_trials_per_nucleon: int = 4
    saca_e_cut: float = -4.0        # MeV/nucleon, N_f >= 3
    saca_e_cut_light: float = 0.0   # MeV/nucleon, N_f < 3 (Puri & Aichelin)
    saca_p_release: float = 0.3
    saca_two_pass: bool = True
    saca_asymmetry: bool = False   # FRIGA's B_asy term on top of SACA's energy
    saca_e_0_asy: float = 23.3     # MeV
    saca_gamma_asy: float = 1.0

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
        # Prefix for every figure written while it is set, so Stage 3 and the
        # Stage 4 baseline can reuse the same _plot_* functions without
        # overwriting each other's files.
        self.fig_prefix = ""

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
        name = f"{self.fig_prefix}{name}"
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
    a_max = cfg.sc_a_max
    if a_max is None:
        table_max = int(df_stable["A"].max())
        a_max = int(np.ceil(table_max + max(0.10 * table_max, 5.0)))
    # One cell per (A, Z) rather than a scatter point.  At HSE scale the grid is
    # ~10k points, and overlapping s=30 markers merged into a solid block; the
    # s=60 "table entries" overlay then covered the ~9% of cells that are green
    # with larger black circles.  That overlay was also pure redundancy —
    # is_stable() is true for exactly the table's (A, Z) pairs — so it hid the
    # only signal in the figure while adding nothing.  Cells make each nucleus
    # one pixel, which stays legible however heavy the table gets.
    img = np.full((a_max + 1, a_max + 1), np.nan)   # NaN => unphysical Z > A
    for A in range(2, a_max + 1):
        for Z in range(0, A + 1):
            img[Z, A] = 1.0 if lut.is_stable(A, Z) else 0.0

    # Two flat colours read better than a continuous map for a binary field:
    # pale grey for "known unstable", saturated green for "in the table".
    cmap = ListedColormap(["#e8e8e8", "#1a9850"])
    cmap.set_bad("white")                            # Z > A stays blank

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=1,
              interpolation="nearest", aspect="auto")
    # Crop the empty Z > A wedge: the valley sits near Z ~ A/2, so plotting up
    # to a_max wastes half the axes on blank space.
    z_hi = int(np.ceil(df_stable["Z"].max() + max(0.10 * df_stable["Z"].max(), 5.0)))
    ax.set_ylim(0, z_hi)
    ax.set_xlim(0, a_max)
    handles = [Patch(facecolor="#1a9850", label=f"in table ({len(df_stable)} nuclei)"),
               Patch(facecolor="#e8e8e8", label="not in table")]
    ax.legend(handles=handles, loc="upper left")
    ax.set(xlabel="A", ylabel="Z", title="Stability lookup table")
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
        pretrainer = MSTPretrainer(
            model,
            loader,
            optimizer=optim.AdamW(model.parameters(), lr=cfg.pretrain_lr),
            device=cfg.device,
            d_cut=cfg.mst_d_cut,
        )
        mst_history = pretrainer.train(
            n_epochs=cfg.pretrain_epochs, log_every=max(1, cfg.pretrain_epochs // 5)
        )
        # Previously the returned history was discarded outright: the warm-start
        # had no Aim series and no figure, only the printed lines.
        _track_mst_history(exp, mst_history, pretrainer.step_history)
        _plot_mst_history(exp, mst_history, pretrainer.step_history,
                          pretrainer.epoch_end_steps)

    # Actor-Critic: the critic's parameters must share the optimizer with the actor.
    critic = (
        SplitValueCritic(input_dim=cfg.input_dim, hidden_dim=cfg.critic_hidden_dim)
        if cfg.use_critic
        else None
    )
    params = list(model.parameters()) + (list(critic.parameters()) if critic else [])
    print(f"--- REINFORCE ({'Actor-Critic' if critic else 'EMA baseline'}) ---")

    optimizer = optim.AdamW(params, lr=cfg.split_lr)
    scheduler = _build_scheduler(optimizer, cfg)
    trainer = KSplitTrainer(
        model,
        loader,
        optimizer=optimizer,
        scheduler=scheduler,
        k=cfg.split_k,
        critic=critic,
        value_coef=cfg.value_coef,
        baseline_momentum=cfg.baseline_momentum,
        grad_clip=cfg.grad_clip,
        type_index=cfg.type_index,
        device=cfg.device,
        reward_type=cfg.reward_type,
        qmd_weight=cfg.qmd_weight,
        energy_scale=cfg.energy_scale,
        bwm_weight=cfg.bwm_weight,
        bwm_form=cfg.bwm_form,
        zeta_spin_factor=cfg.zeta_spin_factor,
        zeta_yukawa=cfg.zeta_yukawa,
        reward_mode=cfg.reward_mode,
    )
    history = trainer.train(n_epochs=cfg.split_epochs)
    torch.save(model.state_dict(), exp.dm_model_path)
    print(f"Saved {exp.dm_model_path}")
    if critic is not None:
        torch.save(critic.state_dict(), exp.critic_model_path)
        print(f"Saved {exp.critic_model_path}")

    _track_split_history(exp, history)
    _track_split_steps(exp, trainer.step_history)
    _plot_split_history(exp, history, trainer.step_history, trainer.epoch_end_steps)
    _plot_grad_norm(exp, history, trainer.step_history)
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
        exp.track(history["grad_norm"][i], name="grad_norm", step=epoch, context=ctx)
        exp.track(history["lr"][i], name="lr", step=epoch, context=ctx)
        # Diagnostics for the falling-`reward` question: q_weighted is the
        # size-weighted twin of `reward`, and n_nodes/node_depth show whether the
        # node population is shifting underneath that unweighted mean.
        for name in ("q_weighted", "n_nodes", "node_depth", "valid_frac",
                     "entropy", "entropy_frac"):
            if name in history:
                exp.track(history[name][i], name=name, step=epoch, context=ctx)
        if use_critic:
            exp.track(history["value_loss"][i], name="value_loss", step=epoch, context=ctx)


def _track_mst_history(
    exp: Ctx, history: Dict[str, List[float]],
    step_history: Dict[str, List[float]] | None = None,
) -> None:
    """Log the MST warm-start to Aim, per epoch and per optimizer step."""
    ctx = {"stage": "pretrain"}
    for i in range(len(history.get("loss", []))):
        exp.track(history["loss"][i], name="mst_loss", step=i + 1, context=ctx)
        exp.track(history["pair_acc"][i], name="mst_pair_acc", step=i + 1, context=ctx)
    sh = step_history or {}
    ctx_s = {"stage": "pretrain", "per": "step"}
    for i, s in enumerate(sh.get("step", [])):
        for key, name in (("loss", "mst_loss"), ("pair_acc", "mst_pair_acc"),
                          ("grad_norm", "mst_grad_norm"),
                          ("entropy_frac", "mst_entropy_frac")):
            v = sh[key][i]
            if v == v:  # skip NaN (batch with no valid pair)
                exp.track(v, name=name, step=int(s), context=ctx_s)


def _plot_mst_history(
    exp: Ctx, history: Dict[str, List[float]],
    step_history: Dict[str, List[float]] | None = None,
    epoch_end_steps: List[int] | None = None,
) -> None:
    """MST warm-start curves on the optimizer-step axis."""
    sh = step_history or {}
    x = sh.get("step") or list(range(1, len(history.get("loss", [])) + 1))
    per_step = bool(sh.get("step"))
    if not x:
        return
    w = max(1, len(x) // 100)
    loss = sh.get("loss") or history["loss"]
    acc = sh.get("pair_acc") or history["pair_acc"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, loss, color="C0", alpha=0.25, lw=0.7)
    ax.plot(x, _rolling(loss, w), color="C0", lw=1.6, label="pair BCE loss")
    ax.set(xlabel="Optimizer step" if per_step else "Epoch", ylabel="Loss",
           title=f"MST warm-start (d_cut={exp.cfg.mst_d_cut} fm)")
    ax2 = ax.twinx()
    ax2.plot(x, acc, color="C2", alpha=0.25, lw=0.7)
    ax2.plot(x, _rolling(acc, w), color="C2", lw=1.6, label="pair accuracy")
    # Normalized slot entropy shares the 0–1 axis: 1.0 = uniform assignment.
    hf = sh.get("entropy_frac", [])
    if any(v == v for v in hf):
        ax2.plot(x, _rolling(hf, w), color="C4", lw=1.4, ls=":", label="slot entropy / log K")
    ax2.set_ylabel("Pair accuracy  /  normalized entropy")
    ax2.set_ylim(0, 1.05)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    exp.save_fig(fig, "mst_history.png")


def _track_split_steps(exp: Ctx, step_history: Dict[str, List[float]]) -> None:
    """Log Stage-2 per-optimizer-step curves to Aim.

    Same metric names as the epoch series but under ``context={"per": "step"}``, so
    Aim keeps them as separate sequences rather than overwriting one with the other.
    """
    ctx = {"stage": "split", "per": "step"}
    steps = step_history.get("step", [])
    names = [n for n in step_history if n != "step"]
    for i in range(len(steps)):
        s = int(steps[i])
        for name in names:
            v = step_history[name][i]
            if v == v:  # skip NaN (batch with no scorable node)
                exp.track(v, name=name, step=s, context=ctx)


def _rolling(y: List[float], w: int) -> np.ndarray:
    """NaN-aware rolling mean, same length as ``y``."""
    a = np.asarray(y, dtype=float)
    if w <= 1 or len(a) < 2:
        return a
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        seg = a[max(0, i - w + 1): i + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = seg.mean()
    return out


def _plot_split_history(
    exp: Ctx, history: Dict[str, List[float]],
    step_history: Dict[str, List[float]] | None = None,
    epoch_end_steps: List[int] | None = None,
) -> None:
    """Stage-2 curves on the optimizer-step axis.

    Per-epoch points are far too few to read once the dataset is large (3 epochs
    over 58k events = 3 points for 3.5 h of compute), so the step series is the
    primary trace.  eval_reward is a full-dataset pass and stays per epoch; it is
    drawn as markers anchored at the step where each epoch ended.
    """
    cfg = exp.cfg
    sh = step_history or {}
    x = sh.get("step") or list(range(1, len(history["reward"]) + 1))
    per_step = bool(sh.get("step"))
    xlabel = "Optimizer step" if per_step else "Epoch"
    # Raw per-step curves are noisy; show them faintly under a rolling mean.
    w = max(1, len(x) // 100)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4.5))
    for key, color, label in (("q_weighted", "C3", "q_weighted (size-weighted)"),
                              ("reward", "C0", "sampled reward (unweighted)")):
        y = sh.get(key) or history.get(key, [])
        if not y:
            continue
        ax1.plot(x, y, color=color, alpha=0.25, lw=0.7)
        ax1.plot(x, _rolling(y, w), color=color, lw=1.6, label=label)
    ex = epoch_end_steps if (per_step and epoch_end_steps) else list(range(1, len(history["eval_reward"]) + 1))
    ax1.plot(ex, history["eval_reward"], color="C2", marker="o", ms=4,
             lw=1.6, label="eval reward (argmax, per epoch)")
    ax1.set(xlabel=xlabel, ylabel="Q [MeV/nucleon]",
            title=f"KSplitTrainer (K={cfg.n_clusters}) — QMD reward")
    ax1.legend(fontsize=8)

    yl = sh.get("loss") or history["loss"]
    ax2.plot(x, yl, color="C0", alpha=0.25, lw=0.7)
    ax2.plot(x, _rolling(yl, w), color="C0", lw=1.6, label="policy loss")
    vl = sh.get("value_loss") or history.get("value_loss", [])
    if cfg.use_critic and any(v for v in vl if v == v):
        # On the z-scored target, value_loss ~= 1 means the critic is no better
        # than predicting the mean; -> 0 means it explains the reward.
        ax2.plot(x, _rolling(vl, w), color="C1", ls="--", lw=1.6, label="critic value loss")
        ax2.axhline(1.0, color="grey", lw=0.8, alpha=0.6)
    ax2.legend(fontsize=8)
    ax2.set(xlabel=xlabel, ylabel="Loss", title=f"KSplitTrainer (K={cfg.n_clusters}) — Losses")

    # Policy entropy.  At log(K) the policy is uniform — it has not committed to
    # any assignment and there is nothing to deploy; a sharp fall is how slot
    # collapse (one slot taking every nucleon) shows up while reward looks fine.
    he = sh.get("entropy") or history.get("entropy", [])
    if any(v == v for v in he):
        log_k = float(np.log(cfg.n_clusters))
        ax3.plot(x, he, color="C4", alpha=0.25, lw=0.7)
        ax3.plot(x, _rolling(he, w), color="C4", lw=1.6, label="policy entropy H")
        ax3.axhline(log_k, color="grey", ls="--", lw=1, label=f"uniform = log K = {log_k:.2f}")
        ax3.set_ylim(0, log_k * 1.08)
        ax3.legend(fontsize=8)
    ax3.set(xlabel=xlabel, ylabel="H [nats]",
            title=f"KSplitTrainer (K={cfg.n_clusters}) — Policy entropy")
    exp.save_fig(fig, "split_history.png")


def _plot_grad_norm(
    exp: Ctx, history: Dict[str, List[float]],
    step_history: Dict[str, List[float]] | None = None,
) -> None:
    """Actor gradient norm per optimizer step (measured before clipping).

    The configured clip threshold is drawn for reference: points above it were
    clipped.  Per-step resolution matters here — the epoch mean hides the spikes
    that clipping exists to suppress.
    """
    sh = step_history or {}
    gn = sh.get("grad_norm") or history.get("grad_norm", [])
    x = sh.get("step") or list(range(1, len(gn) + 1))
    per_step = bool(sh.get("grad_norm"))
    if not gn:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, gn, color="C3", alpha=0.3, lw=0.7)
    ax.plot(x, _rolling(gn, max(1, len(gn) // 100)), color="C3", lw=1.6,
            label="‖grad‖ (actor, pre-clip)")
    ax.axhline(exp.cfg.grad_clip, color="grey", ls="--", lw=1,
               label=f"clip threshold ({exp.cfg.grad_clip:g})")
    finite = [g for g in gn if g == g and g > 0]
    if finite and min(finite) > 0:
        ax.set_yscale("log")
    ax.set(xlabel="Optimizer step" if per_step else "Epoch", ylabel="Gradient norm",
           title="Actor gradient norm (before clipping)")
    ax.legend()
    exp.save_fig(fig, "grad_norm.png")


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
    print("\n=== Stage 3 — Fragment identification & visualization ===")
    fi = build_identifier(exp)
    with torch.no_grad():
        _fragment_report(exp, fi, stage="fragments")


def run_baseline(exp: Ctx) -> None:
    """Stage 4 — MST (+ stability decay) baseline over the same events as Stage 3."""
    cfg = exp.cfg
    print("\n=== Stage 4 — MST baseline"
          f"{' + stability decay' if cfg.baseline_decay else ' (primaries only)'} ===")
    lut = StabilityLookup(cfg.csv_path)
    cuts = []
    if cfg.baseline_metric in ("coord", "mstp"):
        cuts.append(f"d_cut={cfg.baseline_d_cut} fm")
    if cfg.baseline_metric in ("mstp", "momentum"):
        cuts.append(f"p_cut={cfg.baseline_p_cut} MeV/c ({cfg.baseline_p_frame})")
    print(f"algo={cfg.baseline_algo}, metric={cfg.baseline_metric}: "
          f"{', '.join(cuts)}; table: {len(lut)} nuclei")
    common = dict(
        d_cut=cfg.baseline_d_cut,
        p_cut=cfg.baseline_p_cut,
        metric=cfg.baseline_metric,
        p_frame=cfg.baseline_p_frame,
        decay=cfg.baseline_decay,
        emit_rule=cfg.baseline_emit_rule,
        type_index=cfg.type_index,
        trace=cfg.baseline_trace,
    )
    if cfg.baseline_algo == "saca":
        params = SacaParams(
            t_max=cfg.saca_t_max, t_min=cfg.saca_t_min, alpha=cfg.saca_alpha,
            trials_per_nucleon=cfg.saca_trials_per_nucleon, e_cut=cfg.saca_e_cut,
            e_cut_light=cfg.saca_e_cut_light,
            p_release=cfg.saca_p_release, two_pass=cfg.saca_two_pass,
            asymmetry=cfg.saca_asymmetry, e_0_asy=cfg.saca_e_0_asy,
            gamma_asy=cfg.saca_gamma_asy,
        )
        print(f"SACA: T {params.t_max} -> {params.t_min} MeV, alpha={params.alpha}, "
              f"{params.trials_per_nucleon} trials/nucleon, "
              f"e_cut={params.e_cut}/{params.e_cut_light} MeV/A (N_f>=3 / <3), "
              f"two_pass={params.two_pass}")
        baseline = SACABaseline(lut, params=params, seed=int(cfg.seed), **common)
    elif cfg.baseline_algo == "coalescence":
        print(f"Coalescence: cut set {cfg.coal_cut_set}, selection={cfg.coal_selection}")
        baseline = CoalescenceBaseline(lut, cut_set=cfg.coal_cut_set,
                                       selection=cfg.coal_selection,
                                       decay=cfg.baseline_decay,
                                       emit_rule=cfg.baseline_emit_rule,
                                       type_index=cfg.type_index,
                                       trace=cfg.baseline_trace)
    elif cfg.baseline_algo == "mst":
        baseline = MSTDecayBaseline(lut, **common)
    else:
        raise ValueError(f"unknown baseline_algo {cfg.baseline_algo!r}; "
                         f"expected 'mst', 'saca' or 'coalescence'")
    tally = {"n_primary": 0, "n_primary_in_table": 0, "n_evaporated": 0}
    steps: List[DecayStep] = []
    seen = [0]  # event counter, in the order _fragment_report walks them

    def fragment_fn(event: torch.Tensor) -> BaselineResult:
        # Both baselines are numpy/scipy-bound; keep them on the CPU regardless
        # of cfg.device, which only ever helped the neural path.  This is not
        # cosmetic for SACA: its inner loop is thousands of small tensor ops per
        # event, and on MPS the launch overhead and the sync on every .cpu()
        # make it 2.4x slower than the same code on CPU (0.80 vs 0.33 s/event).
        res = baseline(event.cpu())
        for k in tally:
            tally[k] += getattr(res, k)
        for st in res.steps:
            st.event = seen[0]
        steps.extend(res.steps)
        seen[0] += 1
        return res

    exp.fig_prefix = "baseline_"
    try:
        _fragment_report(exp, fragment_fn, stage="baseline")
    finally:
        exp.fig_prefix = ""

    if cfg.baseline_trace:
        _dump_decay_trace(exp, steps)

    frac_known = tally["n_primary_in_table"] / max(1, tally["n_primary"])
    print(f"Primary fragments: {tally['n_primary']}  "
          f"(known nuclei: {tally['n_primary_in_table']}, {frac_known:.1%})")
    print(f"Nucleons evaporated by the decay stage: {tally['n_evaporated']}")
    ctx = {"stage": "baseline"}
    if isinstance(baseline, SACABaseline):
        print(f"SACA energy: {baseline.e_initial:.0f} -> {baseline.e_final:.0f} MeV "
              f"(total over events); moves accepted {baseline.n_accepted}/{baseline.n_proposed}")
        exp.track(baseline.e_initial, name="saca_e_initial", context=ctx)
        exp.track(baseline.e_final, name="saca_e_final", context=ctx)
    exp.track(tally["n_primary"], name="primary_fragments", context=ctx)
    exp.track(tally["n_primary_in_table"], name="primary_fragments_in_table", context=ctx)
    exp.track(frac_known, name="primary_in_table_frac", context=ctx)
    exp.track(tally["n_evaporated"], name="evaporated_nucleons", context=ctx)


def _dump_decay_trace(exp: Ctx, steps: "List[DecayStep]") -> None:
    """Write one row per evaporation step, so the decay rule can be replayed.

    Every column is a quantity the rule actually consulted, in the order it
    consulted it — parent (A, Z), the table verdicts on both daughters, their
    Weizsacker binding, which tie-break decided the channel, and which nucleon
    left.  A row exists only for a fragment that was *not* in the table, which is
    the whole stopping rule: in the table -> emitted untouched, no row.
    """
    fields = [f.name for f in dataclasses.fields(DecayStep)]
    path = exp.out_dir / "baseline_decay_trace.csv"
    with path.open("w") as fh:
        fh.write(",".join(fields) + "\n")
        for st in steps:
            fh.write(",".join(
                f"{getattr(st, f):.4f}" if isinstance(getattr(st, f), float)
                else str(getattr(st, f)) for f in fields) + "\n")
    print(f"Saved {path}  ({len(steps)} decay steps)")


def _fragment_report(exp: Ctx, fragment_fn, *, stage: str) -> None:
    """Run a fragmenter over the visualization events and plot the result.

    ``fragment_fn(event) -> result`` where ``result`` carries ``.fragments``
    (list of ``(n_i, 8)`` tensors) and ``.split_depths``.  Both
    ``FragmentsIdentifier`` and ``MSTDecayBaseline`` satisfy that, so the learned
    and classical paths produce byte-for-byte comparable reports.
    """
    cfg = exp.cfg

    vis_datasets = {
        "SpectatorsLeft": NucleonDataset(cfg.data_path, particle_type="SpectatorsLeft"),
        "SpectatorsRight": NucleonDataset(cfg.data_path, particle_type="SpectatorsRight"),
    }

    nuclei_A: List[int] = []
    nuclei_Z: List[int] = []
    nuclei_eta: List[float] = []
    nucleon_eta: List[float] = []
    n_frags_per_event: List[int] = []
    n_frags_a2_per_event: List[int] = []   # free nucleons excluded — see below
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

    # Progress: Stage 4 with SACA runs for tens of minutes and used to print
    # nothing at all until the very end, so a slow run was indistinguishable
    # from a hung one.
    n_total = sum(min(cfg.n_vis, len(d)) for d in vis_datasets.values())
    t_start = time.time()
    n_done = 0
    budget_hit = False
    budget_s = float(cfg.report_time_budget_min) * 60.0
    report_every = max(1, n_total // 20)

    for ds in vis_datasets.values():
        if budget_hit:
            break
        for idx in range(min(cfg.n_vis, len(ds))):
            if budget_s and (time.time() - t_start) > budget_s:
                print(f"  [budget] stopped after {n_done}/{n_total} events "
                      f"({(time.time() - t_start)/60:.1f} min > "
                      f"{cfg.report_time_budget_min} min budget); "
                      f"everything below is normalized to the events actually "
                      f"processed.", flush=True)
                budget_hit = True
                break
            n_done += 1
            if n_done % report_every == 0:
                el = time.time() - t_start
                print(f"  [{n_done}/{n_total}]  {el/60:.1f} min elapsed, "
                      f"~{(n_total - n_done) * el / n_done / 60:.1f} min left", flush=True)
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

            result = fragment_fn(event)

            n_frags_per_event.append(len(result.fragments))
            n_frags_a2_per_event.append(
                sum(1 for f in result.fragments if f.shape[0] >= 2))
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
                # NaN rather than a skip: nuclei_eta has to stay index-aligned
                # with nuclei_A so the A >= 2 mask below selects the right rows.
                nuclei_eta.append(float(-np.arctanh(p_f[2] / pn)) if pn > 0 else np.nan)
                sum_a += A
                sum_z += Z
                p_out_sum += p_f
                e_out_sum += float(frag_np[:, 3].sum())
                max_z = max(max_z, Z)
                if A >= 2:
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
    n_frags_a2_arr = np.array(n_frags_a2_per_event)
    split_depths_arr = np.array(split_depths)

    # Free protons and neutrons are excluded from every *fragment* plot below.
    # They are ~75-90% of the fragment list and sit in a single (A, Z) cell, so
    # on a log-scaled map they set the colour range on their own and flatten
    # every real nucleus into the bottom decade.  They are still counted
    # everywhere their absence would be a lie: the printed totals, the Aim
    # scalars, fragment_yields.csv, and the conservation residuals — which must
    # see them or they would report a violation that did not happen.
    a2 = nuclei_A_arr >= 2
    nuclei_eta_arr = np.array(nuclei_eta)

    # The number of events actually walked — NOT cfg.n_vis * n_datasets, which
    # over-counts whenever a dataset is shorter than n_vis or the time budget cut
    # the loop short, and would silently deflate every per-event yield.
    n_events_seen = n_done
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
    frag_ctx = {"stage": stage}
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

    _dump_fragment_yields(exp, nuclei_A_arr, nuclei_Z_arr, n_events_seen)

    _plot_n_fragments(exp, n_frags_a2_arr)
    # The baseline performs no tree splits, so this plot would be an empty axes.
    if len(split_depths_arr):
        _plot_splits_per_level(exp, split_depths_arr)
    _plot_eta(exp, np.array(nucleon_eta), nuclei_eta_arr[a2])
    _plot_az_nz(exp, nuclei_A_arr[a2], nuclei_Z_arr[a2], nuclei_N_arr[a2])

    # Event-level diagnostics
    _plot_conservation(exp, np.array(cons_dA), np.array(cons_dZ),
                       np.array(cons_dpx), np.array(cons_dpy),
                       np.array(cons_dpz), np.array(cons_dE))
    _plot_mean_pt(exp, np.array(ev_nucleon_pt), np.array(ev_frag_pt_pn))
    _plot_max_charge(exp, np.array(ev_max_z))
    _plot_n_nucleons(exp, np.array(ev_n_nucleons))
    _plot_pn_ratio(exp, np.array(ev_pn_ratio))


def _dump_fragment_yields(exp: Ctx, A: np.ndarray, Z: np.ndarray,
                          n_events: int) -> None:
    """Write the (A, Z) yield table as CSV next to the figures.

    Comparing two log-scale heatmaps by eye cannot settle how far a produced
    distribution is from the generator's — a point `SESSION_SUMMARY.md` lists as
    still unmeasured.  ``yield_per_event`` is the same normalization the
    generator's own reference plot uses, so the two are directly subtractable
    once mcini exports its numbers.
    """
    pairs, counts = np.unique(np.stack([A, Z], axis=1), axis=0, return_counts=True)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    name = f"{exp.fig_prefix}fragment_yields.csv"
    path = exp.out_dir / name
    lines = ["A,Z,count,yield_per_event"]
    lines += [f"{pairs[i, 0]},{pairs[i, 1]},{counts[i]},{counts[i] / max(1, n_events):.6g}"
              for i in order]
    path.write_text("\n".join(lines) + "\n")
    print(f"Saved {path}  ({len(order)} species)")


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
    ax.set(xlabel="Fragments per event  (A ≥ 2)", ylabel="Events",
           title="Fragment multiplicity  (free nucleons excluded)")
    ax.legend()
    exp.save_fig(fig, "fragment_multiplicity.png")


def _plot_eta(exp: Ctx, nucleon_eta: np.ndarray, nuclei_eta: np.ndarray) -> None:
    # η = -arctanh(pz/|p|) diverges to ±inf for zero-transverse-momentum
    # fragments/nucleons (momentum along the beam, pz/|p| = ±1). Drop non-finite
    # values so the histogram range stays finite.
    nucleon_eta = nucleon_eta[np.isfinite(nucleon_eta)]
    nuclei_eta = nuclei_eta[np.isfinite(nuclei_eta)]
    fig, (ax_nu, ax_nuc) = plt.subplots(1, 2, figsize=(14, 5))
    if len(nucleon_eta):
        ax_nu.hist(nucleon_eta, bins=100, color="steelblue")
    ax_nu.set(xlabel="Pseudorapidity η", ylabel="Count", title="Nucleons (pre-split)")
    if len(nuclei_eta):
        ax_nuc.hist(nuclei_eta, bins=100, color="darkorange")
    ax_nuc.set(xlabel="Pseudorapidity η", ylabel="Count",
               title="Identified fragments (A ≥ 2)")
    exp.save_fig(fig, "fragment_eta.png")


def _plot_az_nz(exp: Ctx, A: np.ndarray, Z: np.ndarray, N: np.ndarray) -> None:
    fig, (ax_AZ, ax_NZ) = plt.subplots(1, 2, figsize=(12, 5))
    # Bins follow the data.  These were fixed at A<=20 / Z<=13, which comfortably
    # covered the amc_5fm table (A 2..18, Z 1..8) but silently truncates HSE,
    # whose fragments reach A~128 / Z~55.  hist2d drops out-of-range entries
    # without warning, so a heavy residue simply vanished from the figure and
    # the run looked like it was over-splitting far worse than it was.
    def _edges(v: np.ndarray) -> np.ndarray:
        hi = int(v.max()) if len(v) else 1
        return np.arange(0, hi + 2) - 0.5

    bins_A = _edges(A)
    bins_Z = _edges(Z)
    bins_N = _edges(N)
    h1 = ax_AZ.hist2d(A, Z, bins=[bins_A, bins_Z], cmap="viridis", norm=LogNorm())
    ax_AZ.set(xlabel="A", ylabel="Z", title="A vs Z  (A ≥ 2)")
    fig.colorbar(h1[3], ax=ax_AZ, label="Count")
    h2 = ax_NZ.hist2d(N, Z, bins=[bins_N, bins_Z], cmap="viridis", norm=LogNorm())
    ax_NZ.set(xlabel="N", ylabel="Z", title="N vs Z  (A ≥ 2)")
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
            label=f"fragments (A ≥ 2) / nucleon  (mean {frag_pt_pn.mean():.3f})")
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
    if 4 in stages:
        run_baseline(exp)


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
    aim_run.add_tag(f"mode:{cfg.reward_mode}")
    if cfg.reward_type == "weizsacker_qmd":
        aim_run.add_tag(f"scale:{cfg.energy_scale}")
    if cfg.reward_type in ("qmd_minus_b", "saca_qmd_minus_b"):
        aim_run.add_tag(f"binding:{cfg.bwm_form}")
        aim_run.add_tag(f"lambda:{cfg.bwm_weight:g}")
    if cfg.reward_type == "zeta_correct":
        aim_run.add_tag(f"lambda:{cfg.bwm_weight:g}")
        aim_run.add_tag(f"spin:{cfg.zeta_spin_factor:g}")
        aim_run.add_tag(f"yukawa:{cfg.zeta_yukawa}")

    exp = Ctx(cfg, out_dir, aim_run)
    try:
        run(exp)
    finally:
        aim_run.close()


if __name__ == "__main__":
    main()
