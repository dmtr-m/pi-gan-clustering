# PiGAN — nuclear multifragmentation clustering

A PyTorch research repo. The single experiment lives in the `clustering` package
(`src/clustering/`): a 3-stage pipeline (Stage 1 stability lookup, Stage 2
REINFORCE split training, Stage 3 fragment identification + visualization).
Experiments are configured with [Hydra](https://hydra.cc) and tracked with
[Aim](https://aimstack.io).

Versions are tracked with three orthogonal tools instead of copied directories:

| What varies | How |
|---|---|
| **Code** (architecture, loss, critic on/off) | a git branch / commit / tag — edit `src/clustering/` in place |
| **Hyperparameters** (K, lr, epochs, d_cut…) | a Hydra config override / preset — no code change |
| **Results & comparison** | an Aim run — metrics + the exact git SHA, compared in one UI |

Every experiment = *(a commit) × (a config) → (a tracked run)*.

## Layout

```
pi-gan/
  pyproject.toml               # editable install + `clustering-run` console script
  requirements.txt
  data/                        # inputs (gitignored)
  .aim/                        # single Aim store (gitignored)
  outputs/                     # per-run artifacts, created by Hydra (gitignored)
  src/clustering/
    experiment.py              # @hydra.main entrypoint (all 3 stages)
    identifier.py              # FragmentsIdentifier
    physics.py
    conf/                      # Hydra configs (config.yaml + experiment/ presets)
    split_prediction/          # dataset.py, model.py, mst.py, trainer.py
    stability/                 # lookup.py (StabilityLookup)
```

## Setup (one-time)

Install the package into the repo venv in editable mode — this puts the
`clustering-run` command on the venv's path and lets you run from anywhere:

```bash
.venv/bin/pip install -r requirements.txt   # deps (already done in Part A)
.venv/bin/pip install -e .                  # register the package + console script
```

## Running experiments

Run from the **repo root** so `outputs/` collects next to `data/` and `.aim/`:

```bash
.venv/bin/clustering-run                     # all 3 stages, Actor-Critic (default)
```

(Equivalently `.venv/bin/python -m clustering.experiment …`.) Hydra composes the
config from `conf/config.yaml` (schema: `ClusteringConfig` in `experiment.py`).
**Every field is overridable on the command line** as `key=value`:

```bash
# Raw REINFORCE (EMA baseline) instead of Actor-Critic — a pure config flip:
.venv/bin/clustering-run use_critic=false

# Retrain the split model + visualize only, K=4, longer training:
.venv/bin/clustering-run 'stages=[2,3]' n_clusters=4 split_epochs=120

# Tiny smoke test (conf/experiment/quick.yaml — note the leading '+'):
.venv/bin/clustering-run +experiment=quick

# Sweep one param, one Aim run each, all in the same store for comparison:
.venv/bin/clustering-run --multirun split_lr=1e-4,3e-4,1e-3
```

> **zsh note:** quote any override containing `[` `]`, e.g. `'stages=[3]'` —
> otherwise zsh tries to glob-expand the brackets and errors with
> `no matches found`.

### Per-run output dir

Hydra creates a fresh timestamped dir per invocation under
`outputs/<date>/<time>/` (sweeps go under `multirun/`), relative to where you
launch the command — hence "run from the repo root". **All** artifacts land
there — `dm_model.pt`, `critic_model.pt`, `figures/`, and `resolved_config.yaml`
(the fully-resolved config plus the git SHA/dirty flag for that run). Nothing is
written to a fixed path, so consecutive runs never overwrite each other.
`outputs/`, `multirun/`, `.aim/`, `*.pt`, and `data/` are gitignored.

### Reuse a checkpoint (skip retraining)

Stage 3 can run standalone against a previous run's split model:

```bash
.venv/bin/clustering-run 'stages=[3]' \
    load_split_from=outputs/2026-07-28/22-55-11
```

`load_split_from` accepts a run dir (it looks for `dm_model.pt` inside) or a
direct `.pt` path.

## Tracking & comparison (Aim)

Every invocation creates one `aim.Run` in a **single store pinned to the project
root** — `Run(repo="<repo_root>/.aim", experiment="clustering")`. This pin is
load-bearing: Hydra changes the per-run output location, and Aim's default
`./.aim` would otherwise create a separate store per run and silently break
cross-run comparison. Each run records:

- `hparams` — the fully-resolved config as a dict.
- **Stage-2 per-epoch** metrics — `reward`, `eval_reward`, `baseline`, `loss`
  (and `value_loss` when `use_critic=true`), under context `{"stage": "split"}`.
- **Stage-3 summary** scalars — fragment counts, fragments/event mean·min·max,
  splits total, max tree level, mean max charge — under `{"stage": "fragments"}`.
- Tags: `sha:<short>`, `dirty` (if the tree was dirty), `particle:<type>`,
  `critic:on|off`. The full SHA + dirty flag are also stored under `git`.

Browse and compare all runs side by side (from the repo root):

```bash
.venv/bin/aim up --repo .
```

Then open the printed URL (default http://127.0.0.1:43800). Aim indexes any new
runs on startup; group/subtract by `hparams.*` and overlay the `stage=split`
curves to compare Actor-Critic vs. raw REINFORCE.

## Reproducing an old result

Open the run in Aim, read its logged git SHA, then:

```bash
git checkout <sha>
```

and rerun with the overrides recorded in that run's `resolved_config.yaml`.
