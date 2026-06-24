# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is **APISAT-Guide-Env**, a Gymnasium-compatible multi-agent reinforcement learning environment for quadrotor UAV delivery task assignment with obstacle avoidance. It is written in Python 3.11 and centers on a single `QuadrotorDeliveryEnv` that can be wrapped for Stable-Baselines3, PettingZoo, or custom CTDE/MAPPO trainers.

Key facts:

- Python version is pinned to **3.11** (`.python-version`).
- The project uses both `pyproject.toml` (primary) and `setup.py` (legacy extras).
- A `.venv` exists at the repository root; VS Code settings point to it.
- `main.py` is a placeholder. Real entry points live in `scripts/`.

## Common commands

### Environment setup

```bash
# Using uv (recommended; uv.lock is present)
uv sync

# Or pip in editable mode
pip install -e .

# With RL + visualization extras (matches README)
pip install -e ".[rl,viz]"
```

The `dev` dependency group in `pyproject.toml` includes `pytest`, `black`, and `ruff`.

### Run tests

```bash
# All tests
pytest tests/ -v

# Run a single test
pytest tests/test_env.py::TestDynamics::test_hover_thrust -v

# Renderer tests require a non-interactive matplotlib backend
MPLBACKEND=Agg pytest tests/test_renderer.py -v

# Without pytest
python tests/test_env.py
```

### Lint and format

```bash
# Format
black .

# Lint
ruff check .
ruff check --fix .
```

### Training

```bash
# Centralized PPO (single policy, flattened obs/actions)
python scripts/train.py --algo ppo --num_drones 4 --num_targets 4

# Independent PPO with parameter sharing (PettingZoo wrapper)
python scripts/train.py --algo ippo --num_drones 4 --num_targets 4

# CTDE MAPPO (custom implementation, actor on local obs + centralized critic)
python scripts/ctde_trainer.py --num_drones 4 --num_targets 4 --total_steps 500000

# Standalone MAPPO baseline with tanh-squashed actions
# Key default: aggregate_phy_steps=20, gamma=0.995, gae_lambda=0.97
python scripts/mappo_train.py --num_drones 4 --num_targets 4 --total_steps 500000

# With obstacles (10 default, increase for harder scenarios)
python scripts/mappo_train.py --num_drones 4 --num_targets 4 --num_obstacles 20

# Delivery mode with more obstacles
python scripts/ctde_trainer.py --num_drones 6 --task_mode delivery --num_obstacles 20

# Override aggregate_phy_steps for finer/coarser control (default 20 = 12 Hz control rate)
python scripts/mappo_train.py --aggregate_phy_steps 10 --max_episode_steps 4000

# SemGAT-MARL: Phase 1 — VAE pretraining (heuristic semantic labels)
python scripts/semgat_pretrain.py --num_drones 4 --num_targets 4 --total_steps 50000

# SemGAT-MARL: Phase 1 — with a different VAE config
python scripts/semgat_pretrain.py --latent_dim 128 --beta 0.05 --lambda_sem 2.0 --vae_epochs 300

# SemGAT-MARL: Phase 2 — GAT-MAPPO training
python scripts/semgat_train.py --num_drones 4 --num_targets 4 --total_steps 500000 \
    --vae_checkpoint models/semgat/vae_pretrained.pt

# SemGAT-MARL: Phase 2 — ablation without VAE (raw obs → GAT)
python scripts/semgat_train.py --no_vae --num_drones 4 --num_targets 4 --total_steps 500000

# SemGAT-MARL: Phase 2 — without semantic reward shaping
python scripts/semgat_train.py --no_semantic_reward --vae_checkpoint models/semgat/vae_pretrained.pt
```

Training writes TensorBoard logs under `logs/` and checkpoints under `models/`. SB3 trainers save `.zip` files; CTDE/MAPPO trainers save `.pt` checkpoints containing `actor` and `critic` `state_dict`s.

### Evaluation and rendering

```bash
# Random policy, live 3D window
python scripts/eval.py --render --num_episodes 1

# Load a trained SB3 model
python scripts/eval.py --model_path models/ppo_final.zip --render

# Record mp4/gif
python scripts/eval.py --record --record_path logs/eval.mp4 --num_episodes 1
python scripts/eval.py --record --record_path logs/eval.gif --num_episodes 1

# 2D top-down view
python scripts/eval.py --top_down --num_episodes 1

# TensorBoard
uv run tensorboard --logdir logs
```

### Running ad-hoc environment code

Because the package is not always installed, scripts insert the repo root into `sys.path`. When running modules directly, prefer:

```bash
PYTHONPATH=. python scripts/eval.py --render
```

The `.env` file already sets `PYTHONPATH=.`.

## High-level architecture

### Core modules (`core/`)

The environment is decomposed into small, stateless-ish modules that are composed by `QuadrotorDeliveryEnv`:

- `dynamics.py` — Newton-Euler quadrotor physics (Crazyflie 2.X defaults). State is 12-DOF `[pos, vel, euler, omega]`; actions are physical `[thrust, tau_x, tau_y, tau_z]`.
- `state.py` — Builds fixed-size local observations per agent and a flattened global observation for centralized critics. Local obs is padded to `max_num_drones`, `max_num_targets`, and `max_num_obstacles`.
- `action.py` — Maps normalized `[-1, 1]^4` policy outputs to physical thrust/torque ranges.
- `target.py` — `Target`, `Obstacle`, and procedural generation. Delivery mode creates PICKUP/DELIVERY pairs linked by `pair_id`.
- `reward.py` — `RewardCalculator`: target reward, collision penalties, energy penalty, distance shaping, completion bonus.
- `termination.py` — `TerminationChecker`: max steps, target reach, collisions, out-of-bounds.
- `semantic.py` — `HeuristicSemanticClassifier` / `LLMSemanticClassifier`: classifies agent-entity pairs into 5 relation types (No-Relation, Target, Contest, Avoid, Separate). Builds sparse semantic interaction graphs. LLM classifier is a placeholder awaiting API wiring.
- `vae.py` — `DualHeadVAE`: variational autoencoder with a physical reconstruction head and a semantic prediction head. Pre-trained offline to produce latent z that fuses dynamics and semantics.
- `gat_policy.py` — `SemanticGATLayer`, `GATActor`, `GATCritic`: graph attention networks with semantic relation-aware attention coefficients. Replace MLP actor/critic in the MAPPO pipeline.

### Environment and wrappers (`envs/`)

- `quadrotor_delivery_v0.py` — `QuadrotorDeliveryEnv` is the canonical Gymnasium env. It returns a `Dict` observation `{"agent_obs": (n_drones, obs_dim), "global_obs": (global_dim,)}` and per-agent rewards/terminated/truncated arrays.
- `pettingzoo_wrapper.py` — `ParallelQuadrotorDelivery` exposes the PettingZoo Parallel API over the base env.
- `sb3_wrapper.py` — `FlattenDictObs`, `FlattenAction`, and `AggregateMARL` turn the multi-agent env into a single-agent-style env for Stable-Baselines3. `make_sb3_env(env)` is the convenience helper.

### Configuration (`utils/config.py`)

All hyperparameters live in dataclasses:

- `EnvConfig`, `QuadrotorParams`, `ObsConfig`, `ActionConfig`, `RewardConfig`, `TerminationConfig`, `TrainingConfig`
- `FullConfig` aggregates them and wires cross-cutting values in `__post_init__` (e.g., `obs.max_num_drones = env.num_drones`, `term.max_steps = env.max_steps`).

Training scripts construct a `FullConfig()` and override fields from CLI args.

### Rendering (`utils/rendering.py`, `utils/visualization.py`, `utils/topdown.py`)

- `Matplotlib3DRenderer` pools matplotlib artists and updates data each frame rather than redrawing. It supports trails, attitude frames, HUD, a 2D top-down inset, and event flashes.
- `SimpleRenderer` is a deprecated alias kept for backward compatibility.
- `utils.visualization` provides `record_episode`, `play_episode`, and `live_render`.
- `utils.topdown` provides a fast 2D overhead renderer used by `render_mode="top_down"`.

### Training scripts (`scripts/`)

- `train.py` — SB3-based centralized PPO or IPPO with parameter sharing. Uses `DummyVecEnv` for PPO and a manual rollout loop for IPPO.
- `ctde_trainer.py` — Custom MAPPO: shared actor on local obs, centralized critic on global obs, per-agent masks for inactive agents.
- `mappo_train.py` — Self-contained MAPPO baseline with tanh-squashed Gaussian actor, clipped value updates, and periodic evaluation.
- `semgat_pretrain.py` — Phase 1 of SemGAT-MARL: collects rollout data with random policy, builds semantic labels (heuristic or LLM), trains the Dual-Head VAE, and saves the frozen encoder.
- `semgat_train.py` — Phase 2 of SemGAT-MARL: loads frozen VAE encoder, constructs semantic graphs at each step, trains GAT actor/critic with MAPPO, and applies semantic-aware reward shaping (paper Eq. 4).
- `eval.py` — Loads SB3 `.zip` models or runs random policy; supports live rendering, recording, and top-down view.

### Task modes

- `reach` — Drones fly to independent target points.
- `delivery` — Targets are PICKUP+DELIVERY pairs. A drone must visit the pickup (setting `carry_status[i] = True`), then the paired delivery target. `pair_id` links the two targets.

Target assignment is greedy nearest-neighbor at reset (`_greedy_assignment`). Targets are marked reached/assigned in `_update_target_status`.

### Observations and actions

- Per-agent local obs: self state (12) + relative drones (3×max_drones) + target info (4×max_targets) + obstacle info (3×max_obstacles) + carry flag (1).
- Global obs: flattened full state including all drone states, target positions/assignment, obstacle positions/radii, and carry status.
- Action per agent: 4 continuous values in `[-1, 1]` mapped to thrust and roll/pitch/yaw torques.

## Important implementation notes

- **Reward design** (tuned for `aggregate_phy_steps=20`): `target_reached=+500`, `completion_bonus=+1000`, `step_penalty=-0.01`, `distance_scale=2.0` (potential-based), `orientation_penalty=0.0` (disabled because quadrotors MUST tilt to move). Per-step shaping (~+0.16 for approaching target) comfortably outweighs the step penalty, so flying toward a target yields net positive per-step reward.
- The environment uses `aggregate_phy_steps` sub-stepping inside `QuadrotorDeliveryEnv.step`, applying the same action repeatedly for finer physics. Default is now 20 (12 Hz control rate); each env step covers 20/240 ≈ 0.083 s of physics.
- `scripts/mappo_train.py` defaults: `gamma=0.995`, `gae_lambda=0.97` (longer temporal credit horizon), `aggregate_phy_steps=20`, `rollout_steps=4096`, `max_episode_steps=2000`.
- Episode terminates early (`terminate_when_impossible=True`) when surviving drones < remaining targets — prevents wasted steps after a drone crashes.
- Reward and termination code contain hardcoded constants (e.g., success radius `2.0`, collision height heuristic `5.0`). Prefer modifying via `RewardConfig` and `TerminationConfig` where possible.
- Observation normalization (`RunningMeanStd` in MAPPO) excludes inactive/terminated agents to prevent zero-vector corruption of statistics.
- Value loss coefficient is applied via `vf_coef` only (no redundant 0.5× scaling).
- The renderer has a monkey-patch for newer matplotlib versions to keep 3D scatter `_sizes3d` as a numpy array (`utils/rendering.py`).
- `scripts/mappo_train.py` and `scripts/ctde_trainer.py` save raw `state_dict` checkpoints, not SB3 models. To evaluate them you must load `actor`/`critic` keys into the network classes defined in those scripts.

## VS Code / debugging

- Default interpreter: `${workspaceFolder}/.venv/bin/python`.
- Tests are configured to use pytest with `tests/` as the target.
- Format-on-save uses Black.
- Launch configurations exist for the current file and for a non-existent `src/main.py`; the real entry points are in `scripts/`.
