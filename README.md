# Quadrotor-Delivery: Multi-Agent UAV Delivery Task Assignment

A Gymnasium-compatible multi-agent reinforcement learning environment for quadrotor delivery drone task assignment with obstacle avoidance.

## Features

- **3D Quadrotor Dynamics**: Newton-Euler rigid-body physics (Crazyflie 2.X parameters)
- **Multi-Agent Support**: Gymnasium flat API + PettingZoo Parallel API wrapper
- **Obstacle Avoidance**: Random 3D obstacles (cylinders, spheres, boxes)
- **Task Assignment**: Drones cooperatively assign to targets (reach or pickup+delivery modes)
- **CTDE Ready**: Local observations per agent + global observation for centralized critic
- **Configurable Reward**: Target completion, distance shaping, collision penalties, energy cost
- **Visualization**: Matplotlib 3D rendering

## Project Structure

```
├── core/                   # Core modules
│   ├── dynamics.py         # Newton-Euler quadrotor dynamics
│   ├── state.py            # Observation space definitions
│   ├── action.py           # Action normalization
│   ├── target.py           # Target & obstacle definitions
│   ├── reward.py           # Reward function
│   └── termination.py      # Termination conditions
├── envs/                   # Environment implementations
│   ├── quadrotor_delivery_v0.py  # Main Gymnasium env
│   ├── pettingzoo_wrapper.py     # PettingZoo Parallel API wrapper
│   └── sb3_wrapper.py            # SB3-compatible wrappers
├── utils/                  # Utilities
│   ├── geometry.py         # Geometric helpers
│   ├── rendering.py        # Matplotlib 3D visualization
│   └── config.py           # Configuration system
├── scripts/                # Entry points
│   ├── train.py            # Centralized & Independent PPO
│   ├── ctde_trainer.py     # CTDE MAPPO (custom implementation)
│   └── eval.py             # Evaluation & visualization
├── tests/                  # Tests
│   └── test_env.py         # Unit & integration tests
└── docs/                   # Documentation
    └── literature_review.md
```

## Quick Start

### Installation

```bash
pip install -e .
# With RL support:
pip install -e ".[rl,viz]"
```

### Basic Usage

```python
from envs import QuadrotorDeliveryEnv

env = QuadrotorDeliveryEnv(
    num_drones=4,
    num_targets=4,
    num_obstacles=10,
    task_mode="reach",  # or "delivery"
)

obs, info = env.reset()
action = env.action_space.sample()

for _ in range(500):
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated.any() or truncated.any():
        break
env.close()
```

### Training

Three training modes available:

```bash
# 1. Centralized PPO (single policy controls all drones)
python scripts/train.py --algo ppo --num_drones 4 --num_targets 4

# 2. Independent PPO (parameter sharing, per-drone policy)
python scripts/train.py --algo ippo --num_drones 4 --num_targets 4

# 3. CTDE MAPPO (centralized critic + decentralized actors) [recommended]
python scripts/ctde_trainer.py --num_drones 4 --num_targets 4 --total_steps 500000

# With delivery mode & more obstacles
python scripts/ctde_trainer.py --num_drones 6 --task_mode delivery --num_obstacles 20
```

### Evaluation

```bash
# Random policy test
python scripts/eval.py --render --num_episodes 3

# Load trained model
python scripts/eval.py --model_path models/ppo_final.zip --render
```

### Running Tests

```bash
pytest tests/ -v
# Or without pytest:
python tests/test_env.py
```

## Configuration

All parameters are configurable via `utils/config.py`:

| Module | Key Parameters |
|--------|---------------|
| `QuadrotorParams` | mass, arm_length, inertia, thrust coeff, dt |
| `ObsConfig` | max agents/targets/obstacles, sensing range, bounds |
| `ActionConfig` | thrust range, torque range, normalization |
| `RewardConfig` | target reward, collision penalty, energy coeff, shaping |
| `TerminationConfig` | max steps, collision kill, out-of-bounds |
| `EnvConfig` | num drones/targets/obstacles, task mode, bounds |

## Environment Description

### Observation (per agent)

| Component | Dim | Description |
|-----------|-----|-------------|
| Self state | 12 | [pos(3), vel(3), euler(3), omega(3)] |
| Relative drones | 3×max_drones | Positions relative to other drones |
| Targets | 4×max_targets | [rel_pos(3), assigned_flag] |
| Obstacles | 3×max_obstacles | Relative positions of nearest obstacles |
| Carry status | 1 | Whether carrying a package |

### Action (per agent)

| Index | Range | Description |
|-------|-------|-------------|
| 0 | [-1, 1] | Thrust (normalized, 0→max_thrust) |
| 1-3 | [-1, 1] | Roll/Pitch/Yaw torques (normalized) |

### Reward

- **+20** per target reached
- **-0.01** per time step
- **-50** obstacle collision
- **-30** drone-drone collision
- **-0.001×control_effort** energy penalty
- **-0.1×distance** shaping reward
- **+100** all targets complete bonus

## Task Modes

### Reach Mode
Drones fly directly to target points. Each target can be assigned to one drone.

### Delivery Mode
Targets are PICKUP+DELIVERY pairs. A drone must first fly to the pickup location, then to the paired delivery location.

## Research References

See `docs/literature_review.md` for related papers and environments.

Key gaps this environment addresses:
1. Continuous quadrotor dynamics + multi-agent task assignment
2. Learned obstacle avoidance (not A*/grid-based)
3. CTDE support with structured observations
4. Delivery semantics (pickup→carry→dropoff)
