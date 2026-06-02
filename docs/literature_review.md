# Literature Review: Multi-UAV Delivery Task Assignment with Obstacles

## 1. Existing Environments

| Environment | Focus | Gap for This Research |
|---|---|---|
| **PettingZoo MPE** (`simple_spread`) | 2D cooperative MARL benchmark, agents navigate to landmarks | 2D only, no quadrotor dynamics, no obstacles |
| **gym-pybullet-drones** (UTIAS DSL) | PyBullet-based quadrotor env with single/MARL, rotor-level physics, obstacle support | Designed for low-level formation control, not high-level task assignment + delivery |
| **PyFlyt** | Fixed-wing and quadrotor Gymnasium envs with PID control and obstacles | Single-agent flight control, no multi-agent task allocation |
| **MAGNNET / HIPPO-MAT** (Skoltech, 2025) | 3D grid env for UAV-UGV task allocation with GNN+PPO; A* path planning | 3D grid only, no continuous quadrotor dynamics |
| ROTORS + Unreal Engine (Ahmed et al., 2025) | LLM-guided DMPC with obstacles | Requires ROS + Unreal, not open-sourced as Gym env |

## 2. Key Papers

1. **"UAV-MARL: Multi-Agent RL for Time-Critical Medical Supply Delivery"** (Guven & Parlak, 2026) — UAV fleet delivering supplies under deadlines. PPO, OSM geography. Closest to this work but no explicit obstacle avoidance.

2. **"HIPPO-MAT: Decentralized Task Allocation Using GraphSAGE"** (Ratnabala et al., Mar 2025) — UAV+UGV task allocation with GraphSAGE embeddings + IPPO. A* for routing. 92.5% conflict-free success rate.

3. **"MAGNNET: Multi-Agent GNN-based Task Allocation with DRL"** (Ratnabala et al., Feb 2025) — GNN + PPO with CTDE for UAV-UGV task assignment on 3D grid.

4. **"MPC of Quadrotor Swarms with Collision Avoidance"** (Bianchi et al., 2026) — Real Crazyflie hardware validated. Obstacle avoidance + coordinated target assignment via MPC (not RL, but hardware-validated dynamics).

5. **"Cluster-Based Multi-Agent Task Scheduling for SAGIN"** (Wang et al., Dec 2024) — CMADDPG algorithm, 25%+ improvement over baselines.

## 3. Research Gap

No existing Gymnasium/PettingZoo environment simultaneously provides:
- Continuous 3D quadrotor dynamics
- Multi-agent task assignment with pickup/dropoff delivery semantics
- Realistic 3D obstacle fields requiring learned avoidance strategies
- Heterogeneous target priorities and delivery time windows
- Full CTDE (Centralized Training Decentralized Execution) MARL support

This motivates the development of a custom `quadrotor-delivery` environment.

## 4. Relevant Codebases

| Repository | What to Reference |
|---|---|
| `gym-pybullet-drones` | Quadrotor Newton-Euler dynamics, thrust/torque action model, Gymnasium interface |
| `PettingZoo/mpe` | Multi-agent wrapper patterns, Parallel/AEC API design |
| MAGNNET/HIPPO-MAT | Decentralized GNN-based task allocation architecture |
| `PettingZoo` custom env tutorial | Pattern for building custom PettingZoo environments |
