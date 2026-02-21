# PNC-GAPPO: Generalizable Petri Net Scheduling with PPO

This project implements a **generalizable scheduling** approach for Petri Net (PN) based manufacturing systems using Proximal Policy Optimization (PPO). The goal is to train a single policy that can generalize across different PN instances—including randomly generated ones—without retraining, achieving robust scheduling decisions that minimize makespan while avoiding deadlocks.

## Project Overview

The project addresses the challenge of **generalization in scheduling**: instead of training a policy for one specific PN instance, we train on diverse PN structures (randomly generated or from a pool) so that the learned policy can be applied to **unseen** PN instances at test time. This is essential for real-world manufacturing where new job-shop configurations frequently arise.

### Key Features

- **Generalizable Policy**: A single PPO-based actor-critic network that outputs transition-conditioned action logits
- **Multiple Training Modes**: `random_gen` (random PN per episode), `gen_pool` (maintain a pool of PNs), `fixed_pool`, or `single_instance`
- **Flexible PN Representation**: Dynamic state (marking, remaining times) + static context (structure, operation times)
- **MLP or GAT Backend**: Configurable network architecture (MLP or Graph Attention) for place-transition aggregation

---

## Installation

### Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn

```bash
pip install torch numpy pandas matplotlib seaborn
```

---

## How to Run

### Training

To start training, run:

```bash
python Train.py
```

Or from the project root:

```bash
cd PNC-GAPPO
python Train.py
```

**Important**: Run from the project directory so that paths to `PNfiles/`, `Training_result/`, and `Eval_result/` resolve correctly.

#### Training Modes (Config.py)

- `TRAIN_MODE = "random_gen"`: Each episode uses a newly generated random PN (best for generalization)
- `TRAIN_MODE = "gen_pool"`: Maintain a pool of PNs, sample from it, and periodically refresh
- `TRAIN_MODE = "fixed_pool"`: Sample from a fixed set of PN files
- `TRAIN_MODE = "single_instance"`: Train on a single PN (for debugging)

#### Key Config Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `EPISODES` | Total training episodes | 5000 |
| `EVAL_INTERVAL` | Validation interval (episodes) | 50 |
| `ROLLOUT_STEPS` | PPO rollout buffer size | 1024 |
| `AC_NET_TYPE` | Network type: `"mlp"` or `"gat"` | `"mlp"` |

### Evaluation

To evaluate a trained model (greedy, argmax policy):

```bash
python Eval.py --model path/to/model.pth
```

If `--model` is omitted, the latest checkpoint in `Training_result/Models/` is used.

**Options:**

- `--bench`: Run evaluation on benchmark PN files (Config.TRAIN_PN_LIST)
- `--no-bench`: Skip benchmark evaluation
- `--random-pns N`: Evaluate on N randomly generated PNs (default: 20)
- `--seed S`: Random seed for PN generation
- `--export-random`: Export generated PNs for reproducibility
- `--gen-stage I`: Use curriculum stage I for the generator

**Examples:**

```bash
# Evaluate on benchmarks and 50 random PNs
python Eval.py --bench --random-pns 50

# Evaluate with exported random PNs
python Eval.py --export-random --seed 42
```

### PN Generator Difficulty Test

To test the difficulty of the random PN generator:

```bash
python PN_Generator.py -n 100
```

Options:

- `-n, --n_tests`: Number of PNs to test (default: 100)
- `--greedy`: Use greedy policy
- `--serial`: Use serial policy (one job at a time)

---

## Project Structure

| File / Directory | Description |
|------------------|-------------|
| **Train.py** | Main training script. Runs PPO training with configurable modes, validation, and final evaluation. |
| **Eval.py** | Evaluation script. Loads a checkpoint and runs greedy evaluation on benchmarks and/or random PNs. |
| **Config.py** | Central configuration: paths, hyperparameters, PN size limits, training mode, PPO settings. |
| **Agent.py** | PPO agent: transition storage, GAE computation, per-episode advantage normalization, PPO update. |
| **Network.py** | Actor-critic networks: MLP-based and GAT-based transition-conditioned networks. |
| **Environment.py** | RL environment wrapping PN: `reset()`, `step()`, state/action mask construction. |
| **PN.py** | Petri Net core: load from files, state transition, enabled transitions, goal/deadlock detection. |
| **PN_Generator.py** | Random PN generator and `GeneratorPNPool` for `gen_pool` mode. Supports curriculum via `configure()`. |
| **Utils.py** | Helpers: benchmark evaluation, random PN evaluation, training/benchmark logging, plotting, curriculum manager. |
| **PNfiles/** | Benchmark PN definitions (`*_init.txt`, `*_matrix.txt`). |
| **Training_result/** | Training outputs: `training_log.csv`, `train_history.pt`, `Models/` (checkpoints). |
| **Eval_result/** | Per-benchmark evaluation history and plots. |

---

## Output Directories

- **Training_result/**: Training logs, plots, and model checkpoints
  - `training_log.csv`: Episode-level metrics
  - `train_history.pt`: Full training history (torch)
  - `Models/model_ep{N}.pth`: Checkpoints
  - `Models/model_best_{name}.pth`: Best model per benchmark
- **Eval_result/**: Per-benchmark CSV and plots
- **Training_result/** is typically excluded from version control (add to `.gitignore`)

---

## Generalization in Scheduling

This project focuses on **generalizable scheduling**:

1. **Problem**: In manufacturing, job-shop configurations (PNs) vary. A policy trained on one PN often fails on another.
2. **Approach**: Train on diverse PNs (random generation or a pool) so the policy learns structure-invariant decision rules.
3. **Representation**: Fixed-size padded state (marking, remaining times, place mask) + static context (incidence matrices, operation times). This allows the same network to handle PNs of different sizes.
4. **Training Modes**:
   - `random_gen`: Maximum diversity; each episode sees a new PN.
   - `gen_pool`: Balance between diversity and repeated exposure for stable learning.

The trained policy can then be applied to **unseen** PN instances without retraining, achieving generalization across the PN distribution.

---

## License

See repository license file.
