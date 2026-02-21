# A Generalisable Learning-Based Scheduler: a Petri-net-conditioned deep reinforcement learning framework with a graph attention network and proximal policy optimisation (PPO) algorithm (PNC-GAPPO)

This project implements a **generalizable scheduling** approach for robotic celluar manufacturing systems (RCMSs) using place-timed Petri Nets (PTPNs). The goal is to train a single policy that can generalize across different RCMS instances—including randomly generated ones—without retraining, achieving transferable scheduling decisions that minimize makespan.

## Project Overview

The project addresses the challenge of **instance-specific scheduling**: instead of training a policy for one specific RCMS instance, we train on diverse systems (randomly generated from a simulator) so that the learned policy can be applied to **unseen** instances in validation. This is essential for real-world manufacturing where new configurations frequently arise.

### Key Contributions

This project focuses on **generalisable scheduling**.
- **PTPN Simulator**: we implement a PTPN simulator supporting multi-robot operations and flexible routing to randomly produce diverse training instances, which enables the scheduler to learn a generalisable policy;<br>
- **GAT Embedding**: we employ the GAT to derive tailored node embeddings within PNs, which learns adaptive attention weights over neighbours for a more nuanced understanding of different RCMSs;<br>
- **PTPN-conditioned Decision Maker**: PPO is introduced to fired transitions based on both dynamic state and static PN contexts. We further design per-episode advantage normalisation, which independently normalises advantages within each training episode to mitigate gradient signal interference facing multiple PNs, thereby promoting cross-instance generalisation;<br>
- **End-to-End Generalisation Framework**: we monitor the scheduler’s decision-making performance on unseen RCMS instances during the end-to-end training paradigm. Through comparative and ablation experiments, we demonstrate the competitive edge of our proposed method. To the best of our knowledge, this is the first work to achieve effective policy generalisation within the context of RCMS scheduling;<br>

### PN-DGS Workflow
The system scheduling workflow is illustrated in the diagram below.<br>
![PNC-GAPPO Workflow](https://github.com/MartiniiiK/PNC-GAPPO/blob/main/Pic/PNC-GAPPO.pdf?raw=true)

## Program Demonstration
The project is developed in Python using a high-performance computing platform equipped with a NVIDIA GeForce RTX 3080 GPU (20 GB VRAM) running on CUDA 12.4. <br>
This demonstration is typically performed in Visual Studio Code. For a detailed explanation of the proposed algorithm, please refer to our paper.<br>
**Due to the requirements of the paper review process, we are currently unable to provide an operation tutorial video for the project. It will be provided at an appropriate time. Please stay tuned.**

## Getting Started

### Requirements
- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- 
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

**Important**: Run from the project directory so that paths to `PNfiles/`, `Training_result/`, and `Eval_result/` resolve correctly. Please manually configure the training mode and validation system in Config.py

#### Training Modes (Config.py)

- `TRAIN_MODE = "random_gen"`: Each episode uses a newly generated random PN (best for generalization)
- `TRAIN_MODE = "gen_pool"`: Maintain a pool of PNs, sample from it, and periodically refresh
- `TRAIN_MODE = "fixed_pool"`: Sample from a fixed set of PN files
- `TRAIN_MODE = "single_instance"`: Train on a single PN (for debugging)

### Evaluation

To evaluate a trained model (greedy, argmax policy):

```bash
python Eval.py --model path/to/model.pth
```

If `--model` is omitted, the latest checkpoint in `Training_result/Models/` is used.

---

## Project Structure

| File / Directory | Description |
|------------------|-------------|
| **Train.py** | Main training script. Runs training with configurable modes, validation, and final evaluation. |
| **Eval.py** | Evaluation script. Loads a checkpoint and runs greedy evaluation on benchmarks and/or random PNs. |
| **Config.py** | Central configuration: paths, hyperparameters, PN size limits, training mode, PPO settings. |
| **Agent.py** | PPO agent: transition storage, GAE computation, per-episode advantage normalization, PPO update. |
| **Network.py** | Actor-critic networks: MLP-based and GAT-based transition-conditioned networks. |
| **Environment.py** | RL environment wrapping PN: `reset()`, `step()`, state/action mask construction. |
| **PN.py** | Petri Net core: load from files. |
| **PN_Generator.py** | Random PN generator and `GeneratorPNPool` for `gen_pool` mode. |
| **Utils.py** | Helpers: benchmark evaluation, random PN evaluation, training/benchmark logging, plotting. |
| **PNfiles/** | Benchmark PN definitions (`*_init.txt`, `*_matrix.txt`). |
| **Training_result/** | Training outputs. |
| **Eval_result/** | Evaluation history and plots. |

---

## Contact

**The contact link cannot be provided at this moment.**
