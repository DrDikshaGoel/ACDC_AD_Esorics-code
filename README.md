# Optimizing Cyber Defense in Dynamic Active Directories through Reinforcement Learning

This repository contains the implementation of the paper:

**Optimizing Cyber Defense in Dynamic Active Directories through Reinforcement Learning**  
Diksha Goel, Kristen Moore, Mingyu Guo, Derui Wang, Minjune Kim, and Seyit Camtepe  
**ESORICS 2024**

## Overview

This work develops an autonomous cyber-defense framework for **dynamic Active Directory (AD) attack graphs**.

The framework models the interaction between an attacker and defender as a Stackelberg game. The attacker attempts to reach the Domain Admin (DA), while the defender aims to reduce the attacker's success probability by strategically blocking edges in the AD attack graph.

The proposed approach consists of:

- **GenRL:** A generalized reinforcement-learning attacker trained across multiple dynamic AD graph snapshots.
- **RL Training Facilitator (TrnF):** Improves RL training through environment and neural-network pruning.
- **RL-EDO:** An RL-assisted Evolutionary Diversity Optimization approach for generating effective and diverse defense strategies.

The attacker is trained using **Proximal Policy Optimization (PPO)** with an actor-critic architecture.

## Repository Structure

The main components of the repository include:

```text
main_r1000.py             Main experiment script
AgentEnv_file_r1000.py    RL environment for the attacker
ppo_policy_r1000.py       PPO attacker implementation
ec_r1000.py               Evolutionary defense optimization
buildgraph.py             AD attack graph construction
setupgraph.py             Graph setup and preprocessing
states.py                 State and transition handling
settings.py               Experimental configuration
utility.py                Supporting utility functions
data/                     Experimental data
I_log_*/                  Independent-distribution experiment logs
P_log_*/                  Positive-correlation experiment logs
N_log_*/                  Negative-correlation experiment logs
sbatch_run.sh             HPC/SLURM execution script
```

Experiments are provided for AD graphs of different sizes, including **r1000, r2000, and r4000**.

## Requirements

The implementation is written in **Python** and uses **PyTorch** and **OpenAI Gym** for reinforcement-learning experiments.

Please ensure that the required Python dependencies are installed before running the experiments.

## Running the Code

The main experiment can be started using:

```bash
python main_r1000.py
```

Experiment parameters and configurations can be adjusted through `settings.py` and the corresponding experiment scripts.

## Paper

This repository accompanies the following publication:

**Diksha Goel, Kristen Moore, Mingyu Guo, Derui Wang, Minjune Kim, and Seyit Camtepe.**  
*"Optimizing Cyber Defense in Dynamic Active Directories through Reinforcement Learning."*  
**29th European Symposium on Research in Computer Security (ESORICS 2024), 2024.**

arXiv: **2406.19596**

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{goel2024optimizing,
  title={Optimizing Cyber Defense in Dynamic Active Directories through Reinforcement Learning},
  author={Goel, Diksha and Moore, Kristen and Guo, Mingyu and Wang, Derui and Kim, Minjune and Camtepe, Seyit},
  booktitle={29th European Symposium on Research in Computer Security (ESORICS 2024)},
  year={2024}
}
```
