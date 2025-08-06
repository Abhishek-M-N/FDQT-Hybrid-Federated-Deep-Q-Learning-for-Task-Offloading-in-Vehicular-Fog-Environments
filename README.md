# FDQT-Hybrid-Federated-Deep-Q-Learning-for-Task-Offloading-in-Vehicular-Fog-Environments
A Federated Deep Reinforcement Learning-based system for optimizing task offloading in Vehicular Fog Computing (VFC). It reduces latency and congestion by training DQN agents on vehicles using FL, with performance visualizations and comparison to baseline methods.
#  VFC-Offloading-RL: Federated Reinforcement Learning for Vehicular Fog Computing

This project implements a **Federated Deep Reinforcement Learning (FL + DRL)** approach for optimizing **task offloading** in **Vehicular Fog Computing (VFC)** systems. It focuses on reducing **latency**, **congestion**, and improving task success rate by training autonomous vehicles using decentralized learning.



##  Key Features

-  **Federated Deep Q-Learning** for offloading decisions
-  Optimizes task delay, queue size, and drop rate
-  Supports multi-vehicle and multi-RSU simulations
-  Visualizations for performance comparisons
-  Evaluation against baseline methods (Cloud, RSU, Random, etc.)



##  Main Components

| File         | Description |
|--------------|-------------|
| `server.py`  | Federated server logic for aggregating DQN weights using FedAvg |
| `simulate.py`| Simulates VFC environment: vehicles, RSUs, and task dispatching |
| `plotfl.py`  | Plots FL-based performance metrics (reward, delay, etc.) |
| `flplot2.py` | Plots comparative results across multiple strategies |
| `train.py`   | Trains a single-agent DQN (non-FL baseline) |
| `train3.py`  | Trains DQN with FL across multiple vehicles |
| `train4.py`  | Variation of FL with RSU coordination |
| `train5.py`  | Enhanced FL with adaptive congestion handling |



###  1. Install Requirements

pip install -r requirements.txt

 2. Run Federated Training

python train3.py
You can also try train4.py and train5.py for alternative FL strategies.

3. Visualize Performance

python plotfl.py
python flplot2.py

## Metrics Tracked
-- Reward vs Epochs
-- Task Delay vs Epochs
-- Queue Size vs Epochs
-- FL vs Non-FL Comparison Charts

## Evaluation Strategies
-- RL (Federated & Local)
-- Cloud-Only Offloading
-- RSU-Only Offloading
-- Random Offloading
-- Greedy Selection
These are compared using logs and plots generated after training.

## Acknowledgements
Inspired by recent research in federated learning and fog computing optimization. Built using:
-- Gym
-- Stable-Baselines3
-- Matplotlib & Pandas
-- CIW (Discrete-event simulation)

🛡 License
This project is for research and academic use. You may adapt and use it with proper citation. License terms can be added here if needed.


