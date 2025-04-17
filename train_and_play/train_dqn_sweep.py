"""
Deep Q-Learning Training Script for Tic Tac Toe Variants

This script trains Deep Q-Learning agents to play Tic Tac Toe on customizable grids
with optional periodic boundary conditions. It supports:

- Parameter sweeps over different game configurations
- Fully convolutional Q-networks
- Evaluation against random agents
- Adaptive exploration strategies
- Optional pretrained model loading
- Model artifact saving
- Weights & Biases logging

Key Classes & Dependencies:
- DeepQLearningAgent: Q-learning agent with optional FullyConvQNetwork
- TicTacToe: Environment handling gameplay with two agents
- evaluate_performance: Evaluates agents against random baselines
"""

import copy
import wandb

from typing import Any

from TicTacToe.TicTacToe import TicTacToe
from TicTacToe.DeepQAgent import DeepQLearningAgent
from TicTacToe.ReplayBuffers import ReplayBuffer
from TicTacToe.Utils import get_param_sweep_combinations, load_pretrained_models, save_model_artifacts, train_and_evaluate

# --- Training Parameters ---
params: dict[str, Any] = {
    "nr_of_episodes": 200000,  # Number of training games
    "rows": 3,  # Board size (rows x rows)
    "learning_rate": 0.0001,  # Optimizer learning rate
    "gamma": 0.95,  # Discount factor for future rewards
    "switching": True,  # Whether players switch turns
    "win_length": 3,  # Number of in-a-row needed to win
    "epsilon_start": 0.925,  # Initial exploration rate
    "epsilon_min": 0.05,  # Minimum exploration rate
    "set_exploration_rate_externally": True,  # Adaptive epsilon enabled
    "epsilon_update_threshold": 0.025,  # Epsilon adjustment sensitivity
    "epsilon_decay": 0.95,  # Decay rate for epsilon
    "win_rate_deque_length": 5,  # Length of win rate deques
    "batch_size": 256,  # Batch size for training updates
    "target_update_frequency": 25,  # Frequency to sync target network
    "evaluation_frequency": 1000,  # Episodes between evaluations
    "evaluation_batch_size": 2000,  # Games to evaluate per round
    "device": "mps",  # Device: "cuda", "mps", or "cpu"
    "replay_buffer_length": 10000,  # Max length of replay buffer
    "wandb": False,  # Enable Weights & Biases logging
    "wandb_logging_frequency": 25,  # Logging frequency (in episodes)
    "load_network": False,  # Whether to load pretrained weights
    "shared_replay_buffer": False,  # Unused flag (placeholder)
    "network_type": "FullyCNN",  # Network architecture
    "periodic": True,  # Periodic boundary conditions
    "save_models": "/Users/jakob/TicTacToe/models/",  # Save weights after training
    "symmetrized_loss": False,  # Use symmetrized loss
    "state_shape": "2D",  # state representation: 'flat' with shape (batch_size, rows * rows), 
                            # '2D' with shape (batch_size, 1, rows, rows), 
                            # 'one-hot' with shape (batch_size, 3, rows, rows)
}

# params["shared_replay_buffer"] = ReplayBuffer(params["replay_buffer_length"], (params["rows"]**2, ), device=params["device"])

# --- Sweep Setup ---
param_sweep = {"state_shape": ["2D"], "symmetrized_loss": [False]}
sweep_combinations, param_keys = get_param_sweep_combinations(param_sweep)
model_metadata = []

# --- Sweep Loop ---
for sweep_idx, combination in enumerate(sweep_combinations):
    print(f"Starting parameter sweep {sweep_idx + 1}/{len(sweep_combinations)}")
    for key, value in zip(param_keys, combination):
        params[key] = value

    paramsX = copy.deepcopy(params)
    paramsO = copy.deepcopy(params)
    paramsX["player"] = "X"  # Player symbol for Agent 1
    paramsO["player"] = "O"  # Player symbol for Agent 2
    paramsX["wandb"] = True  # Log Agent 1 with wandb
    paramsO["wandb"] = False  # Do not log Agent 2

    if params["load_network"]:
        paramsX, paramsO = load_pretrained_models(paramsX, paramsO)

    agent1 = DeepQLearningAgent(paramsX)
    agent2 = DeepQLearningAgent(paramsO)

    game = TicTacToe(
        agent1,
        agent2,
        display=None,
        rows=params["rows"],
        cols=params["rows"],
        win_length=params["win_length"],
        periodic=params["periodic"]
    )

    try:
        train_and_evaluate(game, agent1, agent2, params, wandb_logging=paramsX["wandb"] or paramsO["wandb"])

    finally:
        if params.get("save_models"):
            save_model_artifacts(agent1, agent2, params)

    wandb.finish()
