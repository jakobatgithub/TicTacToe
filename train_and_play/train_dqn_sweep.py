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
from datetime import datetime

from TicTacToe.TicTacToe import TicTacToe
from TicTacToe.DeepQAgent import DeepQLearningAgent
from TicTacToe.ReplayBuffers import ReplayBuffer, PrioritizedReplayBuffer
from TicTacToe.Utils import get_param_sweep_combinations, load_pretrained_models, save_model_artifacts, train_and_evaluate

# --- Training Parameters ---
params: dict[str, Any] = {
    # Game settings
    "player": "X",  # Player symbol for the agent
    "rows": 3,  # Board size (rows x rows)
    "win_length": 3,  # Number of in-a-row needed to win
    "rewards": {
        "W": 1.0,  # Reward for a win
        "L": -1.0,  # Reward for a loss
        "D": 0.5,  # Reward for a draw
    },

    # Training settings
    "nr_of_episodes": 500,  # Number of training games
    "learning_rate": 0.0001,  # Optimizer learning rate
    "gamma": 0.95,  # Discount factor for future rewards
    "switching": True,  # Whether players switch turns
    "target_update_frequency": 25,  # Frequency to sync target network

    # Evaluation settings
    "evaluation_frequency": 100,  # Episodes between evaluations
    "evaluation_batch_size": 200,  # Games to evaluate per round
    "wandb_logging": False,  # Enable Weights & Biases logging
    "wandb_logging_frequency": 25,  # Logging frequency (in episodes)

    # Exploration rate settings
    "epsilon_start": 0.925,  # Initial exploration rate
    "epsilon_min": 0.01,  # Minimum exploration rate
    "set_exploration_rate_externally": True,  # Adaptive epsilon enabled
    "epsilon_update_threshold": 0.025,  # Epsilon adjustment sensitivity
    "epsilon_decay": 0.95,  # Decay rate for epsilon
    "win_rate_deque_length": 5,  # Length of win rate deques

    # Device settings
    "device": "mps",  # Device: "cuda", "mps", or "cpu"

    # Replay buffer settings
    "replay_buffer_type": "prioritized",  # "uniform" or "prioritized"
    "replay_buffer_length": 10000,  # Max length of replay buffer
    "batch_size": 256,  # Batch size for training updates
    "priority_alpha": 0.6,
    "priority_beta": 0.4,
    "shared_replay_buffer": False,  # Share replay buffer between agents

    # Q Network settings
    "network_type": "FullyCNN",  # Network architecture: 'Equivariant', 'FullyCNN', 'FCN', 'CNN'
    "periodic": False,  # Periodic boundary conditions
    "load_network": False,  # Whether to load pretrained weights
    "project_name": "TicTacToe",  # Weights & Biases project name
    "wandb_run_name": "TicTacToe",  # Weights & Biases run name
    "save_models": "/Users/jakob/TicTacToe/models/",  # Save weights after training
    "symmetrized_loss": False,  # Use symmetrized loss
    "state_shape": "one-hot",  # state representation: 'flat' with shape (batch_size, rows * rows), 
                            # '2D' with shape (batch_size, 1, rows, rows), 
                            # 'one-hot' with shape (batch_size, 3, rows, rows)
}

# --- Sweep Setup ---
# param_sweep = {"replay_buffer_type": ["prioritized", "uniform"], "periodic": [True, False], "state_shape": ["one-hot", "flat"]}
param_sweep = {"replay_buffer_type": ["prioritized", "uniform"], "symmetrized_loss": [True, False], "state_shape": ["one-hot", "flat"]}
sweep_combinations, param_keys = get_param_sweep_combinations(param_sweep)

# --- Shared Replay Buffer Setup ---
if params["shared_replay_buffer"]:
    state_shape = params["state_shape"]
    rows = params["rows"]
    if state_shape == "flat":
        shape = (rows**2,)
    elif state_shape == "2D":
        shape = (1, rows, rows)
    elif state_shape == "one-hot":
        shape = (3, rows, rows)
    else:
        raise ValueError(f"Unsupported state shape: {state_shape}")

    buffer_type = params.get("replay_buffer_type", "uniform")
    if buffer_type == "prioritized":
        params["shared_replay_buffer"] = PrioritizedReplayBuffer(
            params["replay_buffer_length"], shape, device=params["device"],
            alpha=params.get("priority_alpha", 0.6),
            beta=params.get("priority_beta", 0.4),
        )
    else:
        params["shared_replay_buffer"] = ReplayBuffer(params["replay_buffer_length"], shape, device=params["device"])


    params["shared_replay_buffer"] = ReplayBuffer(params["replay_buffer_length"], (params["rows"]**2, ), device=params["device"])


# --- Sweep Loop ---
for sweep_idx, combination in enumerate(sweep_combinations):
    print(f"Starting parameter sweep {sweep_idx + 1}/{len(sweep_combinations)}")
    for key, value in zip(param_keys, combination):
        params[key] = value

    params["wandb_run_name"] = f"{datetime.now().strftime("%Y%m%d_%H%M%S")}_{params['rows']}x{params['rows']}x{params['win_length']}_{params['network_type']}_{params['state_shape']}"
    paramsX = copy.deepcopy(params)
    paramsO = copy.deepcopy(params)
    paramsX["player"] = "X"  # Player symbol for Agent 1
    paramsO["player"] = "O"  # Player symbol for Agent 2
    paramsX["wandb_logging"] = True  # Log Agent 1 with wandb
    paramsO["wandb_logging"] = False  # Do not log Agent 2
    params["wandb_logging"] = paramsX["wandb_logging"] or paramsO["wandb_logging"]

    if params["load_network"]:
        paramsX, paramsO = load_pretrained_models(paramsX, paramsO)

    agent1 = DeepQLearningAgent(paramsX)
    agent2 = DeepQLearningAgent(paramsO)

    game = TicTacToe(
        agent1,
        agent2,
        display=None,
        params=params
    )

    try:
        train_and_evaluate(game, agent1, agent2, params)

    finally:
        if params.get("save_models"):
            save_model_artifacts(agent1, agent2, params)

    wandb.finish()
