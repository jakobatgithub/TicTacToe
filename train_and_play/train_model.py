"""
Deep Q-Learning Training Script for Tic Tac Toe Variants

This script trains Deep Q-Learning agents to play Tic Tac Toe on customizable grids
with optional periodic boundary conditions. It supports:

- Parameter sweeps over different game configurations
- Fully convolutional Q-networks
- Evaluation against random agents
- Adaptive exploration strategies
- Optional pretrained model loading
- Weights & artifact saving
- Weights & Biases logging

Key Classes & Dependencies:
- DeepQLearningAgent: Q-learning agent with optional FullyConvQNetwork
- TicTacToe: Environment handling gameplay with two agents
- evaluate_performance: Evaluates agents against random baselines
"""

import copy
import json
import os
import torch
import wandb

from collections import deque
from itertools import product
from pathlib import Path
from typing import Any
from tqdm import tqdm

import numpy as np

from TicTacToe.DeepQAgent import DeepQLearningAgent, FullyConvQNetwork
from TicTacToe.Evaluation import evaluate_performance
from TicTacToe.TicTacToe import TicTacToe


def get_param_sweep_combinations(param_sweep: dict) -> tuple[list[tuple[Any, ...]], list[str]]:
    """
    Generates all combinations of hyperparameter values for sweeping.

    Args:
        param_sweep (dict): Dictionary mapping parameter names to lists of values.

    Returns:
        tuple: A tuple containing:
            - List of all combinations of values as tuples
            - List of parameter keys in the same order
    """
    return list(product(*param_sweep.values())), list(param_sweep.keys())


def load_pretrained_models(paramsX: dict, paramsO: dict) -> tuple[dict, dict]:
    """
    Loads pretrained models from disk and updates the parameter dicts with their paths.

    Args:
        paramsX (dict): Parameters for player X.
        paramsO (dict): Parameters for player O.

    Returns:
        tuple: Updated (paramsX, paramsO) with model paths inserted.
    """
    script_dir = Path(__file__).resolve().parent
    relative_folder = (script_dir / '../models/foundational').resolve()
    model_path_X = f"{relative_folder}/q_network_5x5x5_X_weights.pth"
    model_path_O = f"{relative_folder}/q_network_5x5x5_O_weights.pth"

    if not os.path.exists(model_path_X) or not os.path.exists(model_path_O):
        raise FileNotFoundError(f"Model files {model_path_X} or {model_path_O} do not exist.")

    paramsX["load_network"] = model_path_X
    paramsO["load_network"] = model_path_O
    print(f"Loading model from {model_path_X} and {model_path_O}")
    return paramsX, paramsO


def save_model_artifacts(agent1, agent2, params, model_metadata):
    """
    Saves full models and weight components for both agents and appends metadata.

    Args:
        agent1: Agent playing as 'X'.
        agent2: Agent playing as 'O'.
        rows (int): Grid size.
        win_length (int): Win condition (number of in-a-row).
        params (dict): Parameter configuration.
        model_metadata (list): List to append model metadata entries to.
    """
    script_dir = Path(__file__).resolve().parent
    all_models_folder = (script_dir / '../models/all_models').resolve()
    all_models_folder.mkdir(parents=True, exist_ok=True)
    metadata_file = all_models_folder / "model_metadata.json"

    def save_agent(agent, player):
        model_name = f"q_network_{params["rows"]}x{params["rows"]}x{params["win_length"]}_{player}"
        torch.save(agent.q_network, all_models_folder / f"{model_name}.pth")  # full model
        torch.save(agent.q_network.state_dict(), all_models_folder / f"{model_name}_weights.pth")  # state dict

        base, head = None, None
        if isinstance(agent.q_network, FullyConvQNetwork):
            base = f"{model_name}_periodic_base_weights.pth"
            head = f"{model_name}_periodic_head_weights.pth"
            torch.save(agent.q_network.base.state_dict(), all_models_folder / base)
            torch.save(agent.q_network.head.state_dict(), all_models_folder / head)
        return model_name, f"{model_name}_weights.pth", base, head

    model_X, weights_X, base_X, head_X = save_agent(agent1, "X")
    model_O, weights_O, base_O, head_O = save_agent(agent2, "O")

    model_metadata.append({
        "full_model_X": f"{model_X}.pth",
        "full_model_O": f"{model_O}.pth",
        "weights_X": weights_X,
        "weights_O": weights_O,
        "base_X": base_X,
        "head_X": head_X,
        "base_O": base_O,
        "head_O": head_O,
        "parameters": params.copy(),
    })

    with open(metadata_file, "w") as f:
        json.dump(model_metadata, f, indent=4)

def update_exploration_rate(agent1, agent2, params, eval_data, exploration_rate, win_rate_deques):
    """
    Updates the exploration rate based on smoothed average of recent win rates.
    If the smoothed win rates of both agents are stable, the exploration rate is decreased.
    Args:
        agent1: Agent playing as 'X'.
        agent2: Agent playing as 'O'.
        params (dict): Parameter configuration.
        eval_data (dict): Evaluation data containing win rates.
        exploration_rate (float): Current exploration rate.
        win_rate_deques (tuple): Two deques storing recent win rates for 'X' and 'O'.
    """

    X_win_rates, O_win_rates = win_rate_deques
    X_win, O_win = eval_data["X_against_random: X wins"], eval_data["O_against_random: O wins"]
    X_win_rates.append(X_win)
    O_win_rates.append(O_win)

    if params["set_exploration_rate_externally"] and len(X_win_rates) >= 2:
        # Use smoothed (moving average) win rates
        smoothed_X = np.mean(X_win_rates)
        smoothed_O = np.mean(O_win_rates)

        # Compute smoothed deltas (between current average and previous average)
        if len(X_win_rates) >= 3:
            previous_smoothed_X = np.mean(list(X_win_rates)[:-1])
            previous_smoothed_O = np.mean(list(O_win_rates)[:-1])
        else:
            previous_smoothed_X = smoothed_X
            previous_smoothed_O = smoothed_O

        delta_X = abs(smoothed_X - previous_smoothed_X)
        delta_O = abs(smoothed_O - previous_smoothed_O)

        if delta_X < params["epsilon_update_threshold"] and delta_O < params["epsilon_update_threshold"]:
            exploration_rate = max(exploration_rate * params["epsilon_decay"], params["epsilon_min"])
            agent1.set_exploration_rate(exploration_rate)
            agent2.set_exploration_rate(exploration_rate)
            print(f"Smoothed win rates — X: {smoothed_X:.3f}, O: {smoothed_O:.3f}")
            print(f"New exploration rate: {exploration_rate:.4f}")

def train_and_evaluate(params: dict, sweep_idx: int):
    """
    Trains and evaluates two agents in a Tic Tac Toe game.
    Args:
        params (dict): Parameter configuration.
        sweep_idx (int): Index of the current parameter sweep.
    """

    for episode in tqdm(range(params["nr_of_episodes"])):
        outcome = game.play()
        if outcome:
            outcomes[outcome] += 1

        if episode > 0 and episode % params["evaluation_frequency"] == 0:
            eval_data = evaluate_performance(
                agent1,
                agent2,
                evaluation_batch_size=params["evaluation_batch_size"],
                rows=params["rows"],
                win_length=params["win_length"],
                wandb_logging=paramsX["wandb"] or paramsO["wandb"],
                device=params["device"],
                periodic=params["periodic"]
            )
            if params["set_exploration_rate_externally"]:
                update_exploration_rate(agent1, agent2, params, eval_data, exploration_rate, (X_win_rates, O_win_rates))

    print(f"Outcomes during learning for sweep {sweep_idx + 1}:")
    print(f"X wins: {outcomes['X'] / params["nr_of_episodes"]}, O wins: {outcomes['O'] / params["nr_of_episodes"]}, draws: {outcomes['D'] / params["nr_of_episodes"]}")

# --- Training Parameters ---
params: dict[str, Any] = {
    "nr_of_episodes": 2000,  # Number of training games
    "rows": 3,  # Board size (rows x rows)
    "learning_rate": 0.0001,  # Optimizer learning rate
    "gamma": 0.95,  # Discount factor for future rewards
    "switching": True,  # Whether players switch turns
    "win_length": 3,  # Number of in-a-row needed to win
    "epsilon_start": 0.925,  # Initial exploration rate
    "epsilon_min": 0.025,  # Minimum exploration rate
    "set_exploration_rate_externally": True,  # Adaptive epsilon enabled
    "epsilon_update_threshold": 0.025,  # Epsilon adjustment sensitivity
    "epsilon_decay": 0.99,  # Decay rate for epsilon
    "win_rate_deque_length": 5,  # Length of win rate deques
    "batch_size": 256,  # Batch size for training updates
    "target_update_frequency": 25,  # Frequency to sync target network
    "evaluation_frequency": 25,  # Episodes between evaluations
    "evaluation_batch_size": 200,  # Games to evaluate per round
    "device": "mps",  # Device: "cuda", "mps", or "cpu"
    "replay_buffer_length": 10000,  # Max length of replay buffer
    "wandb": False,  # Enable Weights & Biases logging
    "wandb_logging_frequency": 25,  # Logging frequency (in episodes)
    "load_network": False,  # Whether to load pretrained weights
    "shared_replay_buffer": False,  # Unused flag (placeholder)
    "network_type": "FullyCNN",  # Network architecture
    "periodic": True,  # Periodic boundary conditions
    "save_models": True,  # Save weights after training
}

# --- Sweep Setup ---
param_sweep = {"rows": [5], "win_length": [5]}
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

    outcomes = {"X": 0, "O": 0, "D": 0}
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

    X_win_rates, O_win_rates = deque(maxlen=params["win_rate_deque_length"]), deque(maxlen=params["win_rate_deque_length"])
    exploration_rate = params["epsilon_start"]

    try:
        train_and_evaluate(params, sweep_idx)

    finally:
        if params["save_models"]:
            save_model_artifacts(agent1, agent2, params, model_metadata)

    wandb.finish()
