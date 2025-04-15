"""
This script orchestrates a series of training runs for Deep Q-Learning agents in a custom Tic Tac Toe environment.

Main features:
- Runs parameter sweeps for multiple configurations (currently rows and win_length).
- Trains two agents (X and O) using Deep Q-Learning with either independent or shared replay buffers.
- Optionally loads pretrained networks for initialization.
- Periodically evaluates both agents against random opponents and adjusts exploration rate based on stability.
- Supports detailed experiment tracking with Weights & Biases (wandb).
- Saves model weights and architecture, including periodic base/head splits for FullyConvQNetwork.
- Writes metadata to a JSON file for future reference.

Dependencies:
- TicTacToe.DeepQAgent: Defines DeepQLearningAgent and FullyConvQNetwork.
- TicTacToe.TicTacToe: Game environment supporting periodic boundaries.
- TicTacToe.Evaluation: Evaluation routines against random players.

Ensure all paths (especially model loading/saving) are valid in your project structure before executing.
"""

import copy
import json
import os
from collections import deque
from itertools import product
from pathlib import Path
from typing import Any

import torch
import wandb
from tqdm import tqdm

from TicTacToe.DeepQAgent import DeepQLearningAgent, FullyConvQNetwork
from TicTacToe.Evaluation import evaluate_performance
from TicTacToe.TicTacToe import TicTacToe

# Global configuration
params: dict[str, Any] = {
    "nr_of_episodes": 1000,
    "rows": 3,
    "learning_rate": 0.0001,
    "gamma": 0.95,
    "switching": True,
    "win_length": 3,
    "epsilon_start": 0.9,
    "epsilon_min": 0.1,
    "set_exploration_rate_externally": True,
    "epsilon_update_threshold": 0.025,
    "epsilon_update_factor": 0.99,
    "batch_size": 256,
    "target_update_frequency": 25,
    "evaluation": True,
    "evaluation_frequency": 25,
    "evaluation_batch_size": 200,
    "device": "mps",
    "replay_buffer_length": 10000,
    "wandb": False,
    "wandb_logging_frequency": 25,
    "load_network": False,
    "shared_replay_buffer": False,
    "network_type": "FullyCNN",
    "periodic": True,
    "save_models": True,
}

# Parameter sweep configuration
param_sweep = {
    "rows": [3],
    "win_length": [3],
}

sweep_combinations = list(product(*param_sweep.values()))
param_keys = list(param_sweep.keys())
model_metadata = []

# Begin sweep
for sweep_idx, combination in enumerate(sweep_combinations):
    print(f"Starting parameter sweep {sweep_idx + 1}/{len(sweep_combinations)}")

    for key, value in zip(param_keys, combination):
        params[key] = value

    rows = params["rows"]
    win_length = params["win_length"]
    nr_of_episodes = params["nr_of_episodes"]

    paramsX = copy.deepcopy(params)
    paramsO = copy.deepcopy(params)
    paramsX["player"] = "X"
    paramsO["player"] = "O"
    paramsX["wandb"] = True
    paramsO["wandb"] = False

    if params["load_network"]:
        script_dir = Path(__file__).resolve().parent
        relative_folder = (script_dir / '../models/foundational').resolve()
        model_path_X = f"{relative_folder}/q_network_5x5x5_X_weights.pth"
        model_path_O = f"{relative_folder}/q_network_5x5x5_O_weights.pth"

        paramsX["load_network"] = model_path_X
        paramsO["load_network"] = model_path_O

        print(f"Loading model from {model_path_X} and {model_path_O}")
        if not os.path.exists(model_path_X) or not os.path.exists(model_path_O):
            raise FileNotFoundError(f"Model files {model_path_X} or {model_path_O} do not exist.")

    outcomes = {"X": 0, "O": 0, "D": 0}

    learning_agent1 = DeepQLearningAgent(paramsX)
    learning_agent2 = DeepQLearningAgent(paramsO)

    game = TicTacToe(
        learning_agent1,
        learning_agent2,
        display=None,
        rows=rows,
        cols=rows,
        win_length=win_length,
        periodic=params["periodic"]
    )

    X_win_rates = deque(maxlen=10)
    O_win_rates = deque(maxlen=10)
    current_exploration_rate = params["epsilon_start"]

    try:
        for episode in tqdm(range(nr_of_episodes)):
            outcome = game.play()
            if outcome is not None:
                outcomes[outcome] += 1

            if episode > 0 and episode % params["evaluation_frequency"] == 0:
                evaluation_data = evaluate_performance(
                    learning_agent1,
                    learning_agent2,
                    nr_of_episodes=params["evaluation_batch_size"],
                    rows=rows,
                    win_length=win_length,
                    wandb_logging=paramsX["wandb"] or paramsO["wandb"],
                    device=params["device"],
                    periodic=params["periodic"]
                )
                X_win_rate = evaluation_data["X_against_random: X wins"]
                O_win_rate = evaluation_data["O_against_random: O wins"]

                X_win_rates.append(X_win_rate)
                O_win_rates.append(O_win_rate)

                if (
                    params["set_exploration_rate_externally"]
                    and len(X_win_rates) > 1
                    and len(O_win_rates) > 1
                ):
                    delta_X = abs(X_win_rates[-2] - X_win_rates[-1])
                    delta_O = abs(O_win_rates[-2] - O_win_rates[-1])

                    if delta_X < params["epsilon_update_threshold"] and delta_O < params["epsilon_update_threshold"]:
                        current_exploration_rate = max(
                            current_exploration_rate * params["epsilon_update_factor"],
                            params["epsilon_min"]
                        )
                        learning_agent1.set_exploration_rate(current_exploration_rate)
                        learning_agent2.set_exploration_rate(current_exploration_rate)

                        print(f"X_win_rates = {X_win_rates}, O_win_rates = {O_win_rates}, current_exploration_rate = {current_exploration_rate}")

        print(f"Outcomes during learning for sweep {sweep_idx + 1}:")
        print(f"X wins: {outcomes['X']/nr_of_episodes}, O wins: {outcomes['O']/nr_of_episodes}, draws: {outcomes['D']/nr_of_episodes}")

    finally:
        if params["save_models"]:
            script_dir = Path(__file__).resolve().parent
            all_models_folder = (script_dir / '../models/all_models').resolve()
            all_models_folder.mkdir(parents=True, exist_ok=True)
            metadata_file = all_models_folder / "model_metadata.json"

            model_X_full = f"q_network_{rows}x{rows}x{win_length}_X.pth"
            model_O_full = f"q_network_{rows}x{rows}x{win_length}_O.pth"
            torch.save(learning_agent1.q_network, all_models_folder / model_X_full)
            torch.save(learning_agent2.q_network, all_models_folder / model_O_full)

            model_X_weights = f"q_network_{rows}x{rows}x{win_length}_X_weights.pth"
            model_O_weights = f"q_network_{rows}x{rows}x{win_length}_O_weights.pth"
            torch.save(learning_agent1.q_network.state_dict(), all_models_folder / model_X_weights)
            torch.save(learning_agent2.q_network.state_dict(), all_models_folder / model_O_weights)

            base_X, head_X = None, None
            base_O, head_O = None, None

            if isinstance(learning_agent1.q_network, FullyConvQNetwork):
                base_X = f"q_network_{rows}x{rows}x{win_length}_X_periodic_base_weights.pth"
                head_X = f"q_network_{rows}x{rows}x{win_length}_X_periodic_head_weights.pth"
                torch.save(learning_agent1.q_network.base.state_dict(), all_models_folder / base_X)
                torch.save(learning_agent1.q_network.head.state_dict(), all_models_folder / head_X)

            if isinstance(learning_agent2.q_network, FullyConvQNetwork):
                base_O = f"q_network_{rows}x{rows}x{win_length}_O_periodic_base_weights.pth"
                head_O = f"q_network_{rows}x{rows}x{win_length}_O_periodic_head_weights.pth"
                torch.save(learning_agent2.q_network.base.state_dict(), all_models_folder / base_O)
                torch.save(learning_agent2.q_network.head.state_dict(), all_models_folder / head_O)

            model_metadata.append({
                "full_model_X": model_X_full,
                "full_model_O": model_O_full,
                "weights_X": model_X_weights,
                "weights_O": model_O_weights,
                "base_X": base_X,
                "head_X": head_X,
                "base_O": base_O,
                "head_O": head_O,
                "parameters": params.copy(),
            })

            with open(metadata_file, "w") as f:
                json.dump(model_metadata, f, indent=4)

    wandb.finish()
