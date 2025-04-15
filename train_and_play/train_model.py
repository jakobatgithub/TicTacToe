# %%
# Let the games begin

"""
This script trains two Deep Q-Learning agents to play Tic-Tac-Toe against each other.

Features:
- Configurable training parameters such as learning rate, epsilon decay, and replay buffer size.
- Periodic evaluation of agent performance.
- Saves the trained Q-networks to disk.

Modules:
- TicTacToe.DeepQAgent: Defines the DeepQLearningAgent class.
- TicTacToe.Evaluation: Provides evaluation methods.
- TicTacToe.TicTacToe: Defines the game logic.

Usage:
Run the script to train the agents. Adjust parameters in the `params` dictionary as needed.
"""

import copy
import wandb
import torch
import json
import os

from typing import Any
from tqdm import tqdm
from pathlib import Path
from itertools import product

from TicTacToe.DeepQAgent import DeepQLearningAgent
from TicTacToe.Evaluation import evaluate_performance
from TicTacToe.TicTacToe import TicTacToe

params: dict[str, Any] = {
    "nr_of_episodes": 1000,  # number of episodes for training
    "rows": 3,  # rows of the board, rows = cols
    "epsilon_start": 0.95,  # initial exploration rate
    # "epsilon_min": 0.05,  # minimum exploration rate
    "epsilon_min": 0.1,  # minimum exploration rate
    "learning_rate": 0.0001,  # learning rate
    "gamma": 0.95,  # discount factor
    "switching": True,  # switch between X and O
    "win_length": 3,  # number of symbols in a row to win

    # Parameters for DeepQAgent
    "batch_size": 256,  # batch size for deep learning
    "target_update_frequency": 25,  # target network update frequency
    "evaluation": True,  # save data for evaluation
    "evaluation_frequency": 20,  # frequency of evaluation
    "evaluation_batch_size": 200,  # batch size for evaluation
    "device": "mps",  # device to use, 'cpu' or 'mps' or 'cuda'
    "replay_buffer_length": 10000,  # replay buffer length
    "wandb": False,  # switch for logging with wandb.ai
    "wandb_logging_frequency": 25,  # wandb logging frequency
    "load_network": True,  # file name for loading a PyTorch network
    "shared_replay_buffer": False,  # shared replay buffer
    "network_type": "FullyCNN",  # flag for network type, 'FCN' or 'CNN' or 'Equivariant' or 'FullyCNN'
    "periodic": True,  # periodic boundary conditions
    "save_models": True # flag for saving models
}

# Define parameter sweep ranges
param_sweep = {
    "rows": [5],
    "win_length": [5],
    # "rows": [3, 5],
    # "win_length": [3, 4],
    # "learning_rate": [0.0001, 0.001],
    # "gamma": [0.9, 0.95],
}

# Generate all combinations of parameter values
sweep_combinations = list(product(*param_sweep.values()))

# Map parameter names to their respective indices in the combinations
param_keys = list(param_sweep.keys())

# Create a list to store filenames and corresponding parameter values
model_metadata = []

# Iterate over each parameter combination
for sweep_idx, combination in enumerate(sweep_combinations):
    print(f"Starting parameter sweep {sweep_idx + 1}/{len(sweep_combinations)}")

    # Update parameters with the current combination
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

    game = TicTacToe(learning_agent1, learning_agent2, display=None, rows=rows, cols=rows, win_length=win_length, periodic=params["periodic"])

    try:
        for episode in tqdm(range(nr_of_episodes)):
            outcome = game.play()
            if outcome is not None:
                outcomes[outcome] += 1

            if episode > 0 and episode % params["evaluation_frequency"] == 0:
                evaluate_performance(
                    learning_agent1,
                    learning_agent2,
                    nr_of_episodes=params["evaluation_batch_size"],
                    rows=rows,
                    win_length=win_length,
                    wandb_logging=paramsX["wandb"] or paramsO["wandb"],
                    device = params["device"],
                    periodic=params["periodic"]
                )

        print(f"Outcomes during learning for sweep {sweep_idx + 1}:")
        print(
            f"X wins: {outcomes['X']/nr_of_episodes}, O wins: {outcomes['O']/nr_of_episodes}, draws: {outcomes['D']/nr_of_episodes}"
        )

    finally:
        if params["save_models"]:
            
            # Define a folder to save all models
            script_dir = Path(__file__).resolve().parent
            all_models_folder = (script_dir / '../models/all_models').resolve()
            if not all_models_folder.exists():
                all_models_folder.mkdir(parents=True)

            # Save the metadata to a JSON file after every iteration
            metadata_file = all_models_folder / "model_metadata.json"

            # Save the models with recognizable filenames
            model_X_filename = f"q_network_{rows}x{rows}x{win_length}_X.pth"
            model_O_filename = f"q_network_{rows}x{rows}x{win_length}_O.pth"

            torch.save(learning_agent1.q_network, all_models_folder / model_X_filename) # type: ignore
            torch.save(learning_agent2.q_network, all_models_folder / model_O_filename) # type: ignore
            
            print(f"Models saved as {model_X_filename} and {model_O_filename}")

            # Save the models weights with recognizable filenames
            model_X_filename = f"q_network_{rows}x{rows}x{win_length}_X_weights.pth"
            model_O_filename = f"q_network_{rows}x{rows}x{win_length}_O_weights.pth"

            torch.save(learning_agent1.q_network.state_dict(), all_models_folder / model_X_filename) # type: ignore
            torch.save(learning_agent2.q_network.state_dict(), all_models_folder / model_O_filename) # type: ignore

            # Append metadata to the list
            model_metadata.append({
                "model_X": model_X_filename,
                "model_O": model_O_filename,
                "parameters": params.copy(),
            })

            # Save metadata to file after every iteration
            with open(metadata_file, "w") as f:
                json.dump(model_metadata, f, indent=4)

    wandb.finish()
