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

Structure:
- `initialize_parameters`: Initializes and returns the training parameters.
- `setup_agents`: Sets up the agents and the game environment.
- `evaluate_agents`: Evaluates agent performance and updates exploration rate.
- `save_models`: Saves the trained models and metadata.
- `train_agents`: Main function to train the agents.

Usage:
Run the script to train the agents. Adjust parameters in the `initialize_parameters` function as needed.
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
from collections import deque

from TicTacToe.DeepQAgent import DeepQLearningAgent, FullyConvQNetwork
from TicTacToe.Evaluation import evaluate_performance
from TicTacToe.TicTacToe import TicTacToe

def initialize_parameters() -> dict[str, Any]:
    """Initialize and return the training parameters."""
    return {
        "nr_of_episodes": 1000,  # number of episodes for training
        "rows": 3,  # rows of the board, rows = cols
        "learning_rate": 0.0001,  # learning rate
        "gamma": 0.95,  # discount factor
        "switching": True,  # switch between X and O
        "win_length": 3,  # number of symbols in a row to win

        # Parameters for exploration rate epsilon
        "epsilon_start": 0.9,  # initial exploration rate
        "epsilon_min": 0.1,  # minimum exploration rate
        "set_exploration_rate_externally": True,  # flag for setting exploration rate externally
        "epsilon_update_threshold": 0.025,  # threshold for updating exploration rate
        "epsilon_update_factor": 0.99,  # factor for updating exploration rate

        # Parameters for DeepQAgent
        "batch_size": 256,  # batch size for deep learning
        "target_update_frequency": 25,  # target network update frequency
        "evaluation": True,  # save data for evaluation
        "evaluation_frequency": 25,  # frequency of evaluation
        "evaluation_batch_size": 200,  # batch size for evaluation
        "device": "mps",  # device to use, 'cpu' or 'mps' or 'cuda'
        "replay_buffer_length": 10000,  # replay buffer length
        "wandb": False,  # switch for logging with wandb.ai
        "wandb_logging_frequency": 25,  # wandb logging frequency
        "load_network": False,  # file name for loading a PyTorch network
        "shared_replay_buffer": False,  # shared replay buffer
        "network_type": "FullyCNN",  # flag for network type, 'FCN' or 'CNN' or 'Equivariant' or 'FullyCNN'
        "periodic": True,  # periodic boundary conditions
        "save_models": True,  # flag for saving models
    }

def setup_agents(params: dict[str, Any]) -> tuple[DeepQLearningAgent, DeepQLearningAgent, TicTacToe]:
    """Set up the agents and the game environment."""
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

    learning_agent1 = DeepQLearningAgent(paramsX)
    learning_agent2 = DeepQLearningAgent(paramsO)
    game = TicTacToe(
        learning_agent1, learning_agent2, display=None,
        rows=params["rows"], cols=params["rows"],
        win_length=params["win_length"], periodic=params["periodic"]
    )
    return learning_agent1, learning_agent2, game

def evaluate_agents(
    agent1: DeepQLearningAgent, agent2: DeepQLearningAgent, params: dict[str, Any],
    X_win_rates: deque, O_win_rates: deque, current_exploration_rate: float
) -> float:
    """Evaluate agent performance and update exploration rate if needed."""
    evaluation_data = evaluate_performance(
        agent1,
        agent2,
        nr_of_episodes=params["evaluation_batch_size"],
        rows=params["rows"],
        win_length=params["win_length"],
        wandb_logging=params["wandb"],
        device=params["device"],
        periodic=params["periodic"]
    )
    mode = "X_against_random:"
    X_win_rate = evaluation_data[f"{mode} X wins"]
    X_win_rates.append(X_win_rate)
    mode = "O_against_random:"
    O_win_rate = evaluation_data[f"{mode} O wins"]
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
            agent1.set_exploration_rate(current_exploration_rate)
            agent2.set_exploration_rate(current_exploration_rate)

            print(f"X_win_rates = {X_win_rates}, O_win_rates = {O_win_rates}, current_exploration_rate = {current_exploration_rate}")
    return current_exploration_rate

def save_models(
    agent1: DeepQLearningAgent, agent2: DeepQLearningAgent, params: dict[str, Any],
    model_metadata: list[dict[str, Any]]
):
    """Save the trained models and metadata."""
    script_dir = Path(__file__).resolve().parent
    all_models_folder = (script_dir / '../models/all_models').resolve()
    if not all_models_folder.exists():
        all_models_folder.mkdir(parents=True)

    metadata_file = all_models_folder / "model_metadata.json"

    model_X_filename = f"q_network_{params['rows']}x{params['rows']}x{params['win_length']}_X.pth"
    model_O_filename = f"q_network_{params['rows']}x{params['rows']}x{params['win_length']}_O.pth"

    torch.save(agent1.q_network, all_models_folder / model_X_filename) # type: ignore
    torch.save(agent2.q_network, all_models_folder / model_O_filename) # type: ignore

    model_X_filename = f"q_network_{params['rows']}x{params['rows']}x{params['win_length']}_X_weights.pth"
    model_O_filename = f"q_network_{params['rows']}x{params['rows']}x{params['win_length']}_O_weights.pth"

    torch.save(agent1.q_network.state_dict(), all_models_folder / model_X_filename) # type: ignore
    torch.save(agent2.q_network.state_dict(), all_models_folder / model_O_filename) # type: ignore

    if isinstance(agent1.q_network, FullyConvQNetwork):
        model_X_filename = f"q_network_{params['rows']}x{params['rows']}x{params['win_length']}_X_periodic_base_weights.pth"
        torch.save(agent1.q_network.base.state_dict(), all_models_folder / model_X_filename) # type: ignore
        model_X_filename = f"q_network_{params['rows']}x{params['rows']}x{params['win_length']}_X_periodic_head_weights.pth"
        torch.save(agent1.q_network.head.state_dict(), all_models_folder / model_X_filename) # type: ignore

    if isinstance(agent2.q_network, FullyConvQNetwork):
        model_O_filename = f"q_network_{params['rows']}x{params['rows']}x{params['win_length']}_O_periodic_base_weights.pth"
        torch.save(agent2.q_network.base.state_dict(), all_models_folder / model_O_filename) # type: ignore
        model_O_filename = f"q_network_{params['rows']}x{params['rows']}x{params['win_length']}_O_periodic_head_weights.pth"
        torch.save(agent2.q_network.head.state_dict(), all_models_folder / model_O_filename) # type: ignore

    model_metadata.append({
        "model_X": model_X_filename,
        "model_O": model_O_filename,
        "parameters": params.copy(),
    })

    with open(metadata_file, "w") as f:
        json.dump(model_metadata, f, indent=4)

def train_agents():
    """Main function to train the agents."""
    params = initialize_parameters()
    param_sweep = {
        "rows": [3],
        "win_length": [3],
    }
    sweep_combinations = list(product(*param_sweep.values()))
    param_keys = list(param_sweep.keys())
    model_metadata = []

    for sweep_idx, combination in enumerate(sweep_combinations):
        print(f"Starting parameter sweep {sweep_idx + 1}/{len(sweep_combinations)}")
        for key, value in zip(param_keys, combination):
            params[key] = value

        learning_agent1, learning_agent2, game = setup_agents(params)
        outcomes = {"X": 0, "O": 0, "D": 0}
        X_win_rates = deque(maxlen=10)
        O_win_rates = deque(maxlen=10)
        current_exploration_rate = params["epsilon_start"]

        try:
            for episode in tqdm(range(params["nr_of_episodes"])):
                outcome = game.play()
                if outcome is not None:
                    outcomes[outcome] += 1

                if episode > 0 and episode % params["evaluation_frequency"] == 0:
                    current_exploration_rate = evaluate_agents(
                        learning_agent1, learning_agent2, params,
                        X_win_rates, O_win_rates, current_exploration_rate
                    )

            print(f"Outcomes during learning for sweep {sweep_idx + 1}:")
            print(
                f"X wins: {outcomes['X']/params['nr_of_episodes']}, "
                f"O wins: {outcomes['O']/params['nr_of_episodes']}, "
                f"draws: {outcomes['D']/params['nr_of_episodes']}"
            )

        finally:
            if params["save_models"]:
                save_models(learning_agent1, learning_agent2, params, model_metadata)

    wandb.finish()

if __name__ == "__main__":
    train_agents()
