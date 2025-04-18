import json
import os
import torch
import wandb

from datetime import datetime
from collections import deque
from itertools import product
from pathlib import Path
from typing import Any
from tqdm import tqdm

import numpy as np

from TicTacToe.Agent import Agent
from TicTacToe.TicTacToe import TwoPlayerBoardGame
from TicTacToe.DeepQAgent import DeepQLearningAgent, FullyConvQNetwork
from TicTacToe.Evaluation import evaluate_performance

def get_param_sweep_combinations(param_sweep: dict) -> tuple[list[tuple[Any, ...]], list[str]]:
    """
    Generate all combinations of hyperparameter values for parameter sweeping.

    Args:
        param_sweep (dict): A dictionary where keys are parameter names and values are lists of possible values.

    Returns:
        tuple: A tuple containing:
            - A list of all combinations of parameter values as tuples.
            - A list of parameter keys in the same order as the combinations.
    """
    return list(product(*param_sweep.values())), list(param_sweep.keys())


def load_pretrained_models(paramsX: dict, paramsO: dict) -> tuple[dict, dict]:
    """
    Load pretrained models for players X and O from disk and update their parameter dictionaries.

    Args:
        paramsX (dict): Parameter dictionary for player X.
        paramsO (dict): Parameter dictionary for player O.

    Returns:
        tuple: A tuple containing updated parameter dictionaries for players X and O.
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


def save_model_artifacts(agent1: Agent, agent2: Agent, params: dict):
    """
    Save model artifacts for both agents, including full models, weights, and metadata.

    Args:
        agent1 (Agent): The agent playing as 'X'.
        agent2 (Agent): The agent playing as 'O'.
        params (dict): Configuration parameters for saving models.
    """
    # Prepare folders
    base_folder = Path(params["save_models"]).resolve()
    base_folder.mkdir(parents=True, exist_ok=True)

    # Create unique model folder
    wandb_run_name = params["wandb_run_name"]
    model_folder = base_folder / wandb_run_name
    model_folder.mkdir()

    def save_agent(agent, player):
        base_name = f"{params['rows']}x{params['rows']}x{params['win_length']}_{player}"
        full_model_file = f"q_network_{base_name}.pth"
        weights_file = f"q_network_{base_name}_weights.pth"

        torch.save(agent.q_network, model_folder / full_model_file)
        torch.save(agent.q_network.state_dict(), model_folder / weights_file)

        base, head = None, None
        if isinstance(agent.q_network, FullyConvQNetwork):
            base = f"q_network_{base_name}_periodic_base_weights.pth"
            head = f"q_network_{base_name}_periodic_head_weights.pth"
            torch.save(agent.q_network.base.state_dict(), model_folder / base)
            torch.save(agent.q_network.head.state_dict(), model_folder / head)

        return full_model_file, weights_file, base, head

    full_X, weights_X, base_X, head_X = save_agent(agent1, "X")
    full_O, weights_O, base_O, head_O = save_agent(agent2, "O")

    metadata = {
        "id": wandb_run_name,
        "timestamp": datetime.now().isoformat(),
        "model_folder": wandb_run_name,
        "full_model_X": full_X,
        "full_model_O": full_O,
        "weights_X": weights_X,
        "weights_O": weights_O,
        "base_X": base_X,
        "head_X": head_X,
        "base_O": base_O,
        "head_O": head_O,
        "parameters": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in params.items()
            }
        }

    # Save per-model metadata
    with open(model_folder / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    # Update central index
    index_file = base_folder / "model_index.json"
    if index_file.exists():
        with open(index_file, "r") as f:
            index_data = json.load(f)
    else:
        index_data = []

    # Append a summary entry (you can include more fields if needed)
    index_data.append({
        "id": wandb_run_name,
        "timestamp": metadata["timestamp"],
        "folder": wandb_run_name,
        "rows": params["rows"],
        "win_length": params["win_length"],
    })

    with open(index_file, "w") as f:
        json.dump(index_data, f, indent=4)

def update_exploration_rate_smoothly(agent1: DeepQLearningAgent, agent2: DeepQLearningAgent, params: dict, eval_data: dict, exploration_rate: float, win_rate_deques: tuple[deque, deque], wandb_logging=True):
    """
    Smoothly update the exploration rate based on recent win rates.

    Args:
        agent1 (DeepQLearningAgent): The agent playing as 'X'.
        agent2 (DeepQLearningAgent): The agent playing as 'O'.
        params (dict): Configuration parameters.
        eval_data (dict): Evaluation data containing win rates.
        exploration_rate (float): Current exploration rate.
        win_rate_deques (tuple): Deques storing recent win rates for 'X' and 'O'.
        wandb_logging (bool): Whether to log metrics to Weights & Biases.

    Returns:
        float: The updated exploration rate.
    """

    X_win_rates, O_win_rates = win_rate_deques
    X_win, O_win = eval_data["X_against_random: X wins"], eval_data["O_against_random: O wins"]
    X_win_rates.append(X_win)
    O_win_rates.append(O_win)

    if len(X_win_rates) >= 2:
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
            print(f"Smoothed win rates: X: {smoothed_X:.3f}, O: {smoothed_O:.3f}")
            print(f"delta_X = {delta_X:.3f}, delta_O = {delta_X:.3f}")
            print(f"New exploration rate: {exploration_rate:.4f}")

        data = {
            "smoothed_X_win_rate": smoothed_X,
            "smoothed_O_win_rate": smoothed_O,
            "delta_X": delta_X,
            "delta_O": delta_O,
        }
        if wandb_logging:
            wandb.log(data)

    return exploration_rate

def train_and_evaluate(game: TwoPlayerBoardGame, agent1: DeepQLearningAgent, agent2: DeepQLearningAgent, params: dict):
    """
    Train and evaluate two agents in a Tic Tac Toe game environment.

    Args:
        game (TwoPlayerBoardGame): The game environment instance.
        agent1 (DeepQLearningAgent): The agent playing as 'X'.
        agent2 (DeepQLearningAgent): The agent playing as 'O'.
        params (dict): Configuration parameters for training and evaluation.
    """

    wandb_logging = params["wandb_logging"]

    outcomes = {"X": 0, "O": 0, "D": 0}
    X_win_rates, O_win_rates = deque(maxlen=params["win_rate_deque_length"]), deque(maxlen=params["win_rate_deque_length"])
    exploration_rate = params["epsilon_start"]

    for episode in tqdm(range(params["nr_of_episodes"])):
        outcome = game.play()
        if outcome:
            outcomes[outcome] += 1

        if episode > 0 and episode % params["evaluation_frequency"] == 0:
            eval_data = evaluate_performance(
                agent1,
                agent2,
                params
            )
            if params["set_exploration_rate_externally"]:
                exploration_rate = update_exploration_rate_smoothly(agent1, agent2, params, eval_data, exploration_rate, (X_win_rates, O_win_rates), wandb_logging=wandb_logging)

    print("Outcomes during learning:")
    print(f"X wins: {outcomes['X'] / params['nr_of_episodes']}, O wins: {outcomes['O'] / params['nr_of_episodes']}, draws: {outcomes['D'] / params['nr_of_episodes']}")
