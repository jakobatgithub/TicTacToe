import json
import os
import torch

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
        param_sweep (dict): Dictionary mapping parameter names to lists of values.

    Returns:
        tuple: A tuple containing:
            - List of all combinations of parameter values as tuples.
            - List of parameter keys in the same order as the combinations.
    """
    return list(product(*param_sweep.values())), list(param_sweep.keys())


def load_pretrained_models(paramsX: dict, paramsO: dict) -> tuple[dict, dict]:
    """
    Load pretrained models from disk and update parameter dictionaries with model paths.

    Args:
        paramsX (dict): Parameters for player X.
        paramsO (dict): Parameters for player O.

    Returns:
        tuple: Updated (paramsX, paramsO) with paths to pretrained models.
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


def save_model_artifacts(agent1: Agent, agent2: Agent, params: dict, model_metadata):
    """
    Save full models and weight components for both agents, and append metadata.

    Args:
        agent1 (Agent): Agent playing as 'X'.
        agent2 (Agent): Agent playing as 'O'.
        params (dict): Parameter configuration.
        model_metadata (list): List to append model metadata entries to.
    """
    script_dir = Path(__file__).resolve().parent
    all_models_folder = (script_dir / '../models/all_models').resolve()
    all_models_folder.mkdir(parents=True, exist_ok=True)
    metadata_file = all_models_folder / "model_metadata.json"

    def save_agent(agent, player):
        model_name = f"q_network_{params['rows']}x{params['rows']}x{params['win_length']}_{player}"
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
    if params["shared_replay_buffer"]:
        params["shared_replay_buffer"] = True

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

def update_exploration_rate_smoothly(agent1: DeepQLearningAgent, agent2: DeepQLearningAgent, params: dict, eval_data: dict, exploration_rate: float, win_rate_deques: tuple[deque, deque]):
    """
    Update the exploration rate based on smoothed averages of recent win rates.

    Args:
        agent1 (DeepQLearningAgent): Agent playing as 'X'.
        agent2 (DeepQLearningAgent): Agent playing as 'O'.
        params (dict): Parameter configuration.
        eval_data (dict): Evaluation data containing win rates.
        exploration_rate (float): Current exploration rate.
        win_rate_deques (tuple): Two deques storing recent win rates for 'X' and 'O'.

    Returns:
        float: Updated exploration rate.
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

    return exploration_rate

def train_and_evaluate(game: TwoPlayerBoardGame, agent1: DeepQLearningAgent, agent2: DeepQLearningAgent, params: dict, wandb_logging: bool = True):
    """
    Train and evaluate two agents in a Tic Tac Toe game.

    Args:
        game (TwoPlayerBoardGame): The game environment.
        agent1 (DeepQLearningAgent): Agent playing as 'X'.
        agent2 (DeepQLearningAgent): Agent playing as 'O'.
        params (dict): Parameter configuration.
        wandb_logging (bool): Whether to log to Weights & Biases.
    """

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
                evaluation_batch_size=params["evaluation_batch_size"],
                rows=params["rows"],
                win_length=params["win_length"],
                wandb_logging=wandb_logging,
                device=params["device"],
                periodic=params["periodic"],
                state_shape=params["state_shape"],
            )
            if params["set_exploration_rate_externally"]:
                exploration_rate = update_exploration_rate_smoothly(agent1, agent2, params, eval_data, exploration_rate, (X_win_rates, O_win_rates))

    print("Outcomes during learning:")
    print(f"X wins: {outcomes['X'] / params['nr_of_episodes']}, O wins: {outcomes['O'] / params['nr_of_episodes']}, draws: {outcomes['D'] / params['nr_of_episodes']}")
