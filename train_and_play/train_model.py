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


def get_param_sweep_combinations(param_sweep: dict) -> tuple[list[tuple[Any, ...]], list[str]]:
    return list(product(*param_sweep.values())), list(param_sweep.keys())


def load_pretrained_models(paramsX: dict, paramsO: dict) -> tuple[dict, dict]:
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


def save_model_artifacts(agent1, agent2, rows, win_length, params, model_metadata):
    script_dir = Path(__file__).resolve().parent
    all_models_folder = (script_dir / '../models/all_models').resolve()
    all_models_folder.mkdir(parents=True, exist_ok=True)
    metadata_file = all_models_folder / "model_metadata.json"

    def save_agent(agent, player):
        model_name = f"q_network_{rows}x{rows}x{win_length}_{player}"
        torch.save(agent.q_network, all_models_folder / f"{model_name}.pth")
        torch.save(agent.q_network.state_dict(), all_models_folder / f"{model_name}_weights.pth")

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


params: dict[str, Any] = {
    "nr_of_episodes": 1000,
    "rows": 3,
    "learning_rate": 0.0001,
    "gamma": 0.95,
    "switching": True,
    "win_length": 3,
    "epsilon_start": 0.925,
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

param_sweep = {"rows": [3], "win_length": [3]}
sweep_combinations, param_keys = get_param_sweep_combinations(param_sweep)
model_metadata = []

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
        paramsX, paramsO = load_pretrained_models(paramsX, paramsO)

    outcomes = {"X": 0, "O": 0, "D": 0}
    agent1 = DeepQLearningAgent(paramsX)
    agent2 = DeepQLearningAgent(paramsO)

    game = TicTacToe(agent1, agent2, display=None, rows=rows, cols=rows, win_length=win_length, periodic=params["periodic"])

    X_win_rates, O_win_rates = deque(maxlen=10), deque(maxlen=10)
    exploration_rate = params["epsilon_start"]

    try:
        for episode in tqdm(range(nr_of_episodes)):
            outcome = game.play()
            if outcome:
                outcomes[outcome] += 1

            if episode > 0 and episode % params["evaluation_frequency"] == 0:
                eval_data = evaluate_performance(
                    agent1,
                    agent2,
                    nr_of_episodes=params["evaluation_batch_size"],
                    rows=rows,
                    win_length=win_length,
                    wandb_logging=paramsX["wandb"] or paramsO["wandb"],
                    device=params["device"],
                    periodic=params["periodic"]
                )

                X_win, O_win = eval_data["X_against_random: X wins"], eval_data["O_against_random: O wins"]
                X_win_rates.append(X_win)
                O_win_rates.append(O_win)

                if params["set_exploration_rate_externally"] and len(X_win_rates) > 1:
                    delta_X = abs(X_win_rates[-2] - X_win_rates[-1])
                    delta_O = abs(O_win_rates[-2] - O_win_rates[-1])

                    if delta_X < params["epsilon_update_threshold"] and delta_O < params["epsilon_update_threshold"]:
                        exploration_rate = max(exploration_rate * params["epsilon_update_factor"], params["epsilon_min"])
                        agent1.set_exploration_rate(exploration_rate)
                        agent2.set_exploration_rate(exploration_rate)
                        print(f"X_win_rates = {X_win_rates}, O_win_rates = {O_win_rates}, current_exploration_rate = {exploration_rate}")

        print(f"Outcomes during learning for sweep {sweep_idx + 1}:")
        print(f"X wins: {outcomes['X']/nr_of_episodes}, O wins: {outcomes['O']/nr_of_episodes}, draws: {outcomes['D']/nr_of_episodes}")

    finally:
        if params["save_models"]:
            save_model_artifacts(agent1, agent2, rows, win_length, params, model_metadata)

    wandb.finish()
