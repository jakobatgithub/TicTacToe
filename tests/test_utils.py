# Place this in test_utils.py

import json
import pytest
import torch

from collections import deque
from pathlib import Path
from unittest.mock import patch, MagicMock

from TicTacToe.Agent import Agent
from TicTacToe.DeepQAgent import DeepQLearningAgent
from TicTacToe.Utils import (
    get_param_sweep_combinations,
    load_pretrained_models,
    save_model_artifacts,
    update_exploration_rate_smoothly,
    train_and_evaluate
)

# -------- get_param_sweep_combinations --------

def test_get_param_sweep_combinations():
    param_sweep = {
        "lr": [0.001, 0.01],
        "batch_size": [32, 64]
    }
    combos, keys = get_param_sweep_combinations(param_sweep)
    assert len(combos) == 4
    assert keys == ["lr", "batch_size"]
    assert (0.001, 32) in combos

# -------- load_pretrained_models --------

@patch("TicTacToe.Utils.os.path.exists", return_value=True)
@patch("TicTacToe.Utils.Path")
def test_load_pretrained_models(mock_path, mock_exists):
    mock_script_path = MagicMock()
    mock_path.return_value.resolve.return_value.parent = mock_script_path
    mock_script_path.__truediv__.return_value.resolve.return_value = Path("/fake/models/foundational")

    paramsX = {}
    paramsO = {}
    updatedX, updatedO = load_pretrained_models(paramsX, paramsO)

    assert "load_network" in updatedX
    assert updatedX["load_network"].endswith("X_weights.pth")
    assert updatedO["load_network"].endswith("O_weights.pth")

# -------- save_model_artifacts --------

class DummyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(1, 1)
        self.head = torch.nn.Linear(1, 1)

@pytest.fixture
def dummy_agents():
    agent1 = MagicMock(spec=Agent)
    agent2 = MagicMock(spec=Agent)
    agent1.q_network = DummyNetwork()
    agent2.q_network = DummyNetwork()
    return agent1, agent2

@patch("TicTacToe.Utils.Path")
def test_save_model_artifacts(mock_path_class, dummy_agents, tmp_path):
    agent1, agent2 = dummy_agents

    # Prepare mock paths
    models_dir = tmp_path / "models" / "all_models"
    models_dir.mkdir(parents=True)

    mock_script_dir = MagicMock()
    mock_path_class.return_value.resolve.return_value.parent = mock_script_dir
    mock_script_dir.__truediv__.return_value.resolve.return_value = models_dir

    params = {"rows": 5, "win_length": 5, "shared_replay_buffer": False}
    metadata = []

    save_model_artifacts(agent1, agent2, params, metadata)

    saved_files = list(models_dir.glob("*.pth"))
    assert len(saved_files) >= 4  # full + weights for each
    assert (models_dir / "model_metadata.json").exists()

    with open(models_dir / "model_metadata.json") as f:
        loaded = json.load(f)
        assert loaded[0]["parameters"] == params

# -------- update_exploration_rate_smoothly --------

def test_update_exploration_rate_smoothly():
    agent1 = MagicMock(spec=DeepQLearningAgent)
    agent2 = MagicMock(spec=DeepQLearningAgent)

    params = {
        "epsilon_update_threshold": 0.05,
        "epsilon_decay": 0.9,
        "epsilon_min": 0.1
    }

    eval_data = {
        "X_against_random: X wins": 0.6,
        "O_against_random: O wins": 0.5
    }

    X_win_rates = deque([0.6, 0.61], maxlen=3)
    O_win_rates = deque([0.5, 0.51], maxlen=3)
    current_epsilon = 0.5

    new_epsilon = update_exploration_rate_smoothly(agent1, agent2, params, eval_data, current_epsilon, (X_win_rates, O_win_rates))
    assert new_epsilon < current_epsilon
    agent1.set_exploration_rate.assert_called()
    agent2.set_exploration_rate.assert_called()

# -------- train_and_evaluate --------

@patch("TicTacToe.Utils.evaluate_performance")
@patch("TicTacToe.Utils.tqdm", side_effect=lambda x: x)  # skip progress bar
def test_train_and_evaluate(mock_tqdm, mock_eval_perf):
    agent1 = MagicMock(spec=DeepQLearningAgent)
    agent2 = MagicMock(spec=DeepQLearningAgent)
    game = MagicMock()
    game.play.side_effect = lambda: "X"

    params = {
        "nr_of_episodes": 11,
        "evaluation_frequency": 5,
        "evaluation_batch_size": 4,
        "rows": 3,
        "win_length": 3,
        "epsilon_start": 0.8,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.9,
        "epsilon_update_threshold": 0.05,
        "set_exploration_rate_externally": True,
        "device": "cpu",
        "periodic": False,
        "win_rate_deque_length": 3,
        "state_shape": "flat",
    }

    mock_eval_perf.return_value = {
        "X_against_random: X wins": 0.6,
        "O_against_random: O wins": 0.5
    }

    train_and_evaluate(game, agent1, agent2, params, wandb_logging=False)

    assert game.play.call_count == params["nr_of_episodes"]  # which is 11
    assert mock_eval_perf.call_count == 2
