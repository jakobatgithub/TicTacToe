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

def test_save_model_artifacts(dummy_agents, tmp_path):
    from TicTacToe.Utils import save_model_artifacts

    agent1, agent2 = dummy_agents

    # Set up models directory
    models_dir = tmp_path / "models"
    params = {
        "save_models": models_dir,
        "rows": 5,
        "win_length": 5,
        "shared_replay_buffer": False,
    }

    save_model_artifacts(agent1, agent2, params)

    # Check that the index file was created
    index_file = models_dir / "model_index.json"
    assert index_file.exists()

    with open(index_file) as f:
        index = json.load(f)

    assert isinstance(index, list)
    assert len(index) == 1

    entry = index[0]
    model_subdir = models_dir / entry["folder"]
    assert model_subdir.exists() and model_subdir.is_dir()

    # Check that metadata.json exists in the subdir
    metadata_file = model_subdir / "metadata.json"
    assert metadata_file.exists()

    with open(metadata_file) as f:
        metadata = json.load(f)

    assert metadata["parameters"]["rows"] == 5
    assert metadata["parameters"]["win_length"] == 5
    assert metadata["parameters"]["shared_replay_buffer"] is False

    # Check model files exist
    pth_files = list(model_subdir.glob("*.pth"))
    assert any("X" in f.name and "weights" in f.name for f in pth_files)
    assert any("O" in f.name and "weights" in f.name for f in pth_files)

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

    new_epsilon = update_exploration_rate_smoothly(agent1, agent2, params, eval_data, current_epsilon, (X_win_rates, O_win_rates), wandb_logging = False)
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
        "wandb_logging": False,
        "win_rate_deque_length": 3,
        "state_shape": "flat",
        "rewards": {
            "W": 1.0,  # Reward for a win
            "L": -1.0,  # Reward for a loss
            "D": 0.5,  # Reward for a draw
        },
    }

    mock_eval_perf.return_value = {
        "X_against_random: X wins": 0.6,
        "O_against_random: O wins": 0.5
    }

    train_and_evaluate(game, agent1, agent2, params)

    assert game.play.call_count == params["nr_of_episodes"]  # which is 11
    assert mock_eval_perf.call_count == 2
