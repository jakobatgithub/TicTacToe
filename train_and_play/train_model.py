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

from typing import Any
from tqdm import tqdm
from pathlib import Path

from TicTacToe.DeepQAgent import DeepQLearningAgent
from TicTacToe.Evaluation import evaluate_performance
from TicTacToe.TicTacToe import TicTacToe

params: dict[str, Any] = {
    "nr_of_episodes": 10000,  # number of episodes for training
    "rows": 3,  # rows of the board, rows = cols
    "epsilon_start": 0.75,  # initial exploration rate
    "epsilon_min": 0.05,  # minimum exploration rate
    "learning_rate": 0.0001,  # learning rate
    "gamma": 0.95,  # discount factor
    "switching": True,  # switch between X and O

    # Parameters for DeepQAgent
    "batch_size": 256,  # batch size for deep learning
    "target_update_frequency": 20,  # target network update frequency
    "evaluation": True,  # save data for evaluation
    "device": "cpu",  # device to use, 'cpu' or 'mps' or 'cuda'
    "replay_buffer_length": 10000,  # replay buffer length
    "wandb": False,  # switch for logging with wandb.ai
    "wandb_logging_frequency": 25,  # wandb logging frequency
    "load_network": False,  # file name for loading a PyTorch network
    "shared_replay_buffer": False,  # shared replay buffer
    "network_type": "FCN",  # flag for network type, 'FCN' or 'CNN' or 'Equivariant' or 'FullyCNN'
}

rows = 5
win_length = 3
nr_of_episodes = 1000
evaluation_frequency = 50
params["nr_of_episodes"] = nr_of_episodes
params["rows"] = rows

paramsX = copy.deepcopy(params)
paramsO = copy.deepcopy(params)
paramsX["player"] = "X"
paramsO["player"] = "O"
paramsX["wandb"] = True
paramsO["wandb"] = False

outcomes = {"X": 0, "O": 0, "D": 0}

learning_agent1 = DeepQLearningAgent(paramsX)
learning_agent2 = DeepQLearningAgent(paramsO)

game = TicTacToe(learning_agent1, learning_agent2, display=None, rows=rows, cols=rows, win_length=win_length)

try:
    for episode in tqdm(range(nr_of_episodes)):
        outcome = game.play()
        if outcome is not None:
            outcomes[outcome] += 1

        if episode > 0 and episode % evaluation_frequency == 0:
            evaluate_performance(
                learning_agent1,
                learning_agent2,
                nr_of_episodes=50,
                rows=rows,
                win_length=win_length,
                wandb_logging=paramsX["wandb"] or paramsO["wandb"],
            )

    print("Outcomes during learning:")
    print(
        f"X wins: {outcomes['X']/nr_of_episodes}, O wins: {outcomes['O']/nr_of_episodes}, draws: {outcomes['D']/nr_of_episodes}"
    )

finally:
    script_dir = Path(__file__).resolve().parent
    relative_folder = (script_dir / '../models').resolve()
    if not relative_folder.exists():
        relative_folder.mkdir(parents=True)
    
    # Save the models
    torch.save(learning_agent1.q_network, f"{relative_folder}/q_network_{rows}x{rows}x{win_length}_X.pth") # type: ignore
    torch.save(learning_agent2.q_network, f"{relative_folder}/q_network_{rows}x{rows}x{win_length}_O.pth") # type: ignore

    wandb.finish()
