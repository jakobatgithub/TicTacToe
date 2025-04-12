"""
This script allows a human player to play Tic-Tac-Toe as 'X' against a Deep Q-Learning agent playing as 'O'.

Modules:
- TicTacToe.TicTacToe: Defines the game logic.
- TicTacToe.DeepQAgent: Defines the DeepQPlayingAgent class.
- TicTacToe.Agent: Defines the HumanAgent class.
- TicTacToe.Display: Handles the game display.

Usage:
Run the script to start the game. The human player will make moves as 'X', and the agent will respond as 'O'.
"""

from pathlib import Path

from TicTacToe.TicTacToe import TicTacToe
from TicTacToe.DeepQAgent import DeepQPlayingAgent
from TicTacToe.Agent import HumanAgent
from TicTacToe.Display import ScreenDisplay

# Load the model
script_dir = Path(__file__).resolve().parent
relative_folder = (script_dir / '../models/all_models').resolve()
model_path = f"{relative_folder}/q_network_3x3x3_X.pth"  # Change this path to the desired model

# Set up the game
rows = 3
win_length = 3
agent1 = DeepQPlayingAgent(q_network=model_path, player="O")
agent2 = HumanAgent(player="X")
display = ScreenDisplay(rows=rows, cols=rows, waiting_time=0.5)

game = TicTacToe(agent1, agent2, display=display, rows=rows, cols=rows, win_length=win_length, periodic=True)

# Play the game
game.play()