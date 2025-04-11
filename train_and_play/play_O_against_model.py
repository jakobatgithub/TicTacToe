"""
This script allows a human player to play Tic-Tac-Toe as 'O' against a Deep Q-Learning agent playing as 'X'.

Modules:
- TicTacToe.TicTacToe: Defines the game logic.
- TicTacToe.DeepQAgent: Defines the DeepQPlayingAgent class.
- TicTacToe.Agent: Defines the HumanAgent class.
- TicTacToe.Display: Handles the game display.

Usage:
Run the script to start the game. The human player will make moves as 'O', and the agent will respond as 'X'.
"""

# import torch
from TicTacToe.TicTacToe import TicTacToe
from TicTacToe.DeepQAgent import DeepQPlayingAgent
from TicTacToe.Agent import HumanAgent
from TicTacToe.Display import ScreenDisplay

# Load the model
model_path = "./models/q_network_3x3x3_X.pth"  # Change this path to the desired model

# Set up the game
rows = 3
win_length = 3
agent1 = HumanAgent(player="O")
agent2 = DeepQPlayingAgent(q_network=model_path, player="X")
display = ScreenDisplay(rows=rows, cols=rows, waiting_time=0.5)

game = TicTacToe(agent1, agent2, display=display, rows=rows, cols=rows, win_length=win_length)

# Play the game
game.play()