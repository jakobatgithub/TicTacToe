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