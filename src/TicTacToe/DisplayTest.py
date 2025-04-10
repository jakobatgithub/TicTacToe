# %%
from TicTacToe.Agent import MouseAgent
from TicTacToe.DeepQAgent import DeepQPlayingAgent
from TicTacToe.Display import ScreenDisplay
from TicTacToe.TicTacToe import TicTacToe

rows = 4
win_length = 4
# agent1 = RandomAgent(player='X', switching=False)
# agent1 = HumanAgent(player='X')
agent1 = MouseAgent(player="O")
# agent2 = RandomAgent(player='O', switching=False)
# agent1 = DeepQPlayingAgent(player='X', q_network='models/q_network_4x4x4.pth')
agent2 = DeepQPlayingAgent(player="X", q_network="models/q_network_4x4x4.pth")
# display = ConsoleDisplay(rows=rows, cols=rows, waiting_time=0.5)
display = ScreenDisplay(rows=rows, cols=rows, waiting_time=0.5)
game = TicTacToe(agent1, agent2, display=display, rows=rows, cols=rows, win_length=win_length)
