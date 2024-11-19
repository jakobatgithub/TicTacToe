# %%
from TicTacToe import TicTacToe
from Display import ConsoleDisplay, TicTacToeDisplay
from Agent import RandomAgent, HumanAgent, MouseAgent

rows = 4
win_length = 4
# agent1 = RandomAgent(player='X', switching=False)
# agent1 = HumanAgent(player='X', switching=False)
agent1 = MouseAgent(player='X')
agent2 = RandomAgent(player='O', switching=False)
display = ConsoleDisplay(rows=rows, cols=rows, waiting_time=0.5)
display = TicTacToeDisplay(rows=rows, cols=rows, waiting_time=0.5)
game = TicTacToe(agent1, agent2, display=display, rows=rows, cols=rows, win_length=win_length)