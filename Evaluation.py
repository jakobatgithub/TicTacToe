import numpy as np
import matplotlib.pyplot as plt

from Agent import RandomAgent
from QAgent import QPlayingAgent
from TicTacToe import TicTacToe


def average_array(array, chunk_size=None):
    means = []
    if chunk_size is None:
        chunk_size = max((len(array) // 100, 1))

    for i in range(0, len(array), chunk_size):
        sublist = array[i:i + chunk_size]
        means.append(sum(sublist)/len(sublist))

    return means

def plot_graphs(loss, action_value, rewards):
    # Create a figure with two subplots next to each other
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))  # 1 row, 2 columns
    
    # Plot the first graph: Mean Loss
    chunk_size = max((len(loss) // 100, 1))
    mean_lossX = average_array(loss)
    steps =[i * chunk_size for i in range(len(mean_lossX))]

    axs[0].plot(steps[:-2], mean_lossX[:-2], label='Mean Loss')
    axs[0].set_title(f'Mean loss')
    axs[0].set_xlabel('Training steps')
    axs[0].set_ylabel('Loss')
    axs[0].grid(True)

    # Plot the second graph: Mean Average Action Value
    chunk_size = max((len(action_value) // 100, 1))
    mean_action_valueX = average_array(action_value)
    steps =[i * chunk_size for i in range(len(mean_lossX))]

    axs[1].plot(steps[:-2], mean_action_valueX[:-2], label='Mean Avg Action Value', color='orange')
    axs[1].set_title(f'Mean action value')
    axs[1].set_xlabel('Training steps')
    axs[1].set_ylabel('Action value')
    axs[1].grid(True)

    chunk_size = max((len(rewards) // 100, 1))
    mean_rewards = average_array(rewards, chunk_size)
    steps = [i * chunk_size for i in range(len(mean_rewards))]

    axs[2].plot(steps[:-2], mean_rewards[:-2], label='Mean Rewards', color='blue')
    axs[2].set_title(f'Mean reward per episode')
    axs[2].set_xlabel('Episodes')
    axs[2].set_ylabel('Reward')
    axs[2].grid(True)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

def plot_valid_actions(learning_agent):
    evaluation_data = learning_agent.evaluation_data
    valid_actions = evaluation_data['valid_actions']
    chunk_size = max((len(valid_actions) // 100, 1))
    mean_valid_actions = average_array(valid_actions)
    steps =[i * chunk_size for i in range(len(mean_valid_actions))]
    fig, axs = plt.subplots(1, 1, figsize=(4, 3))  # 1 row, 2 columns
    
    axs.plot(steps[:-2], mean_valid_actions[:-2], label='Valid actions')
    axs.set_title(f'Valid actions')
    axs.set_xlabel('Training steps')
    axs.set_ylabel('Fraction of valid actions')
    axs.grid(True)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

def plot_evaluation_data(learning_agent):
    evaluation_data = learning_agent.evaluation_data
    loss = evaluation_data['loss']
    action_value = evaluation_data['action_value']
    rewards = evaluation_data['rewards']
    print(f"Number of losses: {len(loss)}")
    print(f"Number of action values: {len(action_value)}")
    print(f"Number of rewards: {len(rewards)}")
    plot_graphs(loss, action_value, rewards)

def extract_values(dictionary):
    """Extract all values from a potentially nested dictionary."""
    values = []
    for key, value in dictionary.items():
        if isinstance(value, dict):  # If the value is a dictionary, recurse
            values.extend(extract_values(value))
        else:
            values.append(value)
    return values

def evaluate_and_plot_Q(learning_agent, player):
    Q = learning_agent.Q
    qMatrix = Q.get()
    qValues = extract_values(qMatrix)
    print(qValues)
    print(f"Total number of elements in Q for player {player}: {len(qValues)}")
    
    mean_q = np.mean(qValues)
    median_q = np.median(qValues)
    std_q = np.std(qValues)
    min_q = np.min(qValues)
    max_q = np.max(qValues)

    print(f"Q-value Statistics for player {player}:")
    print(f"Mean: {mean_q}")
    print(f"Median: {median_q}")
    print(f"Standard Deviation: {std_q}")
    print(f"Minimum: {min_q}")
    print(f"Maximum: {max_q}")

    plt.figure(figsize=(10, 6))
    plt.hist(qValues, bins=20, edgecolor='black', alpha=0.7)
    plt.title(f"Histogram of Q-values for player {player}")
    plt.xlabel("Q-value")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def QAgent_plays_against_RandomAgent(Q, player, nr_of_episodes=5000, width=3, height=3, win_length=3):
    playing_agent1 = QPlayingAgent(Q, player=player, switching=False)
    opponent = 'O' if player == 'X' else 'X'
    random_agent1 = RandomAgent(player=opponent, switching=False)
    game = TicTacToe(playing_agent1, random_agent1, display=None, width=width, height=height, win_length=win_length)
    outcomes = {'X' : 0, 'O' : 0, 'D' : 0}
    for _ in range(nr_of_episodes):
        outcome = game.play()
        outcomes[outcome] += 1

    print("Outcomes during playing:")
    print(f"X wins: {outcomes['X']/nr_of_episodes}, O wins: {outcomes['O']/nr_of_episodes}, draws: {outcomes['D']/nr_of_episodes}")

def QAgent_plays_against_QAgent(Q1, player1, Q2, player2=None, nr_of_episodes=5000, width=3, height=3, win_length=3):
    playing_agent1 = QPlayingAgent(Q1, player=player1, switching=False)
    if not player2:
        player2 = 'O' if player1 == 'X' else 'X'

    playing_agent2 = QPlayingAgent(Q2, player=player2, switching=False)
    game = TicTacToe(playing_agent1, playing_agent2, display=None, width=width, height=height, win_length=win_length)
    outcomes = {'X' : 0, 'O' : 0, 'D' : 0}
    for episode in range(nr_of_episodes):
        outcome = game.play()
        outcomes[outcome] += 1

    print("Outcomes during playing:")
    print(f"X wins: {outcomes['X']/nr_of_episodes}, O wins: {outcomes['O']/nr_of_episodes}, draws: {outcomes['D']/nr_of_episodes}")