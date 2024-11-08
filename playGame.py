import random
import time

from IPython.display import clear_output

# random.seed(42)  # Set the random seed

# Initialize the Tic-Tac-Toe board
def initialize_board():
    return [' ' for _ in range(9)]

# Generate all empty positions on the board
def get_empty_positions(board):
    return [i for i, cell in enumerate(board) if cell == ' ']

# Display the board in a 3x3 format
def display_board(board):
    clear_output(wait=True)
    print("\n")
    print(f" {board[0]}  |  {board[1]}  |  {board[2]} ")
    print("----+-----+----")
    print(f" {board[3]}  |  {board[4]}  |  {board[5]} ")
    print("----+-----+----")
    print(f" {board[6]}  |  {board[7]}  |  {board[8]} ")
    print("\n")

# Display the board in a 3x3 format with Q-values on empty fields
def display_board_with_Q(board, Q, Q_initial_value):
    clear_output(wait=True)
    field = {i: board[i] if board[i] != ' ' else Q.get(board, i) - Q_initial_value for i in range(9)}
    def format_cell(value):
        return f"{value:.1f}" if isinstance(value, (int, float)) else f" {value} "

    print("\n")
    print(f"{format_cell(field[0])} | {format_cell(field[1])} | {format_cell(field[2])}")
    print("----+-----+----")
    print(f"{format_cell(field[3])} | {format_cell(field[4])} | {format_cell(field[5])}")
    print("----+-----+----")
    print(f"{format_cell(field[6])} | {format_cell(field[7])} | {format_cell(field[8])}")
    print("\n")

# Check for a winning condition
def check_winner(board, player):
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
        [0, 4, 8], [2, 4, 6]              # diagonals
    ]
    for condition in win_conditions:
        if all(board[pos] == player for pos in condition):
            return True

    return False

# Check for a draw (no empty spaces)
def check_draw(board):
    return ' ' not in board

def q_update(Q, board, action, player, next_board, reward, alpha, gamma):
    old_value = Q.get(board, action)
    if next_board:
        # Calculate max Q-value for the next state over all possible actions
        next_actions = get_empty_positions(next_board)
        future_qs = [Q.get(next_board, next_action) for next_action in next_actions]
        if player == 'X':
            future_q = max(future_qs)
        else:
            future_q = min(future_qs)
    else:
        future_q = 0.0

    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * future_q)
    # print(f"{new_value}, {old_value}, {alpha}, {gamma}, {reward}, {future_q}")
    Q.set(board, action, new_value)
    return abs(old_value - new_value)

# Update Q-values based on the game's outcome, with correct max_future_q
def q_update_backward(Q, history, player, terminal_reward, alpha, gamma):
    diff = 0
    for i in reversed(range(len(history))):
        board, action = history[i]
        if i == len(history) - 1:
            # Update the last state-action pair with the terminal reward
            diff += q_update(Q, board, action, player, None, terminal_reward, alpha, gamma)
        else:
            next_board, _ = history[i + 1]
            diff += q_update(Q, board, action, player, next_board, 0.0, alpha, gamma)
        
    return diff

# Choose an action based on Q-values
def choose_action(Q, board, player, epsilon=0.1):
    if random.uniform(0, 1) < epsilon:
        # Exploration: Choose a random move
        empty_positions = get_empty_positions(board)
        return random.choice(empty_positions)
    else:
        # Exploitation: Choose the best known move
        return int(Q.best_action(board, player))

# Main game loop for two random agents
def play_game(matrices, params, flags):
    if flags is None:
        flags = {}

    training = flags.get('training', True)
    display = flags.get('display', False)
    interactive = flags.get('interactive', False)
    (Q, Visits, Rewards) = matrices
    board = initialize_board()
    current_player = 'X'
    human_player = None
    history = []  # To store state-action pairs
    number_of_actions = 0
    gamma = params['gamma']
    episode = params['episode']
    nr_of_episodes = params['nr_of_episodes']
    epsilon = max(params['epsilon_min'], params['epsilon_start'] / (1 + episode/nr_of_episodes))
    alpha = max(params['alpha_min'], params['alpha_start'] / (1 + episode/nr_of_episodes))
    rewards = params['rewards']

    if interactive:
        while human_player is None:
            user_input = input(f"Choose a player from the set {['X', 'O']}: ")
            human_player = str(user_input)
            if human_player not in ['X', 'O']:
                human_player = None

    while True:
        if display:
            display_board_with_Q(board, Q, params['Q_initial_value'])
            time.sleep(params['waiting_time'])  # Wait a bit before the next move for readability

        empty_positions = get_empty_positions(board)
        if empty_positions:
            if training:
                action = choose_action(Q, board, current_player, epsilon=epsilon)
                history.append((board[:], action))
            else:
                if interactive and current_player == human_player:
                        action = None
                        while action is None:
                            user_input = input(f"Choose a number from the set {empty_positions}: ")
                            action = int(user_input)
                            if action not in empty_positions:
                                action = None
                else:
                    action = choose_action(Q, board, current_player, epsilon=params['eps'][current_player])

            old_board = board[:]
            
            # Update board
            board[action] = current_player
            number_of_actions += 1

            # Update Visits
            visits = Visits.get(old_board, action)
            Visits.set(old_board, action, visits + 1)

        # Check for winner
        if check_winner(board, current_player):
            terminal_reward = rewards[current_player]# if current_player == 'X' else -1.0  # Reward for 'X', penalty for 'O'
            if display:
                display_board(board)
                print(f"Player {current_player} wins!\n")

            if training:
                q_update_backward(Q, history, current_player, terminal_reward, alpha, gamma)

            # Update Rewards
            Rewards.set(old_board, action, terminal_reward)

            # Update outcomes
            outcome = current_player
            break
        
        # Check for draw
        if check_draw(board):
            terminal_reward = rewards['D']
            if display:
                display_board(board)
                print("It's a draw!\n")

            if training:
                q_update_backward(Q, history, current_player, terminal_reward, alpha, gamma)

            # Update Rewards
            Rewards.set(old_board, action, terminal_reward)

            # Update outcomes
            outcome = 'D'
            break
        
        # Switch players
        current_player = 'O' if current_player == 'X' else 'X'

    params['outcomes'][outcome] += 1
    if training:
        params['history'] = history