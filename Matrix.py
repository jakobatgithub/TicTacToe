import dill

from collections import defaultdict
import numpy as np

class Matrix:
    def __init__(self, file=None, default_value=None):
        """
        Initialize a Matrix object.

        Parameters
        ----------
        file : str, optional
            File name to load the matrix from. If None, the matrix is
            initialized with the given default value.
        default_value : object, optional
            Default value to use for initializing the matrix. If None,
            the matrix is loaded from the given file.

        Notes
        -----
        The matrix is stored in a dictionary where the keys represent
        states and the values are dictionaries where the keys are
        actions and the values are q-values.
        """
        self.default_value = 0.0
        if default_value is None and file is None:
            self.q_matrix = defaultdict(self._initialize_q_matrix)
        elif default_value is None and file is not None:
            with open(file, 'rb') as f:
                self.q_matrix = dill.load(f)
        elif file is None and default_value is not None:
            self.default_value = default_value
            self.q_matrix = defaultdict(self._initialize_q_matrix)

    def _initialize_q_matrix(self):
        """
        Initialize an empty dictionary for storing q-values for a given state.

        The dictionary is a defaultdict with the default value set to
        self.default_value.

        Returns
        -------
        state_dict : dict
            An empty dictionary for storing q-values for the given state.
        """
        state_dict = defaultdict(lambda: self.default_value)
        return state_dict    

    def get(self, board=None, action=None):
        """
        Retrieve the q-value(s) for a given state-action pair.

        Parameters
        ----------
        board : list or None, optional
            The current state of the board represented as a list. If None,
            the entire q_matrix is returned.
        action : int or None, optional
            The action taken at the given board state. If None, all q-values
            for the given board state are returned.

        Returns
        -------
        float or dict
            If both board and action are provided, returns the q-value for
            the specified state-action pair. If only the board is provided,
            returns a dictionary of actions and their corresponding q-values
            for the given state. If neither is provided, returns the entire
            q_matrix.
        """
        if board is None and action is None:
            return self.q_matrix
        if action is None and board is not None:
            return self.q_matrix[tuple(board)]
        if action is not None and board is not None:
            return self.q_matrix[tuple(board)][action]

    def set(self, board, action, value):
        """
        Set the q-value for a given state-action pair.

        Parameters
        ----------
        board : list
            The state of the board represented as a list.
        action : int
            The action taken at the given board state.
        value : float
            The q-value to set for the specified state-action pair.
        """
        self.q_matrix[tuple(board)][action] = value


class QMatrix(Matrix):
    def best_actions(self, board, player='X'):
        """
        Choose the best action based on Q-values for the current state.

        Parameters
        ----------
        board : list
            The state of the board represented as a list.
        player : str, optional
            The player whose strategy to use for choosing the best action.
            Defaults to 'X'. If 'X', the action with the highest Q-value is chosen.
            If 'O', the action with the lowest Q-value is chosen.

        Returns
        -------
        best_actions : list
            A list of the best actions based on the given player's strategy.
        """
        actions = [i for i, cell in enumerate(board) if cell == ' ']

        # Retrieve Q-values for all valid actions
        q_values = {action: self.get(board, action) for action in actions}

        # Choose based on player strategy
        if player == 'X':
            max_q = max(q_values.values())
            best_actions = [action for action, q in q_values.items() if q == max_q]
        else:
            min_q = min(q_values.values())
            best_actions = [action for action, q in q_values.items() if q == min_q]

        return best_actions

    def best_action(self, board, player='X'):
        """
        Choose a random best action based on Q-values for the current state.

        Parameters
        ----------
        board : list
            The state of the board represented as a list.
        player : str, optional
            The player whose strategy to use for choosing the best action.
            Defaults to 'X'. If 'X', the action with the highest Q-value is chosen.
            If 'O', the action with the lowest Q-value is chosen.

        Returns
        -------
        best_action : int
            A random best action based on the given player's strategy.
        """
        best_actions = self.best_actions(board, player)
        return np.random.choice(best_actions)