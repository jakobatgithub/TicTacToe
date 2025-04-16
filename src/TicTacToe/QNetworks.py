from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    A neural network for approximating the Q-function.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        Initialize the QNetwork.

        Args:
            input_dim: Dimension of the input state.
            output_dim: Dimension of the output actions.
        """
        super(QNetwork, self).__init__()  # type: ignore
        self.fc = nn.Sequential(
            nn.Linear(input_dim, out_features=49),
            nn.ReLU(),
            nn.Linear(49, 49),
            nn.ReLU(),
            nn.Linear(49, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the QNetwork.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.fc(x)


class CNNQNetwork(nn.Module):
    """
    A convolutional neural network for approximating the Q-function.
    """

    def __init__(self, input_dim: int, rows: int, output_dim: int) -> None:
        """
        Initialize the CNNQNetwork.

        Args:
            input_dim: Dimension of the input state (e.g., number of channels).
            rows: Size of the grid (e.g., 3 for 3x3 grid).
            output_dim: Dimension of the output actions.
        """
        super(CNNQNetwork, self).__init__() # type: ignore
        self.rows = rows

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * rows * rows, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNNQNetwork.

        Args:
            x: Input tensor of shape (batch_size, input_dim, grid_size, grid_size).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        x = x.view(-1, 1, self.rows, self.rows)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = x.view(-1, self.rows * self.rows)  # Flatten the output to (batch_size, rows*rows)
        return x

class PeriodicConvBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class PeriodicQHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, padding_mode='circular')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)  # shape: (batch_size, 1, rows, cols)
        return x.squeeze(1)  # shape: (batch_size, rows, cols)


class FullyConvQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = PeriodicConvBase()
        self.head = PeriodicQHead()

        # ✅ For backward compatibility with tests
        self.conv_layers = nn.Sequential(
            self.base.encoder[0],
            self.base.encoder[1],
            self.base.encoder[2],
            self.base.encoder[3],
            self.head.head
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:  # (batch_size, rows*cols)
            spatial_dim = int(x.size(1) ** 0.5)
            x = x.view(x.size(0), 1, spatial_dim, spatial_dim)
        elif x.ndim == 4:  # already (batch_size, 1, rows, cols)
            pass
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        x = self.base(x)
        x = self.head(x)  # shape: (batch_size, rows, cols)
        return x.view(x.size(0), -1)  # shape: (batch_size, rows * cols)

def permutation_matrix(permutation: list[Any]) -> np.ndarray[Any, Any]:
    """
    Generate a permutation matrix for a given permutation.

    Args:
        permutation (list[Any]): A list representing the permutation.

    Returns:
        np.ndarray[Any, Any]: A permutation matrix.
    """
    n = len(permutation)
    return np.eye(n, dtype=int)[permutation]


def get_number_of_unique_elements(pattern: torch.Tensor) -> int:
    """
    Count the number of unique elements in a tensor.

    Args:
        pattern (torch.Tensor): The input tensor.

    Returns:
        int: The number of unique elements.
    """
    unique_elements = set(pattern.detach().numpy().flatten())  # type: ignore
    return len(unique_elements)


def get_weight_pattern(Bs: List[Any], nn: int, mm: int) -> torch.Tensor:
    """
    Generate a weight pattern based on symmetry transformations.

    Args:
        Bs (List[Any]): List of transformation matrices.
        nn (int): Range for the first dimension.
        mm (int): Range for the second dimension.

    Returns:
        torch.Tensor: A tensor representing the weight pattern.
    """
    # Define ranges
    value_range1 = np.arange(-nn, nn + 1)
    value_range2 = np.arange(-mm, mm + 1)

    # Precompute all combinations of (x, y) indices
    x_combinations = np.array(list(product(value_range1, repeat=2)))
    y_combinations = np.array(list(product(value_range2, repeat=2)))

    # Dictionary to group equivalent elements
    equivalence_classes: Dict[Tuple[float, float, float, float], int] = {}
    class_counter = 1  # Start labeling from 1

    # Output matrix
    weight_pattern = np.zeros((len(x_combinations), len(y_combinations)), dtype=int)

    # Process each pair of indices
    for i, x in enumerate(x_combinations):
        for j, y in enumerate(y_combinations):
            transformed_values: List[Any] | Tuple[Any] = []
            for B in Bs:
                # Apply symmetry transformations
                X = B @ x
                Y = B @ y
                # Use rounded transformed coordinates as a key
                transformed_values.append((round(X[0], 6), round(X[1], 6), round(Y[0], 6), round(Y[1], 6)))

            # Deduplicate transformed values
            transformed_values = tuple(sorted(transformed_values))

            # Assign equivalence class
            if transformed_values not in equivalence_classes:
                equivalence_classes[transformed_values] = class_counter # type: ignore
                class_counter += 1

            # Store the class label in the output matrix
            weight_pattern[i, j] = equivalence_classes[transformed_values] # type: ignore

    # Convert to torch tensor
    return torch.tensor(weight_pattern, dtype=torch.int64)


def get_bias_pattern(Bs: list[Any], mm: int) -> torch.Tensor:
    """
    Generate a bias pattern based on symmetry transformations.

    Args:
        Bs (list[Any]): List of transformation matrices.
        mm (int): Range for the dimension.

    Returns:
        torch.Tensor: A tensor representing the bias pattern.
    """
    # Define ranges
    value_range = np.arange(-mm, mm + 1)

    # Precompute all combinations of (x, y) indices
    x_combinations = np.array(list(product(value_range, repeat=2)))

    # Dictionary to group equivalent elements
    equivalence_classes: Dict[Tuple[float, float, float, float], int] = {}
    class_counter = 1  # Start labeling from 1

    # Output matrix
    bias_pattern = np.zeros(len(x_combinations), dtype=int)

    # Process each pair of indices
    for i, x in enumerate(x_combinations):
        transformed_values: List[Any] | Tuple[Any] = []
        for B in Bs:
            # Apply symmetry transformations
            X = B @ x
            # Use rounded transformed coordinates as a key
            transformed_values.append((round(X[0], 6), round(X[1], 6)))

        # Deduplicate transformed values
        transformed_values = tuple(sorted(transformed_values))

        # Assign equivalence class
        if transformed_values not in equivalence_classes:
            equivalence_classes[transformed_values] = class_counter # type: ignore
            class_counter += 1

        # Store the class label in the output matrix
        bias_pattern[i] = equivalence_classes[transformed_values] # type: ignore

    # Convert to torch tensor
    return torch.tensor(bias_pattern, dtype=torch.int64)


class EquivariantLayer(nn.Module):
    """
    A neural network layer with tied weights and biases based on symmetry patterns.
    """

    def __init__(self, weight_pattern: torch.Tensor, bias_pattern: torch.Tensor):
        """
        Initialize the EquivariantLayer.

        Args:
            weight_pattern (torch.Tensor): Tensor defining the weight tying pattern.
            bias_pattern (torch.Tensor): Tensor defining the bias tying pattern.
        """
        super(EquivariantLayer, self).__init__()  # type: ignore

        # Precompute unique parameter indices
        self.register_buffer("weight_pattern", weight_pattern)
        self.register_buffer("bias_pattern", bias_pattern)

        self.weight_mask = weight_pattern
        self.bias_mask = bias_pattern

        # Fill weights and biases only for non-zero indices
        self.non_zero_weight_mask = self.weight_mask > 0
        self.non_zero_bias_mask = self.bias_mask > 0

        input_size = weight_pattern.size(0)
        bound = 1.0 / np.sqrt(input_size)

        self.weight_params = nn.Parameter(
            torch.empty(self._get_nr_of_unique_nonzero_elements(weight_pattern)).uniform_(-bound, bound)  # type: ignore
        )
        self.bias_params = nn.Parameter(torch.zeros(self._get_nr_of_unique_nonzero_elements(bias_pattern)))  # type: ignore

        self.weight_idx_mask = self.weight_mask[self.non_zero_weight_mask] - 1
        self.bias_idx_mask = self.bias_mask[self.non_zero_bias_mask] - 1

    def _get_nr_of_unique_nonzero_elements(self, pattern: torch.Tensor) -> int:
        """
        Get the number of unique non-zero elements in a pattern.

        Args:
            pattern (torch.Tensor): The input pattern.

        Returns:
            int: The number of unique non-zero elements.
        """
        unique_elements = list(set(pattern.detach().numpy().flatten()))  # type: ignore
        if 0 in unique_elements:
            unique_elements.remove(0)

        return len(unique_elements)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the layer.
        """
        weight = torch.zeros_like(self.weight_mask, dtype=torch.float32)
        bias = torch.zeros_like(self.bias_mask, dtype=torch.float32)

        weight[self.non_zero_weight_mask] = self.weight_params[self.weight_idx_mask]
        bias[self.non_zero_bias_mask] = self.bias_params[self.bias_idx_mask]

        return x @ weight + bias


class EquivariantNN(nn.Module):
    """
    A neural network with multiple equivariant layers.
    """

    def __init__(self, groupMatrices: list[Any], ms: Tuple[int, int, int, int] = (1, 5, 5, 1)) -> None:
        """
        Initialize the EquivariantNN.

        Args:
            groupMatrices (list[Any]): List of transformation matrices.
            ms (Tuple[int, int, int, int]): Dimensions for each layer.
        """
        super(EquivariantNN, self).__init__()  # type: ignore

        m0, m1, m2, m3 = ms

        self.groupMatrices = groupMatrices

        weight_pattern1 = get_weight_pattern(self.groupMatrices, m0, m1)
        bias_pattern1 = get_bias_pattern(self.groupMatrices, m1)
        equivariant_layer1 = EquivariantLayer(weight_pattern=weight_pattern1, bias_pattern=bias_pattern1)

        weight_pattern2 = get_weight_pattern(self.groupMatrices, m1, m2)
        bias_pattern2 = get_bias_pattern(self.groupMatrices, m2)
        equivariant_layer2 = EquivariantLayer(weight_pattern=weight_pattern2, bias_pattern=bias_pattern2)

        weight_pattern3 = get_weight_pattern(self.groupMatrices, m2, m3)
        bias_pattern3 = get_bias_pattern(self.groupMatrices, m3)
        equivariant_layer3 = EquivariantLayer(weight_pattern=weight_pattern3, bias_pattern=bias_pattern3)

        self.fc_equivariant = nn.Sequential(
            equivariant_layer1, nn.ReLU(), equivariant_layer2, nn.ReLU(), equivariant_layer3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the network.
        """
        return self.fc_equivariant(x)
