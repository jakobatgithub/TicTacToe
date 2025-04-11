from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


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
                equivalence_classes[transformed_values] = class_counter
                class_counter += 1

            # Store the class label in the output matrix
            weight_pattern[i, j] = equivalence_classes[transformed_values]

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
            equivalence_classes[transformed_values] = class_counter
            class_counter += 1

        # Store the class label in the output matrix
        bias_pattern[i] = equivalence_classes[transformed_values]

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
