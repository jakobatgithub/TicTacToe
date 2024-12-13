from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


def permutation_matrix(permutation: list[Any]) -> np.ndarray[Any, Any]:
    n = len(permutation)
    return np.eye(n, dtype=int)[permutation]


def get_number_of_unique_elements(pattern: torch.Tensor) -> int:
    unique_elements = set(pattern.detach().numpy().flatten())  # type: ignore
    return len(unique_elements)


def get_weight_pattern(Bs: List[Any], nn: int, mm: int) -> torch.Tensor:
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


# class EquivariantLayer(nn.Module):
#     def __init__(self, weight_pattern: torch.Tensor, bias_pattern: torch.Tensor):
#         super(EquivariantLayer, self).__init__()  # type: ignore

#         nr_of_ties = self._get_nr_of_unique_nonzero_elements(weight_pattern)

#         nr_of_bias_params = self._get_nr_of_unique_nonzero_elements(bias_pattern)

#         input_shape, _ = weight_pattern.shape
#         bound = 1.0 / np.sqrt(input_shape)

#         self.matrix_params = nn.ParameterList(
#             [nn.Parameter(torch.empty(1).uniform_(-bound, bound)) for _ in range(nr_of_ties)]
#         )

#         self.bias_params = nn.ParameterList(
#             [
#                 nn.Parameter(torch.zeros(1))  # Bias initialized to zero
#                 for _ in range(nr_of_bias_params)
#             ]
#         )

#         # Precompute a mapping from tied groups
#         self.register_buffer("weight_pattern", torch.zeros_like(weight_pattern, dtype=torch.float32))
#         self.weight_pattern = weight_pattern

#         self.register_buffer("bias_pattern", torch.zeros_like(bias_pattern, dtype=torch.float32))
#         self.bias_pattern = bias_pattern

#     def _get_nr_of_unique_nonzero_elements(self, pattern: torch.Tensor) -> int:
#         unique_elements = list(set(pattern.detach().numpy().flatten()))  # type: ignore
#         return len(unique_elements)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Construct the weight matrix dynamically
#         weight = torch.zeros_like(self.weight_pattern, dtype=torch.float32)
#         for group_idx, tied_param in enumerate(self.matrix_params):
#             weight[self.weight_pattern == group_idx + 1] = tied_param

#         weight[self.weight_pattern == 0] = 0.0

#         bias = torch.zeros_like(self.bias_pattern, dtype=torch.float32)
#         for group_idx, tied_param in enumerate(self.bias_params):
#             bias[self.bias_pattern == group_idx + 1] = tied_param

#         bias[self.bias_pattern == 0] = 0.0

#         return x @ weight + bias


class EquivariantLayer(nn.Module):
    def __init__(self, weight_pattern: torch.Tensor, bias_pattern: torch.Tensor):
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
            torch.empty(len(torch.unique(weight_pattern[weight_pattern > 0]))).uniform_(-bound, bound)  # type: ignore
        )
        self.bias_params = nn.Parameter(torch.zeros(len(torch.unique(bias_pattern[bias_pattern > 0]))))  # type: ignore

        self.weight_idx_mask = self.weight_mask[self.non_zero_weight_mask] - 1
        self.bias_idx_mask = self.bias_mask[self.non_zero_bias_mask] - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.zeros_like(self.weight_mask, dtype=torch.float32)
        bias = torch.zeros_like(self.bias_mask, dtype=torch.float32)

        weight[self.non_zero_weight_mask] = self.weight_params[self.weight_idx_mask]
        bias[self.non_zero_bias_mask] = self.bias_params[self.bias_idx_mask]

        return x @ weight + bias


class EquivariantNN(nn.Module):
    def __init__(self, groupMatrices: list[Any], ms: Tuple[int, int, int, int] = (1, 5, 5, 1)) -> None:
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

        # number_of_weights = (
        #     get_number_of_unique_elements(weight_pattern1)
        #     + get_number_of_unique_elements(weight_pattern2)
        #     + get_number_of_unique_elements(weight_pattern3)
        # )
        # print(f"Number of matrix weights: {number_of_weights}")

        # number_of_biases = (
        #     get_number_of_unique_elements(bias_pattern1)
        #     + get_number_of_unique_elements(bias_pattern2)
        #     + get_number_of_unique_elements(bias_pattern3)
        # )
        # print(f"Number of biases: {number_of_biases}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_equivariant(x)
