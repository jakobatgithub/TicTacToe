from itertools import product
from typing import Any, Callable

import numpy as np
import sympy as sp
import torch
import torch.nn as nn
from sympy import Abs, Matrix


def permutation_matrix(permutation: list[Any]) -> np.ndarray[Any, Any]:
    n = len(permutation)
    return np.eye(n, dtype=int)[permutation]


def get_matrix_pattern(Bs: list[Any], nn: int, mm: int) -> torch.Tensor:
    Bs = [Matrix(B) for B in Bs]

    def WB(W: Callable[..., Any], B: list[Any], x1: float, x2: float, y1: float, y2: float):
        x = Matrix([x1, x2])  # Input vector x
        y = Matrix([y1, y2])  # Input vector y

        X = B @ x
        Y = B @ y

        det_B = Abs(B.det())  # type: ignore
        return det_B * W(X[0], X[1], Y[0], Y[1])

    # Define WW function
    def WW(W: Callable[..., Any], x1: float, x2: float, y1: float, y2: float, Bs: list[Any]) -> Any:
        Ws = 0  # Initialize the sum
        for B in Bs:
            Ws += WB(W, B, x1, x2, y1, y2)
        return Ws

    def W(x1: float, x2: float, y1: float, y2: float) -> Any:
        return sp.Symbol(f"W({x1},{x2},{y1},{y2})")

    value_range1 = range(-nn, nn + 1)
    value_range2 = range(-mm, mm + 1)

    results: list[Any] = []
    for x1_val, x2_val in product(value_range1, repeat=2):
        for y1_val, y2_val in product(value_range2, repeat=2):
            val = WW(W, x1_val, x2_val, y1_val, y2_val, Bs)
            results.append(sp.simplify(val))  # type: ignore

    # Deduplicate results and map unique values to indices
    unique_results = list(sp.ordered(set(results)))  # type: ignore
    result_mapping = {}
    for idx, val in enumerate(unique_results):
        if val != 0:
            result_mapping[val] = idx + 1
        elif val == 0:
            result_mapping[val] = 0

    # Organize results into a symbolic matrix
    matrix_pattern: list[list[Any]] = []
    for x1_val, x2_val in product(value_range1, repeat=2):
        row = []
        for y1_val, y2_val in product(value_range2, repeat=2):
            val = WW(W, x1_val, x2_val, y1_val, y2_val, Bs)
            row.append(result_mapping[sp.simplify(val)])  # type: ignore

        matrix_pattern.append(row)  # type: ignore

    return torch.tensor(np.array(matrix_pattern, dtype=np.float32))


class EquivariantLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, matrix_pattern: torch.Tensor):
        super(EquivariantLayer, self).__init__()  # type: ignore

        # Get and save mask
        mask = self._get_mask(matrix_pattern)
        self.register_buffer("mask", mask)

        # Create a trainable parameter for each tied group
        nr_of_ties = self._get_nr_of_ties(matrix_pattern)
        self.tied_weights = nn.ParameterList(values=[nn.Parameter(torch.randn(1)) for _ in range(nr_of_ties)])

        # Precompute a mapping from tied groups
        self.register_buffer("weight_matrix", torch.zeros_like(matrix_pattern, dtype=torch.float32))
        self.weight_matrix = matrix_pattern

    def _get_mask(self, matrix_pattern: torch.Tensor) -> torch.Tensor:
        mask = matrix_pattern.clone()
        mask[mask > 0] = 1
        return mask

    def _get_nr_of_ties(self, matrix_pattern: torch.Tensor) -> int:
        ties = list(set(matrix_pattern.detach().numpy().flatten()))  # type: ignore
        tied_groups = [1 for tie in ties if tie != 0]
        return len(tied_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Construct the weight matrix dynamically
        weight = torch.zeros_like(self.weight_matrix)
        for group_idx, tied_param in enumerate(self.tied_weights):
            weight[self.weight_matrix == group_idx + 1] = tied_param

        # Apply the mask to enforce sparsity
        masked_weight = weight * self.mask

        # Compute the output
        return x @ masked_weight  # TODO: Add bias


class EquivariantNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, groupMatrices: list[Any]) -> None:
        super(EquivariantNN, self).__init__()  # type: ignore
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
