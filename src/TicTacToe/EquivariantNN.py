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


def get_weight_pattern(Bs: list[Any], nn: int, mm: int) -> torch.Tensor:
    Bs = [Matrix(B) for B in Bs]

    def W_transformed_by_B(W: Callable[..., Any], B: list[Any], x1: float, x2: float, y1: float, y2: float):
        x = Matrix([x1, x2])  # Input vector x
        y = Matrix([y1, y2])  # Input vector y

        X = B @ x
        Y = B @ y

        det_B = Abs(B.det())  # type: ignore
        return det_B * W(X[0], X[1], Y[0], Y[1])

    # Define WW function
    def W_invariant_under_Bs(W: Callable[..., Any], x1: float, x2: float, y1: float, y2: float, Bs: list[Any]) -> Any:
        Ws = 0  # Initialize the sum
        for B in Bs:
            Ws += W_transformed_by_B(W, B, x1, x2, y1, y2)
        return Ws

    def W(x1: float, x2: float, y1: float, y2: float) -> Any:
        return sp.Symbol(f"W({x1},{x2},{y1},{y2})")

    value_range1 = range(-nn, nn + 1)
    value_range2 = range(-mm, mm + 1)

    results: list[Any] = []
    for x1_val, x2_val in product(value_range1, repeat=2):
        for y1_val, y2_val in product(value_range2, repeat=2):
            val = W_invariant_under_Bs(W, x1_val, x2_val, y1_val, y2_val, Bs)
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
    weight_pattern: list[list[Any]] = []
    for x1_val, x2_val in product(value_range1, repeat=2):
        row = []
        for y1_val, y2_val in product(value_range2, repeat=2):
            val = W_invariant_under_Bs(W, x1_val, x2_val, y1_val, y2_val, Bs)
            row.append(result_mapping[sp.simplify(val)])  # type: ignore

        weight_pattern.append(row)  # type: ignore

    return torch.tensor(np.array(weight_pattern, dtype=np.float32))


def get_bias_pattern(Bs: list[Any], mm: int) -> torch.Tensor:
    Bs = [Matrix(B) for B in Bs]

    def b_transformed_by_B(b: Callable[..., Any], B: list[Any], x1: float, x2: float):
        x = Matrix([x1, x2])  # Input vector x
        X = B @ x
        return b(X[0], X[1])

    def b_invariant_under_Bs(b: Callable[..., Any], x1: float, x2: float, Bs: list[Any]) -> Any:
        bs = 0  # Initialize the sum
        for B in Bs:
            bs += b_transformed_by_B(b, B, x1, x2)
        return bs

    def b(x1: float, x2: float) -> Any:
        return sp.Symbol(f"b({x1},{x2})")

    value_range1 = range(-mm, mm + 1)

    results: list[Any] = []
    for x1_val, x2_val in product(value_range1, repeat=2):
        val = b_invariant_under_Bs(b, x1_val, x2_val, Bs)
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
    bias_pattern: list[list[Any]] = []
    for x1_val, x2_val in product(value_range1, repeat=2):
        val = b_invariant_under_Bs(b, x1_val, x2_val, Bs)
        bias_pattern.append(result_mapping[sp.simplify(val)])  # type: ignore

    return torch.tensor(np.array(bias_pattern, dtype=np.float32))


class EquivariantLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, weight_pattern: torch.Tensor, bias_pattern: torch.Tensor):
        super(EquivariantLayer, self).__init__()  # type: ignore

        # Create a trainable parameter for each tied group
        nr_of_ties = self._get_nr_of_unique_nonzero_elements(weight_pattern)
        self.matrix_params = nn.ParameterList(values=[nn.Parameter(torch.randn(1)) for _ in range(nr_of_ties)])

        nr_of_bias_params = self._get_nr_of_unique_nonzero_elements(bias_pattern)
        self.bias_params = nn.ParameterList(values=[nn.Parameter(torch.randn(1)) for _ in range(nr_of_bias_params)])

        # Precompute a mapping from tied groups
        self.register_buffer("weight_pattern", torch.zeros_like(weight_pattern, dtype=torch.float32))
        self.weight_pattern = weight_pattern

        self.register_buffer("bias_pattern", torch.zeros_like(bias_pattern, dtype=torch.float32))
        self.bias_pattern = bias_pattern

    def _get_nr_of_unique_nonzero_elements(self, pattern: torch.Tensor) -> int:
        unique_elements = list(set(pattern.detach().numpy().flatten()))  # type: ignore
        return len(unique_elements)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Construct the weight matrix dynamically
        weight = torch.zeros_like(self.weight_pattern)
        for group_idx, tied_param in enumerate(self.matrix_params):
            weight[self.weight_pattern == group_idx + 1] = tied_param

        weight[self.weight_pattern == 0] = 0.0

        bias = torch.zeros_like(self.bias_pattern)
        for group_idx, tied_param in enumerate(self.bias_params):
            bias[self.bias_pattern == group_idx + 1] = tied_param

        bias[self.bias_pattern == 0] = 0.0

        return x @ weight + bias


class EquivariantNN(nn.Module):
    def __init__(self, groupMatrices: list[Any]) -> None:
        super(EquivariantNN, self).__init__()  # type: ignore

        n, m1, m2, m3 = 1, 3, 3, 1
        input_dim = (2 * n + 1) ** 2
        output_dim1 = (2 * m1 + 1) ** 2
        output_dim2 = (2 * m2 + 1) ** 2
        output_dim3 = (2 * m3 + 1) ** 2

        self.groupMatrices = groupMatrices

        weight_pattern1 = get_weight_pattern(self.groupMatrices, n, m1)
        bias_pattern1 = get_bias_pattern(self.groupMatrices, m1)
        equivariant_layer1 = EquivariantLayer(
            input_dim=input_dim, output_dim=output_dim1, weight_pattern=weight_pattern1, bias_pattern=bias_pattern1
        )
        weight_pattern2 = get_weight_pattern(self.groupMatrices, m1, m2)
        bias_pattern2 = get_bias_pattern(self.groupMatrices, m2)
        equivariant_layer2 = EquivariantLayer(
            input_dim=output_dim1, output_dim=output_dim2, weight_pattern=weight_pattern2, bias_pattern=bias_pattern2
        )
        weight_pattern3 = get_weight_pattern(self.groupMatrices, m2, m3)
        bias_pattern3 = get_bias_pattern(self.groupMatrices, m3)
        equivariant_layer3 = EquivariantLayer(
            input_dim=output_dim2, output_dim=output_dim3, weight_pattern=weight_pattern3, bias_pattern=bias_pattern3
        )

        self.fc_equivariant = nn.Sequential(
            equivariant_layer1, nn.ReLU(), equivariant_layer2, nn.ReLU(), equivariant_layer3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_equivariant(x)
