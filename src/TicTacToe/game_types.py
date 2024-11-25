from typing import Any, Dict, List, Literal, MutableSequence, Tuple

import numpy as np

Action = int
Actions = List[int]

Board = MutableSequence[str]

History = List[Tuple[Board, Action]]

Player = Literal["X", "O"]
Players = List[Player]

Params = Dict[str, Any]
Outcome = Literal["X", "O", "D"] | None

Reward = float

State = np.ndarray[Any, Any]

StateTransition = Tuple[Board | None, Reward, bool]
StateTransitions = List[StateTransition]

StateTransition2 = Tuple[Board, Action, Board | None, Reward, bool]
StateTransitions2 = List[StateTransition2]
