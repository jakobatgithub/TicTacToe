# TicTacToe/EvaluationMixin.py

import numpy as np
import wandb

class EvaluationMixin:
    def __init__(self, wandb_enabled: bool, wandb_logging_frequency: int):
        self.wandb = wandb_enabled
        self.wandb_logging_frequency = wandb_logging_frequency
        self.train_step_count = 0
        self.episode_count = 0
        self.evaluation_data: dict[str, list[float]] = {
            "loss": [],
            "action_value": [],
            "rewards": [],
            "learning_rate": [],
        }

    def record_eval_data(self, key: str, value: float) -> None:
        if key in self.evaluation_data:
            self.evaluation_data[key].append(value)

    def safe_mean(self, x: list[float]) -> float:
        return float(np.mean(x)) if x else 0.0

    def safe_var(self, x: list[float]) -> float:
        return float(np.var(x)) if x else 0.0

    def maybe_log_metrics(self) -> None:
        if self.train_step_count % self.wandb_logging_frequency != 0:
            return

        if self.wandb:
            wandb.log({
                "loss": self.safe_mean(self.evaluation_data["loss"]),
                "action_value": self.safe_mean(self.evaluation_data["action_value"]),
                "mean_reward": self.safe_mean(self.evaluation_data["rewards"]),
                "var_reward": self.safe_var(self.evaluation_data["rewards"]),
                "episode_count": self.episode_count,
                "train_step_count": self.train_step_count,
                "epsilon": getattr(self, "epsilon", 0),
                "learning_rate": self.safe_mean(self.evaluation_data["learning_rate"]),
            })

        self.reset_evaluation_data()

    def reset_evaluation_data(self) -> None:
        for key in self.evaluation_data:
            self.evaluation_data[key] = []
