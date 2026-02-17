"""井字棋模型实现"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import *
from nn_models.base import BaseModel

from utils.logger import setup_logger

logger = setup_logger(__name__)

class TicTacToeModel(BaseModel):
    """简单 CNN 双头模型"""

    def __init__(self) -> None:
        super().__init__()

        in_channels = HISTORY_LEN + 1 if USE_HISTORY else 2
        
        self._shared_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self._policy_head = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 9),
        )

        self._value_head = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(64 * 3 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    @property
    def shared_layers(self) -> nn.Sequential:
        return self._shared_layers

    def policy_head(self, x: torch.Tensor) -> TensorActions:
        return self._policy_head(x)

    def value_head(self, x: torch.Tensor) -> torch.Tensor:
        return self._value_head(x)
