"""五子棋模型实现"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import *
from nn_models.base import BaseModel

from utils.logger import setup_logger

logger = setup_logger(__name__)

class GomokuModel(BaseModel):
    """CNN 双头模型 - 针对15x15棋盘"""

    def __init__(self) -> None:
        super().__init__()

        in_channels = HISTORY_LEN + 1 if USE_HISTORY else 2
        board_size = 15
        
        # 共享特征提取层
        self._shared_layers = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # 策略头
        self._policy_head = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(128 * board_size * board_size, 512),
            nn.ReLU(),
            nn.Linear(512, 225),
        )

        # 价值头
        self._value_head = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(128 * board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    @property
    def shared_layers(self) -> nn.Sequential:
        return self._shared_layers

    def policy_head(self, x: torch.Tensor) -> TensorActions:
        return self._policy_head(x)

    def value_head(self, x: torch.Tensor) -> torch.Tensor:
        return self._value_head(x)
