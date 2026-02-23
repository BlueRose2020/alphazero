"""五子棋模型实现"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import *
from nn_models.base import BaseModel

from utils.logger import setup_logger

logger = setup_logger(__name__)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self._conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self._bn1 = nn.BatchNorm2d(channels)
        self._conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self._bn2 = nn.BatchNorm2d(channels)
        self._relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self._conv1(x)
        out = self._bn1(out)
        out = self._relu(out)
        out = self._conv2(out)
        out = self._bn2(out)
        out = out + identity
        return self._relu(out)


class GomokuModel(BaseModel):
    """CNN 双头模型 - 针对15x15棋盘"""

    def __init__(self) -> None:
        super().__init__()

        in_channels = HISTORY_LEN + 1 if USE_HISTORY else 2
        board_size = 15

        channels = 64
        blocks = 2

        # 共享特征提取层（AlphaZero 风格）
        self._shared_layers = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            *[ResidualBlock(channels) for _ in range(blocks)],
        )

        # 策略头
        self._policy_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(board_size * board_size, board_size * board_size),
        )

        # 价值头
        self._value_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(2 * board_size * board_size, 64),
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
