"""点格棋模型实现"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import *
from nn_models.base import BaseModel
from utils.logger import setup_logger

from .game import DotsAndBoxesGame

logger = setup_logger(__name__)


class DotsAndBoxesModel(BaseModel):
    """简单 2D CNN 双头模型

    输入维度:
        (batch, HISTORY_LEN+1, C, H, W) 或 (batch, 2, C, H, W)
    """

    def __init__(self) -> None:
        super().__init__()

        depth = DotsAndBoxesGame.STATE_SHAPE[0]
        height = DotsAndBoxesGame.STATE_SHAPE[1]
        width = DotsAndBoxesGame.STATE_SHAPE[2]

        in_channels = HISTORY_LEN * depth + 1 if USE_HISTORY else depth + 1

        self._shared_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self._policy_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(2 * height * width, DotsAndBoxesGame.NUM_ACTION),
        )

        self._value_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(height * width, 64),
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
