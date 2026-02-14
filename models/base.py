from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from torch import nn
from config import *


class NNModel(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[TensorActions, torch.Tensor]:
        raise NotImplementedError
