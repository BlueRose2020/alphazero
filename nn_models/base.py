from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from torch import nn
from config import *


class BaseModel(nn.Module, ABC):    
    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> tuple[TensorActions, torch.Tensor]:
        x = self.shared_layers(x)
        return self.policy_head(x), self.value_head(x)
    
    @property
    @abstractmethod
    def shared_layers(self) -> nn.Sequential:...

    @abstractmethod
    def policy_head(self, x: torch.Tensor) -> TensorActions:...

    @abstractmethod
    def value_head(self, x: torch.Tensor) -> torch.Tensor:...

