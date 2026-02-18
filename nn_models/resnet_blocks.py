from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from torch import nn
from config import *


def create_resnet_block_class(
    layer_class: type[nn.Module],
    norm_class: type[nn.Module],
    layers_num: int = 3,
    active_func: type[nn.Module] = nn.ReLU,
):
    """创建残差层的工厂函数

    Args:
        layer_class (type[nn.Module]): 残差层中采用什么网络，如nn.Conv2d
        norm_class (type[nn.Module]): 采用的BatchNorm
        active_func (type[nn.Module], optional): 采用的激活函数，默认为 nn.ReLU.

    Returns:
        type[nn.Module]: 残差层类
    """

    class ResnetBlock(nn.Module):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            for _ in range(layers_num):
                layers.append(layer_class(**kwargs))
                norm_param = kwargs.get("in_channels") or kwargs.get("in_features")
                layers.append(norm_class(norm_param))
                layers.append(active_func())
            self.model = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.model(x)

    return ResnetBlock


class ResnetConv1d:
    def __init__(
        self,
        in_channels: int,
        layer_num: int = 3,
        kernel_size: int | tuple[int] = 3,
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] = 1,
    ) -> None:
        _ResnetConv1dBase = create_resnet_block_class(
            nn.Conv1d, nn.BatchNorm1d, layers_num=layer_num
        )
        self._block = _ResnetConv1dBase(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            layers_num=layer_num,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._block(x)


class ResnetConv2d:
    def __init__(
        self,
        in_channels: int,
        layer_num: int = 3,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 1,
    ) -> None:
        _ResnetConv2dBase = create_resnet_block_class(nn.Conv2d, nn.BatchNorm2d)
        self._block = _ResnetConv2dBase(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            layers_num=layer_num,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._block(x)


class ResnetLinear:
    def __init__(self, in_features: int, layer_num: int = 3) -> None:
        _ResnetLinearBase = create_resnet_block_class(
            nn.Linear, nn.BatchNorm1d, layers_num=layer_num
        )
        self._block = _ResnetLinearBase(
            in_features=in_features, out_features=in_features, layers_num=layer_num
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._block(x)
