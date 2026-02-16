from typing import Any, Type, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from torch import nn
from config import *

def create_resnet_block_class(
    layer_class: Type[nn.Module],
    norm_class: Type[nn.Module],
    active_func: Type[nn.Module] = nn.ReLU,
):
    """创建残差层的工厂函数

    Args:
        layer_class (Type[nn.Module]): 残差层中采用什么网络，如nn.Conv2d
        norm_class (Type[nn.Module]): 采用的BatchNorm
        active_func (Type[nn.Module], optional): 采用的激活函数，默认为 nn.ReLU.

    Returns:
        Type[nn.Module]: 残差层类
    """

    class ResnetBlock(nn.Module):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            for _ in range(RESNET_LAYERS_NUM):
                layers.append(layer_class(**kwargs))
                norm_param = kwargs.get("in_channels") or kwargs.get("in_features")
                layers.append(norm_class(norm_param))
                layers.append(active_func())
            self.model = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.model(x)

    return ResnetBlock


_ResnetConv1dBase = create_resnet_block_class(nn.Conv1d, nn.BatchNorm1d)
_ResnetConv2dBase = create_resnet_block_class(nn.Conv2d, nn.BatchNorm2d)
_ResnetLinearBase = create_resnet_block_class(nn.Linear, nn.BatchNorm1d)


class ResnetConv1d(_ResnetConv1dBase):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int | tuple[int] = 3,
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )


class ResnetConv2d(_ResnetConv2dBase):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )


class ResnetLinear(_ResnetLinearBase):
    def __init__(self, in_features: int) -> None:
        super().__init__(in_features=in_features, out_features=in_features)
