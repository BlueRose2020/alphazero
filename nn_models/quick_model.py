from __future__ import annotations
from config import *
from config.quick_model_config import *
from nn_models.base import BaseModel
from typing import Callable

from .resnet_blocks import create_resnet_block_class

_Quick_MODEL_RESNET_BASE = create_resnet_block_class(
    nn.Conv2d, nn.BatchNorm2d, layers_num=RESNET_LAYERS_NUM
)


class Quick_MODEL_RESNET(_Quick_MODEL_RESNET_BASE):
    def __init__(
        self,
        in_channels: int = RESNET_CHANNELS,
        kernel_size: int | tuple[int] = RESNET_KERNEL_SIZE,
        stride: int | tuple[int] = RESNET_STRIDE,
        padding: int | tuple[int] = RESNET_PADDING,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )


class QuickModel(BaseModel):
    """该模型由root/config/quick_model_config.py里的参数配置，请勿手动修改

    模型输入为三维，为棋盘历史，玩家通道和棋盘信息，当游戏状态需要多个维度表示时，
    则将其张开并按顺序拼接，如使用两个通道分别表示黑白棋时，状态形如：
    [
        [#历史1
            [1,1,0],
            [0,0,1],
            [0,1,0]
        ],
        [
            [0,0,1],
            [1,1,0],
            [0,0,1]
        ],
        [#历史2
            [1,1,0],
            [0,1,1],
            [0,1,0]
        ],
        [
            [0,0,1],
            [1,1,0],
            [0,0,1]
        ],
        ...
        [#玩家通道
            [-1,-1,-1],
            [-1,-1,-1],
            [-1,-1,-1]
        ]
    ]

    """

    def __init__(self) -> None:
        super().__init__()
        if USE_HISTORY:
            self.in_channels = GAME_STATE_DIM[0] * HISTORY_LEN + 1
        else:
            self.in_channels = GAME_STATE_DIM[0] + 1

        self.layers_dict: dict[str, list[nn.Module]] = {
            "shared_layers": [],
            "policy_head": [],
            "value_head": [],
        }
        if USE_RESNET_BLOCK:
            self._append_shared_layers_with_resnet(in_channels=self.in_channels)
        else:
            self._append_shared_layers_without_resnet(in_channels=self.in_channels)

        self.shared_out_width, self.shared_out_height = self._get_shared_out_size()

        self._append_policy_head()
        self._append_value_head()

        self._shared_layers = nn.Sequential(*self.layers_dict["shared_layers"])
        self._policy_head = nn.Sequential(*self.layers_dict["policy_head"])
        self._value_head = nn.Sequential(*self.layers_dict["value_head"])

    @property
    def shared_layers(self) -> nn.Sequential:
        return self._shared_layers

    def policy_head(self, x: torch.Tensor) -> TensorActions:
        return self._policy_head(x)

    def value_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self._value_head(x)
        return torch.tanh(x)

    def _append_policy_head(self) -> None:
        # 添加策略头卷积层
        policy_channel_list = self._get_channel_list(
            POLICY_CONV2D_CHANNELS, POLICY_CONV2D_NUM
        )
        add_policy_conv = self._make_conv_adder("policy_head", "POLICY_CONV2D")

        out_w, out_h = self.shared_out_width, self.shared_out_height

        forward_wh: Callable[[int, int], tuple[int, int]] = (
            lambda w, h: self._calculate_conv_output_size(
                w,
                h,
                kernel_size=POLICY_CONV2D_KERNEL_SIZE,
                stride=POLICY_CONV2D_STRIDE,
                padding=POLICY_CONV2D_PADDING,
            )
        )

        if POLICY_CONV2D_NUM >= 2:
            add_policy_conv(RESNET_CHANNELS, policy_channel_list[0])
            out_w, out_h = forward_wh(out_w, out_h)
            for i in range(POLICY_CONV2D_NUM - 1):
                add_policy_conv(policy_channel_list[i], policy_channel_list[i + 1])
                out_w, out_h = forward_wh(out_w, out_h)
        else:
            add_policy_conv(RESNET_CHANNELS, policy_channel_list[-1])
            out_w, out_h = forward_wh(out_w, out_h)

        last_conv_out_channels = policy_channel_list[-1]

        # 展平并确保线性层输入维度正确
        self.layers_dict["policy_head"].append(nn.Flatten(1))
        policy_head_input_features = last_conv_out_channels * out_w * out_h

        # 添加策略头全连接层
        policy_linear_list = self._get_feature_list(
            POLICY_LINEAR_FEATURES, POLICY_LINEAR_NUM
        )
        add_policy_linear = self._make_linear_adder("policy_head", "POLICY_LINEAR")
        if POLICY_LINEAR_NUM >= 2:
            add_policy_linear(policy_head_input_features, policy_linear_list[0])
            for i in range(POLICY_LINEAR_NUM - 2):
                add_policy_linear(policy_linear_list[i], policy_linear_list[i + 1])
            add_policy_linear(policy_linear_list[-1], NUM_ACTION, use_activation=False)
        else:
            add_policy_linear(
                policy_head_input_features, NUM_ACTION, use_activation=False
            )

    def _append_value_head(self) -> None:
        # 添加价值头卷积层
        value_channel_list = self._get_channel_list(
            VALUE_CONV2D_CHANNELS, VALUE_CONV2D_NUM
        )
        add_value_conv = self._make_conv_adder("value_head", "VALUE_CONV2D")

        out_w, out_h = self.shared_out_width, self.shared_out_height
        forward_wh: Callable[[int, int], tuple[int, int]] = (
            lambda w, h: self._calculate_conv_output_size(
                w,
                h,
                kernel_size=VALUE_CONV2D_KERNEL_SIZE,
                stride=VALUE_CONV2D_STRIDE,
                padding=VALUE_CONV2D_PADDING,
            )
        )
        if VALUE_CONV2D_NUM >= 2:
            add_value_conv(RESNET_CHANNELS, value_channel_list[0])
            out_w, out_h = forward_wh(out_w, out_h)
            for i in range(VALUE_CONV2D_NUM - 1):
                add_value_conv(value_channel_list[i], value_channel_list[i + 1])
                out_w, out_h = forward_wh(out_w, out_h)
        else:
            add_value_conv(RESNET_CHANNELS, value_channel_list[-1])
            out_w, out_h = forward_wh(out_w, out_h)
        last_conv_out_channels = value_channel_list[-1]

        # 展平，确保线性层输入维度正确
        self.layers_dict["value_head"].append(nn.Flatten(1))
        value_head_input_features = last_conv_out_channels * out_w * out_h

        # 添加价值头全连接层
        value_linear_list = self._get_feature_list(
            VALUE_LINEAR_FEATURES, VALUE_LINEAR_NUM
        )
        add_value_linear = self._make_linear_adder("value_head", "VALUE_LINEAR")
        if VALUE_LINEAR_NUM >= 2:
            add_value_linear(value_head_input_features, value_linear_list[0])
            for i in range(VALUE_LINEAR_NUM - 2):
                add_value_linear(value_linear_list[i], value_linear_list[i + 1])
            add_value_linear(value_linear_list[-1], 1, use_activation=False)
        else:
            add_value_linear(value_head_input_features, 1, use_activation=False)

    def _append_shared_layers_with_resnet(self, in_channels: int) -> None:
        """添加包含 ResNet 块的层结构"""
        if USE_TRANSITION:
            # 添加过渡层：输入 -> 隐藏层 -> ResNet
            add_hidden_layer = self._make_conv_adder("shared_layers", "HIDDEN")
            hidden_channels = self._get_channel_list(HIDDEN_CHANNELS, NUM_HIDDEN_LAYER)

            # 构建隐藏层序列
            if NUM_HIDDEN_LAYER >= 2:
                add_hidden_layer(in_channels, hidden_channels[0])
                for i in range(NUM_HIDDEN_LAYER - 2):
                    add_hidden_layer(hidden_channels[i], hidden_channels[i + 1])
                add_hidden_layer(hidden_channels[-1], RESNET_CHANNELS)
            else:
                add_hidden_layer(in_channels, RESNET_CHANNELS)
        else:
            # 直接连接到 ResNet
            self._append_conv_block(
                model_type="shared_layers",
                in_channels=in_channels,
                out_channels=RESNET_CHANNELS,
                kernel_size=RESNET_KERNEL_SIZE,
                stride=RESNET_STRIDE,
                padding=RESNET_PADDING,
                active_func=RESNET_ACTIVE_FUNC,
            )

        # 添加 ResNet 块（数量由配置 RESNET_BLOCK_NUM 控制）
        for _ in range(RESNET_BLOCK_NUM):
            self.layers_dict["shared_layers"].append(Quick_MODEL_RESNET())

    def _append_shared_layers_without_resnet(self, in_channels: int) -> None:
        """添加纯卷积层结构（不使用 ResNet 块）"""
        add_conv_layer = self._make_conv_adder("shared_layers", "CONV2D")
        conv2d_channels = self._get_channel_list(CONV2D_CHANNELS, NUM_CONV2D)

        # 构建卷积层序列
        if NUM_CONV2D >= 2:
            add_conv_layer(in_channels, conv2d_channels[0])
            for i in range(NUM_CONV2D - 1):
                add_conv_layer(conv2d_channels[i], conv2d_channels[i + 1])
        else:
            add_conv_layer(in_channels, conv2d_channels[-1])

    def _append_conv_block(
        self,
        model_type: str,
        in_channels: int,
        out_channels: int,
        kernel_size: size_2_t,
        stride: size_2_t,
        padding: size_2_t,
        active_func: nn.Module,
    ) -> None:
        """添加一个完整的卷积块：Conv2d -> [BatchNorm] -> Activation -> [Dropout]"""
        # 卷积层
        self.layers_dict[model_type].append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )

        # 批归一化（可选）
        if USE_BATCHNORM:
            self.layers_dict[model_type].append(
                nn.BatchNorm2d(num_features=out_channels)
            )

        # 激活函数
        self.layers_dict[model_type].append(active_func)

        # Dropout（可选）
        if USE_DROPOUT:
            self.layers_dict[model_type].append(nn.Dropout(p=DROPOUT_P))

    def _append_linear_block(
        self,
        model_type: str,
        in_features: int,
        out_features: int,
        active_func: nn.Module,
        use_activation: bool = True,
    ) -> None:
        """添加一个完整的线性块：Linear -> Activation -> [Dropout]"""
        # 线性层
        self.layers_dict[model_type].append(
            nn.Linear(in_features=in_features, out_features=out_features)
        )

        # 激活函数
        if use_activation:
            self.layers_dict[model_type].append(active_func)

        # Dropout（可选）
        if USE_DROPOUT:
            self.layers_dict[model_type].append(nn.Dropout(p=DROPOUT_P))

    def _make_conv_adder(self, model_type: str, layer_type: str):
        config = CONFIGS[layer_type]
        if not isinstance(config, ConvConfig):
            raise ValueError(f"配置类型错误，期望 ConvConfig，但得到 {type(config)}")

        def add_layer(in_channels: int, out_channels: int) -> None:
            self._append_conv_block(
                model_type=model_type,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=config.kernel_size,
                stride=config.stride,
                padding=config.padding,
                active_func=config.active,
            )

        return add_layer

    def _make_linear_adder(self, model_type: str, layer_type: str):
        config = CONFIGS[layer_type]
        if not isinstance(config, LinearConfig):
            raise ValueError(f"配置类型错误，期望 LinearConfig，但得到 {type(config)}")

        def add_layer(
            in_features: int,
            out_features: int,
            use_activation: bool = True,
        ) -> None:
            self._append_linear_block(
                model_type=model_type,
                in_features=in_features,
                out_features=out_features,
                active_func=config.active,
                use_activation=use_activation,
            )

        return add_layer

    def _get_channel_list(self, channles: ChannelsType, channel_num: int) -> list[int]:
        return [channles] * channel_num if isinstance(channles, int) else list(channles)

    def _get_feature_list(self, features: FeaturesType, feature_num: int) -> list[int]:
        return self._get_channel_list(features, feature_num)

    def _get_shared_out_size(self) -> tuple[int, int]:
        """获取共享层输出的特征图尺寸，供后续头部层构建时使用"""
        # 计算卷积层对特征图尺寸的影响
        model = self.layers_dict["shared_layers"]
        input_ = torch.zeros((1, self.in_channels, *GAME_STATE_DIM[1:]))  # 模拟输入
        with torch.no_grad():
            for layer in model:
                input_ = layer(input_)

        return input_.size(-1), input_.size(-2)  # 返回特征图的总尺寸（宽, 高）

    def _calculate_conv_output_size(
        self, width: int, height: int, kernel_size: int, stride: int, padding: int
    ) -> tuple[int, int]:
        """计算卷积层输出特征图的尺寸"""
        out_width = (width + 2 * padding - kernel_size) // stride + 1
        out_height = (height + 2 * padding - kernel_size) // stride + 1
        return out_width, out_height
