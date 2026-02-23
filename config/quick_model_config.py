from torch import nn
from config.type_alias import *
from dataclasses import dataclass

"""必须设置以下两个常量与游戏匹配"""
# ======================================= #
GAME_STATE_DIM = (3, 3)
# 不包括批次维度和历史,例如示例中的井字棋为(3,3)
# 点格棋为(3,4,4)
NUM_ACTION = 9
# ======================================= #

# 以下是模型配置，修改后快速模型会根据这些配置自动调整网络结构，无需修改模型代码
# 可以配置的包括：
# 一. 基础配置：
# 1. 是否使用残差块（USE_RESNET_BLOCK）
# 2. 是否使用批归一化（USE_BATCHNORM）
# 3. 是否使用丢弃（USE_DROPOUT）及丢弃概率（DROPOUT_P）

# 二.卷积配置：
# 分为几个大部分：
# 1. 输入和残差连接间的过渡层配置
# 2. 过渡层卷积配置
# 3. 残差块卷积配置
# 4. 非残差卷积配置（当不使用残差块时）
# 5. 策略头卷积和全连接配置
# 6. 价值头卷积和全连接配置
# 每个部分的卷积配置包括：
# 1. 卷积层数量
# 2. 卷积通道数
# 3. 卷积核大小
# 4. 卷积步幅
# 5. 卷积填充
# 6. 卷积激活函数
# 全连接配置包括：
# 1. 全连接层数量
# 2. 全连接层特征数
# 3. 全连接层激活函数
# 可以通过设置USE_UNIFICATION_xxx为True来统一设置
# 不同层的这些参数（通过设置KERNEL_SIZE，STRIDE，PADDING等）
# 通道数和特征数接受整数或整数列表/元组，如果是列表/元组则需要与
# 注释匹配，且会依次使用列表/元组中的值作为每层的配置

USE_RESNET_BLOCK = True
"""以下两个不建议同时启用"""
USE_BATCHNORM = True
USE_DROPOUT = False

if USE_DROPOUT:
    DROPOUT_P = 0.25  # 丢弃概率


# 配置统一设置参数
# ======================================= #
USE_UNIFICATION_KERNEL_SIZE = True
USE_UNIFICATION_STRIDE = True
USE_UNIFICATION_PADDING = True
USE_UNIFICATION_ACTIVATE_FUNC = True

# kernel_size,stride,padding均支持整数和二元整数元组，包括之后的其他配置
# 以下为统一设置的参数，若对应的USE_UNIFICATION_xxx为True，则所有相关层的该参数均使用这里的设置
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
ACTIVATE_FUNC = nn.ReLU()
# ======================================= #

# ==================共享层配置================= #
# 以下为使用残差块的配置
_RESNET_LAYERS_NUM = 5  # 残差块中卷积层的数量

_USE_TRANSITION = True
# 是否在残差连接和输入间加入过渡的卷积层(残差层数量不包括这层卷积)
# ps:即使不使用默认也会一层卷积用于处理输入和残差的通道数不匹配问题
# 即设为False与设为True且过渡层数量为1是相同的

# 过渡层配置
_NUM_HIDDEN_LAYER = 1
_HIDDEN_CHANNELS = 64
# 从输入到残差连接的过度层通道数，只有在过渡层通道数大于一时才有效
# 若使用元组或列表，则只需要NUM_HIDDEN_LAYER-1个数即可，开头与结
# 尾分别使用神经网络输入和残差层输入
_HIDDEN_KERNEL_SIZE = 3
_HIDDEN_STRIDE = 1
_HIDDEN_PADDING = 1
_HIDDEN_ACTIVE_FUNC = nn.ReLU()

# 残差块配置
_RESNET_BLOCK_NUM = 1
_RESNET_CHANNELS = 128
_RESNET_KERNEL_SIZE = 3
_RESNET_STRIDE = 1
_RESNET_PADDING = 1
_RESNET_ACTIVE_FUNC = nn.ReLU()

# 以下为不使用残差块的配置
_NUM_CONV2D = 12
_CONV2D_CHANNELS = [*([64] * 6), *([128] * 6)]
# 卷积通道数，若使用元组或列表，需要NUM_CONV2D
# 个数即可，开头使用神经网络输入
_CONV2D_KERNEL_SIZE = 3
_CONV2D_STRIDE = 1
_CONV2D_PADDING = 1
_CONV2D_ACTIVE_FUNC = nn.ReLU()
# ====================================== #

# ==================双头配置================= #
# -----策略头配置-----
_POLICY_CONV2D_NUM = 1
_POLICY_CONV2D_CHANNELS = 2
# 卷积通道数，若使用元组或列表，需要POLICY_CONV2D_NUM
# 个数，开头使用共享层输入
_POLICY_CONV2D_KERNEL_SIZE = 3
_POLICY_CONV2D_STRIDE = 1
_POLICY_CONV2D_PADDING = 1
_POLICY_CONV2D_ACTIVE_FUNC = nn.ReLU()

_POLICY_LINEAR_NUM = 2
_POLICY_LINEAR_FEATURES = 64
# 全连接层特征数，若使用元组或列表，需要POLICY_LINEAR_NUM-1个数，
# 开头使用卷积层输出特征数，结尾使用动作数
_POLICY_LINEAR_ACTIVE_FUNC = nn.ReLU()

# -----价值头配置-----
_VALUE_CONV2D_NUM = 1
_VALUE_CONV2D_CHANNELS = 1
# 卷积通道数，若使用元组或列表，需要VALUE_CONV2D_NUM
# 个数，开头使用共享层输入
_VALUE_CONV2D_KERNEL_SIZE = 3
_VALUE_CONV2D_STRIDE = 1
_VALUE_CONV2D_PADDING = 1
_VALUE_CONV2D_ACTIVE_FUNC = nn.ReLU()

_VALUE_LINEAR_NUM = 2
_VALUE_LINEAR_FEATURES = 64
# 全连接层特征数，若使用元组或列表，需要VALUE_LINEAR_NUM-1个数，
# 开头使用卷积层输出特征数，结尾使用1
_VALUE_LINEAR_ACTIVE_FUNC = nn.ReLU()
# ====================================== #


# 以下内容请勿修改，除非你知道自己在做什么
# ======================================= #
@dataclass(frozen=True)
class ConvConfig:
    kernel_size: size_2_t
    stride: size_2_t
    padding: size_2_t
    active: nn.Module


@dataclass(frozen=True)
class LinearConfig:
    active: nn.Module


CONFIGS: dict[str, ConvConfig | LinearConfig] = {}
"""
    "HIDDEN": ConvConfig(
        HIDDEN_KERNEL_SIZE, HIDDEN_STRIDE, HIDDEN_PADDING, HIDDEN_ACTIVE_FUNC
    ),
    "CONV2D": ConvConfig(
        CONV2D_KERNEL_SIZE, CONV2D_STRIDE, CONV2D_PADDING, CONV2D_ACTIVE_FUNC
    ),
    "RESNET": ConvConfig(
        RESNET_KERNEL_SIZE, RESNET_STRIDE, RESNET_PADDING, RESNET_ACTIVE_FUNC
    ),
    "POLICY_CONV2D": ConvConfig(
        POLICY_CONV2D_KERNEL_SIZE,
        POLICY_CONV2D_STRIDE,
        POLICY_CONV2D_PADDING,
        POLICY_CONV2D_ACTIVE_FUNC,
    ),
    "POLICY_LINEAR": LinearConfig(
        POLICY_LINEAR_ACTIVE_FUNC,
    ),
    "VALUE_CONV2D": ConvConfig(
        VALUE_CONV2D_KERNEL_SIZE,
        VALUE_CONV2D_STRIDE,
        VALUE_CONV2D_PADDING,
        VALUE_CONV2D_ACTIVE_FUNC,
    ),
    "VALUE_LINEAR": LinearConfig(VALUE_LINEAR_ACTIVE_FUNC),
"""
# =======================================


# 残差配置
if USE_RESNET_BLOCK:
    RESNET_LAYERS_NUM = _RESNET_LAYERS_NUM  # 残差块中卷积层的数量

    USE_TRANSITION = _USE_TRANSITION
    # 是否在残差连接和输入间加入过渡的卷积层(残差层数量不包括这层卷积)
    # ps:即使不使用默认也会一层卷积用于处理输入和残差的通道数不匹配问题
    # 即与使用且过渡层数量为1是相同的
    if USE_TRANSITION:
        NUM_HIDDEN_LAYER = _NUM_HIDDEN_LAYER
        HIDDEN_CHANNELS: int | tuple[int, ...] | list[int] = _HIDDEN_CHANNELS
        # 从输入到残差连接的过度层通道数，只有在过渡层通道数大于一时才有效
        # 若使用元组或列表，则只需要NUM_HIDDEN_LAYER-1个数即可，开头与结
        # 尾分别使用神经网络输入和残差层输入

        HIDDEN_ACTIVE_FUNC = (
            _HIDDEN_ACTIVE_FUNC
            if not USE_UNIFICATION_ACTIVATE_FUNC
            else ACTIVATE_FUNC  # type:ignore
        )

        HIDDEN_KERNEL_SIZE = (
            _HIDDEN_KERNEL_SIZE
            if not USE_UNIFICATION_KERNEL_SIZE
            else KERNEL_SIZE  # type:ignore
        )
        HIDDEN_STRIDE = (
            _HIDDEN_STRIDE if not USE_UNIFICATION_STRIDE else STRIDE  # type:ignore
        )
        HIDDEN_PADDING = (
            _HIDDEN_PADDING if not USE_UNIFICATION_PADDING else PADDING  # type:ignore
        )

        CONFIGS["HIDDEN"] = ConvConfig(
            kernel_size=HIDDEN_KERNEL_SIZE,
            stride=HIDDEN_STRIDE,
            padding=HIDDEN_PADDING,
            active=HIDDEN_ACTIVE_FUNC,
        )

    RESNET_BLOCK_NUM = _RESNET_BLOCK_NUM
    RESNET_CHANNELS = _RESNET_CHANNELS

    RESNET_KERNEL_SIZE = (
        _RESNET_KERNEL_SIZE
        if not USE_UNIFICATION_KERNEL_SIZE
        else KERNEL_SIZE  # type:ignore
    )
    RESNET_STRIDE = (
        _RESNET_STRIDE if not USE_UNIFICATION_STRIDE else STRIDE  # type:ignore
    )
    RESNET_PADDING = (
        _RESNET_PADDING if not USE_UNIFICATION_PADDING else PADDING  # type:ignore
    )
    RESNET_ACTIVE_FUNC = (
        _RESNET_ACTIVE_FUNC
        if not USE_UNIFICATION_ACTIVATE_FUNC
        else ACTIVATE_FUNC  # type:ignore
    )

    CONFIGS["RESNET"] = ConvConfig(
        kernel_size=RESNET_KERNEL_SIZE,
        stride=RESNET_STRIDE,
        padding=RESNET_PADDING,
        active=RESNET_ACTIVE_FUNC,
    )

else:
    NUM_CONV2D = _NUM_CONV2D
    CONV2D_CHANNELS: ChannelsType = _CONV2D_CHANNELS
    # 卷积通道数，若使用元组或列表，需要NUM_CONV2D
    # 个数即可，开头使用神经网络输入
    CONV2D_KERNEL_SIZE = (
        _CONV2D_KERNEL_SIZE
        if not USE_UNIFICATION_KERNEL_SIZE
        else KERNEL_SIZE  # type:ignore
    )
    CONV2D_STRIDE = (
        _CONV2D_STRIDE if not USE_UNIFICATION_STRIDE else STRIDE  # type:ignore
    )
    CONV2D_PADDING = (
        _CONV2D_PADDING if not USE_UNIFICATION_PADDING else PADDING  # type:ignore
    )
    CONV2D_ACTIVE_FUNC = (
        _CONV2D_ACTIVE_FUNC
        if not USE_UNIFICATION_ACTIVATE_FUNC
        else ACTIVATE_FUNC  # type:ignore
    )

    CONFIGS["CONV2D"] = ConvConfig(
        kernel_size=CONV2D_KERNEL_SIZE,
        stride=CONV2D_STRIDE,
        padding=CONV2D_PADDING,
        active=CONV2D_ACTIVE_FUNC,
    )

# -----策略头配置-----
# 卷积
POLICY_CONV2D_NUM = _POLICY_CONV2D_NUM
POLICY_CONV2D_CHANNELS: ChannelsType = _POLICY_CONV2D_CHANNELS
# 卷积通道数，若使用元组或列表，需要POLICY_CONV2D_NUM
# 个数，开头使用共享层输入
POLICY_CONV2D_KERNEL_SIZE = (
    _POLICY_CONV2D_KERNEL_SIZE
    if not USE_UNIFICATION_KERNEL_SIZE
    else KERNEL_SIZE  # type:ignore
)
POLICY_CONV2D_STRIDE = (
    _POLICY_CONV2D_STRIDE if not USE_UNIFICATION_STRIDE else STRIDE  # type:ignore
)
POLICY_CONV2D_PADDING = (
    _POLICY_CONV2D_PADDING if not USE_UNIFICATION_PADDING else PADDING  # type:ignore
)
POLICY_CONV2D_ACTIVE_FUNC = (
    _POLICY_CONV2D_ACTIVE_FUNC
    if not USE_UNIFICATION_ACTIVATE_FUNC
    else ACTIVATE_FUNC  # type:ignore
)
# 全连接
POLICY_LINEAR_NUM = _POLICY_LINEAR_NUM
POLICY_LINEAR_FEATURES: FeaturesType = _POLICY_LINEAR_FEATURES
# 全连接层特征数，若使用元组或列表，需要POLICY_LINEAR_NUM-1个数，
# 开头使用卷积层输出特征数，结尾使用动作数

POLICY_LINEAR_ACTIVE_FUNC = (
    _POLICY_LINEAR_ACTIVE_FUNC
    if not USE_UNIFICATION_ACTIVATE_FUNC
    else ACTIVATE_FUNC  # type:ignore
)  # 不包括输出层激活函数

CONFIGS["POLICY_CONV2D"] = ConvConfig(
    kernel_size=POLICY_CONV2D_KERNEL_SIZE,
    stride=POLICY_CONV2D_STRIDE,
    padding=POLICY_CONV2D_PADDING,
    active=POLICY_CONV2D_ACTIVE_FUNC,
)
CONFIGS["POLICY_LINEAR"] = LinearConfig(active=POLICY_LINEAR_ACTIVE_FUNC)

# -----价值头配置-----
# 卷积
VALUE_CONV2D_NUM = _VALUE_CONV2D_NUM
VALUE_CONV2D_CHANNELS: ChannelsType = _VALUE_CONV2D_CHANNELS
# 卷积通道数，若使用元组或列表，需要VALUE_CONV2D_NUM
# 个数，开头使用共享层输入
VALUE_CONV2D_KERNEL_SIZE = (
    _VALUE_CONV2D_KERNEL_SIZE
    if not USE_UNIFICATION_KERNEL_SIZE
    else KERNEL_SIZE  # type:ignore
)
VALUE_CONV2D_STRIDE = (
    _VALUE_CONV2D_STRIDE if not USE_UNIFICATION_STRIDE else STRIDE  # type:ignore
)
VALUE_CONV2D_PADDING = (
    _VALUE_CONV2D_PADDING if not USE_UNIFICATION_PADDING else PADDING  # type:ignore
)
VALUE_CONV2D_ACTIVE_FUNC = (
    _VALUE_CONV2D_ACTIVE_FUNC
    if not USE_UNIFICATION_ACTIVATE_FUNC
    else ACTIVATE_FUNC  # type:ignore
)
# 全连接
VALUE_LINEAR_NUM = _VALUE_LINEAR_NUM
VALUE_LINEAR_FEATURES: FeaturesType = _VALUE_LINEAR_FEATURES
# 全连接层特征数，若使用元组或列表，需要VALUE_LINEAR_NUM-1个数，
# 开头使用卷积层输出特征数，结尾使用1
VALUE_LINEAR_ACTIVE_FUNC = (
    _VALUE_LINEAR_ACTIVE_FUNC
    if not USE_UNIFICATION_ACTIVATE_FUNC
    else ACTIVATE_FUNC  # type:ignore
)  # 不包括输出层激活函数

CONFIGS["VALUE_CONV2D"] = ConvConfig(
    kernel_size=VALUE_CONV2D_KERNEL_SIZE,
    stride=VALUE_CONV2D_STRIDE,
    padding=VALUE_CONV2D_PADDING,
    active=VALUE_CONV2D_ACTIVE_FUNC,
)
CONFIGS["VALUE_LINEAR"] = LinearConfig(active=VALUE_LINEAR_ACTIVE_FUNC)

"""检查配置"""
# 检查残差层
if USE_RESNET_BLOCK and USE_TRANSITION:  # type:ignore
    if isinstance(HIDDEN_CHANNELS, list) or isinstance(  # type:ignore
        HIDDEN_CHANNELS, tuple  # type:ignore
    ):
        assert len(HIDDEN_CHANNELS) == (
            NUM_HIDDEN_LAYER - 1  # type:ignore
        ), "通道列表长度与通道数不匹配"
# 检查卷积通道（未使用残差层）
if not USE_RESNET_BLOCK:
    if isinstance(CONV2D_CHANNELS, tuple) or isinstance(  # type:ignore
        CONV2D_CHANNELS, list  # type:ignore
    ):
        assert len(CONV2D_CHANNELS) == (
            NUM_CONV2D  # type:ignore
        ), "通道列表长度与通道数不匹配"
# 检查策略头卷积通道
if isinstance(POLICY_CONV2D_CHANNELS, tuple) or isinstance(  # type:ignore
    POLICY_CONV2D_CHANNELS, list  # type:ignore
):
    assert len(POLICY_CONV2D_CHANNELS) == (
        POLICY_CONV2D_NUM  # type:ignore
    ), "策略头卷积层通道列表长度与通道数不匹配"
# 检查策略头全连接层特征数
if isinstance(POLICY_LINEAR_FEATURES, tuple) or isinstance(  # type:ignore
    POLICY_LINEAR_FEATURES, list  # type:ignore
):
    assert len(POLICY_LINEAR_FEATURES) == (
        POLICY_LINEAR_NUM  # type:ignore
    ), "策略头全连接层特征列表长度与全连接层数不匹配"
# 检查价值头卷积通道
if isinstance(VALUE_CONV2D_CHANNELS, tuple) or isinstance(  # type:ignore
    VALUE_CONV2D_CHANNELS, list  # type:ignore
):
    assert len(VALUE_CONV2D_CHANNELS) == (
        VALUE_CONV2D_NUM  # type:ignore
    ), "价值头卷积层通道列表长度与通道数不匹配"
# 检查价值头全连接层特征数
if isinstance(VALUE_LINEAR_FEATURES, tuple) or isinstance(  # type:ignore
    VALUE_LINEAR_FEATURES, list  # type:ignore
):
    assert len(VALUE_LINEAR_FEATURES) == (
        VALUE_LINEAR_NUM  # type:ignore
    ), "价值头全连接层特征列表长度与全连接层数不匹配"
