from torch import nn
from config.type_alias import *

"""必须设置以下两个常量与游戏匹配"""
# =======================================#
GAME_STATE_DIM = 1  # 用于表示棋盘状态的维度
NUM_ACTION = 9
# =======================================#

USE_RESNET_BLOCK = True
"""以下两个不建议同时启用"""
USE_BATCHNORM = True
USE_DROPOUT = False

if USE_DROPOUT:
    DROPOUT_P = 0.25  # 丢弃概率

USE_UNIFICATION_ACTIVATE_FUNC = True
if USE_UNIFICATION_ACTIVATE_FUNC:
    ACTIVATE_FUNC = nn.ReLU()

# 残差配置
if USE_RESNET_BLOCK:
    # kernel_size,stride,padding支持整数和二元整数元组
    USE_UNIFICATION_KERNEL_SIZE = True
    USE_UNIFICATION_STRIDE = True
    USE_UNIFICATION_PADDING = True

    if USE_UNIFICATION_KERNEL_SIZE:
        KERNEL_SIZE = 3
    if USE_UNIFICATION_STRIDE:
        STRIDE = 1
    if USE_UNIFICATION_PADDING:
        PADDING = 1

    USE_TRANSITION = True
    # 是否在残差连接和输入间加入过渡的卷积层(残差层数量不包括这层卷积)
    # ps:即使不使用默认也会一层卷积用于处理输入和残差的通道数不匹配问题
    # 即与使用且过渡层数量为1是相同的
    if USE_TRANSITION:
        NUM_HIDDEN_LAYER = 1
        HIDDEN_CHANNELS: int | tuple[int, ...] | list[int] = 64
        # 从输入到残差连接的过度层通道数，只有在过渡层通道数大于一时才有效
        # 若使用元组或列表，则只需要NUM_HIDDEN_LAYER-1个数即可，开头与结
        # 尾分别使用神经网络输入和残差层输入

        HIDDEN_ACTIVE_FUNC = (
            nn.ReLU()
            if not USE_UNIFICATION_ACTIVATE_FUNC
            else ACTIVATE_FUNC  # type:ignore
        )

        HIDDEN_KERNEL_SIZE = (
            3 if not USE_UNIFICATION_KERNEL_SIZE else KERNEL_SIZE  # type:ignore
        )
        HIDDEN_STRIDE = 1 if not USE_UNIFICATION_STRIDE else STRIDE  # type:ignore
        HIDDEN_PADDING = 1 if not USE_UNIFICATION_PADDING else PADDING  # type:ignore

    RESNET_NUM = 5
    RESNET_CHANNELS = 128

    RESNET_KERNEL_SIZE = (
        3 if not USE_UNIFICATION_KERNEL_SIZE else KERNEL_SIZE  # type:ignore
    )
    RESNET_STRIDE = 1 if not USE_UNIFICATION_STRIDE else STRIDE  # type:ignore
    RESNET_PADDING = 1 if not USE_UNIFICATION_PADDING else PADDING  # type:ignore
    RESNET_ACTIVE_FUNC = (
        nn.ReLU() if not USE_UNIFICATION_ACTIVATE_FUNC else ACTIVATE_FUNC  # type:ignore
    )

# 非残差配置
else:
    NUM_CONV2D = 12
    CONV2D_CHANNELS: ChannelsType = [*([64] * 6), *([128] * 6)]
    # 卷积通道数，若使用元组或列表，需要NUM_CONV2D
    # 个数即可，开头使用神经网络输入
    CONV2D_KERNEL_SIZE = 3
    CONV2D_STRIDE = 1
    CONV2D_PADDING = 1
    CONV2D_ACTIVE_FUNC = nn.ReLU()

# -----策略头配置-----
# 卷积
POLICY_CONV2D_NUM = 1
POLICY_CONV2D_CHANNELS: ChannelsType = 64
# 卷积通道数，若使用元组或列表，需要POLICY_CONV2D_NUM
# 个数，开头使用共享层输入
POLICY_CONV2D_KERNEL_SIZE = 3
POLICY_CONV2D_STRIDE = 1
POLICY_CONV2D_PADDING = 1
POLICY_CONV2D_ACTIVE_FUNC = nn.ReLU()
# 全连接
POLICY_LINEAR_NUM = 1
POLICY_LINEAR_FEATURES: FeaturesType = NUM_ACTION
# 全连接层特征数，若使用元组或列表，需要POLICY_LINEAR_NUM-1个数，
# 开头使用卷积层输出特征数，结尾使用动作数

POLICY_LINEAR_ACTIVE_FUNC = nn.ReLU()  # 不包括输出层激活函数

# -----价值头配置-----
# 卷积
VALUE_CONV2D_NUM = 1
VALUE_CONV2D_CHANNELS: ChannelsType = 64
# 卷积通道数，若使用元组或列表，需要VALUE_CONV2D_NUM
# 个数，开头使用共享层输入
VALUE_CONV2D_KERNEL_SIZE = 3
VALUE_CONV2D_STRIDE = 1
VALUE_CONV2D_PADDING = 1
VALUE_CONV2D_ACTIVE_FUNC = nn.ReLU()
# 全连接
VALUE_LINEAR_NUM = 1
VALUE_LINEAR_FEATURES: FeaturesType = 1
# 全连接层特征数，若使用元组或列表，需要VALUE_LINEAR_NUM-1个数，
# 开头使用卷积层输出特征数，结尾使用1
VALUE_LINEAR_ACTIVE_FUNC = nn.ReLU()  # 不包括输出层激活函数

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
