"""日志配置"""

import logging

DEFAULT_LOG_LEVEL = logging.DEBUG
TRAIN_LOG_FREQUENCY = 300  # 每多少轮训练输出一次日志
LOSS_LOG_FREQUENCY = 50  # 每多少轮训练输出一次损失日志

# 日志颜色配置
LOSS_COLOR = "orange"  # 训练损失日志颜色
TRAIN_EPOCH_COLOR = "bright_cyan"  # 训练轮数日志颜色
UPDATE_MODEL_COLOR = (100, 255, 100)  # 更新模型日志颜色
PROCESSING_COLOR = "bright_blue"  # 多进程的进程名称日志颜色

SAVE_MODEL_COLOR = "light_blue"  # 保存模型日志颜色
SAVE_EXP_COLOR = "light_blue"  # 保存经验池日志颜色
WORKER_FINISH_COLOR = "bright_magenta"  # 自对弈进程完成日志颜色
FINISH_COLOR = "bright_magenta"  # 训练完成日志颜色
# 你可以在此处添加更多日志类型的颜色配置，并在你的日志输出中使用这些颜色配置

"""经验池参数"""
DEFAULT_CAPACITY = 10000

"""数据增强参数"""
USE_DATA_ENHANCEMENT = True
# 以下参数仅当USE_DATA_ENHANCEMENT为True时有效
_DATA_ENHANCER_CLASS = "DataEnhancer"  # 数据增强器类名，默认为DataEnhancer，需与utils/data_enhancer.py中的类名一致
"""当数据增强器为DataEnhancer时，可以通过
USE_FLIP和USE_ROTATION控制是否使用翻转和
旋转增强，默认为True，若使用自定义数据增强
器，则这两个参数无效，需自行在数据增强器类
中实现相关功能

ps:策略概率的旋转是基于状态的后两维进行的，
如不满足请自行编写旋转函数
"""
# 同时使用旋转和翻转会得到8倍的数据增强效果
# 均为False时则增强器只返回原始数据，不进行任何增强
_USE_FLIP = False  # 是否使用翻转(水平翻转)增强，默认为False
_USE_ROATATION = False  # 是否使用旋转增强，默认为False


# ====================================================================
# 以下内容请勿修改，除非你知道自己在做什么，否则可能会导致程序无法运行
# ====================================================================
if USE_DATA_ENHANCEMENT:
    DATA_ENHANCER_CLASS = _DATA_ENHANCER_CLASS  # 数据增强器类名，默认为DataEnhancer，需与utils/data_enhancer.py中的类名一致
    """当数据增强器为DataEnhancer时，可以通过
    USE_FLIP和USE_ROTATION控制是否使用翻转和
    旋转增强，默认为True，若使用自定义数据增强
    器，则这两个参数无效，需自行在数据增强器类
    中实现相关功能
    
    ps:策略概率的旋转是基于状态的后两维进行的，
    如不满足请自行编写旋转函数
    """
    # 同时使用旋转和翻转会得到8倍的数据增强效果
    # 均为False时则增强器只返回原始数据，不进行任何增强
    USE_FLIP = _USE_FLIP  # 是否使用翻转(水平翻转)增强，默认为False
    USE_ROATATION = _USE_ROATATION  # 是否使用旋转增强，默认为False

if USE_DATA_ENHANCEMENT and not (USE_FLIP or USE_ROATATION):  # type: ignore
    from utils.logger import setup_logger

    logger = setup_logger(__name__)
    logger.warning("数据增强已启用但未启用翻转或旋转增强，将只返回原始数据")
