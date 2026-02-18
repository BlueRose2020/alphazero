"""游戏参数"""

NO_PLAYER = 0
PLAYER1 = 1
PLAYER2 = -1
PLAYERS = (PLAYER1, PLAYER2)
# 如游戏有更多玩家需要手动添加新的常量并重写APP类中的run方法

"""经验池参数"""
DEFAULT_CAPACITY = 10000

"""数据增强参数"""
USE_DATA_ENHANCEMENT = True

if USE_DATA_ENHANCEMENT:
    DATA_ENHANCER_CLASS = "DataEnhancer"  # 数据增强器类名，默认为DataEnhancer，需与utils/data_enhancer.py中的类名一致
    """当数据增强器为DataEnhancer时，可以通过
    USE_FLIP和USE_ROTATION控制是否使用翻转和
    旋转增强，默认为True，若使用自定义数据增强
    器，则这两个参数无效，需自行在数据增强器类
    中实现相关功能
    
    ps:策略概率的旋转是基于状态的后两维进行的，如不满足请自行编写旋转函数
    """
    # 同时使用旋转和翻转会得到8倍的数据增强效果
    # 均为False时则增强器只返回原始数据，不进行任何增强
    USE_FLIP = True  # 是否使用翻转(水平翻转)增强，默认为True
    USE_ROATATION = True  # 是否使用旋转增强，默认为True

if USE_DATA_ENHANCEMENT and not (USE_FLIP or USE_ROATATION): # type: ignore
    from utils.logger import setup_logger

    logger = setup_logger(__name__)
    logger.warning("数据增强已启用但未启用翻转或旋转增强，将只返回原始数据")

"""Alphazero参数"""
C_PUCT = 1.0
ALPHA = 0.3
EPSILON = 0.25
NUM_SIMULATION = 200

USE_HISTORY = True
if USE_HISTORY:
    HISTORY_LEN = 8


"""训练参数"""
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

START_TEMPERATURE = 1.0
END_TEMPERATURE = 0.1
TEMPERATURE_DECAY = 0.99  # 每轮自对弈后温度衰减的系数


"""日志配置"""
import logging

DEFAULT_LOG_LEVEL = logging.DEBUG
TRAIN_LOG_FREQUENCY = 300  # 每多少轮训练输出一次日志
LOSS_LOG_FREQUENCY = 50  # 每多少轮训练输出一次损失日志

"""多进程加速配置"""
from random import randint
SEED_BIAS = randint(0, 10000)  # 多进程时每个进程随机数种子的偏移量，确保不同进程使用不同的随机数序列
NUM_SELF_PLAY_GAMES = 50  # 总共需要进行的自对弈场数
UPDATE_MODEL_FREQUENCY = 5  # 每多少轮自对弈后更新一次模型(单个进程)
MODEL_SAVE_FREQUENCY = 20000  # 每多少轮训练自动保存一次模型状态
EXP_SAVE_FREQUENCY = 10 * 60  # 间隔多长时间（秒）保存一次经验池状态
MIN_EXP_SIZE_FOR_TRAINING = 500  # 训练前经验池中至少需要的样本数量,必须小于经验池容量

if MIN_EXP_SIZE_FOR_TRAINING > DEFAULT_CAPACITY:
    raise ValueError(
        f"MIN_EXP_SIZE_FOR_TRAINING ({MIN_EXP_SIZE_FOR_TRAINING}) 必须小于经验池容量 ({DEFAULT_CAPACITY})"
    )


USE_MULTIPROCESSING = True
if USE_MULTIPROCESSING:
    import torch.multiprocessing as mp

    SELF_PLAY_WORKER_NUM = (
        mp.cpu_count() - 2 if mp.cpu_count() > 2 else 1
    )  # 自对弈进程数量，默认为CPU核心数减2(避免死机)，不建议修改
    TRAINING_THREAD_NUM = 2  # 训练线程数量（进程数为1），默认为2，过多可能导致性能下降
    TRAIN_EPOCHS_AFTER_SELF_PLAY_DONE = 200  # 所有自对弈完成后再进行多少轮训练

TRAIN_FREQUENCY = 10  # 单进程模式下每多少轮自对弈后进行一次训练
TRAIN_EPOCHS = 50  # 单进程模式下每次训练的轮数
