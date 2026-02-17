"""游戏参数"""

NO_PLAYER = 0
PLAYER1 = 1
PLAYER2 = -1
PLAYERS = (PLAYER1, PLAYER2)
# 如游戏有更多玩家需要手动添加新的常量并重写APP类中的run方法

"""Alphazero参数"""
C_PUCT = 1.0
ALPHA = 0.3
EPSILON = 0.25
NUM_SIMULATION = 800

USE_HISTORY = True
if USE_HISTORY:
    HISTORY_LEN = 8

RESNET_LAYERS_NUM = 5  # 默认提供的残差层大小

"""经验池参数"""
DEFAULT_CAPACITY = 10000

"""训练参数"""
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
SELF_PLAY_NUM = 500
LEARNING_RATE = 1e-3


START_TEMPERATURE = 1.0
END_TEMPERATURE = 0.1
TEMPERATURE_DECAY = 0.99  # 每轮自对弈后温度衰减的系数

"""日志配置"""
import logging

DEFAULT_LOG_LEVEL = logging.DEBUG
TRAIN_LOG_FREQUENCY = 30  # 每多少轮训练输出一次日志
LOSS_LOG_FREQUENCY = 30  # 每多少轮训练输出一次损失日志

"""多进程加速配置"""
NUM_SELF_PLAY_GAMES = 800  # 总共需要进行的自对弈场数
UPDATE_MODEL_FREQUENCY = 3  # 每多少轮自对弈后更新一次模型
MODEL_SAVE_FREQUENCY = 1000  # 每多少轮训练保存一次模型状态
MIN_EXP_SIZE_FOR_TRAINING = 1000  # 训练前经验池中至少需要的样本数量,必须小于经验池容量
if MIN_EXP_SIZE_FOR_TRAINING > DEFAULT_CAPACITY:
    raise ValueError(
        f"MIN_EXP_SIZE_FOR_TRAINING ({MIN_EXP_SIZE_FOR_TRAINING}) 必须小于经验池容量 ({DEFAULT_CAPACITY})"
    )


USE_MULTIPROCESSING = True
if USE_MULTIPROCESSING:
    import torch.multiprocessing as mp

    SELF_PLAY_WORKER_NUM = (
        mp.cpu_count() - 1
    )  # 自对弈进程数量，默认为CPU核心数减1，不建议修改
    TRAINING_WORKER_NUM = 1  # 训练进程数量，通常设置为1即可


TRAIN_FREQUENCY = 10  # 单进程模式下每多少轮自对弈后进行一次训练
TRAIN_EPOCHS = 50  # 单进程模式下每次训练的轮数
