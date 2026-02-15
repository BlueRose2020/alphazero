"""游戏配置参数"""

NO_PLAYER = 0
PLAYER1 = 1
PLAYER2 = -1
# 如游戏有更多玩家需要手动添加新的常量

"""Alphazero参数"""
C_PUCT = 1.0
ALPHA = 0.3
EPSILON = 0.25
NUM_SIMULATION = 800

"""神经网络配置"""
USE_HISTORY = True
if USE_HISTORY:
    HISTORY_LEN = 8

RESNET_LAYERS_NUM = 5  # 默认提供的残差层大小

"""经验池参数"""
DEFAULT_CAPACITY = 10000

"""训练参数"""
BATCH_SIZE = 64
SELF_PLAY_NUM = 500

"""日志配置"""
import logging

DEFAULT_LOG_LEVEL = logging.DEBUG

"""多进程加速配置"""
USE_MULTIPROCESSING = True
if USE_MULTIPROCESSING:
    pass
