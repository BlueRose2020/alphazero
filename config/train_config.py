"""训练参数"""
_DEVICE = "auto"
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# 训练过程中温度参数的设置，控制探索程度
START_TEMPERATURE = 1.0
END_TEMPERATURE = 0.1
TEMPERATURE_DECAY = 0.99  # 每轮自对弈后温度衰减的系数

"""多进程加速配置"""
USE_MULTIPROCESSING = True
# 以下几个参数仅当USE_MULTIPROCESSING为True时有效
_SEED_BIAS = "random"
# 多进程时每个进程随机数种子的偏移量，确保不同进程使用不同的随机数序列
# 默认为 "random"，会在每次运行时随机生成一个偏移量，范围为0到10000
# 你也可以设置为一个固定的整数，以确保每次运行使用相同
NUM_SELF_PLAY_GAMES = 50  # 总共需要进行的自对弈场数
UPDATE_MODEL_FREQUENCY = 5  # 每多少轮自对弈后更新一次模型(单个进程)
MODEL_SAVE_FREQUENCY = 20000  # 每多少轮训练自动保存一次模型状态
EXP_SAVE_FREQUENCY = 10 * 60  # 间隔多长时间（秒）保存一次经验池状态
MIN_EXP_SIZE_FOR_TRAINING = 500
# 训练前经验池中至少需要的样本数量,必须小于经验池容量
# 单进程模式下每固定轮数自对弈后进行一次训练，故无需设置此参数

_SELF_PLAY_WORKER_NUM = "auto"
# 自对弈进程数量，默认为 "auto"，会自动设置为CPU核心数减2
# (一个进程用于训练，另一个进程用于避免死机)
# 你可以设置为一个固定的整数
TRAINING_THREAD_NUM = 2  # 训练线程数量（进程数为1），默认为2，过多可能导致性能下降
TRAIN_EPOCHS_AFTER_SELF_PLAY_DONE = 200  
# 所有自对弈完成后再进行多少轮训练，该参数是为了避免
# 最后几轮自对弈得到的经验还未被训练到模型中就结束了训练过程

"""单进程训练配置"""
# 以下参数仅在USE_MULTIPROCESSING为False时有效
TRAIN_FREQUENCY = 10  # 单进程模式下每多少轮自对弈后进行一次训练
TRAIN_EPOCHS = 50  # 单进程模式下每次训练的轮数

# ====================================================================
# 以下内容请勿修改，除非你知道自己在做什么，否则可能会导致程序无法运行
# ====================================================================
from config.basic import USE_HISTORY, _HISTORY_LEN
if USE_HISTORY:
    HISTORY_LEN = _HISTORY_LEN

"""训练参数"""
import torch

if _DEVICE == "auto":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif _DEVICE == "cuda":
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        raise ValueError("未找到可用的CUDA设备，请检查你的环境配置")
elif _DEVICE == "cpu":
    DEVICE = torch.device("cpu")
else:
    raise ValueError(f"无效的DEVICE配置: {_DEVICE}，请使用 'auto', 'cuda' 或 'cpu'")

"""多进程加速配置"""
if _SEED_BIAS == "random":
    from random import randint

    SEED_BIAS = randint(0, 10000)
elif isinstance(_SEED_BIAS, int) and _SEED_BIAS >= 0:
    SEED_BIAS = _SEED_BIAS
else:
    raise ValueError(f"无效的SEED_BIAS配置: {_SEED_BIAS}，请使用 'random' 或一个正整数")

from config.utils_config import DEFAULT_CAPACITY

if MIN_EXP_SIZE_FOR_TRAINING > DEFAULT_CAPACITY:
    raise ValueError(
        f"MIN_EXP_SIZE_FOR_TRAINING ({MIN_EXP_SIZE_FOR_TRAINING}) 必须小于经验池容量 ({DEFAULT_CAPACITY})"
    )

if USE_MULTIPROCESSING:
    if _SELF_PLAY_WORKER_NUM == "auto":
        import torch.multiprocessing as mp

        SELF_PLAY_WORKER_NUM = (
            mp.cpu_count() - 2 if mp.cpu_count() > 2 else 1
        )  # 自对弈进程数量，默认为CPU核心数减2(避免死机)，不建议修改
    elif isinstance(_SELF_PLAY_WORKER_NUM, int) and _SELF_PLAY_WORKER_NUM > 0:
        SELF_PLAY_WORKER_NUM = _SELF_PLAY_WORKER_NUM
    else:
        raise ValueError(
            f"无效的SELF_PLAY_WORKER_NUM配置: {_SELF_PLAY_WORKER_NUM}，请使用 'auto' 或一个正整数"
        )

