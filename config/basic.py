"""游戏参数"""
NO_PLAYER = 0
PLAYER1 = 1
PLAYER2 = -1
PLAYERS = (PLAYER1, PLAYER2)
# 如游戏有更多玩家需要手动添加新的常量并重写APP类中的run方法

"""Alphazero参数"""
USE_HISTORY = True
_HISTORY_LEN = 8  # 历史状态的数量，仅当USE_HISTORY为True时有效

C_PUCT = 1.0
ALPHA = 0.3  # 在根节点添加Dirichlet噪声的分布参数，默认为0.3，范围为[0,1]
EPSILON = 0.25  # 在根节点添加Dirichlet噪声的权重，默认为0.25，范围为[0,1]
# 训练时的模拟次数在train_config.py中配置，
# 推理时的模拟次数在调用时的ai_config中配置

# 虚拟损失相关参数
USE_VIRTUAL_LOSS = False # play时可以用ai_config.use_virtual_loss来覆盖该参数
# 训练初期(模拟次数较小时)不建议使用虚拟损失，
# 测试结果为在井字棋和4*4点格棋中，使用虚拟损
# 失反而会导致训练效果变差，推测原因是虚拟损失
# 引入的数据噪声过大导致训练不稳定，而较小的模
# 拟次数本身就已经足够快了，不需要通过虚拟损失
# 来加速了。
VIRTUAL_LOSS = 0.3 # 虚拟损失，须大于0
# 本框架的虚拟损失用于批量推理叶节点，减少
# 单次推理次数，从而加速训练过程，并未使用
# 多线程
INFER_BATCH_SIZE = 16  # 批量推理叶节点的大小
"""ps: 过大的虚拟损失和过大的批量推理大小可能会
导致数据噪声过大，导致训练不稳定"""


# ====================================================================
# 以下内容请勿修改，除非你知道自己在做什么，否则可能会导致程序无法运行
# ====================================================================
# 检查EPSILON和ALPHA的值是否合法
if USE_HISTORY:
    HISTORY_LEN = _HISTORY_LEN

if not (0 <= EPSILON <= 1):
    raise ValueError(f"EPSILON的值必须在0和1之间，当前值: {EPSILON}")
if not (0 <= ALPHA <= 1):
    raise ValueError(f"ALPHA的值必须在0和1之间，当前值: {ALPHA}")
if VIRTUAL_LOSS <= 0:
    raise ValueError(f"VIRTUAL_LOSS的值必须大于0，当前值: {VIRTUAL_LOSS}")