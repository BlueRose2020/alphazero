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
NUM_SIMULATION = 200  # MCTS每次搜索的模拟次数

# ====================================================================
# 以下内容请勿修改，除非你知道自己在做什么，否则可能会导致程序无法运行
# ====================================================================
# 检查EPSILON和ALPHA的值是否合法
if not (0 <= EPSILON <= 1):
    raise ValueError(f"EPSILON的值必须在0和1之间，当前值: {EPSILON}")
if not (0 <= ALPHA <= 1):
    raise ValueError(f"ALPHA的值必须在0和1之间，当前值: {ALPHA}")