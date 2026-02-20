"""游戏类模板文件

使用说明:
1. 复制此文件到 games/ 目录并重命名
2. 修改类名
3. 实现所有必需的静态方法
"""

from games.base import BaseGame
from config import *


class TemplateGame(BaseGame):
    """游戏类模板

    必须实现的方法:
    - initial_state: 返回初始游戏状态
    - next_state: 执行动作后的状态转移
    - legal_action_mask: 合法动作掩码
    - is_terminal: 判断游戏是否结束
    - terminal_evaluation: 终局评估
    """

    def __init__(self) -> None:
        """初始化游戏实例

        BaseGame 的 __init__ 会自动初始化:
        - self._state: 当前游戏状态 (通过 initial_state() 获取)
        - self._player: 当前玩家 (默认为 PLAYER1)

        子类通常不需要重写此方法，除非需要额外的初始化逻辑
        """
        super().__init__()

    @staticmethod
    def initial_state() -> TensorGameState:
        """返回游戏的初始状态

        Returns:
            TensorGameState: 初始状态张量，shape 应与 GAME_STATE_DIM 匹配
        """
        raise NotImplementedError

    @staticmethod
    def next_state(state: TensorGameState, action: int) -> tuple[TensorGameState, int]:
        """执行动作后的状态转移，注意对于类似于象棋等的游戏，
        状态无需翻转，但UI层为了方便下棋，应实现状态翻转以保
        证玩家在下方，而模型在上方

        Args:
            state: 当前游戏状态
            action: 要执行的动作（0 到 num_action-1）

        Returns:
            tuple[TensorGameState, int]: (下一个状态, 下一个玩家)
        """
        raise NotImplementedError

    @staticmethod
    def legal_action_mask(state: TensorGameState) -> TensorActions:
        """返回当前状态下的合法动作掩码

        Args:
            state: 当前游戏状态

        Returns:
            TensorActions: 合法动作掩码，shape=(1, num_action)，1表示合法，0表示非法
        """
        raise NotImplementedError

    @staticmethod
    def is_terminal(state: TensorGameState) -> bool:
        """判断当前状态是否为终局

        Args:
            state: 当前游戏状态

        Returns:
            bool: True 表示游戏结束，False 表示游戏继续
        """
        raise NotImplementedError

    # 对于玩家数为2的零和游戏，无需实现 terminal_evaluation，BaseGame 已提供默认实现
    # 如果需要自定义评估逻辑，可以重写此方法
    # @classmethod
    # def terminal_evaluation(cls, state: TensorGameState, player: int) -> float:
    #     """评估终局状态的胜负

    #     Args:
    #         state: 终局状态

    #     Returns:
    #         float: 从当前玩家视角的评估值 (1.0=获胜, 0.0=平局, -1.0=失败)
    #     """

    @staticmethod
    def _get_winner(state: TensorGameState) -> int | None:
        """返回赢家，PLAYER1=1, PLAYER2=-1, 平局或未结束返回None
        如有更多玩家，需自行定义返回值含义并重写terminal_evaluation方法

        Args:
            state (TensorGameState): _description_

        Returns:
            int | None: _description_
        """
        raise NotImplementedError
