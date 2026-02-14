from config import *
from typing import Tuple


class BaseGame:
    def __init__(self) -> None:
        self._state = self.initial_state()
        self._player = PLAYER1

    def step(self, action: int) -> Tuple[TensorGameState, int, GameDone]:
        """执行action并返回下一状态和游戏是否结束

        Args:
            action (int): 执行的动作
        """
        next_state, child_player = self.next_state(self._state, action)
        done = self.is_terminal(self._state)
        return next_state, child_player, done

    def get_player(self) -> int:
        return self._player

    def get_state(self) -> TensorGameState:
        return self._state.detach().clone()

    def get_legal_mask(self) -> TensorActions:
        return self.legal_action_mask(self._state)

    def evaluation(self) -> float:
        return self.terminal_evaluation(self._state)

    @staticmethod
    def initial_state() -> TensorGameState:
        """返回一个开局状态

        Raises:
            NotImplementedError: 需要在子类中实现

        Returns:
            StateType: 游戏开局的状态
        """
        raise NotImplementedError

    @staticmethod
    def next_state(state: TensorGameState, action: int) -> tuple[TensorGameState, int]:
        """获取输入的状态执行action后得到的状态即对应的玩家

        Args:
            state (TensorGameState): 游戏状态
            action (int): 动作

        Raises:
            NotImplementedError: 需要在子类中实现

        Returns:
            tuple[TensorGameState, int]: 输入的状态执行action后得到的状态即对应的玩家
        """
        raise NotImplementedError

    @staticmethod
    def legal_action_mask(state: TensorGameState) -> TensorActions:
        """返回合法动作的遮罩，合法为1，非法为0

        Raises:
            NotImplementedError: 需要在子类中实现
        """
        raise NotImplementedError

    @staticmethod
    def is_terminal(state: TensorGameState) -> GameDone:
        """判断游戏是否结束

        Args:
            state (TensorGameState): 游戏状态

        Raises:
            NotImplementedError: 需要在子类中实现

        Returns:
            GameDone: 游戏是否结束
        """
        raise NotImplementedError

    @staticmethod
    def terminal_evaluation(state: TensorGameState) -> float:
        """游戏终局的评估函数

        Raises:
            NotImplementedError: 需要在子类中实现

        Returns:
            float: 返回终局时当前玩家视角的分数
        """
        raise NotImplementedError
