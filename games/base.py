from __future__ import annotations
from config import *
from typing import Tuple

if USE_HISTORY:
    from utils.history_manager import HistoryManager


class BaseGame:
    def __init__(self) -> None:
        if USE_HISTORY:
            self.history_manager = HistoryManager(type(self))  # type: ignore
        self.reset()

    def reset(self) -> None:
        """重置游戏到初始状态"""
        self._state = self.initial_state()
        self._player = PLAYER1
        if USE_HISTORY:
            self.history_manager.reset(type(self))

    def step(self, action: int) -> GameDone:
        """执行action并返回下一状态和游戏是否结束

        Args:
            action (int): 执行的动作
        """
        self._state, self._player = self.next_state(self._state, action)
        
        done = self.is_terminal(self._state)
        if USE_HISTORY:
            self.history_manager.update(self._state)
        return done

    def get_player(self) -> int:
        return self._player

    def get_state(self) -> TensorGameState:
        return self._state.detach().clone()

    def get_history(self) -> StateWithHistory:
        if not USE_HISTORY:
            raise RuntimeError("未启用历史记录功能")
        return self.history_manager.get_state()

    def get_legal_mask(self) -> TensorActions:
        return self.legal_action_mask(self._state)

    def evaluation(self) -> float:
        return self.terminal_evaluation(self._state, self._player)

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

    @classmethod
    def terminal_evaluation(cls, state: TensorGameState, player: int) -> float:
        """对于玩家数为2的零和游戏，无需实现 terminal_evaluation，BaseGame 已提供默认实现
        如果需要自定义评估逻辑，可以重写此方法
        """
        winner = cls._get_winner(state)
        if winner == player:
            return 1.0
        if winner == -player:
            return -1.0
        return 0.0

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
