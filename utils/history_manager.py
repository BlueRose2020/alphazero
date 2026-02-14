from dataclasses import dataclass, field, InitVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from games.base import BaseGame

import collections
from config import *
import torch


@dataclass
class HistoryManager:
    game_cls: InitVar[type[BaseGame]] = BaseGame
    _history: HistoryDeque = field(
        default_factory=lambda: collections.deque(maxlen=HISTORY_LEN)
    )
    _state_shape: tuple[int, ...] = field(init=False)

    def __post_init__(self, game_cls: type[BaseGame]):
        self.reset(game_cls)
        self._state_shape = game_cls.initial_state().shape

    def __len__(self) -> int:
        return HISTORY_LEN

    def update(self, state: TensorGameState) -> None:
        self._history.append(state.clone())

    def reset(self, game_cls: type[BaseGame]) -> None:
        """重置为初始棋盘的状态（非清空）"""
        init_state = game_cls.initial_state()
        for _ in range(HISTORY_LEN):
            self._history.append(init_state)

    def get_state(self) -> StateWithHistory:
        states = [torch.as_tensor(s, dtype=torch.float32) for s in self._history]
        state = torch.stack(states, dim=0).unsqueeze(0)
        return state.detach().clone()

    def get_deque(self) -> HistoryDeque:
        return collections.deque(s.clone() for s in self._history)

    def load(self, history_state: StateWithHistory) -> None:
        self._history = collections.deque(s.clone() for s in history_state)
