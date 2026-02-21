from __future__ import annotations

from dataclasses import dataclass, field, InitVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from games.base import BaseGame

import collections
from config import *
import torch
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class HistoryManager:
    game_cls: InitVar[type[BaseGame]]
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
        states = [s for s in self._history]
        if len(states[0].shape) == 2:
            state = torch.stack(states, dim=0)
        else:
            state = torch.cat(states, dim=0)
        return state.detach().clone()

    def load(self, history_state: StateWithHistory) -> None:
        self._history.clear()
        if len(self._state_shape) == 2:
            for i in range(HISTORY_LEN):
                self._history.append(history_state[i].clone())
        else:
            shape_0 = self._state_shape[0]
            for i in range(HISTORY_LEN):
                start = i * shape_0
                end = start + shape_0
                self._history.append(history_state[start:end].clone())
