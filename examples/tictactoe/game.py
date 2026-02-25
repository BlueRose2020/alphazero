"""井字棋游戏实现"""

from __future__ import annotations

import torch

from config import *
from games.base import BaseGame

from utils.logger import setup_logger

logger = setup_logger(__name__)

class TicTacToeGame(BaseGame):
    """井字棋 (3x3)

    状态表示:
        3x3 的张量，PLAYER1=1, PLAYER2=-1, 空格=0
    """
    @staticmethod
    def initial_state() -> TensorGameState:
        return torch.zeros((3, 3), dtype=torch.float32)

    @staticmethod
    def _current_player(state: TensorGameState) -> int:
        moves = int(torch.count_nonzero(state).item())
        return PLAYER1 if moves % 2 == 0 else PLAYER2

    @staticmethod
    def next_state(state: TensorGameState, action: int, player: int) -> tuple[TensorGameState, int]:
        if action < 0 or action >= 9:
            raise ValueError("action 越界")

        row, col = divmod(action, 3)
        next_state = TicTacToeGame._apply_move(state, row, col, player)
        return next_state, TicTacToeGame._next_player(player)

    @staticmethod
    def legal_action_mask(state: TensorGameState) -> TensorActions:
        mask = (state.view(-1) == 0).float().unsqueeze(0)
        return mask

    @staticmethod
    def is_terminal(state: TensorGameState) -> bool:
        winner = TicTacToeGame._get_winner(state)
        if winner is not None:
            return True
        return torch.count_nonzero(state).item() == 9

    @staticmethod
    def _get_winner(state: TensorGameState) -> int | None:
        lines = []
        lines.extend([state[i, :] for i in range(3)])
        lines.extend([state[:, i] for i in range(3)])
        lines.append(torch.diag(state))
        lines.append(torch.diag(torch.fliplr(state)))

        for line in lines:
            s = int(torch.sum(line).item())
            if s == 3:
                return PLAYER1
            if s == -3:
                return PLAYER2
        return None

    @staticmethod
    def _apply_move(state: TensorGameState, row: int, col: int, player: int) -> TensorGameState:
        if state[row, col].item() != 0:
            raise ValueError("非法落子")

        next_state = state.clone()
        next_state[row, col] = float(player)
        return next_state

    @staticmethod
    def _next_player(player: int) -> int:
        return PLAYER2 if player == PLAYER1 else PLAYER1
