"""五子棋游戏实现"""

from __future__ import annotations

import torch

from config import *
from games.base import BaseGame

from utils.logger import setup_logger

logger = setup_logger(__name__)

class GomokuGame(BaseGame):
    """五子棋 (15x15)

    状态表示:
        15x15 的张量，PLAYER1=1, PLAYER2=-1, 空格=0
        动作编码: action = row * 15 + col (0-224)
    """
    BOARD_SIZE = 15
    WIN_LENGTH = 5

    @staticmethod
    def initial_state() -> TensorGameState:
        return torch.zeros((GomokuGame.BOARD_SIZE, GomokuGame.BOARD_SIZE), dtype=torch.float32)

    @staticmethod
    def current_player(state: TensorGameState) -> int:
        moves = int(torch.count_nonzero(state).item())
        return PLAYER1 if moves % 2 == 0 else PLAYER2

    @staticmethod
    def next_state(state: TensorGameState, action: int) -> tuple[TensorGameState, int]:
        if action < 0 or action >= GomokuGame.BOARD_SIZE ** 2:
            raise ValueError("action 越界")

        row, col = divmod(action, GomokuGame.BOARD_SIZE)
        if state[row, col].item() != 0:
            raise ValueError("非法落子")

        current_player = GomokuGame.current_player(state)
        next_state = state.clone()
        next_state[row, col] = float(current_player)

        next_player = PLAYER2 if current_player == PLAYER1 else PLAYER1
        return next_state, next_player

    @staticmethod
    def legal_action_mask(state: TensorGameState) -> TensorActions:
        mask = (state.view(-1) == 0).float().unsqueeze(0)
        return mask

    @staticmethod
    def is_terminal(state: TensorGameState) -> bool:
        winner = GomokuGame._get_winner(state)
        if winner is not None:
            return True
        return torch.count_nonzero(state).item() == GomokuGame.BOARD_SIZE ** 2

    @staticmethod
    def _get_winner(state: TensorGameState) -> int | None:
        """检查是否有玩家赢了"""
        board = state
        size = GomokuGame.BOARD_SIZE
        win_len = GomokuGame.WIN_LENGTH

        # 检查所有可能的连线
        for row in range(size):
            for col in range(size):
                if board[row, col].item() == 0:
                    continue
                
                player = int(board[row, col].item())
                
                # 水平方向
                if col + win_len <= size:
                    if torch.all(board[row, col:col+win_len] == player):
                        return player
                
                # 竖直方向
                if row + win_len <= size:
                    if torch.all(board[row:row+win_len, col] == player):
                        return player
                
                # 右下对角线
                if row + win_len <= size and col + win_len <= size:
                    diagonal = torch.tensor([board[row+i, col+i].item() for i in range(win_len)])
                    if torch.all(diagonal == player):
                        return player
                
                # 左下对角线
                if row + win_len <= size and col - win_len + 1 >= 0:
                    diagonal = torch.tensor([board[row+i, col-i].item() for i in range(win_len)])
                    if torch.all(diagonal == player):
                        return player
        
        return None
