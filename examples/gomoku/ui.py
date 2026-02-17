from __future__ import annotations

import pygame as pg

from config.constants import PLAYER1, PLAYER2
from config.type_alias import TensorGameState
from ui.board import BoardActionResult, BoardView
from config import *
from ui.app import BaseApp, AIConfig
from nn_models.base import BaseModel
from .game import GomokuGame


class GomokuBoard(BoardView):
    """五子棋棋盘视图 (15x15)"""

    def _get_cell_size(self) -> float:
        """计算网格间距"""
        return self.rect.width / (GomokuGame.BOARD_SIZE - 1)

    def _get_pos_from_mouse(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        """将鼠标位置转换为棋盘坐标，越界返回 None"""
        if not self.rect.collidepoint(pos):
            return None

        cell_size = self._get_cell_size()
        col = int(round((pos[0] - self.rect.left) / cell_size))
        row = int(round((pos[1] - self.rect.top) / cell_size))

        if row < 0 or row >= GomokuGame.BOARD_SIZE or col < 0 or col >= GomokuGame.BOARD_SIZE:
            return None
        return row, col

    def action_from_pos(
        self, pos: tuple[int, int], state: TensorGameState
    ) -> BoardActionResult:
        """将鼠标位置映射为动作"""
        pos_result = self._get_pos_from_mouse(pos)
        if pos_result is None or state[pos_result[0], pos_result[1]].item() != 0:
            return BoardActionResult(None)

        row, col = pos_result
        return BoardActionResult(int(row * GomokuGame.BOARD_SIZE + col))

    def draw(self, surface: pg.Surface, state: TensorGameState) -> None:
        """绘制棋盘与棋子"""
        pg.draw.rect(surface, self.theme.board_bg_color, self.rect, border_radius=8)

        cell_size = self._get_cell_size()
        radius = int(cell_size * 0.35)

        # 绘制网格线
        for i in range(GomokuGame.BOARD_SIZE):
            x = self.rect.left + i * cell_size
            y = self.rect.top + i * cell_size
            pg.draw.line(surface, self.theme.grid_color, (x, self.rect.top), (x, self.rect.bottom), 1)
            pg.draw.line(surface, self.theme.grid_color, (self.rect.left, y), (self.rect.right, y), 1)

        # 颜色映射
        colors = {PLAYER1: self.theme.primary_color, PLAYER2: self.theme.danger_color}

        # 绘制棋子
        for row in range(GomokuGame.BOARD_SIZE):
            for col in range(GomokuGame.BOARD_SIZE):
                value = int(state[row, col].item())
                if value != 0:
                    center_x = self.rect.left + col * cell_size
                    center_y = self.rect.top + row * cell_size
                    pg.draw.circle(surface, colors[value], (int(center_x), int(center_y)), radius)

    def hover(self, surface: pg.Surface, pos: tuple[int, int], state: TensorGameState) -> None:
        """绘制悬浮效果"""
        pos_result = self._get_pos_from_mouse(pos)
        if pos_result is None or state[pos_result[0], pos_result[1]].item() != 0:
            return

        row, col = pos_result
        cell_size = self._get_cell_size()
        radius = int(cell_size * 0.35)
        current_player = GomokuGame.current_player(state)

        color = self.theme.primary_color if current_player == PLAYER1 else self.theme.danger_color
        color = (*color, 100)

        hover_surface = pg.Surface((radius * 2, radius * 2), pg.SRCALPHA)
        pg.draw.circle(hover_surface, color, (radius, radius), radius)

        center_x = self.rect.left + col * cell_size
        center_y = self.rect.top + row * cell_size
        surface.blit(hover_surface, (int(center_x - radius), int(center_y - radius)))


class GomokuAPP(BaseApp):
    """五子棋 UI 应用"""

    def __init__(self, model: BaseModel, ai_config: AIConfig) -> None:
        super().__init__(GomokuGame, model, ai_config)
        self._status_text: str = ""
        self._font = pg.font.SysFont(None, 36)

    def create_board_view(self) -> BoardView:
        margin = 40
        board_size = min(SCREEN_WIDTH, SCREEN_HEIGHT) - margin * 2
        left = (SCREEN_WIDTH - board_size) // 2
        top = (SCREEN_HEIGHT - board_size) // 2
        return GomokuBoard(pg.Rect(left, top, board_size, board_size), self.theme)

    def on_after_step(self, action: int, done: GameDone) -> None:
        if not done:
            self._status_text = ""

    def on_game_over(self, done: GameDone) -> None:
        value = self.game.terminal_evaluation(self._state(), PLAYER1)
        if value > 0:
            self._status_text = "PLAYER1 wins"
        elif value < 0:
            self._status_text = "PLAYER2 wins"
        else:
            self._status_text = "Draw"

    def on_key_down(self, key: int) -> None:
        if key == pg.K_SPACE and self._done():
            self.reset()

    def on_draw_overlay(self, surface: pg.Surface) -> None:
        if not self._status_text:
            return

        text = self._font.render(self._status_text, True, self.theme.text_color)
        surface.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, 24)))

        if self._done():
            hint_font = pg.font.SysFont(None, 28)
            hint_text = hint_font.render("Press SPACE to play again", True, self.theme.text_color)
            surface.blit(hint_text, hint_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 40)))
