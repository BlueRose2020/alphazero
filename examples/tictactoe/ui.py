from __future__ import annotations

import pygame as pg

from config.train_config import PLAYER1, PLAYER2
from config.type_alias import TensorGameState
from ui.board import BoardActionResult, BoardView
from ui.theme import UITheme
from config import *
from ui.app import BaseApp, AIConfig
from nn_models.base import BaseModel
from .game import TicTacToeGame


class TicTacToeBoard(BoardView):
    def __init__(self, rect: pg.Rect, theme: UITheme) -> None:
        super().__init__(rect, theme)

    def action_from_pos(
        self, pos: tuple[int, int], state: TensorGameState
    ) -> BoardActionResult:
        if not self.rect.collidepoint(pos):
            return BoardActionResult(None)
        cell_size = self.rect.width // 3
        col = (pos[0] - self.rect.left) // cell_size
        row = (pos[1] - self.rect.top) // cell_size
        if row < 0 or row > 2 or col < 0 or col > 2:
            return BoardActionResult(None)
        if state[row, col].item() != 0:
            return BoardActionResult(None)
        return BoardActionResult(int(row * 3 + col))

    def draw(self, surface: pg.Surface, state: TensorGameState) -> None:
        pg.draw.rect(surface, self.theme.board_bg_color, self.rect, border_radius=8)
        cell_size = self.rect.width // 3
        for i in range(1, 3):
            pg.draw.line(
                surface,
                self.theme.grid_color,
                (self.rect.left + i * cell_size, self.rect.top),
                (self.rect.left + i * cell_size, self.rect.bottom),
                2,
            )
            pg.draw.line(
                surface,
                self.theme.grid_color,
                (self.rect.left, self.rect.top + i * cell_size),
                (self.rect.right, self.rect.top + i * cell_size),
                2,
            )

        for row in range(3):
            for col in range(3):
                value = int(state[row, col].item())
                if value == 0:
                    continue
                center_x = self.rect.left + col * cell_size + cell_size // 2
                center_y = self.rect.top + row * cell_size + cell_size // 2
                radius = cell_size // 3
                if value == PLAYER1:
                    self._draw_cross(surface, center_x, center_y, radius)
                elif value == PLAYER2:
                    self._draw_circle(surface, center_x, center_y, radius)

    def _draw_cross(self, surface: pg.Surface, cx: int, cy: int, radius: int) -> None:
        color = self.theme.primary_color
        offset = radius
        pg.draw.line(
            surface, color, (cx - offset, cy - offset), (cx + offset, cy + offset), 4
        )
        pg.draw.line(
            surface, color, (cx - offset, cy + offset), (cx + offset, cy - offset), 4
        )

    def _draw_circle(self, surface: pg.Surface, cx: int, cy: int, radius: int) -> None:
        color = self.theme.danger_color
        pg.draw.circle(surface, color, (cx, cy), radius, 4)


class TicTacToeAPP(BaseApp):
    def __init__(
        self,
        model: BaseModel,
        ai_config: AIConfig,
    ) -> None:
        super().__init__(
            TicTacToeGame,
            model,
            ai_config,
        )
        self._status_text: str = ""
        self._font = pg.font.SysFont(None, 36)

    def create_board_view(self) -> BoardView:
        margin = 40
        board_size = min(SCREEN_WIDTH, SCREEN_HEIGHT) - margin * 2
        left = (SCREEN_WIDTH - board_size) // 2
        top = (SCREEN_HEIGHT - board_size) // 2
        rect = pg.Rect(left, top, board_size, board_size)
        return TicTacToeBoard(rect, self.theme)

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
        rect = text.get_rect(center=(SCREEN_WIDTH // 2, 24))
        surface.blit(text, rect)

        if self._done():
            hint_font = pg.font.SysFont(None, 28)
            hint_text = hint_font.render(
                "Press SPACE to play again", True, self.theme.text_color
            )
            hint_rect = hint_text.get_rect(
                center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 40)
            )
            surface.blit(hint_text, hint_rect)
