from __future__ import annotations

import pygame as pg

from config import *
from ui.board import BoardActionResult, BoardView
from ui.app import BaseApp, AIConfig
from ui.theme import UITheme
from nn_models.base import BaseModel
from .game import DotsAndBoxesGame


class DotsAndBoxesBoard(BoardView):
    """点格棋棋盘视图"""

    def __init__(self, rect: pg.Rect, theme: UITheme) -> None:
        super().__init__(rect, theme)

    def _cell_size(self) -> float:
        return self.rect.width / DotsAndBoxesGame.BOX_COLS

    def _dot_pos(self, row: int, col: int) -> tuple[int, int]:
        size = self._cell_size()
        x = self.rect.left + col * size
        y = self.rect.top + row * size
        return int(x), int(y)

    def _edge_from_pos(self, pos: tuple[int, int]) -> tuple[str, int, int] | None:
        if not self.rect.collidepoint(pos):
            return None

        size = self._cell_size()
        rel_x = pos[0] - self.rect.left
        rel_y = pos[1] - self.rect.top
        c_float = rel_x / size
        r_float = rel_y / size

        threshold = size * 0.2

        # candidate horizontal
        h_r = int(round(r_float))
        h_c = int(c_float)
        h_valid = (
            0 <= h_r <= DotsAndBoxesGame.BOX_ROWS
            and 0 <= h_c < DotsAndBoxesGame.BOX_COLS
            and abs(r_float - h_r) * size <= threshold
        )

        # candidate vertical
        v_c = int(round(c_float))
        v_r = int(r_float)
        v_valid = (
            0 <= v_c <= DotsAndBoxesGame.BOX_COLS
            and 0 <= v_r < DotsAndBoxesGame.BOX_ROWS
            and abs(c_float - v_c) * size <= threshold
        )

        if not h_valid and not v_valid:
            return None

        if h_valid and v_valid:
            h_dist = abs(r_float - h_r) * size
            v_dist = abs(c_float - v_c) * size
            if h_dist <= v_dist:
                return "H", h_r, h_c
            return "V", v_r, v_c

        if h_valid:
            return "H", h_r, h_c
        return "V", v_r, v_c

    def action_from_pos(
        self, pos: tuple[int, int], state: TensorGameState
    ) -> BoardActionResult:
        edge = self._edge_from_pos(pos)
        if edge is None:
            return BoardActionResult(None)

        edge_type, r, c = edge
        if edge_type == "H":
            if state[0, r, c].item() != 0:
                return BoardActionResult(None)
        else:
            if state[1, r, c].item() != 0:
                return BoardActionResult(None)

        action = DotsAndBoxesGame._edge_to_action(edge_type, r, c)
        return BoardActionResult(action)

    def draw(self, surface: pg.Surface, state: TensorGameState) -> None:
        pg.draw.rect(surface, self.theme.board_bg_color, self.rect, border_radius=8)

        size = self._cell_size()
        dot_radius = int(size * 0.08)
        line_width = max(2, int(size * 0.12))

        # draw boxes ownership
        for r in range(DotsAndBoxesGame.BOX_ROWS):
            for c in range(DotsAndBoxesGame.BOX_COLS):
                owner = int(state[2, r, c].item())
                if owner == 0:
                    continue
                color = (
                    self.theme.primary_color if owner == PLAYER1 else self.theme.danger_color
                )
                color = (*color, 80)
                box_surface = pg.Surface((int(size), int(size)), pg.SRCALPHA)
                box_surface.fill(color)
                x = self.rect.left + int(c * size)
                y = self.rect.top + int(r * size)
                surface.blit(box_surface, (x, y))

        # draw edges
        for r in range(DotsAndBoxesGame.DOT_ROWS):
            for c in range(DotsAndBoxesGame.BOX_COLS):
                if state[0, r, c].item() == 0:
                    continue
                start = self._dot_pos(r, c)
                end = self._dot_pos(r, c + 1)
                pg.draw.line(surface, self.theme.grid_color, start, end, line_width)

        for r in range(DotsAndBoxesGame.BOX_ROWS):
            for c in range(DotsAndBoxesGame.DOT_COLS):
                if state[1, r, c].item() == 0:
                    continue
                start = self._dot_pos(r, c)
                end = self._dot_pos(r + 1, c)
                pg.draw.line(surface, self.theme.grid_color, start, end, line_width)

        # draw dots
        for r in range(DotsAndBoxesGame.DOT_ROWS):
            for c in range(DotsAndBoxesGame.DOT_COLS):
                pos = self._dot_pos(r, c)
                pg.draw.circle(surface, self.theme.text_color, pos, dot_radius)

    def hover(self, surface: pg.Surface, pos: tuple[int, int], state: TensorGameState) -> None:
        edge = self._edge_from_pos(pos)
        if edge is None:
            return

        edge_type, r, c = edge
        if edge_type == "H":
            if state[0, r, c].item() != 0:
                return
        else:
            if state[1, r, c].item() != 0:
                return

        player = getattr(self, "current_player", PLAYER1)
        color = self.theme.primary_color if player == PLAYER1 else self.theme.danger_color
        color = (*color, 140)

        size = self._cell_size()
        line_width = max(2, int(size * 0.12))

        overlay = pg.Surface(self.rect.size, pg.SRCALPHA)
        if edge_type == "H":
            start = self._dot_pos(r, c)
            end = self._dot_pos(r, c + 1)
        else:
            start = self._dot_pos(r, c)
            end = self._dot_pos(r + 1, c)
        start = (start[0] - self.rect.left, start[1] - self.rect.top)
        end = (end[0] - self.rect.left, end[1] - self.rect.top)
        pg.draw.line(overlay, color, start, end, line_width)
        surface.blit(overlay, self.rect.topleft)


class DotsAndBoxesAPP(BaseApp):
    """点格棋 UI 应用"""

    def __init__(self, model: BaseModel, ai_config: AIConfig) -> None:
        super().__init__(DotsAndBoxesGame, model, ai_config)
        self._status_text: str = ""
        self._font = pg.font.SysFont(None, 36)

    def create_board_view(self) -> BoardView:
        margin = 40
        board_size = min(SCREEN_WIDTH, SCREEN_HEIGHT) - margin * 2
        left = (SCREEN_WIDTH - board_size) // 2
        top = (SCREEN_HEIGHT - board_size) // 2
        rect = pg.Rect(left, top, board_size, board_size)
        return DotsAndBoxesBoard(rect, self.theme)

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
            hint_rect = hint_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 40))
            surface.blit(hint_text, hint_rect)
