from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pygame as pg

from config import *
from .theme import UITheme

from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass(frozen=True)
class BoardActionResult:
    action: Optional[int]

    @property
    def is_valid(self) -> bool:
        return self.action is not None


class BoardView:
    """单个棋盘视图的基类，子类只需实现 `action_from_pos` 与 `draw`"""
    def __init__(self, rect: pg.Rect, theme: UITheme) -> None:
        self.rect = rect
        self.theme = theme

    def action_from_pos(
        self, pos: Position, state: TensorGameState
    ) -> BoardActionResult:
        raise NotImplementedError

    def draw(self, surface: pg.Surface, state: TensorGameState) -> None:
        raise NotImplementedError

    def hover(self, surface: pg.Surface, pos: Position, state: TensorGameState) -> None:
        """可选的悬停渲染，默认不处理。"""


