from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
from config.ui import *
from config.type_alias import Color



@dataclass(frozen=True)
class UITheme:
    background_color: Color
    board_bg_color: Color
    grid_color: Color
    primary_color: Color
    danger_color: Color
    text_color: Color

    _DEFAULT: ClassVar["UITheme"]

    @classmethod
    def default(cls) -> "UITheme":
        return cls(
            background_color=BACKGROUND_COLOR,
            board_bg_color=BOARD_BG_COLOR,
            grid_color=GRID_COLOR,
            primary_color=PRIMARY_COLOR,
            danger_color=DANGER_COLOR,
            text_color=TEXT_COLOR,
        )
