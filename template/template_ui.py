"""UI 模板文件

使用说明:
1. 复制此文件到 examples/<your_game>/ 或 ui/ 目录并重命名
2. 修改类名与导入的游戏类/棋盘视图类
"""

from __future__ import annotations

import pygame as pg

from config.ui import SCREEN_HEIGHT, SCREEN_WIDTH
from ui.board import BoardActionResult, BoardView
from ui.app import BaseApp, AIConfig
from config.type_alias import GameDone, TensorGameState

# TODO: 修改为你的游戏与棋盘视图实现
# from examples.your_game.game import YourGame
# from examples.your_game.ui_board import YourBoardView


class TemplateBoardView(BoardView):
    """棋盘视图模板：实现点击映射与绘制逻辑。"""

    def action_from_pos(
        self, pos: tuple[int, int], state: TensorGameState
    ) -> BoardActionResult:
        """将鼠标位置映射为动作。

        Args:
            pos: 鼠标点击位置 (x, y)
            state: 当前棋盘状态

        Returns:
            BoardActionResult: 有效动作或 None
        """
        # TODO: 根据点击位置返回动作，非法则返回 BoardActionResult(None)
        raise NotImplementedError

    def draw(self, surface: pg.Surface, state: TensorGameState) -> None:
        """绘制棋盘与棋子。

        Args:
            surface: 绘制目标
            state: 当前棋盘状态
        """
        # TODO: 根据 state 绘制棋盘与棋子
        # 注意对于类似于象棋等游戏，视角的显示应保证玩家在下方，模型在上方，这可能需要对状态进行翻转处理
        raise NotImplementedError

    # 可选：实现悬停效果
    # def hover(self, surface: pg.Surface, pos: tuple[int, int], state: TensorGameState) -> None:


class TemplateAPP(BaseApp):
    """UI 模板：实现棋盘创建与可选钩子。"""

    def __init__(self) -> None:
        """初始化 UI，替换为你的游戏类并可设置窗口标题。"""
        # TODO: 替换为你的游戏类
        # super().__init__(YourGame, model, ai_config, caption="Your Game", theme=UITheme.your_theme)
        raise NotImplementedError

    def create_board_view(self) -> BoardView:
        """创建棋盘视图。

        Returns:
            BoardView: 棋盘视图实例
        """
        margin = 40
        board_size = min(SCREEN_WIDTH, SCREEN_HEIGHT) - margin * 2
        left = (SCREEN_WIDTH - board_size) // 2
        top = (SCREEN_HEIGHT - board_size) // 2
        rect = pg.Rect(left, top, board_size, board_size)
        # TODO: 替换为你的棋盘视图
        # return YourBoardView(rect, self.theme)
        raise NotImplementedError

    # def run(self, fps: int = 60) -> None:...
    # 若玩家数量较多或有特殊输入需求可覆盖 run 方法实现自定义事件处理与游戏循环逻辑，默认实现已包含基本的退出与点击处理。

    # 可选：
    # def on_after_step(self, action: int, done: GameDone) -> None:
    #     """每次成功落子后的回调。
    #
    #     Args:
    #         action: 执行的动作
    #         done: 是否终局
    #     """
    #     pass
    #
    # def on_game_over(self, done: GameDone) -> None:
    #     """游戏结束时的回调。
    #
    #     Args:
    #         done: 终局状态
    #     """
    #     pass
    #
    # def on_update(self, dt_ms: int) -> None:
    #     """每帧更新回调。
    #
    #     Args:
    #         dt_ms: 本帧与上一帧的时间间隔（毫秒）
    #     """
    #     pass
    #
    # def on_draw_overlay(self, surface: pg.Surface) -> None:
    #     """覆盖层绘制回调。
    #
    #     Args:
    #         surface: 绘制目标
    #     """
    #     pass
    #
    # def on_key_down(self, key: int) -> None:
    #     """按键按下回调。
    #
    #     Args:
    #         key: pygame 按键常量 (如 pg.K_SPACE, pg.K_r 等)
    #     """
    #     pass

