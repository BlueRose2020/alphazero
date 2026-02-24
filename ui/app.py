import pygame as pg
import torch

from dataclasses import dataclass, InitVar, field
from typing import Optional
from games.base import BaseGame
from nn_models.base import BaseModel
from core.MCTS_alphazero import MCTS
from config import *
from ui.board import BoardView
from ui.theme import UITheme
from utils.logger import setup_logger

import random

logger = setup_logger(__name__)


@dataclass(frozen=True)
class AIConfig:
    play_with_ai: bool = True
    ai_player: Optional[int] = None  # 随机选择一方
    use_mcts: bool = False

    mcts_simulations: InitVar[int] = 800
    simulations: int = field(init=False, default=800)

    use_dirichlet: bool = False  # 是否在MCTS中使用Dirichlet噪声来增加探索
    use_virtual_loss: bool = False 

    def __post_init__(self, mcts_simulations: int) -> None:
        if self.use_mcts:
            object.__setattr__(self, "simulations", mcts_simulations)


class BaseApp:
    def __init__(
        self,
        game_cls: type[BaseGame],
        model: BaseModel,
        ai_config: AIConfig,
        theme: Optional[UITheme] = None,
        caption: Optional[str] = None,
    ) -> None:
        self.game_cls = game_cls
        self.game = game_cls()

        self.device = self._get_model_device(model)
        self.model = model.to(self.device)
        self.model.eval()

        self.theme = theme or UITheme.default()
        self.clock = pg.time.Clock()
        self.board_view = self.create_board_view()

        self.play_with_ai = ai_config.play_with_ai
        if ai_config.play_with_ai:
            self.ai_player = (
                ai_config.ai_player
                if ai_config.ai_player in PLAYERS
                else random.choice(PLAYERS)
            )
            self.use_mcts = ai_config.use_mcts
            if ai_config.use_mcts:
                self.mcts = MCTS(game_cls=game_cls)
                self.mcts_simulations = ai_config.simulations
                self.use_dirichlet = ai_config.use_dirichlet
                self.use_virtual_loss = ai_config.use_virtual_loss

        pg.init()
        self.screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        if caption is not None:
            pg.display.set_caption(caption)
        else:
            pg.display.set_caption(game_cls.__name__)

    @staticmethod
    def _get_model_device(model: BaseModel) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def create_board_view(self) -> BoardView:
        """创建棋盘视图，子类必须实现。"""
        raise NotImplementedError

    # 可选钩子：子类可覆盖以下方法实现额外功能
    # =====================================
    def on_after_step(self, action: int, done: GameDone) -> None:
        pass

    def on_game_over(self, done: GameDone) -> None:
        pass

    def on_update(self, dt_ms: int) -> None:
        pass

    def on_draw_overlay(self, surface: pg.Surface) -> None:
        pass

    def on_key_down(self, key: int) -> None:
        """按键按下回调，子类可覆盖实现自定义按键处理。"""
        pass

    # =====================================

    def get_ai_action(self) -> Optional[int]:
        if self.use_mcts:
            prior = self.mcts.search(
                model=self.model,
                root_state=self._state(),
                root_player=self.game.get_player(),
                history_state=self._history_state(),
                num_simulation=self.mcts_simulations,
                c_puct=C_PUCT,
                use_Dirichlet=self.use_dirichlet,
                use_virtual_loss=self.use_virtual_loss,
            )

            return int(prior.argmax(dim=1).item())
        else:
            state = self._state()
            player_channel = self.game_cls.get_player_channel(
                state, self.game.get_player()
            )

            if USE_HISTORY:
                history_state = self._history_state()
                if history_state is None:
                    raise RuntimeError("启用历史记录功能，但无法获取历史状态")
                model_input = torch.cat(
                    (history_state, player_channel), dim=0
                ).unsqueeze(0)
            else:
                model_input = torch.cat(
                    (state.unsqueeze(0), player_channel), dim=0
                ).unsqueeze(0)

            model_input = model_input.to(self.device)
            policy, value = self.model(model_input)
            policy = policy.squeeze(0).detach().cpu()
            value = value.detach().cpu()
            logger.debug(f"value: {value.item()}")
            legal_mask = self.game.get_legal_mask().squeeze(0)
            masked_policy = policy.masked_fill(legal_mask == 0, float("-inf"))
            return int(masked_policy.argmax().item())

    def reset(self) -> None:
        """重置游戏状态。"""
        self.game = self.game_cls()

    def run(self, fps: int = 60) -> None:
        running = True

        while running:
            dt_ms = self.clock.tick(fps)
            self.on_update(dt_ms)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.MOUSEBUTTONDOWN:
                    self._handel_click(event.pos)
                elif event.type == pg.KEYDOWN:
                    self.on_key_down(event.key)

            self.clear()
            self.update_board()
            self.hover()
            self.on_draw_overlay(self.screen)
            self.display()

            if (
                self.play_with_ai
                and not self._done()
                and self.game.get_player() == self.ai_player
            ):
                ai_action = self.get_ai_action()
                if ai_action is not None:
                    self._apply_action(ai_action)

    def clear(self) -> None:
        self.screen.fill(self.theme.background_color)

    def update_board(self) -> None:
        self.board_view.draw(self.screen, self._state())

    def hover(self) -> None:
        pos = pg.mouse.get_pos()
        self.board_view.hover(self.screen, pos, self._state())

    def display(self) -> None:
        pg.display.flip()

    def _is_action_legal(self, action: int) -> bool:
        legal_mask = self.game.get_legal_mask()
        return legal_mask[0, action].item() == 1

    def _apply_action(self, action: int) -> None:
        if not self._is_action_legal(action):
            return
        done = self.game.step(action)
        self.on_after_step(action, done)
        if done:
            self.on_game_over(done)

    def _handel_click(self, pos: tuple[int, int]) -> None:
        if self._done():
            return
        result = self.board_view.action_from_pos(pos, self._state())
        if result.is_valid:
            self._apply_action(result.action)  # type: ignore

    def _state(self) -> TensorGameState:
        return self.game.get_state()

    def _history_state(self) -> Optional[StateWithHistory]:
        if not USE_HISTORY:
            return None
        return self.game.get_history()

    def _done(self) -> GameDone:
        return self.game.is_terminal(self._state())

    def _current_player(self) -> int:
        return self.game.get_player()
