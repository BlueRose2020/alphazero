from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn

import torch
from torch.distributions import Dirichlet
import math

from config import *

from games.base import BaseGame
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class MCTSNode:
    game_cls: type[BaseGame]
    state: TensorGameState
    player: int
    prior: Optional[TensorActions] = None
    parent: Optional[MCTSNode] = None
    children: ChildType = field(default_factory=lambda: cast(ChildType, {}))
    visits: int = 0
    value: float = 0
    is_expand: bool = False

    def is_terminal(self) -> GameDone:
        """判断当前状态是否为终局

        Returns:
            bool: 终局返回True，否则返回False
        """
        return self.game_cls.is_terminal(self.state)

    def expand(self, prior: TensorActions) -> None:
        """扩展当前节点

        Args:
            prior (TensorActions): 该节点选择各个子节点的概率（注意归一化）
        """
        self.prior = prior
        legal_actions = self._get_vaild_actions()
        for action in legal_actions:
            action_int = int(action.item())
            next_state, child_player = self._next_state(action_int)
            self.children[action_int] = MCTSNode(
                self.game_cls, next_state, child_player, parent=self
            )
        self.is_expand = True

    def select(self, c_puct: float = C_PUCT) -> MCTSNode:
        """根据puct公式选择子节点"""
        if self.prior is None:
            raise ValueError("prior未设置，无法选择子节点")

        best_score = -float("inf")
        best_child: Optional[MCTSNode] = None

        sqrt_N = math.sqrt(self.visits)  # 避免多次开根号
        for action, child in self.children.items():
            q = child.value if self.player == child.player else -child.value
            puct = q + c_puct * self.prior[0, action] * sqrt_N / (child.visits + 1)
            if puct > best_score:
                best_score = puct
                best_child = child

        if best_child is None:
            raise ValueError("未选择子节点")
        return best_child

    def update(self, value: float) -> None:
        """更新访问次数和节点价值

        Args:
            value: 应传入当前节点视角的价值
        """
        self.visits += 1
        # Q_{n+1} = Q_{n} + (v-Q_{n})/(n+1) 避免除0
        self.value += (value - self.value) / self.visits

    def _next_state(self, action: int) -> tuple[TensorGameState, int]:
        """
        Args:
            action (int): 执行动作的索引

        Returns:
            GameState: 从当前状态执行action得到的状态和下一个状态的玩家
            int: 下一个状态的玩家
        """
        next_state, child_player = self.game_cls.next_state(self.state, action)
        return next_state, child_player

    def _get_vaild_actions(self) -> torch.Tensor:
        """根据当前状态获取合法的actions并返回，需要在子类中实现

        Returns:
            torch.Tensor: 需要返回张量类型的actions(一维)
        """
        legal_mask = self.game_cls.legal_action_mask(self.state).squeeze(0)
        return torch.nonzero(legal_mask)


class MCTS:
    def __init__(self, game_cls: type[BaseGame]) -> None:
        self.game_cls = game_cls
        if USE_HISTORY:
            from utils.history_manager import HistoryManager

            self.history_manager = HistoryManager(game_cls)

    def search(
        self,
        model: nn.Module,
        root_state: TensorGameState,
        root_player: int,
        history_state: Optional[StateWithHistory] = None,
        num_simulation: int = TRAIN_NUM_SIMULATION,
        c_puct: float = C_PUCT,
        use_Dirichlet: bool = True,
    ) -> TensorActions:
        if history_state is None and USE_HISTORY:
            raise ValueError("使用历史信息时必须提供history_state参数")

        root_node = MCTSNode(self.game_cls, root_state, root_player)
        for _ in range(num_simulation):
            node: MCTSNode = root_node
            if USE_HISTORY:
                self.history_manager.load(cast(StateWithHistory, history_state))
            # 选择
            while node.is_expand and not node.is_terminal():
                node = node.select(c_puct)
                if USE_HISTORY:
                    self.history_manager.update(
                        node.state
                    )  # 始终保持历史的最新状态与当前状态相同
            # 扩展+评估（alphazero这两步是合在一起的）
            if node.is_terminal():
                value = self.game_cls.terminal_evaluation(node.state, node.player)
            else:
                nn_state = self._node2nn_state(node)
                with torch.inference_mode():
                    policy, value = model(nn_state)
                mask = self.game_cls.legal_action_mask(node.state).to(DEVICE)

                prior = self._get_prior(
                    policy, mask, node is root_node and use_Dirichlet
                )
                node.expand(prior)

                value = value.item()
            # 回溯
            self._bacaward(node, value)

        final_prior = torch.zeros_like(cast(torch.Tensor, root_node.prior))
        for action, child in root_node.children.items():
            final_prior[0, action] = child.visits
        # logger.debug(f"root_value: {root_node.value}")
        return final_prior / final_prior.sum()

    def _get_prior(
        self, policy: torch.Tensor, mask: torch.Tensor, use_Dilichlet: bool
    ) -> torch.Tensor:
        """若节点为根节点且需要噪声，则加入Dirichlet噪声"""

        policy = torch.softmax(policy, dim=-1)
        if use_Dilichlet:
            noise = Dirichlet(torch.ones_like(policy) * ALPHA).sample()
            policy = (1 - EPSILON) * policy + EPSILON * noise

        policy = policy * mask
        prior = policy / policy.sum()
        return prior

    def _bacaward(self, node: MCTSNode, value: float) -> None:
        current: Optional[MCTSNode] = node
        while current is not None:
            current.update(value)
            if current.parent is not None:
                value = value if current.parent.player == current.player else -value
            current = current.parent

    def _node2nn_state(self, node: MCTSNode) -> NNState:
        player_channel = self.game_cls.get_player_channel(node.state, node.player)
        if USE_HISTORY:
            history_state = self.history_manager.get_state()
            state = torch.cat((history_state, player_channel), dim=0)
        else:
            state = torch.cat((node.state.unsqueeze(0), player_channel), dim=0)
        return state.unsqueeze(0).detach().clone().to(DEVICE)
