from __future__ import annotations
from typing import Type, TYPE_CHECKING, cast, Any

if TYPE_CHECKING:
    from games.base import BaseGame
    from nn_models.base import BaseModel

from core.MCTS_alphazero import MCTS
from config import *
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ChessArena:
    def __init__(self, model_cls: Type[BaseModel], game_cls: Type[BaseGame]) -> None:
        self.model = model_cls()
        self.game = game_cls()
        self.mcts = MCTS(game_cls=game_cls)

    def self_play(self, tao: float, experience_pool: ExperiencePoolType) -> None:
        """一次自对弈"""
        done = False
        traj: list[Any] = []
        self.game.reset()

        while not done:
            # 获取状态
            state = self.game.get_state()
            player = self.game.get_player()
            player_channel = type(self.game).get_player_channel(state, player)

            # 搜索
            if USE_HISTORY:
                history_state = self.game.get_history()

                prior = self.mcts.search(
                    self.model,
                    state,
                    player,
                    history_state=history_state,
                )
                nn_state = torch.cat((history_state, player_channel), dim=0).unsqueeze(
                    0
                )
            else:
                prior = self.mcts.search(self.model, state, player)
                if state.shape[0] == 2:
                    nn_state = torch.cat(
                        (state.unsqueeze(0), player_channel), dim=0
                    ).unsqueeze(0)
                else:
                    nn_state = torch.cat((state, player_channel), dim=0).unsqueeze(0)

            # 执行动作
            prior_with_tao = torch.pow(prior, 1 / tao)
            action = cast(int, torch.multinomial(prior_with_tao, num_samples=1).item())
            done = self.game.step(action)

            traj.append((nn_state, prior, player))

        logger.info(f"完成一轮自对弈，轨迹长度: {len(traj)}")
        # 加入经验池
        result = (
            self.game.evaluation()
        )  # 这是下一步棋局的结果，不是nn_state对应的结果，所以需要根据玩家视角进行调整
        result = result if self.game.get_player() == player else -result
        self._generate_experience_date(traj, result, experience_pool)

    def load_model(self, file_path: str) -> None:
        try:
            with open(file_path, "rb") as f:
                self.model.load_state_dict(torch.load(f, weights_only=True))
        except:
            raise FileExistsError(f"模型文件 {file_path} 不存在")

    def _policy2action(self, policy: TensorActions, tao: float) -> int:
        policy = torch.softmax(policy, dim=-1)
        action_mask = self.game.get_legal_mask()
        mask_policy = policy * action_mask

        prior = torch.pow(mask_policy, 1 / tao)
        prior = prior / prior.sum()

        return cast(int, torch.multinomial(prior, num_samples=1).item())

    def _generate_experience_date(
        self, traj: list[Any], result: float, experience_pool: ExperiencePoolType
    ) -> None:
        """生成经验并添加到经验池

        Args:
            traj (list[Any]): 自对弈一轮的轨迹
            result (float): 终局时当前玩家视角的得分
            experience_pool (ExperiencePoolType): 用于存放经验的经验池
        """
        nn_state, prior, child_player = traj[-1]

        experience_pool.put_tupule_experience(
            (
                nn_state.detach().clone(),
                prior.detach().clone(),
                torch.Tensor([result]),
            )
        )

        for nn_state, prior, player in reversed(traj[:-1]):
            result = result if player == child_player else -result
            experience_pool.put_tupule_experience(
                (
                    nn_state.detach().clone(),
                    prior.detach().clone(),
                    torch.Tensor([result]),
                )
            )
            child_player = player
