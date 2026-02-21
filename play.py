from __future__ import annotations

from examples import TicTacToeModel
from examples.tictactoe.ui import TicTacToeAPP

from examples import GomokuModel
from examples.gomoku.ui import GomokuAPP

from examples import DotsAndBoxesModel
from examples.dots_and_boxes.ui import DotsAndBoxesAPP

from nn_models.quick_model import QuickModel
from ui.app import AIConfig
import os
import torch
from config import DEVICE, PLAYER1, PLAYER2, USE_HISTORY
from utils.logger import setup_logger

logger = setup_logger(__name__)


def main() -> None:
    # 配置AI对战参数
    # 参数说明：
    # - play_with_ai: 是否让与AI对战（如果为False，则为自己与自己对战）
    # - ai_player: AI执哪一方（PLAYER1、PLAYER2或None随机选择）
    # - use_mcts: 是否使用蒙特卡洛树搜索来增强AI决策（如果为False，则直接使用模型输出的策略）
    # 当use_mcts为True时，以下参数生效：
    # - mcts_simulations: 每一步AI决策时进行的蒙特卡洛树搜索模拟次数，模拟越多AI决策越强但计算越慢
    # - use_dirichlet: 是否在MCTS中使用Dirichlet噪声来增加开局的多样性(可能会导致AI偶尔做出较差的决策，但能增加游戏的趣味性和不可预测性)
    ai_config = AIConfig(
        play_with_ai=True,
        ai_player=None,
        use_mcts=False,
        mcts_simulations=400,
        use_dirichlet=False,
    )

    # 替换为你的模型类，设备你可以自行指定而无须是DEVICE
    model = QuickModel().to(device=DEVICE)
    # model = TicTacToeModel().to(device=DEVICE)
    # model = GomokuModel().to(device=DEVICE)
    # model = DotsAndBoxesModel().to(device=DEVICE)

    # 替换为你的UI类
    # ui = TicTacToeAPP(model=model, ai_config=ai_config)
    ui = DotsAndBoxesAPP(model=model, ai_config=ai_config)
    # ui = GomokuAPP(model=model, ai_config=ai_config)

    # 根据模型类名和是否使用历史状态来自动加载训练好的模型权重文件，无需修改
    if model.__class__.__name__ == "QuickModel":
        model_name = (
            "Quick"
            + ui.__class__.__name__.replace("APP", "")
            + ("_history" if USE_HISTORY else "")
        )
    else:
        model_name = model.__class__.__name__.replace("Model", "") + (
            "_history" if USE_HISTORY else ""
        )
    model_path = f"./result/models/{model_name}/last_model.pth"

    if ai_config.play_with_ai:
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            logger.warning(
                f"未找到预训练模型，路径: {model_path}，将使用随机初始化的模型"
            )

    ui.run()


if __name__ == "__main__":
    main()
