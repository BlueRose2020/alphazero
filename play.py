from __future__ import annotations

from examples import TicTacToeModel
from examples.tictactoe.ui import TicTacToeAPP

from examples import GomokuModel
from examples.gomoku.ui import GomokuAPP

from nn_models.quick_model import QuickModel
from ui.app import AIConfig
import os
import torch
from config import DEVICE, PLAYER1, PLAYER2, USE_HISTORY
from utils.logger import setup_logger

logger = setup_logger(__name__)


def main() -> None:
    ai_config = AIConfig(
        player_with_ai=True, ai_player=None, use_mcts=True, mcts_simulations=50
    )
    model = QuickModel().to(device=DEVICE)
    # model = TicTacToeModel().to(device=DEVICE)
    # model = GomokuModel().to(device=DEVICE)

    model_name = model.__class__.__name__.replace("Model", "") + ("_history" if USE_HISTORY else "")
    model_path = f"./result/models/{model_name}/last_model.pth"

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        logger.warning(f"未找到预训练模型，路径: {model_path}，将使用随机初始化的模型")
    # ui = TicTacToeAPP(model=model, ai_config=ai_config)
    ui = GomokuAPP(model=model, ai_config=ai_config)
    ui.run()


if __name__ == "__main__":
    main()
