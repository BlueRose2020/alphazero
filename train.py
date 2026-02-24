from __future__ import annotations
from utils.logger import setup_logger

logger = setup_logger(__name__)

"""在./congig/basic.py中配置AlphaZero算法的基本参数"""
"""在./congig/train_config.py中配置训练参数"""
"""训练结束后可以运行play.py来加载模型并与AI对战"""

def main() -> None:
    from examples import TicTacToeGame, TicTacToeModel  # 替换为你的游戏和模型
    from examples import GomokuGame, GomokuModel  # 替换为你的游戏和模型
    from examples import DotsAndBoxesGame, DotsAndBoxesModel  # 替换为你的游戏和模型
    from nn_models.quick_model import (
        QuickModel,
    )  # 可以通过./config/quick_model_config.py配置快速测试模型的参数
    from training.alphazero_trainer import AlphaZeroTrainer

    alphazero_trainer = AlphaZeroTrainer(
        # model_cls=QuickModel,  # 记得修改配置
        # model_cls=TicTacToeModel,  # 替换为你的模型类
        # game_cls=TicTacToeGame,  # 替换为你的游戏类
        # model_cls=DotsAndBoxesModel,  # 替换为你的模型类
        # game_cls=DotsAndBoxesGame,  # 替换为你的游戏类
        model_cls=GomokuModel,  # 替换为你的模型类
        game_cls=GomokuGame,  # 替换为你的游戏类
    )

    alphazero_trainer.train()


if __name__ == "__main__":
    main()
