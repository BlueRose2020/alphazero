# AlphaZero 通用训练框架

一个可复用的 AlphaZero 通用训练框架，内置多进程自对弈、可视化 UI、模板与示例，支持快速接入新博弈环境并进行训练/评估。

## 功能特性

- 通用 AlphaZero 训练流程（自对弈 + 训练 + 评估）
- 多进程自对弈与经验池
- 可视化 UI（棋盘与对局）
- 统一的游戏/模型接口，便于扩展
- 内置示例：井字棋、五子棋
- 模板快速创建新游戏与模型

## 目录概览

- 入口脚本：训练 [train.py](train.py)，对局/演示 [play.py](play.py)
- 核心 MCTS 实现： [core/](core/)
- 游戏接口与抽象： [games/](games/)
- 神经网络模型： [nn_models/](nn_models/)
- 训练流程与自对弈： [training/](training/)
- 经验池/工具： [utils/](utils/)
- UI： [ui/](ui/)
- 示例： [examples/](examples/)
- 模板： [template/](template/)
- 配置： [config/](config/)

## 快速开始

1. 安装依赖

根据 [requirements.txt](requirements.txt) 安装依赖。

2. 训练

使用 [train.py](train.py) 启动训练流程。

3. 对局/演示

使用 [play.py](play.py) 进行对局或 UI 演示。

> 具体参数与默认配置请参考 [config/](config/)。

## 训练流程说明

训练由以下关键模块组成：

- 自对弈与数据生成：见 [training/self_play.py](training/self_play.py)
- 训练调度：见 [training/train_alphazero.py](training/train_alphazero.py)
- 训练器：见 [training/alphazero_trainer.py](training/alphazero_trainer.py)
- 经验池：见 
① [utils/experience_pool.py](utils/experience_pool.py)
② [utils/share_ring_buffer.py](utils/share_ring_buffer.py)
  
整体流程：

1. 使用 MCTS 进行自对弈并产生训练样本
2. 样本进入经验池
3. 训练神经网络
4. 定期更新和保存模型

## UI 与演示

UI 入口在 [ui/app.py](ui/app.py)。棋盘与样式组件见 [ui/board.py](ui/board.py) 与 [ui/theme.py](ui/theme.py)。

示例 UI：

- 井字棋 UI： [examples/tictactoe/ui.py](examples/tictactoe/ui.py)
- 五子棋 UI： [examples/gomoku/ui.py](examples/gomoku/ui.py)

## 扩展新游戏

推荐从模板开始：

- 游戏模板： [template/template_game.py](template/template_game.py)
- 模型模板： [template/template_model.py](template/template_model.py)
- UI 模板： [template/template_ui.py](template/template_ui.py)

主要步骤：

1. 实现游戏状态/动作/胜负判断（参考 [games/base.py](games/base.py) 接口）
2. 实现策略-价值网络（参考 [nn_models/base.py](nn_models/base.py)）
3. 在配置中注册新游戏/模型（见 [config/](config/)）
4. 运行训练脚本开始训练

## 示例

- 井字棋： [examples/tictactoe/](examples/tictactoe/)
- 五子棋： [examples/gomoku/](examples/gomoku/)

## 结果与模型

训练产物默认在 [result/](result/) 下，包括模型权重与经验池等中间结果。

## 贡献

欢迎提交 Issue 或 PR 来完善框架与示例。
