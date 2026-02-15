from __future__ import annotations

from typing import TypeAlias, Optional, TYPE_CHECKING
import collections

if TYPE_CHECKING:
    from utils.experience_pool import ExperiencePool
    from core.MCTS_alphazero import MCTSNode
    import torch
    import torch.multiprocessing as mp

TensorValue: TypeAlias = "torch.Tensor"
"""shape=(1,)"""
TensorActions: TypeAlias = "torch.Tensor"
"""包含批次通道，shape一般为(1,num_actions)"""
TensorGameState: TypeAlias = "torch.Tensor"
"""不包含批次维度"""
NNState: TypeAlias = "torch.Tensor"
"""shape = (1,HISTORY_LEN+1,*state_shape)
用于神经网络
"""
StateWithHistory: TypeAlias = "torch.Tensor"
"""shape = (1,HISTORY_LEN,*state_shape)"""

GameDone: TypeAlias = bool | int
HistoryDeque: TypeAlias = collections.deque[TensorGameState]
ChildType: TypeAlias = dict[int, "MCTSNode"]

size_1_t: TypeAlias = int | tuple[int]
size_2_t: TypeAlias = int | tuple[int, int]

ExperienceDate: TypeAlias = tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]
"""状态，策略，价值的元组

状态(与游戏相同，无批次)，

策略（概率形式，与神经网络输出相同，有批次），

价值张量(shape=(1,))"""
ExperienceDeque: TypeAlias = collections.deque[ExperienceDate]
ExperiencePoolType: TypeAlias = "ExperiencePool | mp.Queue[ExperienceDate]"
ExperienceBatch: TypeAlias = Optional[tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]]
"""包含批次通道的经验数据"""
