from __future__ import annotations

from typing import TypeAlias, TYPE_CHECKING
import collections
import torch

if TYPE_CHECKING:
    from utils.share_ring_buffer import SharedRingBuffer
    from utils.experience_pool import ExperiencePool
    from core.MCTS_alphazero import MCTSNode

TensorValue: TypeAlias = "torch.Tensor"
"""shape=(1,)"""
TensorActions: TypeAlias = "torch.Tensor"
"""包含批次通道，shape=(1,num_actions)"""
TensorGameState: TypeAlias = "torch.Tensor"
"""不包含批次维度"""
NNState: TypeAlias = "torch.Tensor"
"""shape = (1,HISTORY_LEN+1,*state_shape)
用于神经网络
"""
StateWithHistory: TypeAlias = "torch.Tensor"
"""shape = (HISTORY_LEN,*state_shape)"""

GameDone: TypeAlias = bool | int
HistoryDeque: TypeAlias = collections.deque[TensorGameState]
ChildType: TypeAlias = dict[int, "MCTSNode"]

size_1_t: TypeAlias = int | tuple[int]
size_2_t: TypeAlias = int | tuple[int, int]
ShapeType: TypeAlias = tuple[int, ...]

ExperienceDate: TypeAlias = tuple[NNState, TensorActions, TensorValue]
"""包含批次通道的经验数据，形如 (nn_state, prior, value)"""
ExperienceDeque: TypeAlias = collections.deque[ExperienceDate]
ExperiencePoolType: TypeAlias = "ExperiencePool | SharedRingBuffer"
ExperienceBatch: TypeAlias = ExperienceDate | None
"""包含批次通道的经验数据"""


# 快速模型配置的类型
ChannelsType: TypeAlias = int | tuple[int, ...] | list[int]
FeaturesType: TypeAlias = int | tuple[int, ...] | list[int]


"""----UI相关类型----"""
Color = tuple[int, int, int]
Position = tuple[int, int]