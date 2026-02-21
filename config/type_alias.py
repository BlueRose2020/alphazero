from __future__ import annotations

"""!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
"""请勿修改此文件，除非你知道自己在做什么，否则可能会导致程序无法运行"""
"""!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""

from typing import TypeAlias, TYPE_CHECKING
import collections
import torch

if TYPE_CHECKING:
    from utils.share_ring_buffer import SharedRingBufferExperiencePool
    from utils.experience_pool import ExperiencePool
    from core.MCTS_alphazero import MCTSNode

TensorValue: TypeAlias = "torch.Tensor"
"""shape=(1,)"""
TensorActions: TypeAlias = "torch.Tensor"
"""包含批次通道，shape=(1,num_actions)"""
TensorGameState: TypeAlias = "torch.Tensor"
"""不包含批次维度"""
NNState: TypeAlias = "torch.Tensor"
"""
二维:
shape = (1,HISTORY_LEN+1,*state_shape)或
shape = (1,2,*state_shape)，其中HISTORY_LEN为
三维:
shape = (1,HISTORY_LEN*state_shape[0]+1,*state_shape[1:])或
shape = (1,state_shape[0]+1,*state_shape[1:])，其中HISTORY_LEN为
"""
StateWithHistory: TypeAlias = "torch.Tensor"
"""
二维:
shape = (HISTORY_LEN,*state_shape)
三维:
shape = (HISTORY_LEN * state_shape[0],*state_shape[1:])
"""

GameDone: TypeAlias = bool | int
HistoryDeque: TypeAlias = collections.deque[TensorGameState]
ChildType: TypeAlias = dict[int, "MCTSNode"]

size_1_t: TypeAlias = int | tuple[int]
size_2_t: TypeAlias = int | tuple[int, int]
ShapeType: TypeAlias = tuple[int, ...]

ExperienceDate: TypeAlias = tuple[NNState, TensorActions, TensorValue]
"""包含批次通道的经验数据，形如 (nn_state, prior, value)"""
ExperienceDeque: TypeAlias = collections.deque[ExperienceDate]
ExperiencePoolType: TypeAlias = "ExperiencePool | SharedRingBufferExperiencePool"
ExperienceBatch: TypeAlias = ExperienceDate | None
"""包含批次通道的经验数据"""


# 快速模型配置的类型
ChannelsType: TypeAlias = int | tuple[int, ...] | list[int]
FeaturesType: TypeAlias = int | tuple[int, ...] | list[int]


"""----UI相关类型----"""
Color = tuple[int, int, int]
Position = tuple[int, int]
