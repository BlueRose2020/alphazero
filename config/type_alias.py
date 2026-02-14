from typing import TypeAlias, TYPE_CHECKING
import collections

if TYPE_CHECKING:
    from utils.experience_pool import ExperiencePool
    from core.MCTS_alphazero import MCTSNode
    import torch
    import torch.multiprocessing as mp

TensorActions: TypeAlias = torch.Tensor
"""包含批次通道，shape一般为(1,num_actions)"""
TensorGameState: TypeAlias = torch.Tensor
"""不包含批次维度"""
NNState: TypeAlias = torch.Tensor
"""shape = (1,HISTORY_LEN+1,*state_shape)
用于神经网络
"""
StateWithHistory: TypeAlias = torch.Tensor
"""shape = (1,HISTORY_LEN,*state_shape)"""

GameDone: TypeAlias = bool | int
HistoryDeque: TypeAlias = collections.deque[TensorGameState]
ChildType: TypeAlias = dict[int, MCTSNode]

size_1_t: TypeAlias = int | tuple[int]
size_2_t: TypeAlias = int | tuple[int, int]

ExperienceDate: TypeAlias = torch.Tensor | tuple[torch.Tensor,...]
ExperienceDeque: TypeAlias = collections.deque[ExperienceDate]
ExperiencePoolType: TypeAlias = ExperiencePool | mp.Queue[ExperienceDate]
