import collections
from config import *
import random
import pickle
import torch
from dataclasses import dataclass, field, InitVar
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ExperiencePool:
    capacity: InitVar[int] = DEFAULT_CAPACITY
    _date_deque: ExperienceDeque = field(init=False)
    _size: int = 0

    def __post_init__(self, capacity: int) -> None:
        self._date_deque = collections.deque(maxlen=capacity)

    def put_tupule_experience(self, experience: ExperienceDate) -> None:
        logger.debug(
            f"将经验添加到经验池: nn_state={experience[0].shape}, prior={experience[1]}, value={experience[2]}"
        )
        self._date_deque.append(experience)
        self._size += 1 if self._size < len(self) else 0

    def sample(
        self, batch_size: int = BATCH_SIZE, timeout: float = 0.0
    ) -> ExperienceBatch: # timeout 参数目前未使用，是为了统一不同经验池接口预留的
        """从经验池中采样一个批次

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (状态, 策略, 价值)

        Raises:
            ValueError: 如果经验池为空
        """
        if len(self._date_deque) == 0:
            logger.warning("经验池中没有数据可供采样")
            return None

        k = min(batch_size, len(self._date_deque))
        batch = random.sample(self._date_deque, k)
        states, policies, values = zip(*batch)

        states_tensor = torch.cat(states, dim=0).detach().clone()
        policies_tensor = torch.cat(policies, dim=0).detach().clone()
        values_tensor = torch.stack(values).detach().clone()
        return states_tensor, policies_tensor, values_tensor

    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return len(self._date_deque)

    def save(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self._date_deque, f)

    def load(self, filename: str) -> None:
        with open(filename, "rb") as f:
            self._date_deque = pickle.load(f)

    def clear(self) -> None:
        self._date_deque.clear()
        self._size = 0
