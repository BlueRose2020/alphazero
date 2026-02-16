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
    deque_len: InitVar[int] = DEFAULT_CAPACITY
    date_deque: ExperienceDeque = field(init=False)

    def __post_init__(self, deque_len: int) -> None:
        self.date_deque = collections.deque(maxlen=deque_len)

    def put_tupule_experience(self, experience: ExperienceDate) -> None:
        self.date_deque.append(experience)

    def sample(self, batch_size: int = BATCH_SIZE) -> ExperienceBatch:
        """从经验池中采样一个批次

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (状态, 策略, 价值)

        Raises:
            ValueError: 如果经验池为空
        """
        if len(self.date_deque) == 0:
            logger.warning("经验池中没有数据可供采样")
            return None

        k = min(batch_size, len(self.date_deque))
        batch = random.sample(self.date_deque, k)
        states, policies, values = zip(*batch)

        states_tensor = torch.stack(states).detach().clone()
        policies_tensor = torch.cat(policies).detach().clone()
        values_tensor = torch.stack(values).detach().clone()
        return states_tensor, policies_tensor, values_tensor

    def __len__(self) -> int:
        return len(self.date_deque)

    def save(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self.date_deque, f)

    def load(self, filename: str) -> None:
        with open(filename, "rb") as f:
            self.date_deque = pickle.load(f)

    def clear(self) -> None:
        self.date_deque.clear()
