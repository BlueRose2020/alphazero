import collections
from config import *
import random
import pickle
from dataclasses import dataclass, field, InitVar


@dataclass
class ExperiencePool:
    deque_len: InitVar[int] = DEFULT_CAPACITY
    date_deque: ExperienceDeque = field(init=False)

    def __post_init__(self, deque_len: int) -> None:
        self.date_deque = collections.deque(maxlen=deque_len)

    def put(self, experience: ExperienceDate) -> None:
        self.date_deque.append(experience)

    def sample(self, batch_size: int = BATCH_SIZE) -> list[ExperienceDate]:
        k = min(batch_size, len(self.date_deque))
        return random.sample(self.date_deque, k)

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
