from __future__ import annotations
from dataclasses import dataclass, field, InitVar
from typing import Optional, Tuple, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import SynchronizedBase
    from multiprocessing.synchronize import Lock as MpLock, Condition as MpCondition
    from ctypes import c_int
from config import *
import time
import torch
import torch.multiprocessing as mp
from utils.logger import setup_logger


logger = setup_logger(__name__, rate_limit=5.0)  # 相同消息每5秒最多输出一次


@dataclass
class SharedRingBuffer:
    state_shape: InitVar[Tuple[int, ...]]
    action_dim: InitVar[int]

    capacity: int = DEFAULT_CAPACITY
    states: torch.Tensor = field(init=False)
    prior: torch.Tensor = field(init=False)
    values: torch.Tensor = field(init=False)
    write_idx: "SynchronizedBase[c_int]" = field(init=False)
    read_idx: "SynchronizedBase[c_int]" = field(init=False)
    size: "SynchronizedBase[c_int]" = field(init=False)
    total_written: "SynchronizedBase[c_int]" = field(init=False)
    lock: MpLock = field(init=False)
    not_empty: MpCondition = field(init=False)
    not_full: MpCondition = field(init=False)

    def __post_init__(self, state_shape: Tuple[int, ...], action_dim: int):
        self.states = torch.zeros((self.capacity, *state_shape), dtype=torch.float32)
        self.prior = torch.zeros((self.capacity, action_dim), dtype=torch.float32)
        self.values = torch.zeros((self.capacity, 1), dtype=torch.float32)

        self.states.share_memory_()
        self.prior.share_memory_()
        self.values.share_memory_()

        self.write_idx = cast("SynchronizedBase[c_int]", mp.Value("i", 0))
        self.read_idx = cast("SynchronizedBase[c_int]", mp.Value("i", 0))
        self.size = cast("SynchronizedBase[c_int]", mp.Value("i", 0))
        self.total_written = cast("SynchronizedBase[c_int]", mp.Value("i", 0))
        self.lock = mp.Lock()
        self.not_empty = mp.Condition(self.lock)
        self.not_full = mp.Condition(self.lock)

    def put(
        self,
        state: NNState,
        prior: TensorActions,
        value: TensorValue,
        timeout: Optional[float] = None,
    ) -> bool:
        """向缓冲区写入一条数据

        Args:
            state: 游戏状态
            prior: 策略先验概率
            value: 价值评估
            timeout: 超时时间（秒），None 表示无限等待

        Returns:
            bool: 成功写入返回 True，超时返回 False
        """
        end_time = None if timeout is None else time.time() + timeout
        with self.not_full:
            while self.size.value >= self.capacity:  # type: ignore
                if timeout is None:
                    self.not_full.wait()
                else:
                    remaning = cast(float, end_time) - time.time()
                    if remaning <= 0:
                        return False
                    self.not_full.wait(timeout=remaning)

            idx = self.write_idx.value  # type: ignore
            self.states[idx].copy_(state)
            self.prior[idx].copy_(prior)
            self.values[idx].copy_(value)

            self.write_idx.value = (idx + 1) % self.capacity  # type:ignore
            self.size.value = min(self.size.value + 1, self.capacity)  # type:ignore
            self.total_written.value += 1  # type:ignore
            self.not_empty.notify()
            return True

    def get(self, timeout: Optional[float] = None) -> ExperienceDate | None:
        """从缓冲区按顺序获取一条数据

        Args:
            timeout: 超时时间（秒），None 表示无限等待

        Returns:
            ExperienceDate | None: 如果成功获取返回 (state, prior, value)，超时返回 None
        """
        end_time = None if timeout is None else time.time() + timeout
        with self.not_empty:
            while self.size.value <= 0:  # type:ignore
                if timeout is None:
                    self.not_empty.wait()
                else:
                    remaining = cast(float, end_time) - time.time()
                    if remaining <= 0:
                        return None
                    self.not_empty.wait(timeout=remaining)
            idx = self.read_idx.value  # type:ignore
            state = self.states[idx].detach().clone()
            prior = self.prior[idx].detach().clone()
            value = self.values[idx].detach().clone()

            self.read_idx.value = (idx + 1) % self.capacity  # type:ignore
            self.size.value -= 1  # type:ignore
            self.not_full.notify()

            return state, prior, value

    def put_tupule_experience(self, experience: ExperienceDate) -> bool:
        state, prior, value = experience
        return self.put(state, prior, value)

    def sample(
        self, batch_size: int, timeout: Optional[float] = None
    ) -> ExperienceBatch:
        """从缓冲区随机采样一批数据

        Args:
            batch_size: 批次大小
            timeout: 超时时间（秒），None 表示无限等待

        Returns:
            ExperienceBatch | None: 如果成功采样返回 (states, prior, values)，超时返回 None
        """
        end_time = None if timeout is None else time.time() + timeout
        with self.not_empty:
            while self.size.value < batch_size:  # type:ignore
                if timeout is None:
                    logger.debug("共享环形缓冲区采样等待数据")
                    self.not_empty.wait()
                else:
                    remaining = cast(float, end_time) - time.time()
                    if remaining <= 0:
                        logger.warning("共享环形缓冲区采样超时")
                        return None
                    self.not_empty.wait(timeout=remaining)

        size = self.size.value  # type:ignore
        k = min(batch_size, cast(int, size))
        idxs = torch.randint(0, cast(int, size), (k,), dtype=torch.int64)
        states = self.states[idxs].detach().clone()
        prior = self.prior[idxs].detach().clone()
        values = self.values[idxs].detach().clone()
        return states, prior, values
