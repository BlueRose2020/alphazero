from __future__ import annotations
from dataclasses import dataclass, field, InitVar
import os
from typing import Optional, TYPE_CHECKING, cast

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
    state_shape: InitVar[ShapeType]
    num_action: InitVar[ShapeType]

    _capacity: int = DEFAULT_CAPACITY
    _states: torch.Tensor = field(init=False)
    _prior: torch.Tensor = field(init=False)
    _values: torch.Tensor = field(init=False)
    _write_idx: "SynchronizedBase[c_int]" = field(init=False)
    _read_idx: "SynchronizedBase[c_int]" = field(init=False)
    _size: "SynchronizedBase[c_int]" = field(init=False)
    _total_written: "SynchronizedBase[c_int]" = field(init=False)
    _lock: MpLock = field(init=False)
    _not_empty: MpCondition = field(init=False)
    _not_full: MpCondition = field(init=False)

    def __post_init__(self, state_shape: ShapeType, num_action: ShapeType):
        self._states = torch.zeros((self._capacity, *state_shape), dtype=torch.float32)
        self._prior = torch.zeros((self._capacity, *num_action), dtype=torch.float32)
        self._values = torch.zeros((self._capacity, 1), dtype=torch.float32)

        self._states.share_memory_()
        self._prior.share_memory_()
        self._values.share_memory_()
        
        self._write_idx = cast("SynchronizedBase[c_int]", mp.Value("i", 0))
        self._read_idx = cast("SynchronizedBase[c_int]", mp.Value("i", 0))
        self._size = cast("SynchronizedBase[c_int]", mp.Value("i", 0))
        self._total_written = cast("SynchronizedBase[c_int]", mp.Value("i", 0))
        self._lock = mp.Lock()
        self._not_empty = mp.Condition(self._lock)
        self._not_full = mp.Condition(self._lock)

    def put(
        self,
        state: NNState,
        prior: TensorActions,
        value: TensorValue,
    ) -> bool:
        """向缓冲区写入一条数据

        Args:
            state: 游戏状态
            prior: 策略先验概率
            value: 价值评估

        Returns:
            bool: 成功写入返回 True，超时返回 False
        """
        with self._lock:
            if self._size.value >= self._capacity:  # type: ignore
                # 覆盖最旧数据：读指针前移
                self._read_idx.value = (self._read_idx.value + 1) % self._capacity  # type: ignore
                self._size.value = self._capacity  # type: ignore

            idx = self._write_idx.value  # type: ignore
            self._states[idx].copy_(state)
            self._prior[idx].copy_(prior)
            self._values[idx].copy_(value)

            self._write_idx.value = (idx + 1) % self._capacity  # type:ignore
            self._size.value = min(self._size.value + 1, self._capacity)  # type:ignore
            self._total_written.value += 1  # type:ignore
            self._not_empty.notify()
            return True

    def get(self, timeout: Optional[float] = None) -> ExperienceDate | None:
        """从缓冲区按顺序获取一条数据

        Args:
            timeout: 超时时间（秒），None 表示无限等待

        Returns:
            ExperienceDate | None: 如果成功获取返回 (state, prior, value)，超时返回 None
        """
        end_time = None if timeout is None else time.time() + timeout
        with self._not_empty:
            while self._size.value <= 0:  # type:ignore
                if timeout is None:
                    self._not_empty.wait()
                else:
                    remaining = cast(float, end_time) - time.time()
                    if remaining <= 0:
                        return None
                    self._not_empty.wait(timeout=remaining)
            idx = self._read_idx.value  # type:ignore
            state = self._states[idx].detach().clone()
            prior = self._prior[idx].detach().clone()
            value = self._values[idx].detach().clone()
            self._read_idx.value = (idx + 1) % self._capacity  # type:ignore
            self._size.value -= 1  # type:ignore
            self._not_full.notify()

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
        with self._not_empty:
            while self._size.value < batch_size:  # type:ignore
                if timeout is None:
                    logger.debug("共享环形缓冲区采样等待数据")
                    self._not_empty.wait()
                else:
                    remaining = cast(float, end_time) - time.time()
                    if remaining <= 0:
                        logger.warning("共享环形缓冲区采样超时")
                        return None
                    self._not_empty.wait(timeout=remaining)

        size = cast(int,self._size.value)  # type:ignore
        k = min(batch_size, size)
        idxs = torch.randint(0, size, (k,), dtype=torch.int64)
        states = self._states[idxs].squeeze(1).detach().clone()
        prior = self._prior[idxs].squeeze(1).detach().clone()  # squeeze 掉中间的维度
        values = self._values[idxs].detach().clone()
        return states, prior, values

    def __len__(self) -> int:
        return self._capacity
    
    def size(self) -> int:
        return self._size.value  # type:ignore
    
    def save(self, filename: str) -> None:
        """保存缓冲区状态到文件
        
        Args:
            filename: 保存文件路径
        """
        with self._lock:
            checkpoint = {
                'states': self._states.clone(),
                'prior': self._prior.clone(),
                'values': self._values.clone(),
                'write_idx': self._write_idx.value,  # type:ignore
                'read_idx': self._read_idx.value,  # type:ignore
                'size': self._size.value,  # type:ignore
                'total_written': self._total_written.value,  # type:ignore
                'capacity': self._capacity,
            }
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            torch.save(checkpoint, filename)
            logger.info(f"共享环形缓冲区已保存到 {filename}")
        
    def load(self, filename: str) -> None:
        """从文件加载缓冲区状态
        
        Args:
            filename: 加载文件路径
        """
        with self._lock:
            checkpoint = torch.load(filename, weights_only=False)
            
            # 检查容量是否匹配
            if checkpoint['capacity'] != self._capacity:
                raise ValueError(
                    f"容量不匹配: 文件中为 {checkpoint['capacity']}, "
                    f"当前缓冲区为 {self._capacity}"
                )
            
            # 恢复张量数据
            self._states.copy_(checkpoint['states'])
            self._prior.copy_(checkpoint['prior'])
            self._values.copy_(checkpoint['values'])
            
            # 恢复索引状态
            self._write_idx.value = checkpoint['write_idx']  # type:ignore
            self._read_idx.value = checkpoint['read_idx']  # type:ignore
            self._size.value = checkpoint['size']  # type:ignore
            self._total_written.value = checkpoint['total_written']  # type:ignore
            
            logger.info(f"共享环形缓冲区已从 {filename} 加载")
            self._not_empty.notify_all()
            self._not_full.notify_all()