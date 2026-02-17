from config import *
from nn_models.base import BaseModel
import torch
from torch.optim import Optimizer, Adam
from utils.logger import setup_logger
from typing import Optional,Callable

logger = setup_logger(__name__)

def _create_counter()->Callable[[], int]:
    count = 0
    def counter():
        nonlocal count
        count += 1
        return count
    return counter

counter = _create_counter()


class Trainer:
    def __init__(
        self,
        model: BaseModel,
        experience_pool: ExperiencePoolType,
        optim: Optional[Optimizer] = None,
    ) -> None:
        self.model = model.to(DEVICE)
        self.experience_pool = experience_pool
        if optim is None:
            self.optim = Adam(self.model.parameters(), lr=LEARNING_RATE)
        else:
            self.optim = optim

    def train(self, batch_size:int = BATCH_SIZE) -> None:
        """训练一个批次"""
        try:
            self.optim.zero_grad()
            loss = self._calculate_loss(batch_size)
            loss.backward()
            self.optim.step()
        except ValueError as e:
            logger.warning(f"跳过训练步骤: {e}")
            return

    def _calculate_loss(self,batch_size:int = BATCH_SIZE) -> torch.Tensor:
        """计算损失函数

        Returns:
            torch.Tensor: 总损失（策略损失 + 价值损失）

        Raises:
            ValueError: 如果经验池中没有足够的数据
        """

        batch = self.experience_pool.sample(batch_size)
        if batch is None:
            raise ValueError("经验池中没有足够的数据")
        states, target_policies, target_values = batch
        states = states.to(DEVICE)
        target_policies = target_policies.to(DEVICE)
        target_values = target_values.to(DEVICE)

        pred_policies, pred_values = self.model(states)
        log_policies = torch.log_softmax(pred_policies, dim=1)
        policy_loss = -torch.sum(target_policies * log_policies, dim=1).mean()
        value_loss = torch.nn.functional.mse_loss(
            pred_values.squeeze(), target_values.squeeze()
        )

        total_loss = policy_loss + value_loss
        if counter() % LOSS_LOG_FREQUENCY == 0:
            logger.debug(
                f"计算损失: policy_loss={policy_loss.item():.4f}, value_loss={value_loss.item():.4f}, total_loss={total_loss.item():.4f}"
            )
        return total_loss

    def load_model(self, model_path: str) -> None:
        try:
            self.model.load_state_dict(torch.load(model_path))
            logger.info(f"模型已加载: {model_path}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")

    def save_model(self, model_path: str) -> None:
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"模型已保存: {model_path}")
        
