from config import *
from nn_models.base import BaseModel
import torch
from torch.optim import Optimizer, Adam
from utils.logger import setup_logger
from typing import Optional

logger = setup_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: BaseModel,
        experience_pool: ExperiencePoolType,
        optim: Optional[Optimizer] = None,
    ) -> None:
        self.model = model
        self.experience_pool = experience_pool
        if optim is None:
            self.optim = Adam(self.model.parameters(), lr=LEARNING_RATE)
        else:
            self.optim = optim

    def train(self) -> None:
        """训练一个批次"""
        try:
            self.optim.zero_grad()
            loss = self._calculate_loss()
            loss.backward()
            self.optim.step()
        except ValueError as e:
            logger.warning(f"跳过训练步骤: {e}")
            return

    def _calculate_loss(self) -> torch.Tensor:
        """计算损失函数
        
        Returns:
            torch.Tensor: 总损失（策略损失 + 价值损失）
            
        Raises:
            ValueError: 如果经验池中没有足够的数据
        """
        
        batch = self.experience_pool.sample(BATCH_SIZE)
        if batch is None:
            raise ValueError("经验池中没有足够的数据")
        states, target_policies, target_values = batch
        
        pred_policies, pred_values = self.model(states)
        log_policies = torch.log_softmax(pred_policies, dim=1)
        policy_loss = -torch.sum(target_policies * log_policies, dim=1).mean()
        value_loss = torch.nn.functional.mse_loss(
            pred_values.squeeze(), target_values.squeeze()
        )
        total_loss = policy_loss + value_loss
        return total_loss

    def load_model(self, model_path: str) -> None:
        try:
            self.model.load_state_dict(torch.load(model_path))
            logger.info(f"模型已加载: {model_path}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")

    def save_model(self, model_path: str) -> None:
        try:
            torch.save(self.model.state_dict(), model_path)
            logger.info(f"模型已保存: {model_path}")
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
