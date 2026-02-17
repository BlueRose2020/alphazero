"""神经网络模型模板文件

使用说明:
1. 复制此文件到 models/ 目录并重命名
2. 修改类名
3. 实现共享层、策略头和价值头
4. 确保输入输出维度与游戏匹配
"""

from config import *
from nn_models.base import BaseModel
import torch
import torch.nn as nn


class TemplateModel(BaseModel):
    """神经网络模型模板
    
    AlphaZero 使用双头网络架构:
    - 共享层: 提取特征
    - 策略头: 输出动作概率分布 (shape: (batch, num_action))
    - 价值头: 输出状态价值评估 (shape: (batch, 1))
    
    必须实现:
    - __init__: 初始化网络层
    - shared_layers (property): 返回共享层
    - policy_head: 策略头前向传播
    - value_head: 价值头前向传播
    """
    
    def __init__(self) -> None:
        """初始化模型
        
        在这里定义所有网络层：
        - 共享特征提取层
        - 策略头层
        - 价值头层
        """
        super().__init__()
        
        # TODO: 实现共享层
        # 示例: self._shared_layers = nn.Sequential(...)
        raise NotImplementedError("请实现共享层")
        
        # TODO: 实现策略头
        # 示例: self._policy_head = nn.Sequential(...)
        raise NotImplementedError("请实现策略头")
        
        # TODO: 实现价值头
        # 示例: self._value_head = nn.Sequential(...)
        raise NotImplementedError("请实现价值头")
    
    @property
    def shared_layers(self) -> nn.Sequential:
        """返回共享特征提取层
        
        Returns:
            nn.Sequential: 共享层模块
        """
        raise NotImplementedError("请实现 shared_layers 属性")
    
    def policy_head(self, x: torch.Tensor) -> TensorActions:
        """策略头前向传播
        
        Args:
            x: 共享层的输出特征，shape=(batch, channels, height, width)
            
        Returns:
            TensorActions: 动作概率分布（未归一化），shape=(batch, num_action)
            
        注意:
            - 输出不需要经过 softmax，外部会处理
            - 确保输出维度为 num_action
        """
        raise NotImplementedError("请实现 policy_head 方法")
    
    def value_head(self, x: torch.Tensor) -> torch.Tensor:
        """价值头前向传播
        
        Args:
            x: 共享层的输出特征，shape=(batch, channels, height, width)
            
        Returns:
            torch.Tensor: 状态价值评估，shape=(batch, 1)，范围通常在 [-1, 1]
            
        注意:
            - 输出应该经过 tanh 激活函数，将值限制在 [-1, 1]
            - 返回形状为 (batch, 1)
        """
        raise NotImplementedError("请实现 value_head 方法")
