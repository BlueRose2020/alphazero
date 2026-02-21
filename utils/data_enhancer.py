import torch
from config.utils_config import *
from config.type_alias import *


class DataEnhancer:
    """数据增强器，提供了二维和三维游戏状态的八种对称形式（旋转，对称），
    其中三维情况旋转的为最内层两个维度的矩阵

    ps:策略概率的旋转是基于状态的后两维进行的
    """

    @staticmethod
    def get_enhance_data(
        state: NNState, priors: TensorActions, result: TensorValue
    ) -> list[ExperienceDate]:
        """
        若有特殊的数据增强需求，请重写此方法以满足需求，
        但请确保参数和返回的数据格式与原方法一致，以保证与经验池的兼容性

        Args:
            state (NNState): 神经网络输入的状态张量，维度为(N, C, H, W)，其中N为批次大小，C为通道数，H和W为状态的空间维度
            priors (TensorActions): 策略概率张量
            result (TensorValue): 当前状态的结果值，通常为1（当前玩家胜利）、-1（当前玩家失败）或0（平局）

        Returns:
            list[ExperienceDate]: 一个包含增强后经验数据的列表，每个元素都是一个元组，包含增强后的状态、策略概率和结果值
        """
        enhanced_data = []
        if USE_DATA_ENHANCEMENT and USE_ROATATION:
            for k in range(4):
                rotated_state, rotated_priors = DataEnhancer.rotate(state, priors, k)
                enhanced_data.append((rotated_state, rotated_priors, result))
                if USE_FLIP:
                    flipped_state, flipped_priors = DataEnhancer.flip(
                        rotated_state, rotated_priors
                    )
                    enhanced_data.append((flipped_state, flipped_priors, result))
            return enhanced_data
        elif USE_DATA_ENHANCEMENT and USE_FLIP:
            flipped_state, flipped_priors = DataEnhancer.flip(state, priors)
            enhanced_data.append((flipped_state, flipped_priors, result))
            return enhanced_data
        else:
            return [(state, priors, result)]

    @staticmethod
    def rotate(state: NNState, priors: TensorActions, k: int):
        state_shape = state.shape
        for i in (3, 4):
            if len(state_shape) == i:
                rotated_state = torch.rot90(state, k=k, dims=[-2, -1])
                rotated_priors = torch.rot90(
                    priors.reshape(*state_shape[-2:]), k=k, dims=[0, 1]
                ).reshape(-1)

        return rotated_state.detach().clone(), rotated_priors.detach().clone()

    @staticmethod
    def flip(state: NNState, priors: TensorActions):
        state_shape = state.shape
        for i in (3, 4):
            if len(state_shape) == i:
                flipped_state = torch.flip(state, [-1])
                flipped_priors = (
                    torch.flip(priors.reshape(*state_shape[-2:]), [-1])
                    .reshape(-1)
                    .squeeze(0)
                )
        return flipped_state.detach().clone(), flipped_priors.detach().clone()
