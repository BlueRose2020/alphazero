import torch
from config import *

class DataEnhancer:
    """数据增强器，提供了二维和三维游戏状态的八种对称形式（旋转，对称），
    其中三维情况旋转的为最内层两个维度的矩阵，如有特殊需自行编写静态类，
    并在配置文件中将数据增强器改为你的数据增强器类

    ps:策略概率的旋转是基于状态的后两维进行的，如不满足请自行编写旋转函数
    """

    @staticmethod
    def get_enhance_data(
        state: NNState, priors: TensorActions, result: TensorValue
    ) -> list[ExperienceDate]:
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
                flipped_state = torch.flip(state, [-2])
                flipped_priors = (
                    torch.flip(priors.reshape(*state_shape[-2:]), [-2])
                    .reshape(-1)
                    .squeeze(0)
                )
        return flipped_state.detach().clone(), flipped_priors.detach().clone()
