from utils.data_enhancer import DataEnhancer
import torch

d = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
p = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8]])

print("原始数据：")
print(d)
print("原始策略：")
print(p)

print("\n旋转90度：")
rotated_d, rotated_p = DataEnhancer.rotate(d, p, k=1)
print(rotated_d)
print(rotated_p)