import copy
import random
import torch.nn.functional as F
import numpy as np
import torch


def binaryize_similarity_matrix(similarity_matrix, m):
    binary_matrix = []
    for row in similarity_matrix:
        # 将张量转换为普通的Python列表
        row_list = row.tolist()
        # 对相似度排序并获取前m个元素的索引
        indices = sorted(range(len(row_list)), key=lambda k: row_list[k], reverse=True)[:m]
        # 创建当前行的二值化行
        binary_row = [1 if i in indices else 0 for i in range(len(row_list))]
        # 将结果转换回张量类型
        binary_row_tensor = torch.tensor(binary_row)
        binary_matrix.append(binary_row_tensor)
    # 将结果转换为张量类型
    binary_matrix_tensor = torch.stack(binary_matrix)
    return binary_matrix_tensor


def get_manifold(z1, z2, device, args):
    cos_sim_mask = []
    final_manifold_mask = []
    cos_similarity_matrix = []
    return cos_sim_mask, final_manifold_mask, cos_similarity_matrix
