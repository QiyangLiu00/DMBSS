import copy
import random
import torch.nn.functional as F
import numpy as np
import torch


def get_manifold(cos_sim):

    dim = cos_sim.shape[0]
    device = torch.device("cpu")
    # N = 6080
    # nnk = 160
    # alpha = 0.7
    # N = 420
    # nnk = 20
    # alpha = 0.6
    N = 2000
    nnk = 45
    alpha = 0.9
    diag = torch.eye(N, dtype=torch.bool, device=device)
    cos_sim = torch.tensor(cos_sim)
    sim_masked = cos_sim * ~diag
    # cos_sim_mask = binaryize_similarity_matrix(positive_masked, 5)
    sim1 = sim_masked

    # sim 矩阵减去单位矩阵（主对角线元素为1）以去除每个向量与自身的内积。
    # 然后，对于每个向量，从sim 矩阵中选择与之最相似的前 nnk 个向量，并将其他位置上的值设为0
    # sim = sim1 - torch.eye(dim).type(torch.FloatTensor).to(device)
    sim = sim1 * 1.0
    top = torch.rand((1, dim)).type(torch.FloatTensor)
    # 对于sim保留每行数据数值前nnk的数据，其他置为0
    for i in range(dim):
        top[0, :] = sim[i, :]
        top20 = top.sort()[1][0]
        zero = torch.zeros(dim).type(torch.FloatTensor)
        zero[top20[-nnk:]] = 1.0
        #  sim 矩阵第 i 行中排名靠前的 nnk 个元素保留，其他元素置为0
        sim[i, :] = top[0, :] * zero

    # 根据sim矩阵的值将其转换为一个稀疏的对称矩阵 A。
    # 具体来说，将sim 中大于 0.0001 的值设为 1，其他值设为 0，并根据对角线对称进行修改。
    # 接着，找到A 矩阵中每一行和每一列非零元素个数小于等于 0 的行和列。
    # 通过随机采样，从特征矩阵中删除这些行和列，并记录删除的行和列的索引

    # 将sim矩阵中大于0.0001的元素设为1，小于等于0.0001的元素设为0，形成一个二进制矩阵A
    A = (sim > 0.0001).type(torch.FloatTensor).to(device)
    # 将矩阵 A 与其转置相乘，得到一个对称矩阵。
    # 该操作将 A 矩阵变成一个对称的二进制矩阵，其中对角线上的元素为0
    A = A * (A.t())

    # 将对称的二进制矩阵 A 与原始相似度矩阵 sim 对应元素相乘，得到的结果是一个对称的相似度矩阵。
    A = A * sim
    # 计算了 A 矩阵每一行的和，将得到一个列向量，其中第 i 个元素表示矩阵 A 第 i 行的和。
    sum_row = A.sum(1)
    # 首先用 (sum_row > 0) 得到一个二进制列向量，其中大于0的元素对应的位置为1，否则为0。
    # 然后用 (sum_row > 0).sum() 统计该列向量中值为1的元素的个数，即矩阵 A 中非全0行的个数。
    # 最后，用 dim 减去该值得到 aa，即矩阵 A 中全0行的个数。
    aa = dim - (sum_row > 0).sum()
    # 对 sum_row 列向量进行排序，sum_row.sort() 会返回一个元组，其中第一个元素是排序后的值，
    # 第二个元素是排序后的索引。kk 就是排序后的索引，即排序后的行号列表。
    kk = sum_row.sort()[1]
    res_ind = list(range(dim))
    for ind in range(aa):
        res_ind.remove(kk[ind])
    res_ind = random.sample(res_ind, dim - aa)
    # 记录打乱后与打乱前的行索引对应关系
    # 例：{0: 0, 1: 5, 2: 7, 3: 1, 4: 2, 5: 3, 6: 6}
    # 打乱后索引为0的行原来的行索引为0，打乱后索引为1的行原来的行索引为5，
    ind_to_new_id = {}
    for i in range(dim - aa):
        ind_to_new_id[i] = res_ind[i]
    # 这部分代码使用新的索引列表 res_ind，从原始相似度矩阵 sim 中提取对应的子矩阵，并更新 sim 为提取后的子矩阵。
    # 通过这个过程，删除了相似度矩阵中全0行和全0列的元素，得到一个新的相似度矩阵。

    res_ind = (torch.from_numpy(np.asarray(res_ind))).type(torch.LongTensor)
    sim = sim[res_ind, :]
    sim = sim[:, res_ind]

    sim20 = {}
    dim = dim - aa
    top = torch.rand((1, dim)).type(torch.FloatTensor)
    for i in range(dim):
        top[0, :] = sim[i, :]
        top20 = top.sort()[1][0]
        zero = torch.zeros(dim).type(torch.FloatTensor)
        zero[top20[-nnk:]] = 1.0
        k = list(top20[-nnk:])
        sim20[i] = k
        sim[i, :] = top[0, :] * zero
    A = (sim > 0.0001).type(torch.FloatTensor).to(device)
    A = A * (A.t())
    A = A * sim
    sum_row = A.sum(1)

    sum_row = sum_row.pow(-0.5)

    sim = torch.diag(sum_row)
    A = A.mm(sim)
    A = sim.mm(A)

    manifold_sim = (1 - alpha) * torch.inverse(torch.eye(dim).type(torch.FloatTensor) - alpha * A.cpu())

    sim = torch.zeros((N, N))
    for i in range(len(manifold_sim)):
        for j in range(len(manifold_sim)):
            sim[ind_to_new_id[i]][ind_to_new_id[j]] = manifold_sim[i][j]
    return sim
