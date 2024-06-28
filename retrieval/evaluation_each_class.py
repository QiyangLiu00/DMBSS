# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:25:44 2019

@author: 71773
"""

import numpy as np
import os
from typing import Dict, List
import scipy.io as scio
import torch
from PIL import Image

from retrieval.evaluation_manifold import get_manifold

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# KMP_DUPLICATE_LIB_OK = TRUE
# from config import config
def get_text_path(path):
    img_path = []
    targets = []
    # 首先读取图像和图像标签
    with open(path, 'r', encoding='UTF-8') as f:
        x = f.readlines()
        for name in x:
            filepath = name.strip().rsplit(" ", 1)[0]
            target = name.strip().rsplit(" ", 1)[1]
            target = int(target)
            img_path.append(filepath)
            targets.append(target)
    return img_path, targets

    # 保存标签和结果到文件


def save_results_to_file(results, filename):
    with open(filename, 'w') as f:
        for label, result in results.items():
            f.write(f"Label: {label}\n")
            f.write("Evaluation result:\n")
            f.write(f"{result}\n")
            f.write("\n")


def L2_normalization(X):
    """
    —对输入数据（X逐行）进行L2标准化
    """
    X_norm = X.copy()
    n = X.shape[0]
    for i in range(n):
        # print(i)
        c_norm = np.linalg.norm(X[i])  # 求出每行向量的模长
        if c_norm == 0:  # 特殊情况：全0向量
            c_norm = 1
        X_norm[i] = X[i] / c_norm
    return X_norm


def Euclidean_distance(X, Y):
    """
    —欧氏距离
    输入：
    X n*p数组, p为特征维度
    Y m*p数组

    输出：
    D n*m距离矩阵
    """
    n = X.shape[0]
    m = Y.shape[0]
    X2 = np.sum(X ** 2, axis=1)
    Y2 = np.sum(Y ** 2, axis=1)
    D = np.tile(X2.reshape(n, 1), (1, m)) + (np.tile(Y2.reshape(m, 1), (1, n))).T - 2 * np.dot(X, Y.T)
    D[D < 0] = 0
    D = np.sqrt(D)
    return D


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def evaluation(feature, label, distance_func):
    """
    features_path 图像特征库路径
    label_path 图像标签库路径
    distance_func 距离度量函数
    """
    # ----------------------预处理----------------------------
    # -----输入为.npy文件路径
    # img_all = np.load(feature_all_path, allow_pickle=True).item()
    # img_query = np.load(feature_query_path, allow_pickle=True).item()

    img_all_features = feature
    img_all_labels = label
    img_query_features = feature
    img_query_labels = label

    # -----输入为.mat文件路径
    # img_all_features = scio.loadmat(feature_all_path)['features']
    # img_all_labels = scio.loadmat(feature_all_path)['labels']
    # img_query_features = scio.loadmat(feature_query_path)['features']
    # img_query_labels = scio.loadmat(feature_query_path)['labels']

    # 检索结果存放文件夹
    # FE_identity = os.path.splitext(os.path.split(feature_query_path)[1])[0]
    # result_folder = results_root + FE_identity
    # os.makedirs(result_folder, exist_ok=True)
    # 存储评价结果
    results_array = {}

    maxdepth = img_all_labels.size  # 一次查询最大的返回图像数
    # print('maxdepth一次查询最大的返回图像数:', maxdepth)
    nQueries = img_query_labels.size  # 查询次数， 即查询图像数目
    # print('nQueries查询次数， 即查询图像数目:', maxdepth)

    unique_labels = np.unique(img_all_labels)  # 图像库类别数
    # print('unique_labels图像库类别数:', unique_labels)
    rel_docs = np.zeros(nQueries, float)  # 存储查询图像中每幅图像在图像库中的同类图像数目
    rel_classes = []  # 存储查询图像中每一个类别的图像数
    for i in unique_labels:
        res = np.where(img_all_labels == i)[0]  # 找出label为i的所有图像位置(下标)
        rel_docs[np.where(img_query_labels == i)[0]] = res.size
        rel_classes.append((np.where(img_query_labels == i)[0]).size)

    Kq = 2 * rel_docs
    rep_labels = np.tile(img_all_labels.reshape(1, -1), (nQueries, 1))  # 每行存储图像库中图像的label
    # print('rep_labels每行存储图像库中图像的label:\n', rep_labels)
    # reshape（行，列）可以根据指定的数值将数据转换为特定的行数和列数
    gt_labels = np.tile(img_query_labels, (1, maxdepth))  # 每一列存储查询图像的label
    # print('gt_labels每一列存储查询图像的label:\n', gt_labels)
    # np.tile（a,(2)）函数的作用就是将a沿着X轴扩大两倍。(扩大倍数只有一个，默认为X轴)
    # np.tile(a,(2,1))第一个参数为Y轴扩大倍数，第二个为X轴扩大倍数
    distance = []

    imgFea_all_norm = L2_normalization(img_all_features)
    imgFea_query_norm = L2_normalization(img_query_features)

    # print('imgFea_all_norm.shape:', imgFea_all_norm.shape)
    # print('imgFea_query_norm.shape:', imgFea_query_norm.shape)
    if distance_func == 'euclidean_distance':
        # distance = Euclidean_distance(imgFea_query_norm, imgFea_all_norm)  # 距离矩阵,每行是一幅查询图像的检索结果
        distance = CalcHammingDist(imgFea_query_norm, imgFea_all_norm)  # 距离矩阵,每行是一幅查询图像的检索结果
    elif distance_func == 'cos_sim':
        v1 = img_query_features
        v2 = img_all_features
        num = np.dot(v1, np.array(v2).T)  # 向量点乘
        denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
        cos_sim = num / denom
        cos_sim[np.isneginf(cos_sim)] = 0
        sim = 0.5 + 0.5 * cos_sim
        # print('cos_sim:\n', cos_sim)

        # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
        # image_idxs = np.argsort(distance, axis=1)  # 对distance进行按行排序，返回结果为排序后的索引
        image_idxs = np.argsort(sim, axis=1)  # 对cos_sim进行按行排序，返回结果为排序后的索引
        image_idxs = np.flip(image_idxs, axis=1)  # 对cos_sim进行按行翻转，得到余弦相似度由大到小排列后的索引
        # print('image_idxs:\n', image_idxs)

    elif distance_func == 'manifold_sim':
        v1 = img_query_features
        v2 = img_all_features
        num = np.dot(v1, np.array(v2).T)  # 向量点乘
        denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
        cos_sim = num / denom
        cos_sim[np.isneginf(cos_sim)] = 0
        cos_sim = 0.5 + 0.5 * cos_sim
        sim = get_manifold(cos_sim).numpy()

        # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
        # image_idxs = np.argsort(distance, axis=1)  # 对distance进行按行排序，返回结果为排序后的索引
        image_idxs = np.argsort(sim, axis=1)  # 对cos_sim进行按行排序，返回结果为排序后的索引
        image_idxs = np.flip(image_idxs, axis=1)  # 对cos_sim进行按行翻转，得到余弦相似度由大到小排列后的索引
        # print('image_idxs:\n', image_idxs)

    elif distance_func == 'cos+manifold':
        v1 = img_query_features
        v2 = img_all_features
        num = np.dot(v1, np.array(v2).T)  # 向量点乘
        denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
        cos_sim = num / denom
        cos_sim[np.isneginf(cos_sim)] = 0
        sim = 0.5 + 0.5 * cos_sim
        manifold_sim = get_manifold(sim).numpy()
        # top10 = {}
        k = 5
        # 记录余弦相似度下相似度前k的样本索引
        for i in range(cos_sim.shape[1]):
            top_cos = np.zeros((1, cos_sim.shape[1]))
            # 获取第i个查询与其他样本的余弦相似度的排序索引（从小到大）
            top_cos = sim[i, :].argsort()
            # 截取排名索引的后k个（最相似的k个样本索引）
            top10 = top_cos[-k:]
            # 流形相似度中 余弦相似度前k的样本置1
            manifold_sim[i, top10] = 1

        image_idxs = np.argsort(manifold_sim, axis=1)  # 对cos_sim进行按行排序，返回结果为排序后的索引
        image_idxs = np.flip(image_idxs, axis=1)  # 对cos_sim进行按行翻转，得到余弦相似度由大到小排列后的索引
    # ------------------------计算指标_AP--------------------------
    # performance evaluation
    result_labels = np.zeros((nQueries, maxdepth))  # 存储nQuieries次查询排序后图像的标签
    for i in range(nQueries):
        current_labels = rep_labels[i, :]  # 第i次查询图像库中图像的label
        temp_idxs = image_idxs[i, :]  # 第i次查询排序结果
        result_labels[i, :] = current_labels[temp_idxs]  # 第i次查询排序后标签
    results_maxdepth = (result_labels == gt_labels)  # nQueries次查询排序后的图像标签与真实标签比较
    precision = np.zeros((nQueries, maxdepth), float)  # 存储nQueries次查询，返回图像从1:maxdepth时的查准率
    recall = np.zeros((nQueries, maxdepth), float)
    avg_precision = np.zeros((nQueries, 1), float)
    results = []
    for pr in range(maxdepth):  # 计算average precision
        results = (result_labels[:, 0:pr + 1] == gt_labels[:, 0:pr + 1])
        num_tp = np.sum(results, axis=1)  # 返回影像数为pr时的相似图像数
        precision_k = num_tp / (pr + 1)
        recall_k = num_tp / rel_docs
        precision[:, pr] = precision_k
        recall[:, pr] = recall_k
        avg_precision = avg_precision + (precision_k * results_maxdepth[:, pr]).reshape(-1, 1)
    avg_precision = avg_precision / rel_docs
    np.savetxt('avg_precision.txt', avg_precision, delimiter=',', fmt='%.4f')
    # 计算每个类别的MAP
    num_classes = len(np.unique(label))
    MAP_per_class = np.zeros((num_classes,), float)
    for cls in range(num_classes):
        class_indices = np.where(gt_labels == cls)[0]
        class_avg_precision = avg_precision[class_indices].mean()
        MAP_per_class[cls] = class_avg_precision
    np.savetxt('evaluation_results_PatternNet.txt', MAP_per_class, delimiter=',', fmt='%.4f')
    # filename = 'evaluation_results_ucm.txt'
    # with open(filename, 'w') as f:
    #     for label, result in MAP_per_class.items():
    #         f.write(f"Label: {label}\n")
    #         f.write("Evaluation result:\n")
    #         f.write(f"{result}\n")  # 将结果转换为字符串形式写入文件
    #         f.write("\n")


    # ------------------------计算指标_mAP_AVR_NMRR_ANMRR--------------------
    # MAP = np.mean(avg_precision)
    # AVR = np.zeros((nQueries, 1), float)
    # NMRR = np.zeros((nQueries, 1), float)  # nQueries次查询的NMRR
    #
    # for q in range(nQueries):
    #     for n in range(maxdepth):
    #         if ((n + 1) <= Kq[q]):
    #             AVR[q] = AVR[q] + (n + 1) * results[q, n]
    #         else:
    #             AVR[q] = AVR[q] + (1.25 * Kq[q]) * results[q, n]
    #     AVR[q] = AVR[q] / rel_docs[q]
    #     NMRR[q] = (AVR[q] - 0.5 * (1 + rel_docs[q])) / (1.25 * Kq[q] - 0.5 * (1 + rel_docs[q]))
    # ANMRR = np.mean(NMRR)
    # results_array['ANMRR'] = ANMRR
    # results_array['MAP'] = MAP * 100

    # precision_steps = [5, 10, 20, 50, 100]  # 返回图像数，计算P@5,10,20,50,100
    # print('ANMRR为:{:.3f} MAP为:{:.2f}%'.format(ANMRR, MAP * 100))
    # for pr in precision_steps:
    #     # ddd = avg_precision[:, pr - 1]
    #     # dd = np.mean(ddd)*100
    #     prr = np.mean(precision[:, pr - 1]) * 100
    #     print('P@{}:{:.2f}%'.format(pr, prr), end=' ')
    #     results_array['P@{}'.format(pr)] = round(prr, 2)
    # return MAP*100

    # np.save(result_folder + '/results.npy', results_array)  # ——存入.npy文件
    # scio.savemat(result_folder+'/results.mat', {'results': results_array}) #——存入.mat文件

    # with open(result_folder + '/results.txt', "w") as f:
    #     f.write(str(results_array))


if __name__ == "__main__":
    distance_func = "cos_sim"
    distance_func = "manifold_sim"
    # distance_func = "cos+manifold"
    loadData_label = np.load('./特征和标签/la.npy', allow_pickle=True)

    loadData_feature_h = np.load('./特征和标签/fe_high.npy', allow_pickle=True)
    loadData_feature_l = np.load('./特征和标签/fe_low.npy', allow_pickle=True)

    # loadData_feature = torch.Tensor([item for item in loadData_feature_h])
    # loadData_feature = torch.squeeze(loadData_feature)
    # loadData_feature = np.array(loadData_feature)
    #
    # print('\n')
    # print("high")
    # evaluation(loadData_feature, loadData_label, distance_func)

    loadData_feature = torch.Tensor([item for item in loadData_feature_l])
    loadData_feature = torch.squeeze(loadData_feature)
    loadData_feature = np.array(loadData_feature)
    print('\n')
    print("low")
    evaluation(loadData_feature, loadData_label, distance_func)
