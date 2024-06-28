import os

import torch
import torchvision
from PIL import Image
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import transforms

from augmentations import get_aug
from datasets import get_dataset
from datasets.DataSet import MyCustomDataset


def input_data(args):
    # 图像预处理
    trans = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # 自定义数据集

    class_path = '../Data/UCM/class.txt'

    train_data = MyCustomDataset(root_path="../Data/UCM/UCM_train_list.txt", class_path=class_path,
                                 transform=get_aug(train=True, **args.aug_kwargs))
    # test_data = MyCustomDataset(root_path="../Data/UCM/UCM_test_list.txt", class_path=class_path,
    #                             transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs))
    test_data = MyCustomDataset(root_path="../Data/UCM/UCM_test_list.txt", class_path=class_path,
                                transform=trans)

    # class_path = '../Data/AID/class.txt'
    #
    # train_data = MyCustomDataset(root_path="../Data/AID/AID_train_list.txt", class_path=class_path,
    #                              transform=get_aug(train=True, **args.aug_kwargs))
    # # test_data = MyCustomDataset(root_path="../Data/UCM/UCM_test_list.txt", class_path=class_path,
    # #                             transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs))
    # test_data = MyCustomDataset(root_path="../Data/AID/AID_test_list.txt", class_path=class_path,
    #                             transform=trans)

    # class_path = '../Data/PatternNet/class.txt'
    #
    # train_data = MyCustomDataset(root_path="../Data/PatternNet/PatternNet_train_list.txt", class_path=class_path,
    #                              transform=get_aug(train=True, **args.aug_kwargs))
    # # test_data = MyCustomDataset(root_path="../Data/UCM/UCM_test_list.txt", class_path=class_path,
    # #                             transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs))
    # test_data = MyCustomDataset(root_path="../Data/PatternNet/PatternNet_test_list.txt", class_path=class_path,
    #                             transform=trans)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True,
                                                   num_workers=0, pin_memory=True, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0,
                                                  pin_memory=True)

    data_loaders = {
        "train": train_dataloader,
        "test": test_dataloader
    }

    return data_loaders
    #
    # # data_dir = '../../数据集/UCM_swin'  # 样本地址
    # data_dir = '../../数据集/AID_swin'  # 样本地址
    #
    # # 构建训练和验证的样本数据集，字典格
    # # os.path.join实现路径拼接
    #
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])  # data_transforms也是字典格式，
    #                   for x in ['test']}
    # # 分别对训练和验证样本集构建样本加载器，还可以针对minibatch、多线程等进行针对性构建
    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=False, num_workers=0)
    #                for x in ['test']}
    # # 分别计算训练与测试的样本数，字典格式
    # dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}  # 训练与测试的样本数
    #
    # return image_datasets, dataloaders, dataset_sizes
