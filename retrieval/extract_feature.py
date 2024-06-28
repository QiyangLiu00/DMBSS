import os

import numpy as np
import torch
import torchvision
from torch import nn
from torchvision import datasets, models
from torchvision.transforms import transforms
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from datasets import get_dataset
from models import get_model, SimSiam, get_backbone
from retrieval.inputdata import input_data


def main(device, args):
    model = get_model(args.model).to(device)
    model_dict = model.state_dict()
    print(model)
    weights_path = '../cache/simclr-AID_240229203836.pth'
    weights = torch.load(weights_path)['state_dict']  # 读取预训练模型权重
    model.load_state_dict(weights)
    model.eval()

    features_h = []
    features_l = []
    lables = []

    dataloaders = input_data(args=args)
    # image_datasets, dataloaders, dataset_sizes = input_data(args=args)
    with torch.no_grad():
        for image, lable, img_location in tqdm(dataloaders['test']):
            image = image.to(device)
            lable = lable.to(device)
            feat = model(image, image, args, img_location, flag=1)

            feat_high = feat[0].view(-1)
            feat_low = feat[1].view(-1)
            lable = lable.view(-1)
            feat_high = feat_high.cpu().numpy()
            feat_low = feat_low.cpu().numpy()
            lable = lable.cpu().numpy()
            # print('lable:', lable)

            features_h.append(feat_high)
            features_l.append(feat_low)
            lables.append(lable)

    np.save('./特征和标签/fe_high.npy', features_h)
    np.save('./特征和标签/fe_low.npy', features_l)
    np.save('./特征和标签/la.npy', lables)
    print('lables:', lables)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(device=device, args=get_args())

    # weights_path = 'F:/LQY/workspace/SimSiam/models/backbones/retinanet_resnet50_fpn_coco-eeacb38b.pth'
    # weights = torch.load(weights_path)['state_dict']
    # for i in weights:
    #     print(i)
    #
    # x = torch.rand(16, 256, 56, 56)
    # net = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
    # y = net(x)
    # z = torch.flatten(y, 1)
    # print(x)



