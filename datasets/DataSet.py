import numpy as np
import os
from PIL import Image
import torch


class MyCustomDataset():
    def __init__(self, root_path, class_path, transform=None):
        # stuff
        self.imagepath = root_path
        self.classpath = class_path
        # load image path and the corresponding labels
        self.img_path = []
        self.targets = []
        self.classes = []
        self.class_to_id = {}
        self.transform = transform
        # 首先读取图像和图像标签
        with open(self.imagepath, 'r', encoding='UTF-8') as f:
            x = f.readlines()
            for name in x:
                filepath = name.strip().rsplit(" ", 1)[0]
                target = name.strip().rsplit(" ", 1)[1]
                target = int(target)
                self.img_path.append(filepath)
                self.targets.append(target)
        # 读取类别和序号
        with open(self.classpath, 'r', encoding='UTF-8') as f:
            x = f.readlines()
            for name in x:
                classname = name.strip().rsplit(" ", 1)[0]
                classid = name.strip().rsplit(" ", 1)[1]
                self.classes.append(classname)
                self.class_to_id.update({classname: classid})

    # __getitem__() function returns the data and labels. This function is called from dataloader like this:
    def __getitem__(self, index):
        # stuff
        # im_path = self.img_path[index]
        # img = Image.open(im_path)
        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # label = np.array(self.img_label[index])
        # label = torch.from_numpy(label).type(torch.long)
        # return img, label

        img_location, target = self.img_path[index], self.targets[index]
        img = Image.open(img_location).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
            # pos_2 = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        # return pos_1, pos_2, target
        target = np.array(target)
        target = torch.from_numpy(target)
        return img, target, img_location

    def __len__(self):
        return len(self.img_path)
