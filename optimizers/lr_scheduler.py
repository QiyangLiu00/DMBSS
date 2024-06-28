import torch
import numpy as np
from torch.optim import lr_scheduler


# from torch.optim.lr_scheduler import _LRScheduler


class LR_Scheduler(object):
    # optimizer: 优化器，用于更新模型参数
    # warmup_epochs: 热身期轮数，学习率将在这个时间段内从 warmup_lr 逐渐增加到 base_lr
    # warmup_lr: 热身期学习率，学习率在热身期开始时设置为这个值，逐渐增加到 base_lr
    # num_epochs: 总轮数，学习率在这个时间段内从 base_lr 逐渐降到 final_lr
    # base_lr: 基础学习率，学习率在热身期后将从这个值开始降低
    # final_lr: 最终学习率，学习率将在训练结束时逐渐降到这个值
    # iter_per_epoch: 每轮迭代次数，用于计算总的迭代次数
    # constant_predictor_lr: 是否保持特定参数组的学习率不变，通常用于微调预训练模型时冻结大部分模型参数，只更新最后一层时使用。默认值为False。
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch,
                 constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        # 使用热身和余弦退火
        # 热身迭代次数=每轮迭代次数x热身轮次
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
                    1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr


if __name__ == "__main__":
    import torchvision

    model = torchvision.models.resnet50()
    optimizer = torch.optim.SGD(model.parameters(), lr=999)
    epochs = 800
    n_iter = 64
    scheduler = LR_Scheduler(optimizer, 50, 0.003, epochs, 0.03, 0.003, n_iter)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.5)

    import matplotlib.pyplot as plt

    lrs = []
    for epoch in range(epochs):
        for it in range(n_iter):
            lr = scheduler.step()
            print(lr)
            lrs.append(lr)
    plt.plot(lrs)
    plt.show()
