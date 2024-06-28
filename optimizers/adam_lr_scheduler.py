import torch.optim.lr_scheduler as lr_scheduler


class Adam_LRScheduler(object):
    def __init__(self, optimizer, base_lr):
        self.base_lr = base_lr
        self.optimizer = optimizer

    def step(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr'] = self.base_lr
        return lr

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']


class AdamLRScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        super(AdamLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
