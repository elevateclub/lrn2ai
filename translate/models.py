import torch.nn as nn

class Transformer(nn.Module):
    pass

class ScheduledOptim():
    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmpup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        pass