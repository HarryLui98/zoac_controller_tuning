import copy

import numpy as np
import torch

class MeanStdFilter(object):
    def __init__(self, shape):
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.std = torch.ones(shape, dtype=torch.float32)
        self.num = 0

    def forward(self, state):
        # x = np.array(state, dtype=np.float32)
        obs = (state - self.mean) / (self.std + 1e-8)
        return torch.clamp(obs, min=-10, max=10)

    def print(self):
        print("mean")
        print(self.mean)
        print("std")
        print(self.std)

    def save(self, output_dir):
        torch.save(self.mean, output_dir + '/mean')
        torch.save(self.std, output_dir + '/std')

    def set(self, mean, std):
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

    def update(self, incremental_data):
        incre_num = incremental_data.shape[0]
        incre_mean = torch.mean(incremental_data, dim=0)
        if incre_num > 1:
            incre_std = torch.std(incremental_data, dim=0)
        else:
            incre_std = torch.zeros(incre_num)
        ratio = self.num / (self.num + incre_num)
        new_mean = ratio * self.mean + (1 - ratio) * incre_mean
        temp_var_old = self.std ** 2 + (new_mean - self.mean) ** 2
        temp_var_incre = incre_std ** 2 + (new_mean - incre_mean) ** 2
        new_var = ratio * temp_var_old + (1 - ratio) * temp_var_incre
        self.mean = copy.deepcopy(new_mean)
        self.std = copy.deepcopy(torch.sqrt(new_var))
        self.num += incre_num
