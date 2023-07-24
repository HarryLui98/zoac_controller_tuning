import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import collections
import random
from scipy.linalg import toeplitz
import math

global device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


def layer_init(in_size, out_size, nonlinear, init_method='default', actor_last=False):
    gain = nn.init.calculate_gain(nonlinear)
    if actor_last:
        gain *= 0.01
    module = nn.Linear(in_size, out_size)
    if init_method == 'orthogonal':
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0.)
    return module


class LinearActor(object):
    def __init__(self, input_dim, output_dim, max_action, seed=123):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.max_action = max_action
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.para_num = input_dim * output_dim
        self.para_weight = torch.zeros((self.output_dim, self.input_dim), dtype=torch.float32)

    def get_action(self, state, action_noise=False, action_noise_sigma=0.1):
        action = torch.mv(self.para_weight, state).cpu().data.numpy().flatten()
        if not action_noise:
            return action
        noise = np.random.normal(0, 1, self.output_dim)
        return action + action_noise_sigma * noise

    def get_flat_param(self):
        flat_param_weight = copy.deepcopy(self.para_weight).reshape(-1).cpu().data.numpy().flatten()
        return flat_param_weight

    def set_flat_param(self, theta):
        self.para_weight = torch.FloatTensor(copy.deepcopy(theta).reshape((self.output_dim, self.input_dim))).to(device)


class ToeplitzLinearActor(object):
    def __init__(self, input_dim, output_dim, max_action=1, seed=123):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_action = max_action
        self.para_num = self.input_dim + self.output_dim - 1
        self.mat_weight = torch.zeros((self.output_dim, self.input_dim), dtype=torch.float32)

    def get_action(self, state, action_noise=False, action_noise_sigma=0.1):
        action = torch.mv(self.mat_weight, state).cpu().data.numpy().flatten()
        if not action_noise:
            return action
        noise = np.random.normal(0, 1, self.output_dim)
        return action + action_noise_sigma * noise

    def get_flat_param(self):
        first_column = np.array(self.mat_weight[:, 0])
        first_row = np.array(self.mat_weight[0, :])
        if len(first_row) > 1:
            flat_weight = np.concatenate((first_column, first_row[1:]))
        else:
            flat_weight = first_column
        return flat_weight

    def set_flat_param(self, theta):
        first_column = theta[0:self.output_dim]
        first_row = theta[self.output_dim - 1:]
        self.mat_weight = torch.FloatTensor(toeplitz(first_column, first_row))


class NeuralActor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=64, max_action=1, seed=123):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        super().__init__()
        self.l1 = layer_init(input_dim, hidden_size, 'tanh', 'orthogonal')
        self.ln1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.l2 = layer_init(hidden_size, hidden_size, 'tanh', 'orthogonal')
        self.ln2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.l3 = layer_init(hidden_size, output_dim, 'tanh', 'orthogonal', actor_last=True)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_action = max_action
        self.param_shapes = {
            k: tuple(self.state_dict()[k].size())
            for k in sorted(self.state_dict().keys())
        }
        self.para_num = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        a = torch.tanh(self.ln1(self.l1(x)))
        a = torch.tanh(self.ln2(self.l2(a)))
        a = self.max_action * torch.tanh(self.l3(a))
        return a

    def get_action(self, state, action_noise=False, action_noise_sigma=0.1):
        # net_input = torch.FloatTensor(state).to(device)
        action = self.__call__(state).detach().cpu().data.numpy().flatten()
        if not action_noise:
            return action
        noise = np.random.normal(0, 1, self.output_dim)
        return action + action_noise_sigma * noise

    def get_flat_param(self):
        theta_dict = self.state_dict()
        theta_list = []
        for k in sorted(theta_dict.keys()):
            theta_list.append(torch.reshape(theta_dict[k], (-1,)))
        flat_para = torch.cat(theta_list, dim=0).cpu().numpy()
        return flat_para

    def set_flat_param(self, theta):
        pos = 0
        theta_dict = self.state_dict()
        new_theta_dict = collections.OrderedDict()
        for k in sorted(theta_dict.keys()):
            shape = self.param_shapes[k]
            num_params = int(np.prod(shape))
            new_theta_dict[k] = torch.from_numpy(
                np.reshape(theta[pos: pos + num_params], shape)
            )
            pos += num_params
        self.load_state_dict(new_theta_dict)


class Critic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=64, seed=123):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        super().__init__()
        self.l1 = layer_init(input_dim, hidden_size, 'tanh', 'orthogonal')
        self.l2 = layer_init(hidden_size, hidden_size, 'tanh', 'orthogonal')
        self.l3 = layer_init(hidden_size, output_dim, 'linear', 'orthogonal')
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.param_shapes = {
            k: tuple(self.state_dict()[k].size())
            for k in sorted(self.state_dict().keys())
        }
        self.para_num = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        v = torch.tanh(self.l1(x))
        v = torch.tanh(self.l2(v))
        v = self.l3(v)
        return v

    def get_value(self, func_input):
        # net_input = torch.FloatTensor(func_input).to(device)
        value = self.__call__(func_input).detach().cpu().data.numpy().flatten()
        return value

    def get_flat_param(self):
        theta_dict = self.state_dict()
        theta_list = []
        for k in sorted(theta_dict.keys()):
            theta_list.append(torch.reshape(theta_dict[k], (-1,)))
        flat_param = torch.cat(theta_list, dim=0).cpu().numpy()
        return flat_param

    def set_flat_param(self, theta):
        pos = 0
        theta_dict = self.state_dict()
        new_theta_dict = collections.OrderedDict()
        for k in sorted(theta_dict.keys()):
            shape = self.param_shapes[k]
            num_params = int(np.prod(shape))
            new_theta_dict[k] = torch.from_numpy(
                np.reshape(theta[pos: pos + num_params], shape)
            )
            pos += num_params
        self.load_state_dict(new_theta_dict)
