import numpy as np
import torch
import ray
import random


class ActorBuffer(object):
    def __init__(self, state_dim, size):
        self.ptr = 0
        self.max_size = size
        self.advantages = np.zeros(size, dtype=np.float32)
        self.modelerr = np.zeros(size, dtype=np.float32)
        self.deltas = np.zeros(size, dtype=np.int32)
        self.para_nums = np.zeros(size, dtype=np.int32)

    def store(self, adv, err, noise_idx, para_num=0):
        assert self.ptr < self.max_size
        self.advantages[self.ptr] = adv
        self.modelerr[self.ptr] = err
        self.deltas[self.ptr] = noise_idx
        self.para_nums[self.ptr] = para_num
        self.ptr += 1

    def get(self):
        return self.advantages, self.modelerr, self.deltas, self.para_nums

    def reset(self):
        self.ptr = 0


class CriticBuffer(object):
    def __init__(self, state_dim, max_size, gamma, gae_coeff):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.not_deads = np.zeros(max_size, dtype=np.int32)
        self.returns = np.zeros(max_size, dtype=np.float32)
        self.value_pred = np.zeros(max_size, dtype=np.float32)
        self.last_ptr = 0
        self.gamma = gamma
        self.gae_coeff = gae_coeff

    def add(self, state, next_state, reward, not_dead, value_pred):
        assert self.ptr < self.max_size
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.not_deads[self.ptr] = not_dead
        self.value_pred[self.ptr] = value_pred
        self.ptr = self.ptr + 1
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[ind]),
            torch.FloatTensor(self.next_states[ind]),
            torch.FloatTensor(self.rewards[ind]).squeeze(),
            torch.IntTensor(self.not_deads[ind]).squeeze()
        )

    def get(self):
        return (
            torch.FloatTensor(self.states[:self.ptr]),
            torch.FloatTensor(self.returns[:self.ptr])
        )

    def get_buffer_size(self):
        return self.size

    def reset(self):
        self.last_ptr = 0
        self.ptr = 0
        self.size = 0

    def finish_trajs(self, last_val=0):
        path_slice = slice(self.last_ptr, self.ptr)
        value_preds_slice = np.append(self.value_pred[path_slice], last_val)
        rews_slice = self.rewards[path_slice]
        # the next line computes rewards-to-go, to be targets for the value function
        length = len(rews_slice)
        output = np.zeros(length)
        gae = 0.
        for i in reversed(range(length)):
            delta = rews_slice[i] + self.gamma * value_preds_slice[i + 1] - value_preds_slice[i]
            gae = delta + self.gamma * self.gae_coeff * gae
            output[i] = gae + value_preds_slice[i]
        self.returns[path_slice] = output
        self.last_ptr = self.ptr
