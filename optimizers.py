# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

import numpy as np


class Optimizer:
    def __init__(self, policy):
        self.policy = policy
        self.dim = policy.para_num
        self.t = 0

    def update_mean(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        theta = self.policy.get_flat_param(after_map=True)
        ratio = np.linalg.norm(step) / np.linalg.norm(theta)
        return theta + step, ratio

    def update_std(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        theta = np.log(self.policy.get_flat_sigma())
        ratio = np.linalg.norm(step) / np.linalg.norm(theta)
        return theta + step, ratio

    def _compute_step(self, globalg):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, policy, stepsize, momentum=0.0):
        Optimizer.__init__(self, policy)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.stepsize, self.momentum = stepsize, momentum

    def _compute_step(self, globalg):
        self.v = self.momentum * self.v + (1. - self.momentum) * globalg
        step = -self.stepsize * self.v
        return step


class Adam(Optimizer):
    def __init__(self, policy, lr, beta1=0.9, beta2=0.999,
                 epsilon=1e-08):
        Optimizer.__init__(self, policy)
        self.stepsize = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, globalg):
        a = self.stepsize * (np.sqrt(1 - self.beta2**self.t) /
                             (1 - self.beta1**self.t))
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        if self.stepsize > 1e-2:
            self.stepsize -= 2e-4
        return step
