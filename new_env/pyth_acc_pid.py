import random
import math
import numpy as np


class PIDController(object):
    def __init__(self, env, seed):
        random.seed(seed)
        np.random.seed(seed)
        self.action_space = env.action_space
        # Tunable parameters (4):
        # K， Kp, Ki, Kd
        self.tunable_para_high = np.array([10., 10., 10., 10.])
        self.tunable_para_low = np.array([0., 0., 0., 0.])
        self.lin_para_gain = 0.5 * (self.tunable_para_high - self.tunable_para_low)
        self.lin_para_bias = 0.5 * (self.tunable_para_high + self.tunable_para_low)
        self.tunable_para_mapped = np.random.uniform(-1, 1, 4)
        self.tunable_para_unmapped = self.tunable_para_transform(self.tunable_para_mapped, after_map=True)
        self.tunable_para_sigma = 0.01 * np.ones(4)
        self.para_num = 4

    def tunable_para_transform(self, para_in, after_map):
        if after_map:
            lin_para_mapped = para_in
            lin_para_unmapped = self.lin_para_gain * lin_para_mapped + self.lin_para_bias
            para_out = lin_para_unmapped
        else:
            lin_para_unmapped = para_in
            lin_para_mapped = (lin_para_unmapped - self.lin_para_bias) / self.lin_para_gain
            para_out = lin_para_mapped
        return para_out

    def get_flat_param(self, after_map=True):
        if after_map:
            return self.tunable_para_mapped
        else:
            return self.tunable_para_unmapped

    def set_flat_param(self, para, after_map=True):
        if after_map:
            para = np.clip(para, -1., 1.)
            self.tunable_para_mapped = para
            para_unmapped = self.tunable_para_transform(para, after_map)
            self.tunable_para_unmapped = para_unmapped
        else:
            para = np.clip(para, self.tunable_para_low, self.tunable_para_high)
            self.tunable_para_unmapped = para
            para_mapped = self.tunable_para_transform(para, after_map)
            self.tunable_para_mapped = para_mapped

    def get_flat_sigma(self):
        return self.tunable_para_sigma

    def set_flat_sigma(self, para):
        self.tunable_para_sigma = np.clip(para, 1e-2, 0.2)

    def get_action(self, state):
        # state: dd_t, dv_t, a_t, dd_t1, dv_t1, a_t1, dd_t2, dv_t2, a_t2, u_t1
        error = state[1] + self.tunable_para_unmapped[0] * state[0]
        error_1 = state[4] + self.tunable_para_unmapped[0] * state[3]
        error_2 = state[7] + self.tunable_para_unmapped[0] * state[6]
        p_term = error - error_1
        i_term = error
        d_term = error - 2 * error_1 + error_2
        action = self.tunable_para_unmapped[1] * p_term \
                 + self.tunable_para_unmapped[2] * i_term \
                 + self.tunable_para_unmapped[3] * d_term
        return action, None
