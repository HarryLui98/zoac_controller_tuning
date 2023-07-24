import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import os
import pandas as pd


class ACCEnv(gym.Env):
    def __init__(self, **kwargs):
        # state: dd_t, dv_t, a_t, dd_t1, dv_t1, a_t1, dd_t2, dv_t2, a_t2, u_t1
        self.TL = 0.45
        self.KL = 1.0
        self.tau_h = 2.5
        self.d0 = 5
        self.max_episode_steps = kwargs['max_episode_steps']
        self.state = None
        self.simulation_time = 0
        self.action = None
        self.step_T = 0.1
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * 10),
            high=np.array([np.inf] * 10),
            dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1.5]),
                                           high=np.array([0.6]),
                                           dtype=np.float32)
        self.daction_space = gym.spaces.Box(low=np.array([-0.1]),
                                           high=np.array([0.01]),
                                           dtype=np.float32)
        self.seed()
        self.steps = 0
        self.steps_beyond_done = None
        self.head_acc_list = self.set_heading_profile()
        self.head_acc = 0.
        self.window = 30

    def set_heading_profile(self):
        heading_acc = np.zeros(1001)
        for i in range(91):
            heading_acc[i] = 0.05
        for i in range(91, 121):
            heading_acc[i] = 0.15
        for i in range(121, 151):
            heading_acc[i] = -0.1
        for i in range(181, 211):
            heading_acc[i] = -0.2
        for i in range(211, 241):
            heading_acc[i] = 0.15
        for i in range(241, 271):
            heading_acc[i] = 0.05
        return heading_acc

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, manual=False):
        init_dd, init_dv, init_a, init_u = 0., 0., 0., 0.
        self.state = np.array([init_dd, init_dv, init_a, init_dd, init_dv, init_a, init_dd, init_dv, init_a, init_u])
        self.steps_beyond_done = None
        self.steps = 0
        if manual:
            self.head_acc = self.head_acc_list[self.steps]
        else:
            self.head_acc = np.clip(np.random.normal(0, 0.05), -1.5, 0.6)
        return self.state

    def stepPhysics(self, action, manual=False):
        if manual:
            self.head_acc = self.head_acc_list[self.steps]
        else:
            if self.steps % self.window == 0:
                self.head_acc = np.clip(np.random.normal(0, 0.05), -1.5, 0.6)
        self.state[6] = self.state[3]
        self.state[7] = self.state[4]
        self.state[8] = self.state[5]
        self.state[9] = action
        self.state[3] = self.state[0]
        self.state[4] = self.state[1]
        self.state[5] = self.state[2]
        self.state[0] += self.step_T * (self.state[1] - self.tau_h * self.state[2])
        self.state[1] += self.step_T * (-self.state[2] + self.head_acc)
        self.state[2] += self.step_T * (self.KL * action - self.state[2]) / self.TL

    def step(self, d_action, manual=False):
        # state: dd_t, dv_t, a_t, dd_t1, dv_t1, a_t1, dd_t2, dv_t2, a_t2, u_t1
        d_action = np.clip(d_action, self.action_space.low, self.action_space.high)
        self.steps += 1
        action = self.state[9] + d_action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.stepPhysics(action, manual)
        dead = bool(np.abs(self.state[0]) > 5. or np.abs(self.state[1]) > 1.)
        done = bool(dead or self.steps >= self.max_episode_steps)
        L_TE = 0.06 * self.state[0] ** 2 + 0.1 * self.state[1] ** 2
        L_FC = 1.0 * action ** 2 + 0.1 * d_action ** 2
        L_DC = 0.5 * (0.02 * self.state[0] + 0.25 * self.state[1] - self.state[2]) ** 2
        if not done:
            cost = L_TE + L_FC + L_DC
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            cost = L_TE + L_FC + L_DC
            if dead:
                cost += 1000
        else:
            gym.logger.warn("""
                You are calling 'step()' even though this environment has already returned
                done = True. You should always call 'reset()' once you receive 'done = True'
                Any further steps are undefined behavior.
                    """)
            cost = 0
        return self.state.copy(), cost[0], done, {'head_acc': self.head_acc, 'dead': dead}

    def render(self, mode='human'):
        pass
