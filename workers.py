import time
import os
import numpy as np
import gym
import ray
import copy

from utils import *
from models import *
from recording import *
from obs_filter import *
from new_env.pyth_tracking_env import *
from new_env.pyth_tracking_mpc import *
from new_env.pyth_acc_env import *
from new_env.pyth_acc_pid import *

@ray.remote
class Sampler(object):
    def __init__(self, seed, env_name, shift, policy, noise_table, gamma, N_step,
                 para_noise_std, action_noise_std):
        # initialize OpenAI environment for each worker
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env_name = env_name
        if env_name == 'PATHTRACK':
            self.env = PathTrackingEnv(pre_horizon=25, max_episode_steps=200)
        elif env_name == 'ACC':
            self.env = ACCEnv(max_episode_steps=1000)
        self.env.seed(seed)
        self.deltas = SharedNoiseTable(noise_table, seed)
        self.policy = copy.deepcopy(policy)
        self.delta_std = para_noise_std
        self.action_std = action_noise_std
        self.current_state = None
        self.is_done = False
        self.steps = 0
        self.flat_param = self.policy.get_flat_param(after_map=True)
        self.shift = shift
        self.gamma = gamma
        self.N_step = N_step
        self.action_dim = self.env.action_space.shape[0]

    def update_param(self, shared_param):
        new_para = copy.deepcopy(shared_param)
        self.policy.set_flat_param(new_para, after_map=True)
        self.flat_param = new_para

    def init_new_episode(self):
        self.current_state = self.env.reset()
        self.is_done = False
        self.steps = 0

    def rollout_several_step(self, n_step_return=1, shift=0.):
        noise_idx, direction = self.deltas.get_delta(self.policy.para_num)
        perturbed_param = self.flat_param + direction * self.delta_std
        # rollout n steps using perturbed policy
        self.policy.set_flat_param(perturbed_param, after_map=True)
        step = 0
        states = []
        next_states = []
        pred_errs = []
        rewards = []
        not_dead = False
        while step < n_step_return and (not self.is_done):
            action, pred_next_state = self.policy.get_action(self.current_state)
            next_state, reward, is_done, info = self.env.step(action)
            if self.env_name == 'PATHTRACK':
                state_err = next_state[:6] - pred_next_state.squeeze(-1)
                pred_errs.append(0.5 * np.inner(state_err, state_err))
                # reward += 50 * 0.5 * np.inner(state_err, state_err)
            elif self.env_name == 'ACC':
                pred_errs.append(0)
            reward -= shift
            self.steps += 1
            not_dead = not info['dead']
            states.append(self.current_state)
            next_states.append(next_state)
            rewards.append(reward)
            self.current_state = next_state
            self.is_done = is_done
            step += 1
        return {'states': states, 'next_states': next_states, 'rewards': rewards, 'pred_err': pred_errs,
                'not_dead': not_dead, 'noise_idx': noise_idx, 'steps': step, 'is_done': self.is_done
                }


class Evaluator(object):
    def __init__(self, seed, env_name, policy, num_rollouts, gamma, shift=0.):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env_name = env_name
        if env_name == 'PATHTRACK':
            self.env = PathTrackingEnv(pre_horizon=25, max_episode_steps=200)
        elif env_name == 'ACC':
            self.env = ACCEnv(max_episode_steps=1000)
        self.env.seed(seed)
        self.policy = copy.deepcopy(policy)
        self.num_rollouts = num_rollouts
        self.gamma = gamma
        self.shift = shift

    def update_param(self, shared_param):
        new_para = shared_param.copy()
        self.policy.set_flat_param(new_para, after_map=True)

    def eval_rollout_once(self):
        # self.policy.eval()
        total_reward = 0
        total_step = 0
        state, is_done = self.env.reset(), False
        states = []
        pred_errs = []
        while not is_done:
            with torch.no_grad():
                action, pred_next_state = self.policy.get_action(state)
                next_state, reward, is_done, _ = self.env.step(action)
                if self.env_name == 'PATHTRACK':
                    state_err = next_state[:6] - pred_next_state.squeeze(-1)
                    pred_errs.append(0.5 * np.inner(state_err, state_err))
                total_reward += reward
                total_step += 1
                state = next_state
        return total_step, total_reward, states, pred_errs

    def eval_rollouts(self):
        # self.policy.eval()
        rewards = []
        steps = []
        all_pred_errs = []
        for _ in range(self.num_rollouts):
            step, reward, states, pred_errs = self.eval_rollout_once()
            rewards.append(reward)
            steps.append(step)
            all_pred_errs += pred_errs
        return np.array(rewards), np.array(steps), np.array(pred_errs)
