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
from new_env.pyth_pcc_env import *
from new_env.pyth_pcc_mpc import *
from new_env.pyth_tracking_env import *
from new_env.pyth_tracking_mpc import *
from new_env.pyth_acc_env import *
from new_env.pyth_acc_pid import *

@ray.remote
class Sampler(object):
    def __init__(self, seed, env_name, shift, policy, noise_table, gamma, N_step,
                 para_noise_std, action_noise_std, adaptive_std):
        # initialize OpenAI environment for each worker
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if env_name == 'PCC':
            self.env = PythPCCTruck(pre_horizon=25, max_episode_steps=2000)
        elif env_name == 'PATHTRACK':
            self.env = PathTrackingEnv(pre_horizon=25, max_episode_steps=200)
        elif env_name == 'ACC':
            self.env = ACCEnv(max_episode_steps=1000)
        self.env.seed(seed)
        self.deltas = SharedNoiseTable(noise_table, seed)
        self.policy = copy.deepcopy(policy)
        # self.mean_policy = copy.deepcopy(policy)
        self.adaptive_std = adaptive_std
        if adaptive_std:
            self.delta_std = para_noise_std * np.ones(self.policy.para_num)
        else:
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
        if self.adaptive_std:
            self.policy.set_flat_param(new_para[0], after_map=True)
            self.policy.set_flat_sigma(new_para[1])
            self.delta_std = new_para[1]
            self.flat_param = new_para[0]
        else:
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
        if isinstance(self.policy, MaskedNeuralActor):
            para_num = self.policy.eff_para_num
        else:
            para_num = self.policy.para_num
        step = 0
        states = []
        next_states = []
        pred_err = []
        rewards = []
        not_dead = False
        act_diff = np.zeros(self.action_dim)
        while step < n_step_return and (not self.is_done):
            action, pred_next_state = self.policy.get_action(self.current_state)
            # action_nominal = self.mean_policy.get_action(obs, False)
            # act_diff = np.power(action - action_nominal, 2)
            next_state, reward, is_done, info = self.env.step(action)
            state_err = next_state[:6] - pred_next_state.squeeze(-1)
            pred_err.append(0.5 * np.inner(state_err, state_err))
            reward -= shift
            reward += 50 * 0.5 * np.inner(state_err, state_err)
            self.steps += 1
            not_dead = not info['dead']
            states.append(self.current_state)
            next_states.append(next_state)
            rewards.append(reward)
            self.current_state = next_state
            self.is_done = is_done
            step += 1
        mean_pred_err = np.mean(np.array(pred_err))
        return {'states': states, 'next_states': next_states, 'rewards': rewards, 'pred_err': None,
                'not_dead': not_dead, 'noise_idx': noise_idx, 'steps': step, 'is_done': self.is_done,
                'action_diff': act_diff, 'para_num': para_num}


class Evaluator(object):
    def __init__(self, seed, env_name, policy, num_rollouts, gamma, shift=0.):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if env_name == 'PCC':
            self.env = PythPCCTruck(pre_horizon=25, max_episode_steps=2000)
        elif env_name == 'PATHTRACK':
            self.env = PathTrackingEnv(pre_horizon=25, max_episode_steps=200)
        elif env_name == 'ACC':
            self.env = ACCEnv(max_episode_steps=1000)
        self.env.seed(seed)
        self.policy = copy.deepcopy(policy)
        # self.policy.eval()
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
        pred_err = []
        real_state_values = []
        before_real_rewards = []
        after_real_rewards = []
        while not is_done:
            with torch.no_grad():
                action, pred_next_state = self.policy.get_action(state)
                next_state, reward, is_done, _ = self.env.step(action)
                if total_step < 200:
                    states.append(state)
                #     before_real_rewards.append(reward - self.shift)
                # else:
                #     after_real_rewards.append(reward - self.shift)
                # state_err = next_state[:6] - pred_next_state.squeeze(-1)
                # pred_err.append(0.5 * np.inner(state_err, state_err))
                total_reward += reward
                total_step += 1
                state = next_state
        # reward_to_go = 0
        # for i in range(len(after_real_rewards)):
        #     reward_to_go += after_real_rewards[-i - 1]
        #     reward_to_go *= self.gamma
        # for i in range(len(before_real_rewards)):
        #     reward_to_go += before_real_rewards[-i - 1]
        #     reward_to_go *= self.gamma
        #     real_state_values.append(reward_to_go)
        return total_step, total_reward, states, pred_err  # , real_state_values[::-1]

    def eval_rollouts(self):
        # self.policy.eval()
        rewards = []
        steps = []
        all_states = []
        pred_errs = []
        all_state_values = []
        for _ in range(self.num_rollouts):
            step, reward, states, pred_err = self.eval_rollout_once()
            rewards.append(reward)
            steps.append(step)
            all_states += states
            pred_errs += pred_err
            # all_state_values += state_values
        return np.array(rewards), np.array(all_states), np.array(steps), np.array(pred_errs)  # , np.array(all_state_values)
