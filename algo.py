import numpy as np
import torch
import argparse
import os
import time
import parser
import ray
import gym
import copy
import torch.nn.functional as F

from models import *
from utils import *
from workers import *
from recording import *
from optimizers import *
from obs_filter import *
from storage import *
from tensorboardX import SummaryWriter
from new_env.pyth_tracking_env import *
from new_env.pyth_tracking_mpc import *
from new_env.pyth_acc_env import *
from new_env.pyth_acc_pid import *

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


class ZOACLearner(object):
    def __init__(self, algo_params):
        self.critic_info = None
        self.actor_info = None
        self.start = None
        logdir = algo_params['dir_path']
        env_name = algo_params['env_name']
        seed = algo_params['seed']
        configure_output_dir(logdir)
        save_params(algo_params)
        self.logdir = logdir
        self.shift = algo_params['shift']
        if env_name == 'PATHTRACK':
            env = PathTrackingEnv(pre_horizon=25, max_episode_steps=200)
        elif env_name == 'ACC':
            env = ACCEnv(max_episode_steps=1000)
        env.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])
        self.total_steps = 0
        self.total_episodes = 0
        self.max_iter = algo_params['max_iter']
        self.N_step_return = algo_params['N_step_return']
        self.n_workers = algo_params['n_workers']
        self.actor_step_size = algo_params['actor_step_size']
        self.critic_step_size = algo_params['critic_step_size']
        self.para_noise_std = algo_params['para_noise_std']
        self.action_noise_std = algo_params['action_noise_std']
        self.avg_act_dist = 0
        self.train_freq = algo_params['train_freq']
        self.topn_directions = round(algo_params['topn_directions'] * self.n_workers * self.train_freq)
        self.eval_freq = algo_params['eval_freq']
        self.discount_factor = algo_params['discount_factor']
        self.gae_lambda = algo_params['gae_lambda']
        self.actor_type = algo_params['actor_type']
        if self.actor_type == "LinearActor":
            self.policy = LinearActor(self.state_dim, self.action_dim, self.max_action, seed)
        elif self.actor_type == "NeuralActor":
            self.policy = NeuralActor(self.state_dim, self.action_dim, algo_params['actor_network_hidden_size'],
                                      self.max_action, seed)
        elif self.actor_type == "ModelPredictiveController":
            self.policy = ModelPredictiveController(env, seed)
        elif self.actor_type == "PIDController":
            self.policy = PIDController(env, seed)
        else:
            raise NotImplementedError
        print("State dim: %i" % self.state_dim)
        print("Action dim: %i" % self.action_dim)
        print("Parameter dim: %i" % self.policy.para_num)
        self.policy_mean_optimizer = Adam(self.policy, lr=self.actor_step_size)
        self.policy_std_optimizer = Adam(self.policy, lr=self.actor_step_size)
        self.critic = Critic(self.state_dim, 1, algo_params['critic_network_hidden_size'], seed)
        deltas_id = create_shared_noise.remote(seed)
        self.deltas = SharedNoiseTable(ray.get(deltas_id), seed=seed)
        self.noise_dim = self.policy.para_num
        self.actor_batchsize = self.n_workers * self.train_freq
        self.actor_train_buffer = ActorBuffer(self.state_dim, self.actor_batchsize)
        self.critic_train_buffer = [CriticBuffer(self.state_dim, self.train_freq * self.N_step_return,
                                                 self.discount_factor, self.gae_lambda)
                                    for _ in range(self.n_workers)]
        self.sampler_set = [Sampler.remote(seed=seed + 3 * i, env_name=env_name, shift=self.shift, policy=self.policy,
                                           noise_table=deltas_id, N_step=self.N_step_return,
                                           para_noise_std=self.para_noise_std, action_noise_std=self.action_noise_std,
                                           gamma=self.discount_factor)
                            for i in range(self.n_workers)]
        self.evaluator = Evaluator(seed=seed + 5, env_name=env_name, policy=self.policy,
                                   num_rollouts=10, gamma=self.discount_factor, shift=self.shift)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_step_size,
                                                 weight_decay=0.)
        self.critic_epoch = algo_params['critic_update_epoch']
        self.critic_minibatch_size = algo_params['critic_minibatch_size']
        self.time_cost_rollouts = 0
        self.time_cost_actor_update = 0
        self.time_cost_critic_update = 0
        self.time_cost_total = 0
        self.writer = SummaryWriter(logdir)
        self.critic = self.critic.to(device)
        self.etype = algo_params['episode_type']
        self.policy.set_flat_sigma(self.para_noise_std * np.ones(self.policy.para_num))

    def fill_act_buffer(self, result):
        length = result['steps']
        states_torch = torch.FloatTensor(np.array(result['states'])).to(device)
        next_states_torch = torch.FloatTensor(np.array(result['next_states'])).to(device)
        not_deads = torch.ones(length)
        not_deads[-1] = result['not_dead']
        with torch.no_grad():
            values_pert = torch.FloatTensor(result['rewards']).to(device)
            values_pert += self.discount_factor * self.critic(next_states_torch).squeeze() * not_deads
            values_init = self.critic(states_torch).squeeze()
        deltas = values_pert - values_init
        deltas_to_go = 0.
        for j in range(length):
            deltas_to_go *= (self.gae_lambda * self.discount_factor)
            deltas_to_go += deltas[length - j - 1]
        self.actor_train_buffer.store(deltas_to_go, result['noise_idx'])

    def fill_cri_buffer(self, result, worker_num):
        length = result['steps']
        states = result['states']
        next_states = result['next_states']
        not_deads = torch.ones(result['steps'])
        not_deads[-1] = result['not_dead']
        rewards = result['rewards']
        states_torch = torch.FloatTensor(np.array(result['states'])).to(device)
        value_pred = self.critic(states_torch)
        for j in range(length):
            self.critic_train_buffer[worker_num].add(states[j], next_states[j],
                                                     rewards[j], not_deads[j], value_pred[j].item())

    def gen_experience(self, idx):
        results = ray.get(
            [worker.rollout_several_step.remote(self.N_step_return, self.shift) for worker in self.sampler_set])
        filter_incre_data = []
        for i in range(self.n_workers):
            result = results[i]
            self.total_steps += result['steps']
            # concat states to update observation filter
            filter_incre_data += result['states']
            # calculate advantage and fill in actor train buffer
            self.fill_act_buffer(result)
            self.fill_cri_buffer(result, i)
            # a trajectory is finished
            if self.etype == 'Truncated':
                if result['is_done'] is True or idx == self.train_freq - 1:
                    # initialize a new episode in the corresponding worker
                    ray.get(self.sampler_set[i].init_new_episode.remote())
            elif self.etype == 'Full':
                if result['is_done'] is True:
                    # initialize a new episode in the corresponding worker
                    ray.get(self.sampler_set[i].init_new_episode.remote())
                    self.total_episodes += 1
            if result['is_done'] is True or idx == self.train_freq - 1:
                with torch.no_grad():
                    # calculate rewards-to-go as target value
                    last_state = torch.FloatTensor(result['next_states'][-1]).to(device)
                    last_value = self.critic(last_state) * result['not_dead']
                    self.critic_train_buffer[i].finish_trajs(last_value.item())

    def update_critic(self):
        total_obses, total_targets = [], []
        for i in range(self.n_workers):
            states, targets = self.critic_train_buffer[i].get()
            total_obses.append(states)
            total_targets.append(targets)
        total_obses = torch.cat(total_obses)
        total_targets = torch.cat(total_targets)
        nbatch = total_targets.shape[0]
        inds = np.arange(nbatch)
        for i in range(self.critic_epoch):
            # Randomize the indexes
            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            for start in range(0, nbatch - self.critic_minibatch_size,
                               self.critic_minibatch_size):
                end = start + self.critic_minibatch_size
                if nbatch - end < self.critic_minibatch_size:
                    end = nbatch
                mbinds = inds[start:end]
                mb_obs = total_obses[mbinds]
                mb_targets = total_targets[mbinds]
                self.critic.train()
                mb_estvalue = self.critic(mb_obs).squeeze()
                critic_loss = F.mse_loss(mb_estvalue, mb_targets, reduction='mean')
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5, norm_type=2)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

    def update_actor(self):
        advs, noise_idxes = self.actor_train_buffer.get()
        advs_max_idx = np.argsort(np.abs(advs))
        top_advs = advs[advs_max_idx][-self.topn_directions:]
        top_noise_idxes = noise_idxes[advs_max_idx][-self.topn_directions:]
        top_adv_mean = np.mean(top_advs) * np.ones(self.topn_directions)
        top_adv_norm = (top_advs - top_adv_mean) / (np.std(top_advs) + 1e-8)
        top_noises = np.array([self.deltas.get(idx, self.policy.para_num) for idx in top_noise_idxes])
        p_grad_hat = np.dot(top_adv_norm, top_noises) / self.topn_directions
        new_mean_para, _ = self.policy_mean_optimizer.update_mean(p_grad_hat)  # minimize
        self.policy.set_flat_param(new_mean_para, after_map=True)

    def update_workers(self):
        para_id = ray.put(self.policy.get_flat_param(after_map=True))
        ray.get([worker.update_param.remote(para_id) for worker in self.sampler_set])

    def evaluate_policy(self, iter_num):
        self.evaluator.update_param(self.policy.get_flat_param(after_map=True))
        rewards, steps, errs = self.evaluator.eval_rollouts()
        log_tabular("Time", time.time() - self.start)
        log_tabular("TotalSteps", self.total_steps)
        log_tabular("TotalEpisodes", self.total_episodes)
        log_tabular("Iteration", iter_num + 1)
        log_tabular("AverageReward", np.mean(rewards))
        log_tabular("AverageModelErr", np.mean(errs))
        log_tabular("AverageEpsLength", np.mean(steps))
        log_tabular("StdRewards", np.std(rewards))
        log_tabular("MaxRewardRollout", np.max(rewards))
        log_tabular("MinRewardRollout", np.min(rewards))
        dump_tabular()
        self.writer.add_scalar('Evaluate/AverageReward', np.mean(rewards), self.total_steps)
        self.writer.add_scalar('Evaluate/AverageModelErr', np.mean(errs), self.total_steps)
        self.writer.add_scalar('Evaluate/AverageEpsLength', np.mean(steps), self.total_steps)
        self.writer.add_scalar("Evaluate/StdRewards", np.std(rewards), self.total_steps)

    def save_data(self, iter_num):
        policy_save_path = os.path.join(self.logdir, 'policy/')
        if not (os.path.exists(policy_save_path)):
            os.makedirs(policy_save_path)
        critic_save_path = os.path.join(self.logdir, 'critic/')
        if not (os.path.exists(critic_save_path)):
            os.makedirs(critic_save_path)
        value_func_path = os.path.join(critic_save_path, 'value_func_epoch_{}.pth'.format(iter_num + 1))
        torch.save(self.critic.state_dict(), value_func_path)
        if self.actor_type == "LinearActor":
            np.save(policy_save_path + 'policy_epoch_{}'.format(iter_num + 1), self.policy.get_flat_param())
        elif self.actor_type == "NeuralActor":
            policy_path = os.path.join(policy_save_path, 'policy_epoch_{}.pth'.format(iter_num + 1))
            torch.save(self.policy.state_dict(), policy_path)
        elif self.actor_type == "ModelPredictiveController":
            np.save(policy_save_path + 'policy_epoch_{}'.format(iter_num + 1),
                    self.policy.get_flat_param(after_map=False))
        elif self.actor_type == "PIDController":
            np.save(policy_save_path + 'policy_epoch_{}'.format(iter_num + 1),
                    self.policy.get_flat_param(after_map=False))

    def train(self):
        self.start = time.time()
        ray.wait([worker.init_new_episode.remote() for worker in self.sampler_set])
        for iter_num in range(self.max_iter):
            time1 = time.time()
            for i in range(self.n_workers):
                self.critic_train_buffer[i].reset()
            self.actor_train_buffer.reset()
            self.update_workers()
            # generate experience
            for i in range(self.train_freq):
                self.gen_experience(i)
            time2 = time.time()
            old_policy_para = self.policy.get_flat_param(after_map=True)
            old_critic_para = self.critic.get_flat_param()
            # policy evaluation
            self.update_critic()
            time3 = time.time()
            # policy improvement
            self.update_actor()
            time4 = time.time()
            new_policy_para = self.policy.get_flat_param(after_map=True)
            new_critic_para = self.critic.get_flat_param()
            actor_grad_norm = np.linalg.norm(new_policy_para - old_policy_para)
            critic_grad_norm = np.linalg.norm(new_critic_para - old_critic_para)
            current = time.time()
            self.time_cost_rollouts += time2 - time1
            self.time_cost_critic_update += time3 - time2
            self.time_cost_actor_update += time4 - time3
            self.time_cost_total = current - self.start
            self.writer.add_scalar('Time/TimeRollouts', self.time_cost_rollouts, self.total_steps)
            self.writer.add_scalar('Time/TimeCriticUpdate', self.time_cost_critic_update, self.total_steps)
            self.writer.add_scalar('Time/TimeActorUpdate', self.time_cost_actor_update, self.total_steps)
            self.writer.add_scalar('Time/TimeTotal', self.time_cost_total, self.total_steps)
            self.writer.add_scalar('Update/ActorGradNorm', actor_grad_norm, self.total_steps)
            self.writer.add_scalar('Update/CriticGradNorm', critic_grad_norm, self.total_steps)
            if iter_num % self.eval_freq == 0:
                self.evaluate_policy(iter_num)
                if self.total_steps >= int(5e5):
                    break
            if iter_num % (1 * self.eval_freq) == 0:
                self.save_data(iter_num)
                print("Mean")
                print(self.policy.get_flat_param(after_map=False))
                print("Std")
                print(self.policy.get_flat_sigma())
        self.writer.close()


def run_algo(algo_params):
    dir_path = algo_params['dir_path']
    if not (os.path.exists(dir_path)):
        os.makedirs(dir_path)
    learner = ZOACLearner(algo_params)
    learner.train()
