import numpy as np
import argparse
import os
import time
import parser
import copy
import torch.nn.functional as F
import nevergrad as ng
from workers import *
from recording import *
from optimizers import *
from obs_filter import *
from storage import *
from tensorboardX import SummaryWriter
from new_env.pyth_pcc_env import *
from new_env.pyth_pcc_mpc import *
from new_env.pyth_tracking_env import *
from new_env.pyth_tracking_mpc import *
from new_env.pyth_acc_env import *
from new_env.pyth_acc_pid import *


class ControllerCostMinimizer(object):
    def __init__(self, seed, env_name, opt_name, budget, worker_num, logdir):
        random.seed(seed)
        np.random.seed(seed)
        if env_name == 'PCC':
            env = PythPCCTruck(pre_horizon=25, max_episode_steps=2000)
            self.opt_para = ng.p.Array(init=np.random.uniform(-1, 1, 12), lower=-1, upper=1)
        elif env_name == 'PATHTRACK':
            env = PathTrackingEnv(pre_horizon=25, max_episode_steps=200)
            self.opt_para = ng.p.Array(init=np.random.uniform(-1, 1, 19), lower=-1, upper=1)
        elif env_name == 'ACC':
            env = ACCEnv(max_episode_steps=1000)
            self.opt_para = ng.p.Array(init=np.random.uniform(-1, 1, 4), lower=-1, upper=1)
        env.seed(seed)
        self.policy = PIDController(env, seed)
        self.trainer = [Evaluator(seed=seed + 3 * i, env_name=env_name, policy=self.policy,
                                  num_rollouts=1, gamma=1.) for i in range(worker_num)]
        self.evaluator = Evaluator(seed=seed + 5, env_name=env_name, policy=self.policy,
                                   num_rollouts=10, gamma=1.)
        self.worker_num = worker_num
        self.budget = budget
        self.opt_name = opt_name
        if opt_name == 'CMA':
            self.optimizer = ng.optimizers.CMA(parametrization=self.opt_para, budget=budget, num_workers=worker_num)
        elif opt_name == 'PSO':
            self.optimizer = ng.optimizers.PSO(parametrization=self.opt_para, budget=budget, num_workers=worker_num)
        elif opt_name == 'BO':
            self.optimizer = ng.optimizers.BO(parametrization=self.opt_para, budget=budget, num_workers=1)
        self.logdir = logdir
        configure_output_dir(logdir)
        self.writer = SummaryWriter(logdir)
        self.total_steps = 0
        self.total_episodes = 0
        self.start = 0
        self.iter_num = 0

    def fitness(self, para, *args, **kwargs):
        thread_id = kwargs['thread_id']
        self.trainer[thread_id].update_param(para)
        step, cost, _, _ = self.trainer[thread_id].eval_rollout_once()
        self.total_steps += step
        self.total_episodes += 1
        return cost

    def evaluate(self, current_para, *args, **kwargs):
        self.policy.set_flat_param(current_para, after_map=True)
        self.evaluator.update_param(self.policy.get_flat_param(after_map=True))
        rewards, states, steps, errs = self.evaluator.eval_rollouts()
        log_tabular("Time", time.time() - self.start)
        log_tabular("TotalSteps", self.total_steps)
        log_tabular("TotalEpisodes", self.total_episodes)
        log_tabular("AverageReward", np.mean(rewards))
        # log_tabular("AverageModelErr", np.mean(errs))
        log_tabular("AverageEpsLength", np.mean(steps))
        log_tabular("StdRewards", np.std(rewards))
        log_tabular("MaxRewardRollout", np.max(rewards))
        log_tabular("MinRewardRollout", np.min(rewards))
        dump_tabular()
        self.writer.add_scalar('Evaluate/AverageReward', np.mean(rewards), self.total_steps)
        # self.writer.add_scalar('Evaluate/AverageModelErr', np.mean(errs), self.total_steps)
        self.writer.add_scalar('Evaluate/AverageEpsLength', np.mean(steps), self.total_steps)
        self.writer.add_scalar("Evaluate/StdRewards", np.std(rewards), self.total_steps)
        K, Kp, Ki, Kd = self.policy.get_flat_param(after_map=False).tolist()
        self.writer.add_scalar('Parameter/K', K, self.total_steps)
        self.writer.add_scalar('Parameter/Kp', Kp, self.total_steps)
        self.writer.add_scalar('Parameter/Ki', Ki, self.total_steps)
        self.writer.add_scalar('Parameter/Kd', Kd, self.total_steps)
        # Cf, Cr, a, b, m, Iz, dy_w, dp_w, yaw_w, str_w, acc_w = self.policy.get_flat_param(after_map=True).tolist()
        # Cf0, Cr0, a0, b0, m0, Iz0, dy_w0, dp_w0, yaw_w0, str_w0, acc_w0 = self.policy.tunable_para_expert_mapped.tolist()
        # self.writer.add_scalar('ModelParameter/Cf', Cf - Cf0, self.iter_num + 1)
        # self.writer.add_scalar('ModelParameter/Cr', Cr - Cr0, self.iter_num + 1)
        # self.writer.add_scalar('ModelParameter/a', a - a0, self.iter_num + 1)
        # self.writer.add_scalar('ModelParameter/b', b - b0, self.iter_num + 1)
        # self.writer.add_scalar('ModelParameter/m', m - m0, self.iter_num + 1)
        # self.writer.add_scalar('ModelParameter/Iz', Iz - Iz0, self.iter_num + 1)
        # self.writer.add_scalar('CostParameter/dy_w', dy_w - dy_w0, self.iter_num + 1)
        # self.writer.add_scalar('CostParameter/dp_w', dp_w - dp_w0, self.iter_num + 1)
        # self.writer.add_scalar('CostParameter/yaw_w', yaw_w - yaw_w0, self.iter_num + 1)
        # self.writer.add_scalar('CostParameter/str_w', str_w - str_w0, self.iter_num + 1)
        # self.writer.add_scalar('CostParameter/acc_w', acc_w - acc_w0, self.iter_num + 1)
        policy_save_path = os.path.join(self.logdir, 'policy/')
        if not (os.path.exists(policy_save_path)):
            os.makedirs(policy_save_path)
        np.save(policy_save_path + 'policy_epoch_{}'.format(self.total_episodes),
                self.policy.get_flat_param(after_map=False))

    def optimize(self):
        if self.opt_name == 'CMA' or self.opt_name == 'PSO':
            self.start = time.time()
            for u in range(self.budget // self.worker_num):
                x = []
                for _ in range(self.worker_num):
                    x.append(self.optimizer.ask())
                y = []
                for i in range(self.worker_num):
                    y.append(self.fitness(*x[i].args, **x[i].kwargs, thread_id=i))
                for thread_id in range(self.worker_num):
                    self.optimizer.tell(x[thread_id], y[thread_id])
                recommendation = self.optimizer.recommend()
                self.iter_num = u + 1
                print(self.iter_num)
                if self.total_episodes % 10 == 0 or self.iter_num == 1:
                    self.evaluate(*recommendation.args, **recommendation.kwargs)
        if self.opt_name == 'BO':
            self.start = time.time()
            for u in range(self.budget):
                x = self.optimizer.ask()
                y = self.fitness(*x.args, **x.kwargs, thread_id=0)
                self.optimizer.tell(x, y)
                recommendation = self.optimizer.recommend()
                self.iter_num = u + 1
                print(self.iter_num)
                if self.total_episodes % 10 == 0 or self.iter_num == 1:
                    self.evaluate(*recommendation.args, **recommendation.kwargs)


for seed in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    learner = ControllerCostMinimizer(seed=seed, env_name='ACC', opt_name='CMA', budget=600, worker_num=10,
                                      logdir='./acc/baseline/cma-600/' + str(seed))
    learner.optimize()
    learner = ControllerCostMinimizer(seed=seed, env_name='ACC', opt_name='BO', budget=600, worker_num=1,
                                      logdir='./acc/baseline/bo-600/' + str(seed))
    learner.optimize()
