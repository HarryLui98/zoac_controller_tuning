import sys
import time
from algo import *
from datetime import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '-env', type=str, default='PATHTRACK')
    parser.add_argument('--shift', '-sh', type=float, default=0.)
    parser.add_argument('--seed', '-se', type=int, default=300)
    parser.add_argument('--max_iter', type=int, default=501)
    parser.add_argument('--n_workers', '-n', type=int, default=10)
    parser.add_argument('--N_step_return', '-N', type=int, default=20)
    parser.add_argument('--train_freq', '-tf', type=int, default=5)
    parser.add_argument('--topn_directions', '-topn', type=float, default=1.0)
    parser.add_argument('--critic_update_epoch', '-ck', type=int, default=10)
    parser.add_argument('--critic_minibatch_size', '-cb', type=int, default=128)
    parser.add_argument('--gae_lambda', '-lam', type=float, default=0.95)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--discount_factor', '-gamma', type=float, default=0.99)
    parser.add_argument('--actor_type', '-at', type=str, default='ModelPredictiveController')
    parser.add_argument('--actor_network_hidden_size', '-ah', type=int, default=64)
    parser.add_argument('--critic_network_hidden_size', '-ch', type=int, default=256)
    parser.add_argument('--actor_step_size', '-as', type=float, default=5e-2)
    parser.add_argument('--critic_step_size', '-cs', type=float, default=5e-4)
    parser.add_argument('--para_noise_std', '-pstd', type=float, default=0.1) 
    parser.add_argument('--action_noise_std', '-astd', type=float, default=0.)
    parser.add_argument('--dir_path', '-dir', type=str, default='./test/' + str(datetime.now()))
    parser.add_argument('--episode_type', '-etype', type=str, default='Full') 
    args = parser.parse_args()
    algo_params = vars(args)

    ray.init(local_mode=False)
    run_algo(algo_params)
    ray.shutdown()
