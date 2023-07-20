import sys
import time
from algo import *
from datetime import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '-env', type=str, default='ACC')
    parser.add_argument('--shift', '-sh', type=float, default=0.)
    parser.add_argument('--seed', '-se', type=int, default=100)
    parser.add_argument('--max_iter', type=int, default=101)
    parser.add_argument('--n_workers', '-n', type=int, default=10)  # 1 2 4 8 16
    parser.add_argument('--N_step_return', '-N', type=int, default=20)  # 10 20 40 60
    parser.add_argument('--train_freq', '-tf', type=int, default=5)  # 16 32 64
    parser.add_argument('--topn_directions', '-topn', type=float, default=1.0)
    parser.add_argument('--critic_update_epoch', '-ck', type=int, default=8)  # 2 3 4
    parser.add_argument('--critic_minibatch_size', '-cb', type=int, default=64)
    parser.add_argument('--gae_lambda', '-lam', type=float, default=0.95)  # 0.8 0.9 0.95
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--discount_factor', '-gamma', type=float, default=0.99)
    parser.add_argument('--actor_type', '-at', type=str, default='ModelPredictiveController')  # ToeplitzLinearActor, LinearActor, NeuralActor
    parser.add_argument('--actor_network_hidden_size', '-ah', type=int, default=64)
    parser.add_argument('--critic_network_hidden_size', '-ch', type=int, default=256)
    parser.add_argument('--actor_step_size', '-as', type=float, default=5e-3)  # 5e-3 1e-2 2e-2
    parser.add_argument('--critic_step_size', '-cs', type=float, default=5e-4)  # 1e-4 3e-4 5e-4
    parser.add_argument('--para_noise_std', '-pstd', type=float, default=0.04)  # 0.01 0.02 0.04 0.08
    parser.add_argument('--action_noise_std', '-astd', type=float, default=0.)
    parser.add_argument('--adaptive_noise_std', '-adpt', type=int, default=0)
    parser.add_argument('--perf_model_coeff', '-pmc', type=float, default=0.)
    parser.add_argument('--dir_path', '-dir', type=str, default='./test/1')
    parser.add_argument('--episode_type', '-etype', type=str, default='Full')  # Truncated, Full
    args = parser.parse_args()
    algo_params = vars(args)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    ray.init(local_mode=False)
    run_algo(algo_params)
    ray.shutdown()
