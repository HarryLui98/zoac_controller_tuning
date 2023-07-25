# Performance-Driven Controller Tuning via Derivative-Free Reinforcement Learning

Implementation of the paper *Performance-Driven Controller Tuning via Derivative-Free Reinforcement Learning* presented at 21st 2022 IEEE 61st Conference on Decision and Control (CDC). The paper can be downloaded either from [IEEE](https://ieeexplore.ieee.org/abstract/document/9993026) or [arXiv](https://arxiv.org/abs/2209.04854).

## Requirements

```
pip install -r requirements.txt
```

## Tuning Controllers

We provide two tuning examples in this repository:

(1) PID controller for adaptive cruise control problem:

```
python3 main.py -env "ACC" -at "PIDController"
```

(2) MPC controller for path tracking problem, where tunable parameters include uncalibrated model parameters and cost weights:

```
python3 main.py -env "PATHTRACK" -at "ModelPredictiveController"
```

For more details, please refer to our [conference paper](https://ieeexplore.ieee.org/abstract/document/9993026).

We leverage [NeverGrad](https://github.com/facebookresearch/nevergrad) as the baseline of derivative-free optimization (see ```run_baseline.py```).

## Acknowledgements

Our code take reference from [ARS](https://github.com/modestyachts/ARS) and [OpenAI-ES](https://github.com/openai/evolution-strategies-starter).


## Citation

```
@inproceedings{lei2022performance,
  title={Performance-Driven Controller Tuning via Derivative-Free Reinforcement Learning},
  author={Lei, Yuheng and Chen, Jianyu and Li, Shengbo Eben and Zheng, Sifa},
  booktitle={2022 IEEE 61st Conference on Decision and Control (CDC)},
  pages={115--122},
  year={2022},
  organization={IEEE}
}
```