import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy.signal import savgol_filter
import os
import pandas as pd
from gym.wrappers.time_limit import TimeLimit


class ReferencePath(object):
    def __init__(self):
        self.curve_list = [(7.5, 200, 0.), (2.5, 300., 0.), (-5., 400., 0.)]
        self.period = 1200.

    def compute_path_y(self, x):
        y = 0
        for curve in self.curve_list:
            magnitude, T, shift = curve
            y += magnitude * np.sin((x - shift) * 2 * np.pi / T)
        return y

    def compute_path_phi(self, x):
        deriv = 0
        for curve in self.curve_list:
            magnitude, T, shift = curve
            deriv += magnitude * 2 * np.pi / T * np.cos((x - shift) * 2 * np.pi / T)
        return np.arctan(deriv)

    def compute_delta_y(self, x, y):
        y_ref = self.compute_path_y(x)
        return y - y_ref

    def compute_delta_phi(self, x, phi):
        phi_ref = self.compute_path_phi(x)
        delta_phi = phi - phi_ref
        if delta_phi > math.pi:
            delta_phi -= 2 * math.pi
        if delta_phi <= -math.pi:
            delta_phi += 2 * math.pi
        return delta_phi

    def compute_y(self, x, delta_y):
        y_ref = self.compute_path_y(x)
        return delta_y + y_ref

    def compute_phi(self, x, delta_phi):
        phi_ref = self.compute_path_phi(x)
        phi = delta_phi + phi_ref
        if phi > math.pi:
            phi -= 2 * math.pi
        if phi <= -math.pi:
            phi += 2 * math.pi
        return phi


class PathTrackingEnv(gym.Env):
    def __init__(self, **kwargs):
        # ref: y1, phi1, ..., yn, phin
        # dynamic_state: x, u, v, yaw, y, phi
        self.vehicle_params = dict(C_f=-128915.5,  # front wheel cornering stiffness [N/rad]
                                   C_r=-85943.6,  # rear wheel cornering stiffness [N/rad]
                                   a=1.06,  # distance from CG to front axle [m]
                                   b=1.85,  # distance from CG to rear axle [m]
                                   mass=1412.,  # mass [kg]
                                   I_z=1536.7,  # Polar moment of inertia at CG [kg*m^2]
                                   g=9.81  # acceleration of gravity [m/s^2]
                                   )
        a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
                        self.vehicle_params['mass'], self.vehicle_params['g']
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        self.vehicle_params.update(dict(F_zf=F_zf,
                                        F_zr=F_zr))
        self.u_target = 18  # [m/s]
        self.Np = kwargs['pre_horizon']
        self.max_episode_steps = kwargs['max_episode_steps']
        self.path = ReferencePath()
        self.state = None
        self.dynamic_state = None
        self.simulation_time = 0
        self.ref_points = None
        self.action = None
        self.dynamic_T = 0.1
        self.step_T = 0.1
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * (6 + int(self.Np) * 3)),
            high=np.array([np.inf] * (6 + int(self.Np) * 3)),
            dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1.2 * math.pi / 9, -3]),
                                           high=np.array([1.2 * math.pi / 9, 3]),
                                           dtype=np.float32)
        self.seed()
        self.steps = 0
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, manual=False):
        if manual:
            init_x = 8
            init_delta_y = 0.25
            init_y = self.path.compute_y(init_x, init_delta_y)
            init_delta_phi = 0.04
            init_phi = self.path.compute_phi(init_x, init_delta_phi)
            init_u = 17.6
            beta = 0.03
            init_v = init_u * np.tan(beta)
            init_yaw = 0.05
        else:
            init_x = np.random.uniform(0, 100)
            init_delta_y = np.clip(np.random.normal(0, 0.2), -0.5, 0.5)
            init_y = self.path.compute_y(init_x, init_delta_y)
            init_delta_phi = np.clip(np.random.normal(0, 0.05), -0.1, 0.1)
            init_phi = self.path.compute_phi(init_x, init_delta_phi)
            init_u = np.random.uniform(17.5, 18.5)
            beta = np.clip(np.random.normal(0, 0.05), -0.1, 0.1)
            init_v = init_u * np.tan(beta)
            init_yaw = np.clip(np.random.normal(0, 0.05), -0.1, 0.1)
        self.dynamic_state = np.array([init_x, init_u, init_v, init_yaw, init_y, init_phi])
        self.ref_points = self.compute_ref_points(self.dynamic_state[0])
        self.steps_beyond_done = None
        self.steps = 0
        return np.concatenate((self.dynamic_state, self.ref_points))

    def stepPhysics(self, action, repeat=1):
        for _ in range(repeat):
            x, v_x, v_y, r, y, phi = self.dynamic_state.tolist()
            steer, a_x = action.tolist()
            C_f = self.vehicle_params['C_f']
            C_r = self.vehicle_params['C_r']
            a = self.vehicle_params['a']
            b = self.vehicle_params['b']
            mass = self.vehicle_params['mass']
            I_z = self.vehicle_params['I_z']
            tau = self.dynamic_T
            self.dynamic_state = np.array([x + tau * (v_x * math.cos(phi) - v_y * math.sin(phi)),
                                           v_x + tau * a_x,
                                           (mass * v_y * v_x + tau * (a * C_f - b * C_r) * r -
                                            tau * C_f * steer * v_x - tau * mass * (v_x ** 2) * r)
                                           / (mass * v_x - tau * (C_f + C_r)),
                                           (I_z * r * v_x + tau * (a * C_f - b * C_r) * v_y
                                            - tau * a * C_f * steer * v_x) /
                                           (I_z * v_x - tau * ((a ** 2) * C_f + (b ** 2) * C_r)),
                                           y + tau * (v_x * math.sin(phi) + v_y * math.cos(phi)),
                                           phi + tau * r])

    def step(self, action):
        # ref: y1, phi1, ..., yn, phin
        # dynamic_state: x, u, v, yaw, y, phi
        # action: steer, a_x
        # obs: x, delta_u, v, yaw, delta_y, delta_phi, y1-y0, phi1-phi0, ..., yn-y0, phin-phi0
        self.steps += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)
        env_step = int(self.step_T / self.dynamic_T)
        self.stepPhysics(action, repeat=env_step)
        xs, v_xs, v_ys, yaw_s, ys, phis = self.dynamic_state.tolist()
        delta_ys = self.path.compute_delta_y(xs, ys)
        delta_phis = self.path.compute_delta_phi(xs, phis)
        delta_vxs = v_xs - self.u_target
        self.ref_points = self.compute_ref_points(xs)
        dead = bool(np.abs(delta_ys) > 4 or np.abs(delta_phis) > np.pi / 4. or np.abs(delta_vxs) > 4.)
        done = bool(dead or self.steps >= self.max_episode_steps)
        obs = np.concatenate((self.dynamic_state, self.ref_points))
        if not done:
            cost = 0.01 * delta_vxs ** 2 + 0.04 * delta_ys ** 2 + 0.1 * delta_phis ** 2 \
                + 0.02 * self.dynamic_state[3] ** 2 + 5 * action[0] ** 2 + 0.05 * action[1] ** 2
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            cost = 0.01 * delta_vxs ** 2 + 0.04 * delta_ys ** 2 + 0.1 * delta_phis ** 2 \
                + 0.02 * self.dynamic_state[3] ** 2 + 5 * action[0] ** 2 + 0.05 * action[1] ** 2
            if dead:
                cost += 10
        else:
            gym.logger.warn("""
                You are calling 'step()' even though this environment has already returned
                done = True. You should always call 'reset()' once you receive 'done = True'
                Any further steps are undefined behavior.
                    """)
            cost = 0
        return obs, 100 * cost, done, {'dead': dead}

    def render(self, mode='human'):
        pass

    def compute_ref_points(self, x0):
        ref_points = []
        y0 = self.path.compute_path_y(x0)
        phi0 = self.path.compute_path_phi(x0)
        x = x0
        for i in range(self.Np):
            phi = self.path.compute_path_phi(x)
            x += self.u_target * self.step_T * math.cos(phi)
            y = self.path.compute_path_y(x)
            ref_points.append(x)
            ref_points.append(y)
            phi = self.path.compute_path_phi(x)
            ref_points.append(phi)
        return np.array(ref_points)
