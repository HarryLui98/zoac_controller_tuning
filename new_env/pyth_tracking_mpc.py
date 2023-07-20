import random
import math
import numpy as np
import casadi


class ModelPredictiveController(object):
    def __init__(self, env, seed):
        random.seed(seed)
        np.random.seed(seed)
        self.Np = env.Np
        self.step_T = env.step_T
        self.action_space = env.action_space
        self.u_target = env.u_target
        self.path = env.path
        # Tunable parameters (19):
        # model - Cf, Cr, a, b, m, Iz
        # stage cost - dx_w, dy_w, dphi_w, v_w, yaw_w, str_w, acc_w (du_w is set as 0.01) - log space
        # terminal cost - dx_w, dy_w, dphi_w, v_w, yaw_w, du_w - log space
        self.tunable_para_high = np.array([-8e4, -8e4, 2.2, 2.2, 2000, 2000,
                                           1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2,
                                           1e2, 1e2, 1e2, 1e2, 1e2, 1e2])
        self.tunable_para_low = np.array([-16e4, -16e4, 0.8, 0.8, 1000, 1000,
                                          1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6,
                                          1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
        self.x0 = ([0, 0, 0, 0, 0, 0, 0, 0] * (self.Np + 1))
        self.x0.pop(-1)
        self.x0.pop(-1)
        self.ref_p = None
        self.lin_para_gain = 0.5 * (self.tunable_para_high[:6] - self.tunable_para_low[:6])
        self.lin_para_bias = 0.5 * (self.tunable_para_high[:6] + self.tunable_para_low[:6])
        self.log_para_gain = 0.5 * (np.log10(self.tunable_para_high[6:]) - np.log10(self.tunable_para_low[6:]))
        self.log_para_bias = 0.5 * (np.log10(self.tunable_para_high[6:]) + np.log10(self.tunable_para_low[6:]))
        self.tunable_para_mapped = np.random.uniform(-1, 1, 19)
        self.tunable_para_unmapped = self.tunable_para_transform(self.tunable_para_mapped, after_map=True)
        self.tunable_para_sigma = 0.01 * np.ones(19)
        self.tunable_para_expert = np.array([-128915.5, -85943.6, 1.06, 1.85, 1412, 1536.7,
                                             1e-6, 0.04, 0.1, 1e-6, 0.02, 5, 0.05,
                                             1e-6, 0.04, 0.1, 1e-6, 0.02, 0.01])
        self.tunable_para_expert_mapped = self.tunable_para_transform(self.tunable_para_expert, after_map=False)
        self.para_num = 19
        self.model_para_num = 6
        self.gamma = 0.99

    def tunable_para_transform(self, para_in, after_map):
        if after_map:
            lin_para_mapped = para_in[:6]
            log_para_mapped = para_in[6:]
            lin_para_unmapped = self.lin_para_gain * lin_para_mapped + self.lin_para_bias
            log_para_unmapped = np.power(10, self.log_para_gain * log_para_mapped + self.log_para_bias)
            para_out = np.concatenate((lin_para_unmapped, log_para_unmapped))
        else:
            lin_para_unmapped = para_in[:6]
            log_para_unmapped = para_in[6:]
            lin_para_mapped = (lin_para_unmapped - self.lin_para_bias) / self.lin_para_gain
            log_para_mapped = (np.log10(log_para_unmapped) - self.log_para_bias) / self.log_para_gain
            para_out = np.concatenate((lin_para_mapped, log_para_mapped))
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

    def step_forward(self, state, action):
        x, v_x, v_y, r, y, phi = state[0], state[1], state[2], state[3], state[4], state[5]
        steer, a_x = action[0], action[1]
        C_f, C_r, a, b, mass, I_z = self.tunable_para_unmapped[:6].tolist()
        tau = self.step_T
        next_state = [x + tau * (v_x * casadi.cos(phi) - v_y * casadi.sin(phi)),
                      v_x + tau * a_x,
                      (mass * v_y * v_x + tau * (a * C_f - b * C_r) * r -
                       tau * C_f * steer * v_x - tau * mass * (v_x ** 2) * r)
                      / (mass * v_x - tau * (C_f + C_r)),
                      (I_z * r * v_x + tau * (a * C_f - b * C_r) * v_y
                       - tau * a * C_f * steer * v_x) /
                      (I_z * v_x - tau * ((a ** 2) * C_f + (b ** 2) * C_r)),
                      y + tau * (v_x * casadi.sin(phi) + v_y * casadi.cos(phi)),
                      phi + tau * r]
        return next_state

    def get_action(self, initial_state):
        self.ref_p = []
        x = casadi.SX.sym('x', 6)
        u = casadi.SX.sym('u', 2)
        # Create empty NLP
        w = []
        lbw = []
        ubw = []
        lbg = []
        ubg = []
        ref_list = []
        G = []
        J = 0
        Xk = casadi.MX.sym('X0', 6)
        w += [Xk]
        lbw += [initial_state[0], initial_state[1], initial_state[2], initial_state[3], initial_state[4],
                initial_state[5]]
        ubw += [initial_state[0], initial_state[1], initial_state[2], initial_state[3], initial_state[4],
                initial_state[5]]
        for k in range(1, self.Np + 1):
            f = casadi.vertcat(*self.step_forward(x, u))
            F = casadi.Function("F", [x, u], [f])
            Uname = 'U' + str(k - 1)
            Uk = casadi.MX.sym(Uname, 2)
            w += [Uk]
            lbw += [self.action_space.low[0], self.action_space.low[1]]
            ubw += [self.action_space.high[0], self.action_space.high[1]]
            Fk = F(Xk, Uk)
            Xname = 'X' + str(k)
            Xk = casadi.MX.sym(Xname, 6)
            w += [Xk]
            ubw += [casadi.inf, casadi.inf, casadi.inf, casadi.inf, casadi.inf, casadi.inf]
            lbw += [-casadi.inf, -casadi.inf, -casadi.inf, -casadi.inf, -casadi.inf, -casadi.inf]
            # dynamic_state: x, u, v, yaw, y, phi
            G += [Fk - Xk]
            ubg += [0., 0., 0., 0., 0., 0.]
            lbg += [0., 0., 0., 0., 0., 0.]
            REFname = 'REF' + str(k)
            REFk = casadi.MX.sym(REFname, 3)
            ref_list += [REFk]
            self.ref_p += [initial_state[6 + (k - 1) * 3], initial_state[6 + (k - 1) * 3 + 1],
                           initial_state[6 + (k - 1) * 3 + 2]]
            if k < self.Np:
                ref_cost = 0.01 * casadi.power(w[k * 2][1] - self.u_target, 2)  # u
                ref_cost += self.tunable_para_unmapped[6] * casadi.power(w[k * 2][0] - ref_list[k - 1][0], 2)  # x
                ref_cost += self.tunable_para_unmapped[7] * casadi.power(w[k * 2][4] - ref_list[k - 1][1], 2)  # y
                ref_cost += self.tunable_para_unmapped[8] * casadi.power(w[k * 2][5] - ref_list[k - 1][2], 2)  # phi
                ref_cost += self.tunable_para_unmapped[9] * casadi.power(w[k * 2][2], 2)  # v
                ref_cost += self.tunable_para_unmapped[10] * casadi.power(w[k * 2][3], 2)  # yaw
                ref_cost *= casadi.power(self.gamma, k)
            else:
                ref_cost = self.tunable_para_unmapped[16] * casadi.power(w[k * 2][1] - self.u_target, 2)  # u
                ref_cost += self.tunable_para_unmapped[11] * casadi.power(w[k * 2][0] - ref_list[k - 1][0], 2)  # x
                ref_cost += self.tunable_para_unmapped[12] * casadi.power(w[k * 2][4] - ref_list[k - 1][1], 2)  # y
                ref_cost += self.tunable_para_unmapped[13] * casadi.power(w[k * 2][5] - ref_list[k - 1][2], 2)  # phi
                ref_cost += self.tunable_para_unmapped[14] * casadi.power(w[k * 2][2], 2)  # v
                ref_cost += self.tunable_para_unmapped[15] * casadi.power(w[k * 2][3], 2)  # yaw
                ref_cost *= casadi.power(self.gamma, k)
            act_cost = self.tunable_para_unmapped[11] * casadi.power(w[k * 2 - 1][0], 2)  # steer
            act_cost += self.tunable_para_unmapped[12] * casadi.power(w[k * 2 - 1][1], 2)  # ax
            act_cost *= casadi.power(self.gamma, k-1)
            J += (ref_cost + act_cost)
        nlp = dict(f=J, g=casadi.vertcat(*G), x=casadi.vertcat(*w), p=casadi.vertcat(*ref_list))
        S = casadi.nlpsol('S', 'ipopt', nlp,
                          {'ipopt.max_iter': 200, 'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0})
        r = S(lbx=lbw, ubx=ubw, x0=self.x0, lbg=lbg, ubg=ubg, p=self.ref_p)
        X = np.array(r['x']).tolist()
        action = np.array([X[6][0], X[7][0]])
        self.x0 = casadi.DM(
            X[8:] + X[-8] + X[-7] + X[-6] + X[-5] + X[-4] + X[-3] + X[-2] + X[-1])  # for faster optimization
        return action, np.array(X[8:14])
