import numpy as np
from typing import Callable
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/hkhedr/Haitham/projects/TLLnet/")
from utils.utils import compute_mu, diff_drive_delta_tau

SEED = 1351
EPS = 1e-10

np.random.seed(SEED)


class SimpleController:

    def __init__(self):

        self.ctrl_gains = np.array([10, 50, 10]) /10

    @property
    def k_x(self):
        return self.ctrl_gains[0]

    @property
    def k_y(self):
        return self.ctrl_gains[1]

    @property
    def k_phi(self):
        return self.ctrl_gains[2]

    def __call__(self, state, ref_vel):

        v_ref, _ = ref_vel
        e_x, e_y, e_phi = state
        v = self.k_x * e_x
        w = (
            self.k_y * v_ref * np.sinc(e_phi / np.math.pi) * e_y + self.k_phi * e_phi
        )  # Use sinc to handle e_phi = 0

        return (v, w)


class DiffDrive:
    def __init__(
        self,
        state_interval=None,
        ctrl_interval=None,
        controller: Callable = None,
        init_state=None,
        sampling_time=0.01,
    ):

        self._state_dim = 3
        self._ctrl_dim = 2

        if state_interval is not None:
            assert (
                len(state_interval) == self._state_dim
            ), f"State interval should have {self._state_dim} dimensions"
            self.state_interval = state_interval
        else:
            MAX_X = MAX_Y = 2
            ub = np.array([MAX_X, MAX_Y, 2 * np.pi]).reshape((-1, 1))
            self.state_interval = np.hstack((-ub, ub))
        
        if init_state is not None:
            assert (
                len(init_state) == self._state_dim
            ), f"Initial state should have {self._state_dim} dimensions"
            self.state = init_state
        else:
            self.state = np.random.uniform(
                low=self.state_interval[:, 0], high=self.state_interval[:, 1]
            ).reshape((-1, 1))

        if ctrl_interval is not None:
            assert (
                len(ctrl_interval) == self._ctrl_dim
            ), f"Control interval should have {self._ctrl_dim} dimensions"
            self.ctrl_interval = ctrl_interval
        else:
            MAX_U = 1
            ub = np.array([MAX_U, MAX_U]).reshape((-1, 1))
            self.ctrl_interval = np.hstack((-ub, ub))

        if controller is not None:
            self.controller = controller
        else:
            self.controller = SimpleController()

        self.Ts = sampling_time


    @property
    def state_dim(self):
        return self._state_dim

    def step(self, ref_vel, time_steps=1):
        trace = {"state": [], "control": [], "V": []}
        trace["state"].append(self.state.copy())
        V = (
            self.controller.k_y * 0.5 * (self.state[0] ** 2 + self.state[1] ** 2)
            + 0.5 * self.state[2] ** 2
        )
        trace["V"].append(V)
        v_ref, w_ref = ref_vel
        for _ in range(time_steps):

            v, w = self.controller(self.state, ref_vel)
            e_x, e_y, e_phi = self.state.copy()
            d_x = w_ref * e_y - v + e_y * w
            d_y = -w_ref * e_x + v_ref * np.math.sin(e_phi) - w * e_x
            d_phi = -w

            self.state[0] += d_x * self.Ts
            self.state[1] += d_y * self.Ts
            self.state[2] += d_phi * self.Ts
            V = (
                self.controller.k_y * 0.5 * (self.state[0] ** 2 + self.state[1] ** 2)
                + 0.5 * self.state[2] ** 2
            )
            trace["V"].append(V)
            trace["control"].append([v, w])
            trace["state"].append(self.state.copy())

        trace["state"] = np.array(trace["state"])
        trace["control"] = np.array(trace["control"])
        trace["V"] = np.array(trace["V"])
        return trace

    def compute_Lip_bounds(self, ref_pt):
        # Need to compute bounds on Kx, Ku, Kpsi

        # K_u Lipschitz constant of the dynamics w.r.t u
        # \nabla_u f(x,u) =
        # [-1,   e_y]
        # [ 0,  -e_x]
        # [ 0,    -1]

        ex_ub = max(np.abs(self.state_interval[0, :]))
        ey_ub = max(np.abs(self.state_interval[1, :]))
        # K_u = 2 * max(1, ey_ub, ex_ub)
        K_u = np.sqrt(2 + ey_ub ** 2 + ex_ub ** 2).item()

        # K_x Lipschitz constant of the dynamics w.r.t x
        # \nabla_x f(x,u) =
        # [0,            w_ref + w,         0        ]
        # [-w_ref- w,        0,      v_ref cos(e_phi)]
        # [ 0,               0,              0       ]

        v_ref, w_ref = ref_pt
        a12 = max(np.abs(w_ref + self.ctrl_interval[1, :]))
        # K_x = np.sqrt(3) * max(a12, abs(v_ref)).item()
        K_x = np.sqrt(2*a12 + v_ref**2).item()

        # K_psi depends on control gains, ref point, and bounds on states
        # K_psi Lipschitz constant of the U w.r.t x
        # \nabla_x U =
        # [k_x,              0,                                              0        ]
        # [0,        k_y * v_ref * sin(e_phi)/e_phi,      k_y * v_ref *e_y * (e_phi * cos(e_phi) - sin(e_phi)) / e_phi^2]

        a11 = abs(self.controller.k_x)
        a22 = abs(self.controller.k_y * v_ref)
        a23 = abs(
            self.controller.k_phi
            + self.controller.k_y * v_ref * max(np.abs(self.state_interval[1, :])) * 0.5
        )
        # K_psi = np.sqrt(3) * max(a11, a22, a23).item()
        K_psi = np.sqrt(self.controller.k_x**2 + a22**2 + a23**2).item()

        return (K_x, K_u, K_psi)

    
if __name__ == "__main__":

    q = np.array([1.1, 0.8, 0]).reshape((-1, 1))
    q_ref = np.array([2, 2.4, -0.25]).reshape((-1, 1))
    q_diff = q_ref - q
    # v_ref = np.sqrt(q_diff[0] ** 2 + q_diff[1] ** 2)
    v_ref = 2
    w_ref = 0.1
    ref_vel = (v_ref, w_ref)
    Ts = 0.01
    diff_drive = DiffDrive(init_state=q_diff, sampling_time=Ts)
    time_steps = 3000
    trace = diff_drive.step(ref_vel, time_steps=time_steps)
    plt.plot(Ts * np.arange(len(trace["state"])), trace["state"][:, 0], label="e_x")
    plt.plot(Ts * np.arange(len(trace["state"])), trace["state"][:, 1], label="e_y")
    plt.plot(Ts * np.arange(len(trace["state"])), trace["state"][:, 2], label="e_phi")
    plt.plot(Ts * np.arange(len(trace["V"])), trace["V"], label="V")
    plt.xlabel("Time (s)")
    plt.ylabel("Error")
    plt.legend()
    plt.savefig("error")

    K_x, K_u, K_psi = diff_drive.compute_Lip_bounds(ref_vel)
    print(K_x, K_u, K_psi)

    delta = 0.00001
    epsilon = 1.0
    tau = diff_drive_delta_tau(diff_drive, ref_vel, delta, epsilon)
    mu = compute_mu(diff_drive, (K_x, K_u, K_psi), delta, tau)
    pass
