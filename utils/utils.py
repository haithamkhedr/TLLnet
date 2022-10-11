import cvxpy as cp
import numpy as np
import sys
sys.path.append('/home/hkhedr/Haitham/projects/TLLnet/')
# from system_examples.diff_drive import DiffDrive, SimpleController



def optimize(objective, constraints):
    prob = cp.Problem(objective, constraints)
    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    return result

def diff_drive_delta_tau(sys, ref_pt, delta, epsilon):
    
    # Construct optimization variables
    # delta = 1E-3
    # epsilon = 1E-5
    c_outer = 1000
    controller = sys.controller
    v_ref, w_ref = ref_pt
    x = cp.Variable(sys.state_dim)
    state_interval = sys.state_interval
    k_x, k_y, k_phi = controller.k_x, controller.k_y, controller.k_phi

    c_inner = 0.5 * k_y  * (np.sqrt(2 * c_outer / k_y) - delta) ** 2
    # e_y LB in D . E(\epsilon)
    e_y_lb = abs(np.sqrt((2 * c_inner - epsilon**2) / k_y - epsilon**2))

    # e_x\dot lb in D . E(\epsilon)
    e_x_dot_lb =  w_ref * e_y_lb + k_y * v_ref * e_y_lb **2 + k_phi * epsilon * e_y_lb - k_x * epsilon
    tau_prime = epsilon / e_x_dot_lb

    # Max V\dot in D \ E(\epsilon)
    V_dot_max = - (k_x * k_y + k_phi) * (epsilon**2)

    tau = tau_prime + (c_outer - c_inner) / abs(V_dot_max)
    return tau


def compute_mu(sys, Lip, delta, tau):
    (K_x, K_u, K_psi) = Lip
    exponent = (K_x + 2 * K_u * K_psi) * tau
    denominator = exponent + np.log(tau) + np.log(K_u)
    mu = np.exp(np.log(delta) - denominator)
    return mu

# if __name__ == "__main__":
   
#     q = np.array([1.1, 0.8, 0]).reshape((-1, 1))
#     q_ref = np.array([2, 2.4, -0.25]).reshape((-1, 1))
#     q_diff = q_ref - q
#     v_ref = 2
#     w_ref = 0.1
#     ref_vel = (v_ref, w_ref)
#     Ts = 0.01
#     delta = 1E-1
#     epsilon = 1E-2
#     diff_drive = DiffDrive(init_state=q_diff, sampling_time=Ts)
#     diff_drive_delta_tau(diff_drive, ref_vel, delta, epsilon)