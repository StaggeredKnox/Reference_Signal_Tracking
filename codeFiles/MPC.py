"""
System Dynamics (Koopman Linear System)

z+ = A.z + B.u
y = C.z
"""

import numpy as np
from casadi import *
import casadi as ca

from config.cfg import config_vars as cfg


def mpc(A, B, C, uref):
    n = A.shape[0]
    r = C.shape[0]
    nu = B.shape[1]

    prediction_horizon = cfg["prediction_horizon"]
    time_step = cfg["time_step"]
    N = int(np.round(prediction_horizon/time_step))

    z = ca.SX.sym('z', n, 1)
    u = ca.vertcat(ca.SX.sym('u1', 1, 1), ca.SX.sym('u2', 1, 1))

    zp = (A @ z) + (B @ u)
    y = C @ z
    f = ca.Function('get_zp', [z, u], [zp])
    ff = ca.Function('get_y', [z], [y])

    # y = C @ ((A @ z) + (B @ u))
    # f = ca.Function('f', [z, u], [y])


    P = ca.SX.sym('P', n+r)   # P = [initial_state_z, refrence_signal or reference_state]
    X = ca.SX.sym('X', n, N+1)
    U = ca.SX.sym('U', nu, N)

    obj = 0
    g = []
    Q = np.eye(r)
    R = np.eye(nu)

    st_z = X[:, 0]
    st = ff(st_z)
    g.append(st-P[:r])
    yr = P[-r:]
    for k in range(N):
        st_z = X[:, k]
        st = ff(st_z)
        con = U[:, k]
        obj += (st.T @ Q @ st) + (con.T @ R @ con) - 2*(yr.T @ Q @ st) - 2*(uref.T @ R @ con)
        st_z_next = X[:, k+1]
        st_next = ff(st_z_next)
        st_z_next_pred = f(st_z, con)
        st_next_pred = ff(st_z_next_pred)
        g.append(st_next-st_next_pred)

    OPT_variables = ca.vertcat(ca.reshape(X, n*(N+1), 1), ca.reshape(U, nu*N, 1))

    nlp_prob = {'f': obj, 'x': OPT_variables, 'g':ca.vertcat(*g), 'p':P}

    opts_setting = {'ipopt.max_iter':200, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6 }
    
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

    return [solver, Q]


""" END """
