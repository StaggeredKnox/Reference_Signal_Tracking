import numpy as np

from helpFDM import get_Ds

def solve_it(initial_state, u, N = 100, s_nt = 20, s_dt = 0.01, s_nu = 0.01):
    X = np.zeros((N, s_nt))  
    X[:, 0] = initial_state
    D, D2 = get_Ds()
    for t in range(1, s_nt):
        v = X[:, t-1]
        R = s_dt*(D @ v)*v + (np.eye(N) - s_dt*s_nu*D2) @ v - X[:, t-1] - s_dt*u
        J = 2*s_dt*(D @ np.diag(v)) + (np.eye(N)-s_dt*s_nu*D2); 
        v = v - np.linalg.solve(J, R);
        X[:, t] = v

    X = X[:, -1]
    return X


""" END """
