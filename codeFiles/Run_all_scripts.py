import numpy as np
import numpy.matlib
import scipy.io
import matplotlib.pyplot as plt

from config.cfg import config_vars as cfg

from utility.utils import reduce_spatial_dim, build_koopman_state

from EDMD import edmd_algorithm as edmd

from MPC import mpc

data = scipy.io.loadmat(cfg["path"])
X, Y, U = data["X"], data["Y"], data["U"]
print(X.shape, Y.shape, U.shape)

keylist = [key for key in data]
print(keylist)

N = cfg["N"]
Ntraj = cfg["Ntraj"]
trajLen = cfg["trajLen"]
nd = cfg["nd"]

A, B, C = edmd(X, Y, U, Ntraj, trajLen, nd)
print(f"A : {A.shape},  B : {B.shape},  C : {C.shape}")

# dat = scipy.io.loadmat(cfg["path2"])
# A, B, C = dat["A"], dat["B"], dat["C"]

def get_ref_control(Y_ref, A, B, C):
    U_ref = np.linalg.pinv(B) @ ( (np.linalg.pinv(C) @ Y_ref) - (A @ (np.linalg.pinv(C) @ Y_ref)) )
    return U_ref


"""CONSTRAINTS"""

Np = int(np.round(cfg["prediction_horizon"]/cfg["time_step"]))
n = A.shape[1]  # from first equation of dynamics [ zp = A.z + B.u ] 
nu = B.shape[1]

umin, umax = -0.1, 0.1
lbg, ubg = 0, 0
lbx, ubx = [], []

for i in range(n*(Np+1)):
    lbx.append(-np.inf)
    ubx.append(np.inf)

for i in range(nu*Np):
    lbx.append(umin)
    ubx.append(umax)


"""SIMULATION SETUP"""

Tsim = cfg["Tsim"]
sim_dt = cfg["sim_dt"]
Nsim = Tsim/sim_dt


x = np.linspace(0, 1, N)
ic1 = np.exp((-1)*((x-0.5)*5)**2)
ic2 = (np.sin(4*np.pi*x))**2
a = 0.2
ic = a*ic1 + (1-a)*ic2


# Refrence signal (could also be changed)

# x_ref = np.ones(int(Nsim))*(-0.2)
# for i in range(int(Nsim/6)):
#     x_ref[i] = 0.75
#     x_ref[i+int(Nsim/6)] = 0.5
#     x_ref[i+int(2*Nsim/6)] = 0.25
#     x_ref[i+int(3*Nsim/6)] = 0.40
#     x_ref[i+int(4*Nsim/6)] = 0

# X_ref = np.ones((N, 1))*x_ref


x_ref = np.ones(int(Nsim))*0.5
for i in range(int(Nsim/3)):
    x_ref[i+int(Nsim/3)] = 1

X_ref = np.ones((N, 1))*x_ref


"""SIMULATION LOOP"""

from solveNumericallyFDM import solve_it

f1 = np.exp(-(((x)-.25)*15)**2);
f2 = np.exp(-(((x)-.75)*15)**2);

U = np.zeros((nu, nd))
X = np.zeros((N, nd+1))
X[:, 0] = ic

for i in range(nd):
    X[:, i+1] = solve_it(X[:, i], U[:, i][0]*f1+U[:, i][1]*f2)

output_error = []

plt.ion()
figure, axis = plt.subplots(figsize=(10, 6))

# setup for ploting refrence signal and controlled trajectory
line1, = axis.plot(x, np.linspace(0, 1, N), '.')
line2, = axis.plot(x, np.linspace(0, 1, N), linewidth=2.5)

plt.title("refrence_signal / trajectory (evolution wrt time)", fontsize=10)
 
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# setup for ploting control input
figure2, axis2 = plt.subplots(figsize=(10, 6))

line1_2, = axis2.plot(x, np.linspace(umin, umax, N), 'r', linewidth=2.5)
line2_2, = axis2.plot(x, np.linspace(umin, umax, N), 'g', linewidth=2.5)

plt.title("control inputs (evolution wrt time)", fontsize=10)
 
plt.xlabel("X-axis")
plt.ylabel("control i/p value")

Y_ref = reduce_spatial_dim(X_ref)

init_control = np.reshape(np.zeros(nu*Np), (-1, 1))
for i in range(int(Nsim)):
    yref = Y_ref[:, i]

    a = yref-reduce_spatial_dim(X[:, -1])
    uref = np.array([2*umax*np.sum(a[:5]), 2*umax*np.sum(a[5:])])

    z = build_koopman_state(X, U, nd)

    P = np.vstack( (np.reshape(z, (-1, 1)), np.reshape(yref, (-1, 1))) )

    init_state = np.reshape(numpy.matlib.repmat(z, 1, Np+1), (-1, 1))
    init = np.vstack( (init_state, init_control) )
    solver, Q = mpc(A, B, C, np.reshape(uref, (-1, 1)))

    res = solver(x0=init, p=P, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)

    estimated_opt = res['x'].full()

    u = np.reshape(estimated_opt[-nu*Np:], (nu, Np))

    X = np.hstack( (X, np.reshape(solve_it(X[:, -1], u[:, 0][0]*f1+u[:, 0][1]*f2), (-1, 1))) )
    U = np.hstack( (U, np.reshape(u[:, 0], (-1, 1))) )

    init_control[:Np*nu-nu] = np.reshape(u[:, 1:], (-1, 1))

    # trajectory plot
    line1.set_xdata(x)
    line1.set_ydata(X_ref[:, i])

    line2.set_xdata(x)
    line2.set_ydata(X[:, -1])

    figure.canvas.draw()
    figure.canvas.flush_events()

    # control inputs plot
    line1_2.set_xdata(x)
    line1_2.set_ydata(u[:, 0][0]*f1)

    line2_2.set_xdata(x)
    line2_2.set_ydata(u[:, 0][1]*f2)

    figure2.canvas.draw()
    figure2.canvas.flush_events()

    # ynow = reduce_spatial_dim(X[:, -1])
    # output_error.append( (ynow-yref).T @ Q @ (ynow-yref) )

    print(str(u[:, 0])+"  control input for actual system   ---   "+str((i+1)*100/Nsim)+" % Completed ...", end='\r')


""" END """




