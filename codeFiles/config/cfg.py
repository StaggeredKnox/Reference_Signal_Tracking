config_vars = {

    "path" : 'C:\\Users\\premsha73866453\\Desktop\\Burgers_pde\\Data\\BurgersTrajectoryData.mat',

    "path2" : 'C:\\Users\\premsha73866453\\Desktop\\Burgers_pde\\codeFiles\\koopman_matrices.mat',
    

    "N": 100,  # spatial points

    "Ntraj": 100,  # number of trajectories (in Dataset)

    "trajLen": 200,  # length of a trajectory

    "nd": 5,  # delay embedding param



   # """ ------ RECEDING HORIZON PARAMS (for MPC) ------""" #

    "prediction_horizon": 0.15,

    "time_step": 0.05,

    # note: number of future predictions for MPC is (prediction_horizon / time_step) #



   # """ ------ SIMULATION SETUP PARAMS (for MPC) ------""" #

    "Tsim": 6,  # simulation time

    "sim_dt": 0.01,  # steps taken to complete Tsim
   
}


"""
NOTE 1: 

Koopman linear system dynamics equations -
z+ = A.z + B.u
x = C.z

z+ and zp are just the same.
"""


"""
NOTE 2:

In EDMD.py script
dynamics equations
z+ = A.z + B.u
x = C.z
y = D.z+

but last equation is never used.
"""


"""
NOTE 3: data matrices are X, Y and U

       X -> (100, 200*100)   [row - spatial dimension (N in our case)] 
       Y -> (100, 200*100)   [col - each col is 20th time step solution for single trajectory with single corres. input sequence]
       for X, Y for col - first 200 cols are for first trajectory(which means first and single initial condition)
       within those 200 columns the trajectory is evolving wrt control input.

       U -> (2, 200*100) 
"""


""" END """
