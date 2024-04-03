import numpy as np


def fd(N, dd, x, nd):
    Dm = np.zeros((N, N))
    
    for n in range(0, int((dd - 1) / 2)):
        w = weights(x[n], x[0:dd], nd)
        Dm[n, 0:dd] = w[nd, :]
    
    for n in range(int((dd - 1) / 2), N - int((dd - 1) / 2)):
        w = weights(x[n], x[n - int((dd - 1) / 2):n + int((dd - 1) / 2) + 1], nd)
        Dm[n, n - int((dd - 1) / 2):n + int((dd - 1) / 2) + 1] = w[nd, :]
    
    for n in range(N - int((dd - 1) / 2), N):
        w = weights(x[n], x[N - dd:N], nd)
        Dm[n, N - dd:N] = w[nd, :]
    
    return Dm



def weights(z, x, m):
    n = len(x)
    w = np.zeros((m + 1, n))
    c1 = 1
    c4 = x[0] - z
    w[0, 0] = 1
    
    for i in range(1, n):
        mn = min(i+1, m + 1)
        c2 = 1
        c5 = c4
        c4 = x[i] - z
        
        for j in range(0, i):
            c3 = x[i] - x[j]
            c2 *= c3
            
            if j == i - 1:
                w[1:mn, i] = c1 * ((np.arange(1, mn).T * w[0:mn - 1, i - 1]) - c5 * w[1:mn, i - 1]) / c2
                w[0, i] = -c1 * c5 * w[0, i - 1] / c2
                
            w[1:mn, j] = (c4 * w[1:mn, j] - (np.arange(1, mn).T * w[0:mn - 1, j])) / c3
            w[0, j] = c4 * w[0, j] / c3
        
        c1 = c2
    
    return w



def get_Ds(N=100):
    # Parameters
    N = 100
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # Compute D2
    D2 = fd(N, 3, x, 2)

    # Create C matrix
    C = np.zeros((N, N))
    C[0:N-1, 1:N] = np.eye(N - 1)
    C[N-1, 0] = 1

    # Apply boundary conditions to D2
    D2[0, :] = np.dot(D2[1, :], C.T)
    D2[N - 1, :] = np.dot(D2[N - 2, :], C)

    # Compute D
    D = fd(N, 3, x, 1)

    # Apply boundary conditions to D
    D[0, :] = np.dot(D[1, :], C.T)
    D[N - 1, :] = np.dot(D[N - 2, :], C)

    return [D, D2]


""" END """
