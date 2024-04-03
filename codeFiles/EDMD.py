import numpy as np

from utility.utils import reduce_spatial_dim, squaredAverage_rowed


def delay_embed_block(X, nd):
    n, m = X.shape[0], X.shape[1]
    H = np.zeros((nd*n, m-nd+1))
    for j in range(m-nd+1):
        H[:, j] = np.reshape(X[:, j:j+nd], (-1))
    return H


def preparation(X, Y, U, Ntraj, trajLen, nd):
    XX = reduce_spatial_dim(X)
    YY = reduce_spatial_dim(Y)
    nm = XX.shape[0]

    Xtilde, Ytilde, Utilde = [], [], []

    if nd <= 1:
        Xtilde, Ytilde, Utilde = XX, YY, U
    
    else :
        for i in range(Ntraj):
            trajIndex = np.arange(0, trajLen, 1) + i*trajLen
            UIndex = np.arange(nd-1, trajLen, 1) + i*trajLen

            Xe = delay_embed_block(XX[:, trajIndex], nd)
            Ye = delay_embed_block(YY[:, trajIndex], nd)
            Ue = delay_embed_block(U[:, trajIndex], nd-1)
            XXe = XX[:, 0 : trajLen-nd+1]

            if len(Xtilde)==0:
                Xtilde = np.vstack( (Xe, Ue[:, 0 : Ue.shape[1]-1]) )
                Ytilde = np.vstack( (Ye, Ue[:, 1 : Ue.shape[1]]) )
                Utilde = U[:, UIndex]
            else :
                Xtilde = np.hstack( (Xtilde, np.vstack((Xe, Ue[:, 0 : Ue.shape[1]-1]))) )
                Ytilde = np.hstack( (Ytilde, np.vstack((Ye, Ue[:, 1 : Ue.shape[1]]))) )
                Utilde = np.hstack( (Utilde, U[:, UIndex]) )

    z = np.vstack( (Xtilde, squaredAverage_rowed(Xtilde, nd, nm), np.ones((1, Xtilde.shape[1]))) )
    zp = np.vstack( (Ytilde, squaredAverage_rowed(Ytilde, nd, nm), np.ones((1, Ytilde.shape[1]))) )
    u = Utilde

    return [z, zp, u, XX.shape[0]]



def edmd_algorithm(X, Y, U, Ntraj, trajLen, nd):

    z, zp, u, nm = preparation(X, Y, U, Ntraj, trajLen, nd)

    a = np.vstack( (z, u) )
    
    G = a @ a.T
    V = zp @ a.T
    # V = M G
    M = V @ np.linalg.pinv(G)

    r = z.shape[0]

    A = M[:, 0:r]
    B = M[:, r:]
    C = np.zeros((nm, A.shape[0]))
    C[:, (nd-1)*nm : nd*nm] = np.eye((nm))

    print(f"z : {z.shape},  z+ : {zp.shape},  u : {u.shape},  nm : {nm}")

    return [A, B, C]




