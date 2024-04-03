import numpy as np


def reduce_spatial_dim(X, start=9, gap=10):
    return X[start : X.shape[0]+1 : gap]


def squaredAverage_rowed(X, nd, nm):
    return np.sum(np.abs(X[(nd-1)*nm : nd*nm, :]**2), axis=0)/nm


def delay_embed(X, nd):
    return np.reshape(X[:, -nd:], (-1, 1))


def create_zeta(X, U, nd):
    x = reduce_spatial_dim(X)
    xe = delay_embed(x, nd)
    u = delay_embed(U[:, -nd:-1], nd-1)
    return np.vstack( (xe, u) )


def build_koopman_state(X, U, nd):
    x = reduce_spatial_dim(X[:, -1])
    gx = np.sum(np.abs(x)**2)/len(x)
    z = np.vstack( ( create_zeta(X, U, nd), np.reshape(gx, (-1, 1)), np.ones((1, 1)) ) )
    return z


""" END """
