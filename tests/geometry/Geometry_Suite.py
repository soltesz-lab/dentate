import numpy as np
import scipy.integrate as integrate

import rbf
import rbf.basis
from rbf.interpolate import RBFInterpolant


def get_x(vol, p, diff=None):
    return vol._xvol(p, diff=diff)
def get_y(vol, p, diff=None):
    return vol._yvol(p, diff=diff)
def get_z(vol, p, diff=None):
    return vol._zvol(p, diff=diff)
def get_xyz(vol, p, diff=None):
    return get_x(vol, p, diff), get_y(vol, p, diff), get_z(vol, p, diff)

def integrand(x, *args):
    a1, a2, vol, axis = args
    p = None
    if axis == 1:
        p = np.asarray([a1, x, a2], dtype=np.float64)
    elif axis == 0:
        p = np.asarray([x, a1, a2], dtype=np.float64)
    p = np.array([p], dtype=np.float64)
    diff = np.array([1,0,0]) if axis == 0 else np.array([0,1,0])
    dx, dy, dz = get_xyz(vol, p, diff)
    dxyz = np.array([dx, dy, dz], dtype=np.float64)
    g11 = np.sum(np.multiply(dxyz,dxyz))
    return np.sqrt(g11)

def distance_quadrature(vol, p1, p2, dimension, time=False):
    if dimension == 'u':
        axis = 0
    elif dimension == 'v':
        axis = 1

    lb, ub = p1[axis], p2[axis]
    y,args = None, None
    if axis == 0:
        args = (p1[0], p1[2], vol, axis)
    elif axis == 1:
        args = (p1[1], p1[2], vol, axis)

    import time
    tic = time.time()
    y = integrate.quad(integrand, lb, ub, args=args)
    toc = time.time()
    elapsed_time = toc - tic
    if time:
        return y, elapsed_time
    else:
        return y, None

def distance_euclidean_vanilla(vol, U, V, L, axis):
    import time
    tic = time.time()
    y = vol.point_distance(U, V, L, axis=axis, return_zeros=True)
    toc = time.time()
    elapsed_time = toc - tic
    return y, elapsed_time
    

def choose_trajectory(vol, u=None, v=None, l=None, ur=1, vr=1):
    U, V, L = None, None, None
    if u is None:
        U = vol.u
        V = np.asarary([v],dtype=np.float64)
        L = np.asarray([l],dtype=np.float64)
    elif v is None:
        U = np.asarray([u],dtype=np.float64)
        V = vol.v
        L = np.asarray([l],dtype=np.float64)
    elif l is None:
        U = np.asarray([u],dtype=np.float64)
        V = np.asarray([v],dtype=np.float64)
        L = vol.l
    pts = vol(U,V,L)
    return pts, pts.shape
