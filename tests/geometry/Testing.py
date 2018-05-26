
import os, sys
import numpy as np
import scipy.integrate as integrate
from scipy import interpolate

import rbf
from rbf.interpolate import RBFInterpolant
import rbf.basis

sys.path.insert(0, '/home/dhadjia1/soltesz-lab/dentate')
from DG_volume import make_volume
from connection_generator import get_volume_distances

import Geometry_Suite as geom


class Testing(object):

    def __init__(self, vol, cache):
        self.vol = vol
        self.cache = cache

    def init_test(self, pos1, pos2, dimension):
        trajectory, shape, axis = None, None, None

        if dimension == 'v':
            trajectory, shape = geom.choose_trajectory(self.vol, u=u[pos1], l=l[pos2])
            axis = 1
        elif diension == 'u':
            trajectory, shape = geom.choose_trajectory(self.vol, v=v[pos1], l=l[pos2])
            axis = 0
        trajectory_reshape = np.reshape(trajectory, (3,-1)).T
        uvl_t = self.vol.inverse(trajectory_reshape)
   
        return uvl_t, trajectory, trajectory_reshape, shape, axis

    def euclidean_vanilla_test(self, pos1, pos2, dimension, cache=None):
        if cache is None:
            cache = self.cache

        u, v, l, nu, nv, nl = cache
        uvl_t, trajectory, trajectory_reshape, shape, axis = self.init_test(pos1, pos2, dimension)
        U, V, L = None, None, None
        if dimension == 'v':
            U, V, L = uvl_t[0,0], np.asarray(uvl_t[:,1],dtype=np.float64), uvl_t[0,2]
        elif dimension == 'u':
            U, V, L = np.asarray(uvl_t[:,0], dtype=np.float64), uvl_t[0,1], uvl_t[0,2]
        distances, elapsed_time = geom.distance_euclidean_vanilla(self.vol, U,V,L,axis)
        return distances, distances[0][0][-1], trajectory, elapsed_time

    def euclidean_interpolate_test(self, pos1, pos2, dimension, res=5):
        u, v, l, nu, nv, nl = self.cache
        hru, hrv = self.vol._resample_uv(res, res)
        self.vol.v = hrv
        self.vol.u = hru
        nu *= res
        nv *= res
        cache = (hru, hrv, l, nu, nv, nl)
        d, d1, t, et = self.euclidean_vanilla_test(pos1, pos2, dimension, cache=cache)
        self.vol.u = u
        self.vol.v = v
        return d, d1, t, et
        
        
        
    def quad_test(self,pos1, pos2, dimension):
        
        u, v, l, nu, nv, nl = self.cache
        uvl_t, trajectory, trajectory_reshape, shape, axis = self.init_test(pos1, pos2, dimension)
        
        U, V, L = None, None, None
        lb, ub = None, None
        p1, p2 = None, None

        if dimension == 'v':
            U, V, L = uvl_t[0,0], np.asarray(uvl_t[:,1],dtype=np.float64), uvl_t[0,2]
            lb, ub = V[0], V[-1]
            p1, p2 = np.array([U, lb, L], dtype=np.float64), np.array([U, ub, L], dtype=np.float64)
        elif dimension == 'u':
            U, V, L = np.asarray(uvl_t[:,0], dtype=np.float64), uvl_t[0,1], uvl_t[0,2]
            lb, ub = U[0], U[-1]
            p1, p2 = np.array([lb, V, L], dtype=np.float64), np.array([ub, V, L], dtype=np.float64)
        y, elapsed_time = geom.distance_quadrature(self.vol, p1, p2, dimension, time=True)
        return y, elapsed_time
        

if __name__ == '__main__':
    ures, vres, lres = 20, 20, 10
    print('Making volume')
    ip_vol, xyz = make_volume(-3.95, 3.2, ures=ures, vres=vres, lres=lres)
    print('Volume created')
    u, v, l = ip_vol.u, ip_vol.v, ip_vol.l
    nu, nv, nl = len(u), len(v), len(l)
    cache = (u,v,l,nu,nv,nl)

    testing = Testing(ip_vol, cache)
    y, elapsed_time = testing.quad_test(int(nu/2),int(nl/2),'v')
    print(y, elapsed_time)

    y2, y2_end, traj, elapsed_time = testing.euclidean_vanilla_test(int(nu/2), int(nl/2), 'v')
    print(y2_end, elapsed_time)

    y3, y3_end, traj2, elapsed_time = testing.euclidean_interpolate_test(int(nu/2), int(nl/2), 'v',res=20)
    print(y3_end, elapsed_time)

