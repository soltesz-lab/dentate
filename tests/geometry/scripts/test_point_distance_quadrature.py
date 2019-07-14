
import logging
import os
import sys

import numpy as np

import rbf
import rbf.basis
#from dentate.DG_volume import make_volume
from connection_generator import get_volume_distances
from DG_volume import make_volume
from mayavi import mlab
from rbf.interpolate import RBFInterpolant

#import dentate
sys.path.insert(0,'/home/dhadjia1/soltesz-lab/dentate')

logging.basicConfig()
script_name = 'test_point_distances_quadrature.py'
logger = logging.getLogger(script_name)
logger.setLevel(logging.INFO)

def uvl_trajectory(vol, lpos=3.2, ur=1, vr=1):
    U, V = vol._resample_uv(ur, vr)
    L = np.asarray([lpos])
    nupts = U.shape[0]
    nvpts = V.shape[0]

    upts = vol(U,V[int(nvpts/2)],L)
    vpts = vol(U[int(nupts/2)], V, L)
    return upts, vpts

def insert_iso(vol, uvl):

    upts, vpts = uvl_trajectory(vol)
    vol.mplot_surface(color=(0,1,0),opacity=1.0, ures=4,vres=4)
    mlab.points3d(*upts, scale_factor=100.0, color=(1,1,0))
    mlab.points3d(*vpts, scale_factor=100.0, color=(1,1,0))

def pick_points(vol=None, num_points=2, lpos=3.2, ur=1, vr=1):
    if vol is None:
        return
    upts, vpts = uvl_trajectory(vol)
    u = upts[:,:,:]
    return u

def mayavi_plot(): 
    mlab.show()    


def parametric_map(u,v,l):
    x = -500. * np.cos(u) * (5.3 - np.sin(u) + (1. + 0.138 * l) * np.cos(v))
    y = 750. * np.sin(u) * (5.5 - 2. * np.sin(u) + (0.9 + 0.114 * l) * np.cos(v))
    z = 2500. * np.sin(u) + (663. + 114. * l) * np.sin(v - 0.13 * (np.pi - u))
    return x,y,z

def integrand(u, v, l, vol):
    p = np.array([u,v,l])
    p = np.array([p])
    dxdu = vol._xvol(p, diff=np.array([1,0,0]))
    dydu = vol._yvol(p, diff=np.array([1,0,0]))
    dzdu = vol._zvol(p, diff=np.array([1,0,0]))
   
    x,y,z = parametric_map(u,v,l)
    return np.sqrt(dxdu ** 2 + dydu  ** 2 + dzdu ** 2)

def distance(p1, p2):
    u1, v1, l1 = p1
    u2, v2, l2 = p2

    x1, y1, z1 = parametric_map(u1, v1, l1)
    x2, y2, z2 = parametric_map(u2, v2, l2)
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


if __name__ == '__main__':

    logger.info('Making volume')
    ip_vol, xyz = make_volume(-3.95, 3.2, ures=20, vres=20, lres=10) 
    logger.info('Volume created')
    insert_iso(ip_vol, xyz)
    xyz_u = pick_points(vol=ip_vol)
    xyz_u = np.reshape(xyz_u,(3,-1)).T
    print(xyz_u)
    mlab.plot3d(xyz_u[:,0], xyz_u[:,1], xyz_u[:,2], opacity=1.0, color=(1,1,1), tube_radius=10, line_width=10)

    uvl = ip_vol.inverse(xyz_u)
    U = np.asarray(uvl[:,0])
    V = np.asarray([uvl[0,1]])
    L = np.asarray([uvl[0,2]])

    logger.info('...computing U distances')
    distances_u = ip_vol.point_distance(U,V,L,axis=0)
    distances_u = distances_u[0]
    print(distances_u[0])


    p1 = uvl[0,:]
    p2 = uvl[10,:]
    lb, ub = p1[0], p2[0]
    print('Lower bound: %f. Upper bound: %f' % (lb, ub))
    import scipy.integrate as integrate
    from timeit import default_timer as timer
    
    tic = timer()
    y, abserr, infodict = integrate.quad(lambda x: integrand(x, p1[1], p1[2], ip_vol), lb, ub, full_output=1, limit=20)
    toc = timer()
    print('Time to perform adaptive integration: %f' % (toc - tic))
    print(y, abserr)

    K = infodict['last']
    print('Subintervals: %d' % K)
    a_list = np.sort(infodict['alist'][0:K], axis=0)
    b_list = np.sort(infodict['blist'][0:K], axis=0)
    integration_pts = np.array([ [i, p1[1], p1[2]] for i in a_list] + [[b_list[-1], p1[1], p1[2]]])
    #integration_pts = np.sort(integration_pts,axis=0)
    xyz_integration = ip_vol(integration_pts[:,0], integration_pts[0,1], integration_pts[0,2])
    xyz_integration_2 = np.reshape(xyz_integration, (3,-1)).T
    mlab.plot3d(xyz_integration_2[:,0], xyz_integration_2[:,1], xyz_integration_2[:,2], opacity=1.0, color=(0,0,1), line_width=10, tube_radius=10)
    mlab.points3d(*xyz_integration, scale_factor=100.0, opacity=0.5, color=(0,0,1))

    mlab.show()  
