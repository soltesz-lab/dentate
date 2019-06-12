import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

import quadpy
import rbf
import rbf.basis
from connection_generator import get_volume_distances
from DG_volume import make_volume
from mayavi import mlab
from rbf.interpolate import RBFInterpolant

sys.path.insert(0,'/home/dhadjia1/soltesz-lab/dentate')

logging.basicConfig()
script_name = 'test_point_distances_quadrature_2.py'
logger = logging.getLogger(script_name)
logger.setLevel(logging.INFO)


def choose_trajectory(vol, u=None, v=None, l=None, ur=1, vr=1):
    U,V,L = None, None, None
    if u is None:
        U = vol.u
        V = np.asarray([v])
        L = np.asarray([l])
    elif v is None:
        U = np.asarray([u])
        V = vol.v
        L = np.asarray([l])
    elif l is None:
        U = np.asarray([u])
        V = np.asarray([v])
        L = vol.l

    pts = vol(U,V,L)
    return pts, pts.shape

def distance_calculations(ip_vol, trajectory_size, trajectories, dimension):
    Nt, _, _ = trajectories.shape
    distances = {}
    for i in range(0, Nt):
        for j in range(0, i):
            uvl_t1 = ip_vol.inverse(trajectories[i,:,:])
            uvl_t2 = ip_vol.inverse(trajectories[j,:,:])

            axis = None
            U1, V1, L1 = None, None, None
            U2, V2, L2 = None, None, None
            if dimension == 'v':
                axis = 1
                U1, V1, L1 = uvl_t1[0,0], np.asarray(uvl_t1[:,1]), uvl_t1[0,2]
                U2, V2, L2 = uvl_t2[0,0], np.asarray(uvl_t2[:,1]), uvl_t2[0,2]
            elif dimension == 'u':
                axis = 0
                U1, V1, L1 = np.asarray(uvl_t1[:,0]), uvl_t1[0,1], uvl_t1[0,2]
                U2, V2, L2 = np.asarray(uvl_t2[:,0]), uvl_t2[0,1], uvl_t2[0,2]

            d1 = ip_vol.point_distance(U1, V1, L1, axis=axis, return_zeros=True)
            d2 = ip_vol.point_distance(U2, V2, L2, axis=axis, return_zeros=True)
            td1, fd1 = d1[0][0][trajectory_size], d1[0][0][-1]
            td2, fd2 = d2[0][0][trajectory_size], d2[0][0][-1]
            nd1, nd2 = float(td1) / fd1, float(td2) / fd2
            diff_nd12 = np.abs(nd2 - nd1)

            distances[(i,j)] = (td1, fd1, nd1, td2, fd2, nd2, diff_nd12)

    return distances          

def integrand(x, *args):/
    a1, a2, vol, spat_dev, axis = args

    p = None
    if axis == 1:
        u, v, l = a1, x, a2
        for vi in v:
            p = [np.asarray([u,v0,l]) for v0 in vi]
    elif axis == 0:
        u, v, l = x, a1, a2
        p = [np.asarray([u1,v,l]) for u1 in u[0]]
    p = np.asarray(p)
    print(v)

	
    #dxdu = vol._xvol(p, diff=spat_dev)
    #dydu = vol._yvol(p, diff=spat_dev)
    #dzdu = vol._zvol(p, diff=spat_dev)

    #return np.sqrt(dxdu ** 2 + dydu ** 2 + dzdu ** 2)

    

def plot_volume(vol):
    vol.mplot_surface(color=(0,1,0), opacity=1.0, ures=4, vres=4)

def plot_trajectory(pts):
    mlab.points3d(*pts, scale_factor=100.0, color=(1,1,0))

def plot_plot3d(points):
    mlab.plot3d(points[:,0], points[:,1], points[:,2], opacity=1.0, color=(0,0,1), line_width=10, tube_radius=10)

def show_mayavi():
    mlab.show()

def distance_quadrature(vol, p1, p2, axis, epsabs=1.49e-08, epsrel=1.49e-08):
    lb = p1[axis]
    ub = p2[axis]
  
    print(p1)
    print(p2)
  
    y = None
    if axis == 0:
        spat_dev = np.array([1,0,0])
        y = integrate.fixed_quad(lambda x: integrand(x, p1[1], p1[2], vol, spat_dev), lb, ub, n=2)
    elif axis == 1:
        spat_dev = np.array([0,1,0])
        args = (p1[1], p1[2], vol, spat_dev, axis)
def show_mayavi():
    mlab.show()

def distance_euclidean_vanilla(ip_vol, cache, dimension=None):
    if dimension is None:
        return None, None

    u, v, l, nu, nv, nl = cache
    U, V, L = None, None, None
    if dimension == 'v':
        axis = 1
        ttest, trajectory_shape = choose_trajectory(ip_vol, u=u[int(nu/2)], l=l[int(nl/2)])
    elif dimension == 'u':
        axis = 0
        ttest, trajectory_shape = choose_trajectory(ip_vol, v=v[int(nv/2)], l=l[int(nl/2)])
    ttest_reshape = np.reshape(ttest, (3,-1)).T
    uvl_t = ip_vol.inverse(ttest_reshape)

    if dimension == 'v':
        U, V, L = uvl_t[0,0], np.asarray(uvl_t[:,1]), uvl_t[0,2]
    elif dimension == 'u':
        U, V, L = np.asarray(uvl_t[:,0]), uvl_t[0,1], uvl_t[0,2]

    distances = ip_vol.point_distance(U, V, L, axis=axis, return_zeros=True)
    return distances, distances[0][0][-1], ttest

def distance_euclidean_highres(ip_vol, trajectory, res, dimension=None):
    if dimension is None or res <= 0:
        return None, None

    print(trajectory.shape)

def distance_quadrature(vol, p1, p2, axis, epsabs=1.49e-08, epsrel=1.49e-08):
    lb = p1[axis]
    ub = p2[axis]
  
    print(p1)
    print(p2)
  
    y = None
    if axis == 0:
        spat_dev = np.array([1,0,0])
        y = integrate.fixed_quad(lambda x: integrand(x, p1[1], p1[2], vol, spat_dev), lb, ub, n=2)
    elif axis == 1:
        spat_dev = np.array([0,1,0])
        args = (p1[1], p1[2], vol, spat_dev, axis)
        #y = quadpy.line_segment.integrate_adaptive(lambda x: integrand(x, *args), [lb, ub], 1e-3)
        y = integrate.quadrature(integrand, lb, ub, args=(p1[0], p1[2], vol, spat_dev, axis), tol=epsabs, rtol=epsrel)
    elif axis == 2:
        spat_dev = np.array([0,0,1])
        y = integrate.fixed_quad(lambda x: integrand(p1[0], p1[1], x, vol, spat_dev), lb, ub, n=2)
    return y

def trajectory_calculations(ip_vol, cache, dimension=None):
    if dimension is None:
        return None

    u, v, l, nu, nv, nl = cache
    trajectories, Nx = None, None
    if dimension == 'v':
        trajectories = [ [] for _ in np.arange(nu)]
        Nx = nu
    elif dimension == 'u':
        trajectories = [ [] for _ in np.arange(nv)]
        Nx = nv
    trajectory_shape = None

    for i in range(0, int(lres)):
        for j in range(0, Nx):
            if dimension == 'v':
                trajectory, trajectory_shape = choose_trajectory(ip_vol, u=u[j], l=l[i])
            elif dimension == 'u':
                trajectory, trajectory_shape = choose_trajectory(ip_vol, v=v[j], l=l[i])
            #plot_trajectory(trajectory)
            trajectory_reshape = np.reshape(trajectory, (3,-1)).T
            trajectories[j].append(trajectory_reshape)
 
    trajectories = np.asarray(trajectories)
    Nx, Nt, Np, D = trajectories.shape
    trajectory_size = np.arange(Np)
    
    #trajectory_points = np.zeros((Nu, Nt, trajectory_size, D))
    #for i in range(0, Nu):
    #    for j in range(0, Nt):
    #        trajectory_points[i,j,:,:] = trajectories[i,j,0:trajectory_size,:]
    #        #plot_plot3d(trajectory_points[i,j,:,:])

    for ts in trajectory_size:
        distances = [ {} for _ in np.arange(Nx) ]
        for upos in range(0,Nx): #Nx
            print('%d out of %d u points complete for ts: %d' % (upos+1,Nx,ts))
            distances[upos] = distance_calculations(ip_vol, ts, trajectories[upos,:,:,:], dimension)
        
        distance_matrix = np.zeros((Nx, Nt, Nt))
        for d in range(0,len(distances)):    
            for (i,j) in distances[d]:
                td1, fd1, nd1, td2, fd2, nd2, diff_nd12 = distances[d][(i,j)]
                distance_matrix[d][i][j] = diff_nd12 
        distance_matrix = np.asarray(distance_matrix)
        
        save_fn = 'ts-'+str(ts+1)
        write_files(save_fn, distance_matrix, Nx, dimension)
    return distance_matrix

def write_files(save_fn, distance_matrix, Nx, dimension):

        save_fn_txt = save_fn + '.txt'
        f = open(save_fn_txt, 'w')
        fig = plt.figure()
        pos = 1
        for i in range(0, Nx):
            plt.subplot(5,4,pos)
            pos += 1
            curr = distance_matrix[i,:,:]
            for x in range(0, curr.shape[0]):
                for y in range(0, curr.shape[1]):
                    f.write('%0.3f\t' % curr[x,y])
                f.write('\n')
            f.write('###################\n')
            plt.imshow(distance_matrix[i,:,:])
            plt.axis('off')
            if dimension == 'v':
                plt.title('u = %0.3f' % u[i])
            elif dimension == 'u':
                plt.title('v = %0.3f' % v[i])
            plt.colorbar()
            save_fn_png = save_fn + '.png'
            plt.savefig(save_fn_png)
        f.close()

def quad_test(ip_vol, cache, dimension=None):
    if dimension is None:
        return None, None, None

    u, v, l, nu, nv, nl = cache

    ttest, trajectory_shape, axis = None, None, None
    if dimension == 'v':
        ttest, trajectory_shape = choose_trajectory(ip_vol, u=u[int(nu/2)], l=l[int(nl/2)])
        axis = 1
    elif dimension == 'u':
        ttest, trajectory_shape = choose_trajectory(ip_vol, v=v[int(nv/2)], l=l[int(nl/2)])
        axis = 0
    ttest_reshape = np.reshape(ttest, (3,-1)).T
    uvl_t = ip_vol.inverse(ttest_reshape)
    
    U, V, L = None, None, None
    lb, ub = None, None
    p1, p2 = None, None
    if dimension == 'v':
        U, V, L = uvl_t[0,0], np.asarray(uvl_t[:,1]), uvl_t[0,2]
        print(V)
        lb, ub = V[0], V[-1]
        p1, p2 = np.array([U, lb, L]), np.array([U, ub, L])
    elif dimension == 'u':
        U, V, L = np.asarray(uvl_t[:,0]), uvl_t[0,1], uvl_t[0,2]
        lb, ub = U[0], U[-1]
        p1, p2 = np.array([lb, V, L]), np.array([ub, V, L])


    import time
    tic = time.time()
    y = distance_quadrature(ip_vol, p1, p2, axis, epsabs=1.5e-3, epsrel=1.5e-3)
    toc = time.time()
    print('Time to calculate distance using quadrature: %0.3f' % (toc - tic))

    tic = time.time()
    euclidean_distance = ip_vol.point_distance(U, V, L, axis=axis, return_zeros=True)
    toc = time.time()
    print('Time to calculate distance using euclidean method: %0.3f' % (toc - tic))

    return y, euclidean_distance[0][0], ttest

if __name__ == '__main__':


    logger.info('Making volume...')
    ures, vres, lres = 20, 20, 10
    ip_vol, xyz = make_volume(-3.95, 3.2, ures=ures, vres=vres, lres=lres)
    logger.info('Volume created...')

    u,v,l = ip_vol.u, ip_vol.v, ip_vol.l
    nu, nv, nl = len(u), len(v), len(l)
    cache = (u, v, l, nu, nv, nl)

    #y, euclidean_distance, trajectory = quad_test(ip_vol, cache, dimension='v')
    #print('Distance calculated by gaussian quad: %0.3f' % y[0])
    #print('GQuad error: %f' % y[1])
    #print('Distance calculated by euclidean distance: %0.3f' % euclidean_distance[-1])

    distance_all, distance, trajectory = distance_euclidean_vanilla(ip_vol, cache, dimension='v')
    print('Distance from end to end is: %0.3f' % distance)
    plot_volume(ip_vol)
    plot_trajectory(trajectory)
    mlab.show()
    sys.exit(1)

    
    #ttest1 = np.reshape(ttest1, (3,-1)).T
    #ttest2  = np.reshape(ttest2, (3,-1)).T
    #uvl_t1 = ip_vol.inverse(ttest1)
    #uvl_t2 = ip_vol.inverse(ttest2)
 
    #U1, V1, L1 = uvl_t1[0,0], np.asarray(uvl_t1[:,1]), uvl_t1[0,2]
    #U2, V2, L2 = uvl_t2[0,0], np.asarray(uvl_t2[:,1]), uvl_t2[0,2]
    #d1 = ip_vol.point_distance(U1, V1, L1, axis=1, return_zeros=True)
    #d2 = ip_vol.point_distance(U2, V2, L2, axis=1, return_zeros=True)
    #td1, fd1 = d1[0][0][5], d1[0][0][-1]
    #td2, fd2 = d2[0][0][5], d2[0][0][-1]
    #nd1, nd2 = float(td1) / fd1, float(td2) / fd2
    #print(np.abs(nd1 - nd2))
    #print(td1, fd1, nd1, td2, fd2, nd2)
    #mlab.show()

    
  
    # Using middle u coordinate 
    #plot_volume(ip_vol)
    #v_distance_matrix = trajectory_calculations(ip_vol, cache, dimension='v')
    u_distance_matrix = trajectory_calculations(ip_vol, cache, dimension='u')

    trajectories = [ [] for _ in np.arange(nu)]
    trajectory_shape = None
    for i in range(0, int(lres)):
        for j in range(0, nu):
            trajectory, trajectory_shape = choose_trajectory(ip_vol, u=u[j], l=l[i])
            #plot_trajectory(trajectory)
            trajectory_reshape = np.reshape(trajectory, (3,-1)).T
            trajectories[j].append(trajectory_reshape)
 
    trajectories = np.asarray(trajectories)
    Nu, Nt, Np, D = trajectories.shape
    trajectory_size = np.arange(Np)
