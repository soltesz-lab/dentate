
import os
import sys
import logging
import numpy as np
from mayavi import mlab
import scipy.integrate as integrate
import matplotlib.pyplot as plt

import rbf
from rbf.interpolate import RBFInterpolant
import rbf.basis

sys.path.insert(0,'/home/dhadjia1/soltesz-lab/dentate')
from DG_volume import make_volume
from connection_generator import get_volume_distances

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

def integrand(u, v, l, vol, spat_dev):
    p = np.array([u,v,l])
    p = np.array([p])
    dxdu = vol._xvol(p, diff=spat_dev)
    dydu = vol._yvol(p, diff=spat_dev)
    dzdu = vol._zvol(p, diff=spat_dev)

    return np.sqrt(dxdu ** 2 + dydu ** 2 + dzdu ** 2)

def plot_volume(vol):
    vol.mplot_surface(color=(0,1,0), opacity=1.0, ures=4, vres=4)

def plot_trajectory(pts):
    mlab.points3d(*pts, scale_factor=100.0, color=(1,1,0))

def plot_plot3d(points):
    mlab.plot3d(points[:,0], points[:,1], points[:,2], opacity=1.0, color=(0,0,1), line_width=10, tube_radius=10)

def show_mayavi():
    mlab.show()

def distance_quadrature(vol, p1, p2, axis):
    lb = p1[axis]
    ub = p2[axis]
  
    y = None
    if axis == 0:
        spat_dev = np.array([1,0,0])
        y = integrate.quad(lambda x: integrand(x, p1[1], p1[2], vol, spat_dev), lb, ub, full_output=0)
    elif axis == 1:
        spat_dev = np.array([0,1,0])
        y = integrate.quad(lambda x: integrand(p1[0], x, p1[2], vol, spat_dev), lb, ub, full_output=0)
    elif axis == 2:
        spat_dev = np.array([0,0,1])
        y = integrate.quad(lambda x: integrand(p1[0], p1[1], x, vol, spat_dev), lb, ub, full_output=0)
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
            for (i,j) in distances[d].keys():
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


if __name__ == '__main__':


    logger.info('Making volume...')
    ures, vres, lres = 20, 20, 10
    ip_vol, xyz = make_volume(-3.95, 3.2, ures=ures, vres=vres, lres=lres)
    logger.info('Volume created...')

    u,v,l = ip_vol.u, ip_vol.v, ip_vol.l
    nu, nv, nl = len(u), len(v), len(l)
    cache = (u, v, l, nu, nv, nl)
    #ttest1,_ = choose_trajectory(ip_vol, u=u[int(nu/2)], l=l[0])
    #ttest2,_ = choose_trajectory(ip_vol, u=u[int(nu/2)], l=l[-1])
    #plot_volume(ip_vol)
    #plot_trajectory(ttest1)
    #plot_trajectory(ttest2)
    
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
    sys.exit(1)

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
    
    #trajectory_points = np.zeros((Nu, Nt, trajectory_size, D))
    #for i in range(0, Nu):
    #    for j in range(0, Nt):
    #        trajectory_points[i,j,:,:] = trajectories[i,j,0:trajectory_size,:]
    #        #plot_plot3d(trajectory_points[i,j,:,:])

    import matplotlib.pyplot as plt
    for ts in trajectory_size:
        distances = [ {} for _ in np.arange(Nu) ]
        for upos in range(0,Nu): #Nu
            print('%d out of %d u points complete for ts: %d' % (upos+1,Nu, ts))
            distances[upos] = distance_calculations(ip_vol, ts, trajectories[upos,:,:,:])
        
        distance_matrix = np.zeros((Nu, Nt, Nt))
        for d in range(0,len(distances)):    
            for (i,j) in distances[d].keys():
                td1, fd1, nd1, td2, fd2, nd2, diff_nd12 = distances[d][(i,j)]
                distance_matrix[d][i][j] = diff_nd12 
        distance_matrix = np.asarray(distance_matrix)

        
        save_fn = 'ts-'+str(ts+1)
        save_fn_txt = save_fn + '.txt'
        f = open(save_fn_txt, 'w')
        fig = plt.figure()
        pos = 1
        for i in range(0, Nu):
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
            plt.title('u = %0.3f' % u[i])
            plt.colorbar()
            save_fn_png = save_fn + '.png'
            plt.savefig(save_fn_png)
        f.close()

           

