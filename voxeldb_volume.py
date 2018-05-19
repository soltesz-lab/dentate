
import numpy as np
import rbf
from rbf.interpolate import RBFInterpolant
from rbf.nodes import snap_to_boundary,disperse,menodes
from rbf.geometry import contains
import dentate
from dentate.alphavol import alpha_shape
from dentate.rbf_volume import RBFVolume
import h5py
from mayavi import mlab


#def make_volume(input_path, ns='Hippocampus', basis=rbf.basis.phs3, res=15):
input_path = './datasets/Hippocampus_voxeldb.h5'
ns  =  'Hippocampus'
ures =  500
vres =  200
lres =  100
f   =  h5py.File(input_path)

pty = f['/voxeldb/%s/Type' % ns][:]
    
idxs = np.isin(pty, [1,2,3,4,5,6])
print 'idxs shape: ', idxs.shape    
X = f['/voxeldb/%s/X' % ns][idxs]
Y = f['/voxeldb/%s/Y' % ns][idxs]
Z = f['/voxeldb/%s/Z' % ns][idxs]

obs_u = np.linspace(1, X.size-1, ures, dtype=np.int)
obs_v = np.linspace(1, Y.size-1, vres, dtype=np.int)
obs_l = np.linspace(1, Z.size-1, lres, dtype=np.int)

u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
obs_uvl = np.array([u.ravel(),v.ravel(),l.ravel()]).T
obs_x = X[obs_uvl[:,0]]
obs_y = Y[obs_uvl[:,1]]
obs_z = Z[obs_uvl[:,2]]
obs_xyz = np.array([obs_x, obs_y, obs_z]).reshape(3, u.size).T
#s = np.linspace(0., 1., obs_u.size)
#obs_uvl = np.array([s],dtype=np.float32).T

#print obs_uvl
#print X[obs_u]

#xvol = RBFInterpolant(obs_uvl,X[obs_u],basis='imq',penalty=0.1,extrapolate=False)

#mlab.mesh(X[obs_u], Y[obs_v], Z[obs_l], representation='wireframe', color=(0, 0, 0))
#mlab.points3d(X[obs_u], Y[obs_v], Z[obs_l], scale_factor=100.0, color=(1, 1, 0))


    
vol = RBFVolume(np.asarray(obs_u.ravel(), dtype=np.float32), \
                np.asarray(obs_v.ravel(), dtype=np.float32), \
                np.asarray(obs_l.ravel(), dtype=np.float32), \
                obs_xyz, basis='imq', order=1)


#xyz = make_volume()

#vol.mplot_surface(color=(0, 1, 0), ures=2, vres=2, opacity=0.33)

#xyz = np.array([X[obs_u].ravel(), Y[obs_v].ravel(), Z[obs_l].ravel()]).reshape(3, obs_u.size).T

#mlab.points3d(obs_xyz[:,0], obs_xyz[:,1], obs_xyz[:,2], scale_factor=100.0, color=(1, 1, 0))

#mlab.show()
